"""Exercise the production CPU scheduler with controlled native work."""

import asyncio
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import patch

from app.core.config import settings
from app.services.asr.engines import ASRSegmentResult, WordToken
from app.services.asr.long_audio import (
    OfflineASRRequest,
    PreparedLongAudio,
    prepare_long_audio,
)
from app.services.asr.runtime.router import (
    RuntimeFamily,
    RuntimeRouter,
)
from app.utils.audio_splitter import AudioSegment, AudioSplitter


class CPUSegmentSchedulingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.created = []
        self.finished = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.addCleanup(self.release.set)
        self.calls = []
        self.segment_count = 4
        self.operation = lambda path: Path(path).name
        owner = self

        class Engine:
            def __init__(self) -> None:
                self.busy = threading.Lock()
                self.realtime = False
                owner.created.append(self)

            def transcribe_segments(
                self, segments: list[AudioSegment], **kwargs: object
            ) -> list[ASRSegmentResult]:
                assert not self.realtime, "Offline work used a realtime lease"
                assert self.busy.acquire(blocking=False), "Concurrent native use"
                try:
                    path = segments[0].temp_file
                    owner.calls.append(path)
                    text = owner.operation(path)
                    assert Path(path).exists(), "Input was removed during native work"
                    return [ASRSegmentResult(text=text, start_time=0, end_time=1)]
                finally:
                    self.busy.release()

        @contextmanager
        def prepare(audio_path: str, *args: object) -> Iterator[PreparedLongAudio]:
            with tempfile.TemporaryDirectory(dir=self.directory.name) as directory:
                segments = []
                for index in range(self.segment_count):
                    path = Path(directory) / f"{audio_path}-{index}"
                    path.touch()
                    segments.append(
                        AudioSegment(
                            index * 1000, (index + 1) * 1000, temp_file=str(path)
                        )
                    )
                yield PreparedLongAudio(segments, 4.0)
            self.finished.append(audio_path)

        patches = [
            patch("app.services.asr.runtime.router.get_model_manager"),
            patch("app.services.asr.long_audio.prepare_long_audio", prepare),
            patch.object(settings, "QWEN_RUST_CPU_WORKERS", 2),
            patch.object(settings, "ASR_BATCH_SIZE", 4),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.router = RuntimeRouter()
        self.router._manager.create_engine.side_effect = lambda _: Engine()
        self.router._resolve_family = lambda _: RuntimeFamily.QWEN_RUST_CPU

    async def run_request(self, name: str = "audio") -> object:
        return await self.router.run_offline(OfflineASRRequest("model", name))

    async def test_single_request_uses_full_budget_and_preserves_order(self) -> None:
        barrier = threading.Barrier(2)

        def work(path: str) -> str:
            barrier.wait(timeout=3)
            return Path(path).name

        self.operation = work
        result = await asyncio.wait_for(self.run_request(), 5)
        self.assertEqual(result.text, "audio-0\naudio-1\naudio-2\naudio-3")
        self.assertEqual(len(self.created), 2)
        self.assertEqual(self.finished, ["audio"])

    async def test_realtime_lease_is_exclusive_and_waiters_cancel_cleanly(self) -> None:
        first = await self.router.acquire_engine("model")
        first.engine.realtime = True
        try:
            await asyncio.wait_for(self.run_request("mixed"), 3)
            second = await self.router.acquire_engine("model")
            second.engine.realtime = True
            try:
                waiting = asyncio.create_task(self.run_request("waiting"))
                await asyncio.sleep(0.03)
                self.assertFalse(waiting.done())
                waiting.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await waiting
                self.assertEqual(len(self.calls), 4)
            finally:
                second.engine.realtime = False
                await second.close()
        finally:
            first.engine.realtime = False
            await first.close()
        await asyncio.wait_for(self.run_request("after"), 3)
        self.assertEqual(len(self.created), 2)
        self.assertFalse(list(Path(self.directory.name).iterdir()))

    async def test_waiting_realtime_is_not_starved_by_later_batches(self) -> None:
        self.segment_count = 12
        realtime_entered = threading.Event()
        barrier = threading.Barrier(2)

        def work(path: str) -> str:
            index = int(path.rsplit("-", 1)[1])
            if index < 2:
                barrier.wait(timeout=3)
                self.entered.set()
                if not self.release.wait(3):
                    raise TimeoutError("Initial segments were not released")
            if index >= 4:
                assert realtime_entered.is_set(), "Later batch bypassed realtime waiter"
            return Path(path).name

        self.operation = work
        task = asyncio.create_task(self.run_request())
        lease_task = None
        lease = None
        try:
            self.assertTrue(await asyncio.to_thread(self.entered.wait, 3))

            async def acquire_realtime() -> object:
                acquired = await self.router.acquire_engine("model")
                acquired.engine.realtime = True
                realtime_entered.set()
                return acquired

            lease_task = asyncio.create_task(acquire_realtime())
            await asyncio.sleep(0)
            self.release.set()
            lease = await asyncio.wait_for(lease_task, 3)
            await asyncio.wait_for(task, 3)
        finally:
            self.release.set()
            if lease is not None:
                lease.engine.realtime = False
                await lease.close()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(len(self.calls), 12)

    async def test_cancel_discards_queued_segments_and_drains_active_work(self) -> None:
        barrier = threading.Barrier(2)

        def work(path: str) -> str:
            barrier.wait(timeout=3)
            self.entered.set()
            if not self.release.wait(3):
                raise TimeoutError("Native work was not released")
            return Path(path).name

        self.operation = work
        task = asyncio.create_task(self.run_request())
        try:
            self.assertTrue(await asyncio.to_thread(self.entered.wait, 3))
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0.03)
            self.assertFalse(task.done())
            self.assertTrue(all(Path(path).exists() for path in self.calls))
        finally:
            self.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(list(Path(self.directory.name).iterdir()))
        self.operation = lambda path: Path(path).name
        await asyncio.wait_for(self.run_request("after"), 3)

    async def test_failure_drains_sibling_and_does_not_return_partial_success(
        self,
    ) -> None:
        barrier = threading.Barrier(2)

        def work(path: str) -> str:
            barrier.wait(timeout=3)
            if path.endswith("-0"):
                self.entered.set()
                raise ValueError("Native failure")
            if not self.release.wait(3):
                raise TimeoutError("Sibling was not released")
            return Path(path).name

        self.operation = work
        with patch.object(settings, "ASR_BATCH_SIZE", 2):
            task = asyncio.create_task(self.run_request())
            try:
                self.assertTrue(await asyncio.to_thread(self.entered.wait, 3))
                await asyncio.sleep(0.03)
                self.assertFalse(task.done())
                self.assertTrue(all(Path(path).exists() for path in self.calls))
            finally:
                self.release.set()
            with self.assertRaisesRegex(ValueError, "Native failure"):
                await asyncio.wait_for(task, 3)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(list(Path(self.directory.name).iterdir()))

    async def test_shared_vad_inference_is_serialized(self) -> None:
        active = 0
        overlap = threading.Event()
        lock = threading.Lock()

        def generate(**kwargs: object) -> list[dict[str, list[list[int]]]]:
            nonlocal active
            with lock:
                active += 1
                if active > 1:
                    overlap.set()
            self.entered.set()
            try:
                if not self.release.wait(3):
                    raise TimeoutError("VAD was not released")
                return [{"value": [[0, 1000]]}]
            finally:
                with lock:
                    active -= 1

        splitter = AudioSplitter(device="cpu")
        with patch(
            "app.services.asr.engines.get_global_vad_model",
            return_value=SimpleNamespace(generate=generate),
        ):
            first = asyncio.create_task(
                asyncio.to_thread(splitter.get_vad_segments, "first")
            )
            second = None
            try:
                self.assertTrue(await asyncio.to_thread(self.entered.wait, 3))
                second = asyncio.create_task(
                    asyncio.to_thread(splitter.get_vad_segments, "second")
                )
                await asyncio.sleep(0.03)
                self.assertFalse(overlap.is_set(), "Shared VAD was used concurrently")
            finally:
                self.release.set()
                results = await asyncio.gather(first, *([second] if second else []))
            self.assertEqual(results, [[(0, 1000)], [(0, 1000)]])

    async def test_borrowed_input_survives_success_and_timestamps_are_scaled(
        self,
    ) -> None:
        source = Path(self.directory.name) / "source.wav"
        source.touch()
        segment = AudioSegment(1000, 2000, temp_file=str(source), speaker_id="speaker")
        with (
            patch.object(settings, "TEMP_DIR", self.directory.name),
            patch("app.services.asr.long_audio.get_audio_duration", return_value=2.0),
            patch(
                "app.utils.audio_splitter.AudioSplitter.split_audio_file",
                return_value=[segment],
            ),
        ):
            with prepare_long_audio(str(source), "cpu", False, "model") as audio:
                result = audio.finish(
                    [
                        ASRSegmentResult(
                            "word", 0, 1, word_tokens=[WordToken("word", 0.1, 0.2)]
                        )
                    ],
                    2.0,
                )
        self.assertEqual(result.duration, 4.0)
        self.assertEqual(result.segments[0].start_time, 2.0)
        self.assertEqual(result.segments[0].end_time, 4.0)
        self.assertEqual(result.segments[0].speaker_id, "speaker")
        self.assertEqual(result.segments[0].word_tokens[0].start_time, 0.2)
        self.assertEqual(result.segments[0].word_tokens[0].end_time, 0.4)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [source])

    async def test_partial_preparation_is_removed_without_deleting_source(self) -> None:
        source = Path(self.directory.name) / "source.wav"
        source.touch()

        def split(path: str, output_dir: str) -> list[AudioSegment]:
            (Path(output_dir) / "partial.wav").touch()
            raise ValueError("Decode failed")

        with (
            patch.object(settings, "TEMP_DIR", self.directory.name),
            patch("app.services.asr.long_audio.get_audio_duration", return_value=1.0),
            patch(
                "app.utils.audio_splitter.AudioSplitter.split_audio_file",
                side_effect=split,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Decode failed"):
                with prepare_long_audio(str(source), "cpu", False, "model"):
                    self.fail("Preparation should fail")
        self.assertEqual(list(Path(self.directory.name).iterdir()), [source])


if __name__ == "__main__":
    unittest.main()
