"""Resource ownership checks without loading model weights."""

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.core.config import settings
from app.services.asr.engines import ASRFullResult, ASRSegmentResult, WordToken
from app.services.asr.long_audio import OfflineASRRequest, prepare_long_audio
from app.services.asr.r2t2_engine import R2T2Engine
from app.services.asr.runtime.router import (
    RuntimeRouter,
)

from app.utils.audio_splitter import AudioSegment
from app.utils.speaker_diarizer import DiarizationResult


class RuntimeOwnershipTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        manager = SimpleNamespace(create_engine=lambda _: object())
        patcher = patch(
            "app.services.asr.runtime.router.get_model_manager", return_value=manager
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_cancelled_inference_keeps_lease_until_worker_finishes(self) -> None:
        entered = asyncio.Event()
        second_entered = threading.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        class Engine:
            def transcribe_long_audio(
                self, *, audio_path: str, **kwargs: object
            ) -> ASRFullResult:
                if audio_path == "first":
                    loop.call_soon_threadsafe(entered.set)
                    if not release.wait(3):
                        raise TimeoutError("Worker was not released")
                else:
                    second_entered.set()
                return ASRFullResult(text=audio_path, segments=[], duration=1.0)

        router = RuntimeRouter()
        with (
            patch.object(router, "resolve_model_id", return_value="confucius4-r2t2"),
            patch.object(
                router,
                "_get_engine",
                return_value=Engine(),
            ),
        ):
            first = asyncio.create_task(
                router.run_offline(OfflineASRRequest("model", "first"))
            )
            second = None
            try:
                await asyncio.wait_for(entered.wait(), 1)
                first.cancel()
                await asyncio.sleep(0)
                first.cancel()
                second = asyncio.create_task(
                    router.run_offline(OfflineASRRequest("model", "second"))
                )
                await asyncio.sleep(0.03)
                self.assertFalse(
                    second_entered.is_set(), "Cancelled worker's engine was reused"
                )
                self.assertFalse(
                    first.done(), "Cancellation escaped before worker completion"
                )
            finally:
                release.set()
                await asyncio.gather(
                    first, *([second] if second else []), return_exceptions=True
                )
            self.assertTrue(first.cancelled())
            self.assertEqual(second.result().text, "second")

    async def test_failed_model_is_not_reported_loaded(self) -> None:
        router = RuntimeRouter()
        with (
            patch.object(router, "resolve_model_id", return_value="confucius4-r2t2"),
            patch.object(
                router._manager, "create_engine", side_effect=ValueError("Load failed")
            ),
        ):
            with self.assertRaises(ValueError):
                router.warmup_model("model")
        self.assertEqual(router.get_loaded_model_ids(), [])

    async def test_cancelled_diarized_pipeline_keeps_chunks_until_worker_finishes(
        self,
    ) -> None:
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        engine = R2T2Engine.__new__(R2T2Engine)
        engine.device = "cuda:0"
        engine.model_id = "confucius4-r2t2"
        engine.aligner = SimpleNamespace(align_transcript=Mock(return_value=[]))

        def split(audio_path: str, output_dir: str) -> list[AudioSegment]:
            chunk = Path(output_dir) / "chunk.wav"
            chunk.touch()
            return [AudioSegment(0, 1000, temp_file=str(chunk))]

        def recognize(samples: np.ndarray, context: str) -> str:
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(3):
                raise TimeoutError("Worker was not released")
            return "Finished."

        empty = DiarizationResult([], np.zeros((100, 8)), 0.01, 1, (None,) * 8)
        router = RuntimeRouter()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(settings, "TEMP_DIR", directory),
            patch.object(router, "resolve_model_id", return_value=engine.model_id),
            patch.object(router, "_get_engine", return_value=engine),
            patch("app.services.asr.long_audio.get_audio_duration", return_value=1),
            patch(
                "app.utils.speaker_diarizer.get_speaker_diarizer",
                return_value=SimpleNamespace(diarize=lambda _: empty),
            ),
            patch(
                "app.utils.audio_splitter.AudioSplitter.split_audio_file",
                side_effect=split,
            ),
            patch(
                "app.services.asr.r2t2_engine._load_audio",
                return_value=np.zeros(16000),
            ),
            patch(
                "app.services.asr.r2t2_engine.transcribe_segment",
                side_effect=recognize,
            ),
        ):
            source = Path(directory) / "source.wav"
            source.touch()
            task = asyncio.create_task(
                router.run_offline(
                    OfflineASRRequest(
                        "model", str(source), enable_itn=False, enable_punctuation=False
                    )
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), 1)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertEqual(len(list(Path(directory).rglob("chunk.wav"))), 1)
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
            self.assertTrue(task.cancelled())
            self.assertEqual(list(Path(directory).iterdir()), [source])

    async def test_borrowed_input_survives_success_and_timestamps_are_scaled(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = temporary.name
        source = Path(directory) / "source.wav"
        source.touch()
        segment = AudioSegment(1000, 2000, temp_file=str(source))
        with (
            patch.object(settings, "TEMP_DIR", directory),
            patch("app.services.asr.long_audio.get_audio_duration", return_value=2.0),
            patch(
                "app.utils.audio_splitter.AudioSplitter.split_audio_file",
                return_value=[segment],
            ),
        ):
            with prepare_long_audio(str(source), "cuda:0", False, "model") as audio:
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
        self.assertIsNone(result.segments[0].speaker_id)
        self.assertEqual(result.segments[0].word_tokens[0].start_time, 0.2)
        self.assertEqual(result.segments[0].word_tokens[0].end_time, 0.4)
        self.assertEqual(list(Path(directory).iterdir()), [source])

    async def test_partial_preparation_is_removed_without_deleting_source(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = temporary.name
        source = Path(directory) / "source.wav"
        source.touch()

        def split(path: str, output_dir: str) -> list[AudioSegment]:
            (Path(output_dir) / "partial.wav").touch()
            raise ValueError("Decode failed")

        with (
            patch.object(settings, "TEMP_DIR", directory),
            patch("app.services.asr.long_audio.get_audio_duration", return_value=1.0),
            patch(
                "app.utils.audio_splitter.AudioSplitter.split_audio_file",
                side_effect=split,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Decode failed"):
                with prepare_long_audio(str(source), "cuda:0", False, "model"):
                    self.fail("Preparation should fail")
        self.assertEqual(list(Path(directory).iterdir()), [source])


if __name__ == "__main__":
    unittest.main()
