"""Resource ownership checks without loading model weights."""

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.config import settings
from app.services.asr.engines import ASRFullResult, ASRSegmentResult, WordToken
from app.services.asr.long_audio import OfflineASRRequest, prepare_long_audio
from app.services.asr.runtime.router import (
    RuntimeRouter,
)

from app.utils.audio_splitter import AudioSegment


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

    async def test_borrowed_input_survives_success_and_timestamps_are_scaled(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = temporary.name
        source = Path(directory) / "source.wav"
        source.touch()
        segment = AudioSegment(1000, 2000, temp_file=str(source), speaker_id="speaker")
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
        self.assertEqual(result.segments[0].speaker_id, "speaker")
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
