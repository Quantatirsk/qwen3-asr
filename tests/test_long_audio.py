"""Shared preparation ownership and VAD concurrency regressions."""

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.config import settings
from app.services.asr.long_audio import prepare_long_audio
from app.services.asr.results import ASRSegmentResult, WordToken
from app.utils.audio_splitter import AudioSegment, AudioSplitter


class LongAudioTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.entered = threading.Event()
        self.release = threading.Event()

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
