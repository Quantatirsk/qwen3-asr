"""Nemotron keeps overlap and recording-local identity without segmenting ASR audio."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from app.core.exceptions import DefaultServerErrorException
from app.utils.speaker_diarizer import (
    SpeakerDiarizer,
    close_speaker_diarizer,
    get_speaker_diarizer,
)


class FakeInputs(dict[str, torch.Tensor]):
    @property
    def attention_mask(self) -> torch.Tensor:
        return self["attention_mask"]

    def to(self, device: torch.device, dtype: torch.dtype) -> "FakeInputs":
        return self


class SpeakerDiarizerTest(unittest.TestCase):
    def setUp(self) -> None:
        logger_patch = patch("app.utils.speaker_diarizer.logger")
        logger_patch.start()
        self.addCleanup(logger_patch.stop)

    def make_diarizer(
        self, frames: int = 100, raw_segments: list[dict] | None = None
    ) -> SpeakerDiarizer:
        diarizer = SpeakerDiarizer()
        diarizer._model = Mock(
            device=torch.device("cpu"),
            dtype=torch.float32,
            return_value=SimpleNamespace(logits=torch.zeros(1, frames, 8)),
        )
        inputs = FakeInputs(
            {
                "input_features": torch.zeros(1, frames, 128),
                "attention_mask": torch.ones(1, frames, dtype=torch.long),
            }
        )
        diarizer._processor = Mock(return_value=inputs)
        diarizer._processor.extract_speaker_dict.return_value = [raw_segments or []]
        return diarizer

    def test_overlap_short_interjection_and_late_return_keep_identity(self) -> None:
        diarizer = self.make_diarizer(
            30100,
            [
                {"Start": 300.0, "End": 301.01, "Speaker": 6},
                {"Start": 0.2, "End": 0.23, "Speaker": 2},
                {"Start": 0.0, "End": 0.8, "Speaker": 6},
            ],
        )
        with patch(
            "app.utils.speaker_diarizer.librosa.load",
            return_value=(np.zeros(301 * 16000, dtype=np.float32), 16000),
        ):
            result = diarizer.diarize("meeting.wav")
        self.assertEqual(len(result.segments), 3)
        self.assertEqual(
            [s.speaker_id for s in result.segments],
            ["\u8bf4\u8bdd\u4eba1", "\u8bf4\u8bdd\u4eba2", "\u8bf4\u8bdd\u4eba1"],
        )
        self.assertEqual(result.segments[1].end_sec, 0.23)
        self.assertEqual(result.segments[-1].end_sec, 301)
        self.assertEqual(result.speaker_ids[6], "\u8bf4\u8bdd\u4eba1")
        self.assertIsNone(result.speaker_ids[0])
        self.assertEqual(result.probabilities.shape, (30100, 8))
        self.assertAlmostEqual(result.segments[0].confidence, 0.5)
        diarizer._model.assert_called_once()
        self.assertNotIn("speaker_cache", diarizer._model.call_args.kwargs)
        self.assertNotIn("num_lookahead_frames", diarizer._model.call_args.kwargs)

    def test_silence_and_empty_recording(self) -> None:
        diarizer = self.make_diarizer()
        for length in (16000, 0):
            with (
                self.subTest(length=length),
                patch(
                    "app.utils.speaker_diarizer.librosa.load",
                    return_value=(np.zeros(length, dtype=np.float32), 16000),
                ),
            ):
                result = diarizer.diarize("silence.wav")
                self.assertEqual(result.segments, [])
                self.assertEqual(result.speaker_ids, (None,) * 8)
                self.assertEqual(result.duration, length / 16000)
        diarizer._model.assert_called_once()

    def test_multiple_requests_are_isolated_and_serialized(self) -> None:
        diarizer = self.make_diarizer()
        guard = threading.Lock()
        concurrent = 0
        maximum = 0

        def infer(**inputs: torch.Tensor) -> SimpleNamespace:
            nonlocal concurrent, maximum
            self.assertEqual(set(inputs), {"input_features", "attention_mask"})
            with guard:
                concurrent += 1
                maximum = max(maximum, concurrent)
            time.sleep(0.01)
            with guard:
                concurrent -= 1
            return SimpleNamespace(logits=torch.zeros(1, 100, 8))

        diarizer._model.side_effect = infer
        diarizer._processor.extract_speaker_dict.side_effect = [
            [[{"Start": 0.0, "End": 1.0, "Speaker": column}]] for column in (7, 1, 2, 3)
        ]
        with (
            patch(
                "app.utils.speaker_diarizer.librosa.load",
                return_value=(np.zeros(16000, dtype=np.float32), 16000),
            ),
            ThreadPoolExecutor(4) as executor,
        ):
            results = list(executor.map(diarizer.diarize, ["a.wav"] * 4))
        self.assertEqual(maximum, 1)
        self.assertEqual(diarizer._model.call_count, 4)
        for result in results:
            self.assertEqual(result.segments[0].speaker_id, "\u8bf4\u8bdd\u4eba1")
            self.assertEqual(sum(s is not None for s in result.speaker_ids), 1)

    def test_bad_model_output_fails_without_poisoning_next_request(self) -> None:
        diarizer = self.make_diarizer()
        invalid_outputs = (
            torch.full((1, 100, 8), float("nan")),
            torch.zeros(1, 100, 7),
            torch.zeros(1, 99, 8),
        )
        with patch(
            "app.utils.speaker_diarizer.librosa.load",
            return_value=(np.zeros(16000, dtype=np.float32), 16000),
        ):
            for logits in invalid_outputs:
                diarizer._model.return_value = SimpleNamespace(logits=logits)
                with self.assertRaises(DefaultServerErrorException):
                    diarizer.diarize("audio.wav")
            diarizer._model.return_value = SimpleNamespace(
                logits=torch.zeros(1, 100, 8)
            )
            self.assertEqual(diarizer.diarize("audio.wav").segments, [])

    def test_bad_intervals_and_probabilities_are_rejected(self) -> None:
        diarizer = SpeakerDiarizer()
        probabilities = np.full((100, 8), 0.8, dtype=np.float32)
        for raw in (
            {"Start": -0.1, "End": 0.5, "Speaker": 0},
            {"Start": 0.0, "End": float("nan"), "Speaker": 0},
            {"Start": 0.5, "End": 0.5, "Speaker": 0},
            {"Start": 0.0, "End": 0.5, "Speaker": 8},
            {"Start": 0.0, "End": 0.5, "Speaker": 0.5},
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                diarizer._make_result([raw], probabilities, 1.0)
        for invalid in (np.zeros((100, 7)), probabilities * 2):
            with self.assertRaises(ValueError):
                diarizer._make_result([], invalid, 1.0)

    def test_singleton_release_waits_and_disallows_stale_instance_reuse(self) -> None:
        close_speaker_diarizer()
        diarizer = get_speaker_diarizer()
        self.assertIs(diarizer, get_speaker_diarizer())
        close_speaker_diarizer()
        with self.assertRaises(RuntimeError):
            diarizer.warmup()
        self.assertIsNot(diarizer, get_speaker_diarizer())
        close_speaker_diarizer()


if __name__ == "__main__":
    unittest.main()
