"""Independent ASR chunking respects inference limits without dropping tails."""

import tempfile
import unittest
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.core.config import settings
from app.utils.audio_splitter import AudioSplitter


class SegmentBoundsTest(unittest.TestCase):
    def test_long_vad_and_fixed_duration_preserve_all_intervals(self) -> None:
        with patch.object(settings, "MAX_SEGMENT_SEC", 60):
            splitter = AudioSplitter()
        for duration in (500, 60500, 120500, 125000):
            for segments in (
                splitter.merge_segments_greedy([(0, duration)], duration),
                splitter._split_by_fixed_duration(duration),
            ):
                self.assertEqual(segments[0][0], 0)
                self.assertEqual(segments[-1][1], duration)
                self.assertTrue(
                    all(0 < end - start <= 60000 for start, end in segments)
                )
                self.assertEqual(sum(end - start for start, end in segments), duration)
                self.assertTrue(all(a[1] == b[0] for a, b in pairwise(segments)))

    def test_one_sample_past_limit_is_split_and_retained(self) -> None:
        audio = np.zeros(60 * 16000 + 1, dtype=np.float32)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(settings, "MAX_SEGMENT_SEC", 60),
            patch(
                "app.utils.audio_splitter.librosa",
                SimpleNamespace(load=Mock(return_value=(audio, 16000))),
            ),
            patch.object(AudioSplitter, "get_vad_segments", return_value=[]),
            patch("app.utils.audio_splitter.sf.write"),
        ):
            segments = AudioSplitter().split_audio_file("original.wav", directory)
            self.assertGreater(len(segments), 1)
            self.assertTrue(
                all(len(item.audio_data) <= 60 * 16000 for item in segments)
            )
            self.assertEqual(sum(len(item.audio_data) for item in segments), len(audio))


if __name__ == "__main__":
    unittest.main()
