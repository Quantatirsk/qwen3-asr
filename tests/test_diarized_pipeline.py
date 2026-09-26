"""Exercise independent chunk recognition, alignment, and speaker attribution."""

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.core.config import settings
from app.services.asr.engines import ASRFullResult
from app.services.asr.r2t2_engine import R2T2Engine
from app.utils.audio_splitter import AudioSegment
from app.utils.speaker_diarizer import DiarizationResult, SpeakerSegment


class DiarizedPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.directory = Path(self.context.enter_context(tempfile.TemporaryDirectory()))
        self.source = self.directory / "recording.wav"
        self.source.touch()
        self.engine = R2T2Engine.__new__(R2T2Engine)
        self.engine.device = "cuda:0"
        self.engine.model_id = "confucius4-r2t2"
        self.engine.aligner = Mock()
        self.engine.aligner.align_transcript.side_effect = [
            [
                {"text": "First", "start_ms": 0, "end_ms": 700},
                {"text": "Yes", "start_ms": 1000, "end_ms": 1100},
                {"text": "Mixed", "start_ms": 2000, "end_ms": 3000},
            ],
            [{"text": "After", "start_ms": 0, "end_ms": 1000}],
        ]
        self.spans = [
            SpeakerSegment(10, 11, "speaker-1", 0.9),
            SpeakerSegment(11, 11.1, "speaker-2", 0.9),
            SpeakerSegment(11.1, 14, "speaker-1", 0.9),
            SpeakerSegment(12, 13, "speaker-2", 0.9),
        ]
        probabilities = np.zeros((1400, 8), dtype=np.float32)
        for span in self.spans:
            column = 0 if span.speaker_id == "speaker-1" else 1
            probabilities[
                int(span.start_sec * 100) : int(span.end_sec * 100), column
            ] = 0.9
        self.diarization = DiarizationResult(
            segments=self.spans,
            probabilities=probabilities,
            frame_seconds=0.01,
            duration=14,
            speaker_ids=("speaker-1", "speaker-2", None, None, None, None, None, None),
        )
        self.diarizer = Mock()
        self.diarizer.diarize.return_value = self.diarization
        self.context.enter_context(
            patch.object(settings, "TEMP_DIR", str(self.directory))
        )
        self.context.enter_context(
            patch("app.services.asr.long_audio.get_audio_duration", return_value=14)
        )
        self.get_diarizer = self.context.enter_context(
            patch(
                "app.utils.speaker_diarizer.get_speaker_diarizer",
                return_value=self.diarizer,
            )
        )
        self.splitter = self.context.enter_context(
            patch(
                "app.utils.audio_splitter.AudioSplitter.split_audio_file",
                side_effect=self.split,
            )
        )
        self.context.enter_context(
            patch(
                "app.services.asr.r2t2_engine._load_audio",
                return_value=np.zeros(16000, dtype=np.float32),
            )
        )
        self.recognize = self.context.enter_context(
            patch(
                "app.services.asr.r2t2_engine.transcribe_segment",
                side_effect=["First. Yes! Mixed.", "After."],
            )
        )

    def split(self, audio_path: str, output_dir: str) -> list[AudioSegment]:
        self.assertEqual(audio_path, str(self.source))
        chunks = []
        for index, (start, end) in enumerate(((10000, 13000), (13000, 14000))):
            path = Path(output_dir) / f"chunk-{index}.wav"
            path.touch()
            chunks.append(AudioSegment(start, end, temp_file=str(path)))
        return chunks

    def transcribe(self, **kwargs: object) -> ASRFullResult:
        return self.engine.transcribe_long_audio(
            str(self.source), enable_punctuation=False, **kwargs
        )

    def assert_cleaned(self) -> None:
        self.assertEqual(list(self.directory.iterdir()), [self.source])

    def test_overlap_does_not_duplicate_asr_and_scaled_times_preserve_short_turn(
        self,
    ) -> None:
        result = self.transcribe(word_timestamps=True, timestamp_scale=2)
        self.assertEqual(self.recognize.call_count, 2)
        self.splitter.assert_called_once()
        self.diarizer.diarize.assert_called_once_with(str(self.source))
        self.assertEqual(result.text, "First. Yes! Mixed.\nAfter.")
        self.assertEqual(
            [segment.text for segment in result.segments],
            ["First. ", "Yes! ", "Mixed.", "After."],
        )
        self.assertEqual(
            [segment.speaker_id for segment in result.segments],
            ["speaker-1", "speaker-2", None, "speaker-1"],
        )
        self.assertEqual(
            result.segments[2].speaker_candidates, ["speaker-1", "speaker-2"]
        )
        np.testing.assert_allclose(
            [(segment.start_time, segment.end_time) for segment in result.segments],
            [(20, 21.4), (22, 22.2), (24, 26), (26, 28)],
        )
        self.assertEqual(result.duration, 28)
        self.assertEqual(
            [word.text for segment in result.segments for word in segment.word_tokens],
            ["First", "Yes", "Mixed", "After"],
        )
        self.assertTrue(
            all(segment.word_tokens[0].start_time == 0 for segment in result.segments)
        )
        self.assertAlmostEqual(result.segments[1].word_tokens[0].end_time, 0.2)
        self.assertEqual(
            [(span.start_sec, span.end_sec) for span in result.speaker_segments],
            [(20, 22), (22, 22.2), (22.2, 28), (24, 26)],
        )
        self.assertEqual(self.spans[0].start_sec, 10)
        self.assert_cleaned()

    def test_internal_alignment_does_not_expose_unrequested_words(self) -> None:
        result = self.transcribe(word_timestamps=False)
        self.assertEqual(self.engine.aligner.align_transcript.call_count, 2)
        self.assertTrue(all(segment.word_tokens is None for segment in result.segments))
        self.assertEqual(result.segments[1].speaker_id, "speaker-2")
        self.assertEqual(
            result.segments[2].speaker_candidates, ["speaker-1", "speaker-2"]
        )
        self.assertEqual(result.speaker_segments, self.spans)
        self.assert_cleaned()

    def test_disabled_diarization_and_words_skip_both_models(self) -> None:
        result = self.transcribe(
            enable_speaker_diarization=False, word_timestamps=False
        )
        self.get_diarizer.assert_not_called()
        self.engine.aligner.align_transcript.assert_not_called()
        self.assertEqual(len(result.segments), 2)
        self.assertTrue(
            all(
                segment.speaker_id is None and segment.word_tokens is None
                for segment in result.segments
            )
        )
        self.assertIsNone(result.speaker_segments)
        self.assert_cleaned()

    def test_requested_words_still_work_without_diarization(self) -> None:
        result = self.transcribe(enable_speaker_diarization=False, word_timestamps=True)
        self.get_diarizer.assert_not_called()
        self.assertEqual(self.engine.aligner.align_transcript.call_count, 2)
        self.assertEqual(result.segments[0].word_tokens[1].start_time, 1)
        self.assertEqual(result.segments[0].start_time, 10)
        self.assertIsNone(result.speaker_segments)
        self.assert_cleaned()

    def test_empty_diarization_preserves_independent_asr(self) -> None:
        self.diarizer.diarize.return_value = DiarizationResult(
            [], np.zeros((1400, 8)), 0.01, 14, (None,) * 8
        )
        result = self.transcribe(word_timestamps=True)
        self.assertEqual(self.recognize.call_count, 2)
        self.assertEqual(result.text, "First. Yes! Mixed.\nAfter.")
        self.assertTrue(all(segment.speaker_id is None for segment in result.segments))
        self.assertEqual(result.speaker_segments, [])
        self.assert_cleaned()

    def test_silent_recognition_returns_empty_without_fabricated_speakers(self) -> None:
        self.recognize.side_effect = ["", ""]
        self.engine.aligner.align_transcript.side_effect = [[], []]
        result = self.transcribe()
        self.assertEqual(result.text, "")
        self.assertEqual(result.segments, [])
        self.assert_cleaned()

    def test_recognition_or_alignment_failure_cleans_owned_chunks(self) -> None:
        self.recognize.side_effect = RuntimeError("Recognition failed")
        with self.assertRaisesRegex(RuntimeError, "Recognition failed"):
            self.transcribe()
        self.assert_cleaned()
        self.recognize.side_effect = ["First. Yes! Mixed.", "After."]
        self.engine.aligner.align_transcript.side_effect = RuntimeError(
            "Alignment failed"
        )
        with self.assertRaisesRegex(RuntimeError, "Alignment failed"):
            self.transcribe()
        self.assert_cleaned()

    def test_diarization_failure_does_not_start_asr_or_leave_owned_directory(
        self,
    ) -> None:
        self.diarizer.diarize.side_effect = RuntimeError("Diarization failed")
        with self.assertRaisesRegex(RuntimeError, "Diarization failed"):
            self.transcribe()
        self.splitter.assert_not_called()
        self.recognize.assert_not_called()
        self.assert_cleaned()


if __name__ == "__main__":
    unittest.main()
