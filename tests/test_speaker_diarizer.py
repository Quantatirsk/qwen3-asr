"""Only an empty VAD result bypasses speaker clustering."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.core.exceptions import DefaultServerErrorException
from app.utils.speaker_diarizer import SpeakerDiarizer


class SpeakerDiarizerTest(unittest.TestCase):
    def test_no_speech_returns_no_segments_without_clustering(self) -> None:
        pipeline = Mock()
        pipeline.preprocess.return_value = []
        with patch(
            "app.utils.speaker_diarizer.get_global_diarization_pipeline",
            return_value=pipeline,
        ):
            self.assertEqual(SpeakerDiarizer().diarize("silence.wav"), [])
        pipeline.preprocess.assert_called_once_with("silence.wav")
        pipeline.assert_not_called()

    def test_short_speech_and_pipeline_errors_are_not_silence(self) -> None:
        speech = [[0.0, 0.5, np.ones(8000, dtype=np.float32)]]
        for error in (
            AssertionError("The effective audio duration is too short"),
            RuntimeError("Embedding inference failed"),
        ):
            with self.subTest(error=error):
                pipeline = Mock(side_effect=error)
                pipeline.preprocess.return_value = speech
                with patch(
                    "app.utils.speaker_diarizer.get_global_diarization_pipeline",
                    return_value=pipeline,
                ):
                    with self.assertRaises(DefaultServerErrorException):
                        SpeakerDiarizer().diarize("short.wav")
                pipeline.preprocess.assert_called_once_with("short.wav")
                self.assertIs(pipeline.call_args.args[0], speech)


if __name__ == "__main__":
    unittest.main()
