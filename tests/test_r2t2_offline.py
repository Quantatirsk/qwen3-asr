"""Offline recognition sends fresh audio to the shared engine before alignment."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import numpy as np

from app.core.config import settings
from app.services.asr.r2t2_engine import R2T2Engine
from app.services.realtime.client import get_engine_capabilities, transcribe_segment
from app.services.realtime.protocol import (
    MODEL_ID,
    PROTOCOL_VERSION,
    SAMPLE_RATE,
    StreamError,
)


class SharedOfflineClientTest(unittest.TestCase):
    def test_capabilities_require_shared_engine_protocol(self) -> None:
        capabilities = {
            "model": MODEL_ID,
            "sample_rate": SAMPLE_RATE,
            "protocol_version": PROTOCOL_VERSION,
            "offline_transcription": True,
        }
        with patch.object(settings, "R2T2_URL", "http://engine:8001"):
            for offline in (True, False, "true", None):
                capabilities["offline_transcription"] = offline
                with patch(
                    "app.services.realtime.client.urlopen",
                    return_value=io.BytesIO(json.dumps(capabilities).encode()),
                ):
                    if offline is True:
                        self.assertEqual(get_engine_capabilities(), capabilities)
                    else:
                        with self.assertRaises(StreamError):
                            get_engine_capabilities()

    def test_raw_audio_context_and_internal_auth(self) -> None:
        audio = np.arange(16000, dtype=np.float32) / 16000
        with (
            patch.object(settings, "R2T2_URL", "http://engine:8001"),
            patch.object(settings, "R2T2_INTERNAL_TOKEN", "private-token"),
            patch(
                "app.services.realtime.client.urlopen",
                return_value=io.BytesIO(json.dumps({"text": "Fresh result."}).encode()),
            ) as send,
        ):
            self.assertEqual(
                transcribe_segment(audio, "Names: Ada & Alan"), "Fresh result."
            )
        request = send.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(urlsplit(request.full_url).path, "/v1/transcribe")
        self.assertEqual(
            parse_qs(urlsplit(request.full_url).query),
            {"context": ["Names: Ada & Alan"]},
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer private-token")
        self.assertEqual(request.data, audio.astype("<f4").tobytes())
        self.assertEqual(send.call_args.kwargs["timeout"], 180)

    def test_bad_audio_and_context_are_rejected_before_network(self) -> None:
        with patch("app.services.realtime.client.urlopen") as send:
            for audio in (
                np.zeros(0),
                np.zeros((2, 2)),
                np.zeros(960001),
                np.array([np.nan]),
            ):
                with self.assertRaises(ValueError):
                    transcribe_segment(audio)
            with self.assertRaises(ValueError):
                transcribe_segment(np.zeros(16000), "x" * 2049)
            send.assert_not_called()

    def test_bad_responses_and_network_errors_do_not_fall_back(self) -> None:
        with patch.object(settings, "R2T2_URL", "http://engine:8001"):
            for payload in (
                b"{}",
                b"[]",
                b"null",
                b'{"text": 42}',
                b'{"text": "partial", "error": "failed"}',
                b"x" * (1024 * 1024 + 1),
            ):
                with (
                    patch(
                        "app.services.realtime.client.urlopen",
                        return_value=io.BytesIO(payload),
                    ),
                    self.assertRaises(StreamError),
                ):
                    transcribe_segment(np.zeros(16000))
            for error in (
                TimeoutError("slow"),
                HTTPError("engine", 503, "unavailable", {}, None),
            ):
                with (
                    patch("app.services.realtime.client.urlopen", side_effect=error),
                    self.assertRaises(StreamError),
                ):
                    transcribe_segment(np.zeros(16000))


class R2T2OfflineTest(unittest.TestCase):
    def test_asr_chunks_keep_text_and_relative_word_times(self) -> None:
        engine = R2T2Engine.__new__(R2T2Engine)
        engine.aligner = SimpleNamespace(
            align_transcript=Mock(
                return_value=[{"text": "fresh", "start_ms": 200, "end_ms": 800}]
            )
        )
        audio = np.zeros(16000, dtype=np.float32)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("app.services.asr.r2t2_engine._load_audio", return_value=audio),
            patch(
                "app.services.asr.r2t2_engine.transcribe_segment",
                side_effect=["fresh!", "second."],
            ) as recognize,
            patch(
                "app.services.asr.r2t2_engine.normalize_asr_text",
                side_effect=lambda text, **kwargs: text,
            ) as normalize,
        ):
            paths = [
                str(Path(directory) / name) for name in ("first.wav", "second.wav")
            ]
            for path in paths:
                Path(path).touch()
            segments = [
                SimpleNamespace(
                    temp_file=paths[0], start_sec=5.0, end_sec=8.0, speaker_id=None
                ),
                SimpleNamespace(
                    temp_file=paths[1], start_sec=9.0, end_sec=11.0, speaker_id=None
                ),
            ]
            results = engine.transcribe_segments(
                segments, hotwords="Ada", word_timestamps=True
            )
            self.assertEqual([r.text for r in results], ["fresh!", "second."])
            self.assertEqual([r.speaker_id for r in results], [None, None])
            self.assertEqual(results[0].start_time, 5.0)
            self.assertEqual(results[0].word_tokens[0].start_time, 0.2)
            self.assertEqual(results[0].word_tokens[0].text, "fresh")
            self.assertEqual(recognize.call_count, 2)
            self.assertIs(recognize.call_args.args[0], audio)
            self.assertEqual(recognize.call_args.args[1], "Ada")
            normalize.assert_any_call("fresh!", enable_itn=True)
            engine.aligner.align_transcript.assert_any_call(
                audio_path=paths[0], text="fresh!", audio=audio
            )
        with self.assertRaises(FileNotFoundError):
            engine.transcribe_segments(segments)

    def test_no_timestamps_skips_aligner(self) -> None:
        engine = R2T2Engine.__new__(R2T2Engine)
        engine.aligner = Mock()
        with (
            tempfile.NamedTemporaryFile() as source,
            patch(
                "app.services.asr.r2t2_engine._load_audio", return_value=np.zeros(16000)
            ),
            patch(
                "app.services.asr.r2t2_engine.transcribe_segment",
                return_value="Fresh text.",
            ),
        ):
            result = engine.transcribe_segments(
                [
                    SimpleNamespace(
                        temp_file=source.name, start_sec=0, end_sec=1, speaker_id=None
                    )
                ],
                enable_itn=False,
            )
        self.assertEqual(result[0].text, "Fresh text.")
        self.assertIsNone(result[0].word_tokens)
        engine.aligner.align_transcript.assert_not_called()


if __name__ == "__main__":
    unittest.main()
