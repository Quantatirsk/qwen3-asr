from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from app.core.config import settings
from app.services.asr.manager import _supports_qwen_realtime_on_device
from app.services.asr.model_capabilities import get_enabled_qwen_huggingface_assets
from app.services.asr.qwen3_engine import Qwen3ASREngine
from app.services.asr.qwen3_remote_vllm import Qwen3RemoteVLLMBackend


class Qwen3RemoteVLLMBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = Qwen3RemoteVLLMBackend(
            base_url="http://qwen-npu:8000",
            model="qwen3-asr",
            api_key="secret",
            timeout_sec=30.0,
            max_inference_batch_size=4,
        )

    @patch("app.services.asr.qwen3_remote_vllm.requests.post")
    def test_transcribe_text_uploads_audio_to_vllm(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {"text": "hello world"}
        post.return_value = response

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            audio_path.write_bytes(b"RIFF")
            text = self.backend.transcribe_text(
                str(audio_path),
                context="product names",
                language="en",
                enable_itn=False,
            )

        self.assertEqual(text, "hello world")
        self.assertEqual(
            post.call_args.args[0], "http://qwen-npu:8000/v1/audio/transcriptions"
        )
        self.assertEqual(post.call_args.kwargs["data"]["model"], "qwen3-asr")
        self.assertEqual(post.call_args.kwargs["data"]["prompt"], "product names")
        self.assertEqual(post.call_args.kwargs["data"]["language"], "en")
        self.assertEqual(
            post.call_args.kwargs["headers"], {"Authorization": "Bearer secret"}
        )
        self.assertEqual(post.call_args.kwargs["timeout"], 30.0)
        response.raise_for_status.assert_called_once_with()

    def test_word_timestamps_fail_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            audio_path.write_bytes(b"RIFF")

            with self.assertRaisesRegex(RuntimeError, "word timestamps"):
                self.backend.transcribe_raw(
                    str(audio_path),
                    word_timestamps=True,
                )

    def test_remote_runtime_does_not_require_local_qwen_assets(self) -> None:
        with patch.object(
            settings,
            "QWEN_VLLM_BASE_URL",
            "http://qwen-npu:8000",
        ):
            self.assertEqual(get_enabled_qwen_huggingface_assets(), [])

    def test_engine_selects_remote_backend_without_realtime_claim(self) -> None:
        backend = Mock()
        with (
            patch.object(
                settings,
                "QWEN_VLLM_BASE_URL",
                "http://qwen-npu:8000",
            ),
            patch(
                "app.services.asr.qwen3_engine.Qwen3RemoteVLLMBackend",
                return_value=backend,
            ) as backend_class,
        ):
            engine = Qwen3ASREngine(
                model_path="Qwen/Qwen3-ASR-1.7B",
                device="cpu",
                forced_aligner_path="Qwen/Qwen3-ForcedAligner-0.6B",
            )

        self.assertEqual(engine._backend, "remote_vllm")
        self.assertFalse(engine.supports_realtime)
        self.assertIs(engine.model, backend)
        backend_class.assert_called_once()

    def test_remote_runtime_is_not_advertised_as_realtime(self) -> None:
        with patch.object(
            settings,
            "QWEN_VLLM_BASE_URL",
            "http://qwen-npu:8000",
        ):
            self.assertFalse(_supports_qwen_realtime_on_device("cpu"))

    @patch("app.services.asr.qwen3_remote_vllm.requests.post")
    def test_http_failures_include_remote_endpoint(self, post: Mock) -> None:
        post.side_effect = requests.ConnectionError("connection refused")

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            audio_path.write_bytes(b"RIFF")

            with self.assertRaisesRegex(RuntimeError, "qwen-npu:8000"):
                self.backend.transcribe_text(str(audio_path))

    @patch("app.services.asr.qwen3_remote_vllm.requests.post")
    def test_invalid_json_includes_remote_endpoint(self, post: Mock) -> None:
        response = Mock()
        response.json.side_effect = requests.JSONDecodeError("invalid", "x", 0)
        post.return_value = response

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            audio_path.write_bytes(b"RIFF")

            with self.assertRaisesRegex(RuntimeError, "qwen-npu:8000"):
                self.backend.transcribe_text(str(audio_path))


if __name__ == "__main__":
    unittest.main()
