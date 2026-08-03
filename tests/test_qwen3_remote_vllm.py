from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from app.services.asr.manager import ASCEND_MODEL_ID, ModelManager
from app.services.asr.model_capabilities import (
    QWEN_ASCEND_REVISION,
    get_enabled_qwen_huggingface_assets,
)
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
            text = self.backend.transcribe_text(str(audio_path), language="en")

        self.assertEqual(text, "hello world")
        self.assertEqual(
            post.call_args.args[0], "http://qwen-npu:8000/v1/audio/transcriptions"
        )
        self.assertEqual(post.call_args.kwargs["data"]["model"], "qwen3-asr")
        self.assertEqual(post.call_args.kwargs["data"]["to_language"], "en")
        self.assertEqual(
            post.call_args.kwargs["headers"], {"Authorization": "Bearer secret"}
        )
        response.raise_for_status.assert_called_once_with()

    @patch("app.services.asr.qwen3_remote_vllm.requests.post")
    def test_context_hints_fail_instead_of_being_ignored(self, post: Mock) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            audio_path.write_bytes(b"RIFF")
            with self.assertRaisesRegex(RuntimeError, "context hints"):
                self.backend.transcribe_text(str(audio_path), context="product names")
        post.assert_not_called()

    @patch("app.services.asr.qwen3_remote_vllm.requests.get")
    def test_readiness_probes_remote_health_endpoint(self, get: Mock) -> None:
        get.return_value = Mock()
        self.backend.ensure_ready()
        get.assert_called_once_with(
            "http://qwen-npu:8000/health",
            headers={"Authorization": "Bearer secret"},
            timeout=10.0,
        )
        get.return_value.raise_for_status.assert_called_once_with()

    def test_catalog_contains_only_pinned_ascend_model(self) -> None:
        entries = ModelManager().list_declared_entries()
        assets = get_enabled_qwen_huggingface_assets()
        self.assertEqual([entry["id"] for entry in entries], [ASCEND_MODEL_ID])
        self.assertEqual([asset.model_id for asset in assets], ["Qwen/Qwen3-ASR-1.7B"])
        self.assertEqual(assets[0].revision, QWEN_ASCEND_REVISION)

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
