import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.v1 import api_router
from app.core.config import settings
from app.services.asr.engines import ASRFullResult, ASRSegmentResult, WordToken
from app.services.realtime.protocol import MODEL_ID, StreamError


class APIContractTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(api_router)
        self.app = app
        self.client = TestClient(app)
        result = ASRFullResult(
            "hello",
            [ASRSegmentResult("hello", 0, 1, "speaker1", [WordToken("hello", 0, 1)])],
            1,
        )

        async def start_transcription(**kwargs):
            async def transcribe():
                return result

            return asyncio.create_task(transcribe())

        self.service = SimpleNamespace(
            start_transcription=AsyncMock(side_effect=start_transcription),
        )

    def test_offline_all_response_formats(self):
        with patch(
            "app.api.v1.openai_compatible.get_offline_transcription_service",
            return_value=self.service,
        ):
            for fmt in ("json", "verbose_json", "text", "srt", "vtt"):
                with self.subTest(format=fmt):
                    response = self.client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("test.wav", b"fake", "audio/wav")},
                        data={
                            "model": MODEL_ID,
                            "response_format": fmt,
                            "word_timestamps": "true",
                        },
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertIn("hello", response.text)
                    if fmt == "verbose_json":
                        self.assertEqual(
                            response.json()["segments"][0]["speaker"], "speaker1"
                        )
                        self.assertEqual(response.json()["words"][0]["word"], "hello")
                    if fmt == "vtt":
                        self.assertTrue(response.text.startswith("WEBVTT"))
        self.assertEqual(self.service.start_transcription.await_count, 5)

    def test_aliyun_offline_contract(self):
        with patch(
            "app.api.v1.asr.get_offline_transcription_service",
            return_value=self.service,
        ):
            response = self.client.post(
                "/stream/v1/asr?word_timestamps=true", content=b"fake"
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["result"], "hello")
        self.assertEqual(response.json()["segments"][0]["speaker_id"], "speaker1")
        self.assertEqual(
            response.json()["segments"][0]["word_tokens"][0]["text"], "hello"
        )
        self.service.start_transcription.assert_awaited_once()

    def test_offline_default_model(self) -> None:
        with patch(
            "app.api.v1.openai_compatible.get_offline_transcription_service",
            return_value=self.service,
        ):
            response = self.client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", b"fake")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.service.start_transcription.assert_awaited_once()

    def test_r2t2_model_with_audio_url(self) -> None:
        with patch(
            "app.api.v1.openai_compatible.get_offline_transcription_service",
            return_value=self.service,
        ):
            response = self.client.post(
                "/v1/audio/transcriptions",
                data={
                    "model": MODEL_ID,
                    "audio_address": "https://example.test/audio.wav",
                    "response_format": "verbose_json",
                    "word_timestamps": "true",
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["words"][0]["word"], "hello")
        self.service.start_transcription.assert_awaited_once()
        options = self.service.start_transcription.call_args.kwargs
        self.assertIsNone(options["audio_data"])
        self.assertEqual(options["audio_address"], "https://example.test/audio.wav")

    def test_any_model_value_uses_configured_transcription_service(self) -> None:
        names = (
            "",
            "arbitrary-model",
            "qwen3-asr",
            "qwen3-asr-0.6b",
            "qwen3-asr-1.7b",
            "Qwen/Qwen3-ASR-1.7B",
            "whisper-1",
            "paraformer-large",
            "custom-local-model",
            "netease-youdao/Confucius4-R2T2",
            "Confucius4-R2T2",
        )
        with patch(
            "app.api.v1.openai_compatible.get_offline_transcription_service",
            return_value=self.service,
        ) as get_service:
            for model in names:
                with self.subTest(model=model):
                    response = self.client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("test.wav", b"fake")},
                        data={"model": model},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["text"], "hello")
        self.assertEqual(get_service.call_count, len(names))
        self.assertEqual(self.service.start_transcription.await_count, len(names))

    def test_models_do_not_probe_remote_availability(self) -> None:
        with patch(
            "app.services.realtime.client.get_capabilities",
            side_effect=StreamError("unavailable", "test", 503),
        ) as get_capabilities:
            response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual([model["id"] for model in response.json()["data"]], [MODEL_ID])
        self.assertEqual(response.json()["data"][0]["owned_by"], "netease-youdao")
        get_capabilities.assert_not_called()

    def test_declared_models_share_one_offline_and_realtime_entry(self) -> None:
        runtime = SimpleNamespace(
            resolve_model_id=Mock(return_value=MODEL_ID),
            get_loaded_model_ids=Mock(return_value=[MODEL_ID]),
        )
        with patch("app.api.v1.asr.get_runtime_router", return_value=runtime):
            response = self.client.get("/stream/v1/asr/models")
        self.assertEqual(response.status_code, 200, response.text)
        metadata = response.json()
        self.assertEqual(metadata["declared_count"], 1)
        entry = metadata["declared_entries"][0]
        self.assertEqual(entry["id"], MODEL_ID)
        self.assertTrue(entry["supports_realtime"])
        self.assertEqual(entry["offline_model"], entry["realtime_model"])
        self.assertEqual(metadata["runtime"]["loaded_model_ids"], [MODEL_ID])

    def test_health_does_not_borrow_busy_offline_engine(self):
        runtime = SimpleNamespace(
            resolve_model_id=Mock(return_value=MODEL_ID),
            get_loaded_model_ids=Mock(return_value=[MODEL_ID]),
            get_memory_usage=Mock(return_value={}),
            acquire_engine=AsyncMock(
                side_effect=AssertionError("health borrowed engine")
            ),
        )
        with patch("app.api.v1.asr.get_runtime_router", return_value=runtime):
            with patch("app.api.v1.asr.detect_device", return_value="cuda:0"):
                self.assertTrue(
                    self.client.get("/stream/v1/asr/health").json()["model_loaded"]
                )
                runtime.get_loaded_model_ids.return_value = []
                self.assertFalse(
                    self.client.get("/stream/v1/asr/health").json()["model_loaded"]
                )
        runtime.acquire_engine.assert_not_called()

    def test_realtime_auth_and_removed_chat_route(self):
        with patch.object(settings, "API_KEY", "secret-token-123"):
            self.assertEqual(self.client.get("/v1/config").status_code, 401)
            self.assertEqual(
                self.client.post("/v1/chat/completions", json={}).status_code, 404
            )
            response = self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret-token-123"},
                json={
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [],
                },
            )
            self.assertEqual(response.status_code, 404)
        with patch.object(settings, "R2T2_URL", ""):
            with self.client.websocket_connect("/v1/stream") as ws:
                ws.send_json({})
                self.assertEqual(ws.receive_json()["code"], "realtime_unavailable")
        for path in ("/ws/v1/asr", "/ws/v1/asr/funasr", "/ws/v1/asr/qwen"):
            with self.assertRaises(WebSocketDisconnect):
                with self.client.websocket_connect(path):
                    pass
