import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services.realtime.engine import Model
from app.services.realtime.protocol import (
    OFFLINE_CONCURRENCY,
    OFFLINE_TAIL_SAMPLES,
    StreamConfig,
    StreamError,
)
from app.services.realtime.server import create_app, offline_result
from test_realtime import FakeModel


class SharedServerTest(unittest.TestCase):
    def test_one_model_serves_streaming_and_complete_offline_audio(self) -> None:
        model = FakeModel()
        model.transcribe = AsyncMock(return_value="independent offline text")
        factory = Mock(return_value=model)
        with TestClient(create_app(factory, max_sessions=4)) as client:
            with client.websocket_connect("/v1/stream") as ws:
                ws.send_json({"context": "live"})
                self.assertTrue(ws.receive_json()["offline_transcription"])
                pcm = np.full(16000, 0.25, dtype="<f4").tobytes()
                result = client.post(
                    "/v1/transcribe?context=offline",
                    content=pcm,
                    headers={"Content-Type": "application/octet-stream"},
                )
                self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(result.json()["text"], "independent offline text")
                audio, context = model.transcribe.call_args.args
                np.testing.assert_array_equal(audio, np.full(16000, 0.25))
                self.assertEqual(context, "offline")
                ws.send_bytes(b"\0" * 10240)
                self.assertEqual(ws.receive_json()["delta"], "live")
                ws.send_text("end")
                self.assertEqual(ws.receive_json()["text"], "live!")
        factory.assert_called_once_with(4)

    def test_private_offline_validation_and_admission(self) -> None:
        model = FakeModel()
        model.transcribe = AsyncMock(return_value="ok")
        with patch.dict(os.environ, {"R2T2_INTERNAL_TOKEN": "private"}):
            app = create_app(lambda _: model)
        headers = {
            "Authorization": "Bearer private",
            "Content-Type": "application/octet-stream",
        }
        with TestClient(app) as client:
            self.assertEqual(
                client.post("/v1/transcribe", content=b"0000").status_code, 401
            )
            for body in (
                b"",
                b"odd",
                np.array([np.nan], dtype="<f4").tobytes(),
                b"0" * (60 * 16000 * 4 + 4),
            ):
                with self.subTest(length=len(body)):
                    response = client.post(
                        "/v1/transcribe", content=body, headers=headers
                    )
                    self.assertIn(response.status_code, (400, 413))
                    self.assertFalse(app.state.offline_active)
            app.state.offline_active = OFFLINE_CONCURRENCY
            self.assertEqual(
                client.post(
                    "/v1/transcribe", content=b"0000", headers=headers
                ).status_code,
                503,
            )
            app.state.offline_active = OFFLINE_CONCURRENCY - 1
            self.assertEqual(
                client.post(
                    "/v1/transcribe", content=b"0000", headers=headers
                ).status_code,
                200,
            )
        model.transcribe.assert_awaited_once()

    def test_one_async_llm_has_room_and_priority_for_both_modes(self) -> None:
        llm = SimpleNamespace(from_engine_args=Mock())
        args = Mock(side_effect=lambda **kwargs: kwargs)
        processor = SimpleNamespace(from_pretrained=Mock())
        modules = {
            "vllm": SimpleNamespace(SamplingParams=Mock(side_effect=lambda **kw: kw)),
            "vllm.engine.arg_utils": SimpleNamespace(AsyncEngineArgs=args),
            "vllm.transformers_utils.processors.qwen3_asr": SimpleNamespace(
                Qwen3ASRProcessor=processor
            ),
            "vllm.v1.engine.async_llm": SimpleNamespace(AsyncLLM=llm),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {}, clear=True),
            patch.object(settings, "DEVICE", "cuda:0"),
        ):
            model = Model(4)
        llm.from_engine_args.assert_called_once()
        options = llm.from_engine_args.call_args.args[0]
        self.assertEqual(options["max_num_seqs"], 4 + OFFLINE_CONCURRENCY)
        self.assertEqual(options["scheduling_policy"], "priority")
        self.assertEqual(options["max_model_len"], 16384)
        self.assertEqual(model.offline_sampling["max_tokens"], 4096)


class SharedGenerationTest(unittest.IsolatedAsyncioTestCase):
    def model(self) -> Model:
        model = Model.__new__(Model)
        model.chunk_samples = 2560
        model.processor = SimpleNamespace(
            apply_chat_template=lambda messages, **kw: messages[0]["content"]
        )
        model.tokenizer = SimpleNamespace(
            encode=lambda s, **kw: list(s), decode=lambda ids: "".join(ids)
        )
        model.sampling = {16: 16, 128: 128}
        model.offline_sampling = 4096
        return model

    async def test_offline_uses_same_engine_without_live_prefix(self) -> None:
        model = self.model()
        calls = []
        outputs = iter(
            [
                "language English<asr_text>fresh text.|",
                "language English<asr_text>fresh|",
                " text.",
            ]
        )

        async def generate(prompt, sampling, **kwargs):
            calls.append((prompt, sampling, kwargs))
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=next(outputs),
                        finish_reason="stop",
                    )
                ]
            )

        model.engine = SimpleNamespace(generate=generate, abort=AsyncMock())
        session = model.new_session(StreamConfig(context="live"))
        await model.push(session, np.ones(5120, dtype=np.float32))
        text = await model.transcribe(np.ones(16000, dtype=np.float32), "offline")
        self.assertEqual(text, "fresh text.")
        self.assertEqual(
            [c[0]["prompt"] for c in calls],
            ["live", "offline", "offlinelanguage English<asr_text>fresh"],
        )
        self.assertEqual([c[2]["priority"] for c in calls], [0, 10, 10])
        self.assertEqual([c[1] for c in calls], [16, 4096, 128])
        original = calls[1][0]["multi_modal_data"]["audio"][0]
        padded = calls[2][0]["multi_modal_data"]["audio"][0]
        self.assertEqual(len(original), 16000)
        np.testing.assert_array_equal(padded[:16000], original)
        np.testing.assert_array_equal(padded[16000:], np.zeros(OFFLINE_TAIL_SAMPLES))
        self.assertNotEqual(calls[0][2]["request_id"], calls[1][2]["request_id"])

    async def test_cancelling_finalization_aborts_only_the_pending_request(
        self,
    ) -> None:
        model = self.model()
        entered = asyncio.Event()
        ids = []

        async def generate(*args, request_id, **kwargs):
            ids.append(request_id)
            if len(ids) == 1:
                yield SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="language English<asr_text>Pending|",
                            finish_reason="stop",
                        )
                    ]
                )
            else:
                entered.set()
                await asyncio.Future()

        model.engine = SimpleNamespace(generate=generate, abort=AsyncMock())
        task = asyncio.create_task(model.transcribe(np.ones(16000), ""))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        model.engine.abort.assert_awaited_once_with(ids[-1])

    async def test_silence_does_not_generate_an_unprompted_tail(self) -> None:
        model = self.model()
        calls = []

        async def generate(*args, **kwargs):
            calls.append(kwargs)
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="language None<asr_text>", finish_reason="stop"
                    )
                ]
            )

        model.engine = SimpleNamespace(generate=generate, abort=AsyncMock())
        self.assertEqual(await model.transcribe(np.zeros(16000), ""), "")
        self.assertEqual(len(calls), 1)

    async def test_disconnect_aborts_shared_engine_offline_request(self) -> None:
        model = self.model()

        async def generate(*args, **kwargs):
            await asyncio.Future()
            yield None

        model.engine = SimpleNamespace(generate=generate, abort=AsyncMock())
        request = SimpleNamespace(
            receive=AsyncMock(return_value={"type": "http.disconnect"})
        )
        with self.assertRaises(StreamError) as error:
            await offline_result(request, model, np.ones(16000), "")
        self.assertEqual(error.exception.code, "client_disconnected")
        model.engine.abort.assert_awaited_once()
        self.assertTrue(model.engine.abort.call_args.args[0].startswith("offline-"))
