import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.services.realtime.engine import DISCARD_SAMPLES, WINDOW_SAMPLES, Model, Session
from app.services.realtime.protocol import (
    CHUNK_SAMPLES,
    MODEL_ID,
    AudioQueue,
    StreamConfig,
    StreamError,
    validate_pcm,
)
from app.services.realtime.server import create_app


class FakeModel:
    def __init__(self, capacity=4):
        self.chunk_samples = CHUNK_SAMPLES
        self.aborted = []
        self.calls = []

    async def warmup(self):
        pass

    async def shutdown(self):
        pass

    def new_session(self, config):
        return Session(config.context)

    async def push(self, session, audio, *, final=False):
        self.calls.append((session.id, len(audio), final))
        session.append(audio)
        session.steps += 1
        delta = session.prompt if len(audio) else "!" if session.text else ""
        session.text += delta
        await asyncio.sleep(0)
        return delta

    async def abort(self, session):
        self.aborted.append(session.id)


class ProtocolTest(unittest.TestCase):
    def test_pcm_and_config(self):
        self.assertEqual(validate_pcm(b"\0\0" * 2560), 2560)
        for data in (b"", b"a", b"\0" * 32002):
            with self.assertRaises(StreamError):
                validate_pcm(data)
        for config in (
            {"language": "Chinese"},
            {"temperature": 1},
            {"context": "x" * 2049},
        ):
            with self.assertRaises(ValidationError):
                StreamConfig.model_validate(config)

    def test_rolling_audio_and_text_stay_aligned(self):
        state = Session("")
        for i in range(250):
            count = state.next_samples
            state.append(np.full(count, i, dtype=np.float32))
            prefix = "".join(s for _, s in state.parts)
            state.commit(prefix + f" word{i}", prefix)
            state.steps += 1
            self.assertLessEqual(state.size, WINDOW_SAMPLES)
            self.assertEqual(state.size + state.offset, state.samples)
            self.assertTrue(all(end > state.offset for end, _ in state.parts))
            self.assertLessEqual(len(state.parts), 100)
        self.assertGreaterEqual(state.offset, 3 * DISCARD_SAMPLES)
        self.assertEqual(state.text, "".join(f" word{i}" for i in range(250)))

    def test_multiple_clients_capacity_and_disconnect_recovery(self):
        model = FakeModel()
        app = create_app(lambda _: model, max_sessions=2)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/stream") as first:
                first.send_json({"context": "中文"})
                self.assertEqual(first.receive_json()["model"], MODEL_ID)
                with client.websocket_connect("/v1/stream") as second:
                    second.send_json({"context": " English"})
                    second.receive_json()
                    with client.websocket_connect("/v1/stream") as third:
                        self.assertEqual(
                            third.receive_json()["code"], "capacity_exceeded"
                        )
                    first.send_bytes(b"\0" * (4 * CHUNK_SAMPLES))
                    second.send_bytes(b"\0" * (4 * CHUNK_SAMPLES))
                    self.assertEqual(first.receive_json()["delta"], "中文")
                    self.assertEqual(second.receive_json()["delta"], " English")
                    first.send_text("end")
                    final = first.receive_json()
                    self.assertEqual(final["text"], "中文!")
                    self.assertTrue(final["done"])
                    self.assertEqual(final["audio_ms"], 320)
                    self.assertEqual((final["utterance"], final["utterance_end"]), (0, True))
                # TestClient context exit waits for the disconnected handler.
            self.assertEqual(client.get("/v1/config").json()["active_sessions"], 0)
            self.assertEqual(len(set(model.aborted)), 2)
            with client.websocket_connect("/v1/stream") as ws:
                ws.send_json({})
                self.assertTrue(ws.receive_json()["ready"])
                ws.send_text("end")
                self.assertEqual(ws.receive_json()["text"], "")

    def test_short_tail_invalid_input_and_auth(self):
        model = FakeModel()
        with TestClient(create_app(lambda _: model)) as client:
            with client.websocket_connect("/v1/stream") as ws:
                ws.send_json({"context": "short"})
                ws.receive_json()
                ws.send_bytes(b"\0" * 22)
                ws.send_text("end")
                self.assertEqual(ws.receive_json()["text"], "short")
            self.assertEqual(model.calls[-1][1:], (11, True))
            with client.websocket_connect("/v1/stream") as ws:
                ws.send_json({})
                ws.receive_json()
                ws.send_bytes(b"odd")
                self.assertEqual(ws.receive_json()["code"], "invalid_audio")
            with client.websocket_connect("/v1/stream") as ws:
                ws.send_json({"language": "en"})
                self.assertEqual(ws.receive_json()["code"], "invalid_config")
        with patch.dict(os.environ, {"R2T2_INTERNAL_TOKEN": "private"}):
            app = create_app(lambda _: model)
        with TestClient(app) as client:
            self.assertEqual(client.get("/v1/config").status_code, 401)
            self.assertEqual(
                client.get(
                    "/v1/config", headers={"Authorization": "Bearer private"}
                ).status_code,
                200,
            )


class AsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_slow_decode_cadence_still_flushes_short_pauses(self):
        model = Model.__new__(Model)
        model.sampling = {128: object()}

        async def generate(*args, **kwargs):
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="language English<asr_text>Hello.", finish_reason="stop"
                    )
                ]
            )

        model.engine = SimpleNamespace(generate=generate, abort=AsyncMock())
        state = Session("prompt", chunk_samples=10240)
        self.assertEqual(await model.push(state, np.ones(5120)), "")
        self.assertEqual(await model.push(state, np.ones(2560)), "")
        self.assertEqual(await model.push(state, np.zeros(2560)), "")
        self.assertEqual(await model.push(state, np.zeros(2560)), "Hello.")
        self.assertEqual(state.size, 0)
        self.assertEqual(state.header, "")
        self.assertEqual(state.next_decode, state.samples + 20480)

    async def test_pause_flushes_and_releases_previous_language_prefix(self):
        model = Model.__new__(Model)
        model.tokenizer = SimpleNamespace(
            encode=lambda s, **_: list(s), decode=lambda ids: "".join(ids)
        )
        model.sampling = {n: n for n in range(4, 129)}
        outputs = iter(
            [
                "language English<asr_text>Hello ",
                " world|",
                "d.|",
                ".",
                "language Chinese<asr_text>你好！|",
            ]
        )
        prompts = []

        async def generate(prompt, *args, **kwargs):
            prompts.append(prompt["prompt"])
            yield SimpleNamespace(
                outputs=[SimpleNamespace(text=next(outputs), finish_reason="stop")]
            )

        model.engine = SimpleNamespace(generate=generate, abort=AsyncMock())
        state = Session("prompt")
        deltas = [await model.push(state, np.full(5120, 0.1, dtype=np.float32))]
        for _ in range(2):
            deltas.append(await model.push(state, np.zeros(2560, dtype=np.float32)))
        self.assertEqual(state.text, "Hello world.")
        self.assertEqual(state.size, 0)
        self.assertEqual(state.offset, state.samples)
        self.assertEqual(state.utterance, 1)
        self.assertEqual(state.next_samples, 5120)
        deltas.append(await model.push(state, np.full(5120, 0.1, dtype=np.float32)))
        self.assertEqual(prompts[-1], "prompt")
        self.assertEqual(state.text, "Hello world.你好")
        self.assertEqual("".join(deltas), state.text)

    async def test_bounded_queue_abort_wakes_blocked_producer(self):
        queue = AudioQueue(4)
        await queue.put(b"1234")
        waiting = asyncio.create_task(queue.put(b"56"))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        await queue.abort()
        with self.assertRaises(StreamError):
            await waiting
        self.assertIsNone(await queue.get())
        self.assertEqual(queue.size, 0)

    async def test_native_async_engine_flush_preserves_english_spaces(self):
        model = Model.__new__(Model)
        model.tokenizer = SimpleNamespace(
            encode=lambda s, **_: list(s), decode=lambda ids: "".join(ids)
        )
        model.sampling = {n: n for n in range(4, 129)}
        outputs = iter(["language English<asr_text>Hello |", " world|", "d."])

        async def generate(*args, **kwargs):
            yield SimpleNamespace(
                outputs=[SimpleNamespace(text=next(outputs), finish_reason="stop")]
            )

        model.engine = SimpleNamespace(generate=generate, abort=AsyncMock())
        state = Session("prompt")
        await model.push(state, np.zeros(5120, dtype=np.float32))
        await model.push(state, np.zeros(2560, dtype=np.float32))
        await model.push(state, np.empty(0, dtype=np.float32), final=True)
        self.assertEqual(state.text, "Hello world.")
        self.assertEqual(state.samples, 7680)

    async def test_cancel_aborts_vllm_request(self):
        entered = asyncio.Event()

        async def generate(*args, **kwargs):
            entered.set()
            await asyncio.Future()
            yield None

        model = Model.__new__(Model)
        model.sampling = {16: object()}
        model.engine = SimpleNamespace(generate=generate, abort=AsyncMock())
        state = Session("prompt")
        work = asyncio.create_task(model.push(state, np.zeros(5120, dtype=np.float32)))
        await entered.wait()
        work.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await work
        model.engine.abort.assert_awaited_once_with(state.id + "-0")
        self.assertIsNone(state.request_id)

    async def test_none_language_header_survives_silence_and_mixed_speech(self):
        model = Model.__new__(Model)
        model.tokenizer = SimpleNamespace(
            encode=lambda s, **_: list(s), decode=lambda ids: "".join(ids)
        )
        model.sampling = {n: n for n in range(4, 129)}
        outputs = iter(["language None<asr_text>", "之前有|", "hello!|"])
        prompts = []

        async def generate(prompt, *args, **kwargs):
            prompts.append(prompt["prompt"])
            yield SimpleNamespace(
                outputs=[SimpleNamespace(text=next(outputs), finish_reason="stop")]
            )

        model.engine = SimpleNamespace(generate=generate, abort=AsyncMock())
        state = Session("prompt")
        for n in (5120, 2560, 2560):
            await model.push(state, np.zeros(n, dtype=np.float32))
        self.assertEqual(state.text, "之前hello")
        self.assertIn("language None<asr_text>之前", prompts[-1])
