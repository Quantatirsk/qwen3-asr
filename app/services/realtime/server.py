"""One async inference engine and one admission counter for all API workers."""

import asyncio
import hmac
import logging
import os
import time
from contextlib import asynccontextmanager, suppress

import numpy as np
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.core.config import settings

from .engine import Model
from .protocol import (
    CHUNK_SAMPLES,
    MAX_SECONDS,
    MODEL_ID,
    MODEL_REVISION,
    OFFLINE_MAX_BYTES,
    PROTOCOL_VERSION,
    SAMPLE_RATE,
    AudioQueue,
    StreamConfig,
    StreamError,
    supervise,
    validate_pcm,
)

logger = logging.getLogger(__name__)


async def offline_result(
    request: Request, model: Model, audio: np.ndarray, context: str
) -> str:
    async def disconnected() -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass

    inference = asyncio.create_task(model.transcribe(audio, context))
    disconnect = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait(
            {inference, disconnect}, timeout=120, return_when=asyncio.FIRST_COMPLETED
        )
        if inference in done:
            return inference.result()
        if disconnect in done:
            raise StreamError("client_disconnected", "Offline client disconnected", 499)
        raise StreamError("inference_timeout", "Offline inference timed out", 504)
    finally:
        for task in (inference, disconnect):
            task.cancel()
        await asyncio.gather(inference, disconnect, return_exceptions=True)


def create_app(model_factory=Model, *, max_sessions=None):
    capacity = (
        max_sessions
        if max_sessions is not None
        else int(
            os.getenv("R2T2_MAX_SESSIONS", "1" if settings.DEVICE == "cpu" else "4")
        )
    )
    if not 1 <= capacity <= 64:
        raise ValueError("R2T2_MAX_SESSIONS must be between 1 and 64")
    token = os.getenv("R2T2_INTERNAL_TOKEN", "")

    @asynccontextmanager
    async def lifespan(app):
        model = model_factory(capacity)
        app.state.model = model
        app.state.chunk_samples = model.chunk_samples
        try:
            await model.warmup()
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            await model.shutdown()

    app = FastAPI(title="R2T2 inference", lifespan=lifespan)
    app.state.active = 0
    app.state.ready = False
    app.state.offline_active = False
    app.state.chunk_samples = CHUNK_SAMPLES

    def authorized(headers):
        return not token or hmac.compare_digest(
            headers.get("authorization", ""), f"Bearer {token}"
        )

    def capabilities():
        return {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "ready": app.state.ready,
            "protocol_version": PROTOCOL_VERSION,
            "sample_rate": SAMPLE_RATE,
            "format": "int16_le",
            "channels": 1,
            "chunk_seconds": app.state.chunk_samples / SAMPLE_RATE,
            "max_frame_seconds": 1,
            "max_session_seconds": MAX_SECONDS,
            "max_sessions": capacity,
            "active_sessions": app.state.active,
            "language": "auto",
            "word_timestamps": False,
            "speaker_diarization": False,
            "offline_transcription": True,
            "offline_max_seconds": 60,
        }

    @app.get("/health")
    async def health():
        return JSONResponse(
            {"ready": app.state.ready, "active_sessions": app.state.active},
            status_code=200 if app.state.ready else 503,
        )

    @app.get("/v1/config")
    async def config(request: Request):
        if not authorized(request.headers):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return capabilities()

    @app.post("/v1/transcribe")
    async def transcribe(request: Request, context: str = Query("", max_length=2048)):
        if not authorized(request.headers):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        if not app.state.ready or app.state.offline_active:
            return JSONResponse(
                {"error": "Offline inference capacity is exhausted"}, status_code=503
            )
        # Reserve before reading the body, bounding both retained audio and GPU jobs.
        app.state.offline_active = True
        try:
            if request.headers.get("content-type") != "application/octet-stream":
                raise StreamError("invalid_audio", "Expected float32 PCM audio")

            async def read_audio() -> bytes:
                data = bytearray()
                async for chunk in request.stream():
                    if len(data) + len(chunk) > OFFLINE_MAX_BYTES:
                        raise StreamError(
                            "invalid_audio", "Offline segment exceeds 60 seconds", 413
                        )
                    data.extend(chunk)
                if not data or len(data) % 4:
                    raise StreamError("invalid_audio", "Expected nonempty float32 PCM")
                return bytes(data)

            data = await asyncio.wait_for(read_audio(), 30)
            audio = np.frombuffer(data, dtype="<f4")
            if not np.isfinite(audio).all():
                raise StreamError("invalid_audio", "Audio samples must be finite")
            text = await offline_result(request, app.state.model, audio, context)
            return {"text": text}
        except StreamError as error:
            return JSONResponse(
                {"error": str(error), "code": error.code}, status_code=error.status
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                {"error": "Offline audio upload timed out"}, status_code=408
            )
        except Exception:
            logger.exception("Offline R2T2 inference failed")
            return JSONResponse({"error": "Offline inference failed"}, status_code=500)
        finally:
            app.state.offline_active = False

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket):
        if not authorized(ws.headers):
            await ws.close(code=1008)
            return
        # Check and reserve without awaiting: atomic on the engine event loop.
        if not app.state.ready or app.state.active >= capacity:
            await ws.accept()
            await ws.send_json(
                {"error": "Realtime capacity is exhausted", "code": "capacity_exceeded"}
            )
            await ws.close(code=1013)
            return
        app.state.active += 1
        queue = AudioQueue()
        session = None
        model = app.state.model
        try:
            await ws.accept()
            config = StreamConfig.model_validate(
                await asyncio.wait_for(ws.receive_json(), 10)
            )
            session = model.new_session(config)
            await asyncio.wait_for(
                ws.send_json(dict(capabilities(), session_id=session.id)), 5
            )

            async def receive():
                samples = 0
                ended = False
                while True:
                    message = await asyncio.wait_for(ws.receive(), 30)
                    if message["type"] == "websocket.disconnect":
                        return
                    if ended:
                        raise StreamError(
                            "session_finished", "No messages are allowed after end"
                        )
                    data = message.get("bytes")
                    if data is not None:
                        samples += validate_pcm(data)
                        if samples > MAX_SECONDS * SAMPLE_RATE:
                            raise StreamError(
                                "session_limit", "Maximum session duration exceeded"
                            )
                        await asyncio.wait_for(queue.put(data), 5)
                    elif message.get("text") == "end":
                        ended = True
                        await queue.finish()
                    else:
                        raise StreamError("invalid_message", "Expected PCM or end")

            async def consume():
                buffer = bytearray()

                async def decode(data, final=False):
                    started = time.perf_counter()
                    utterance = session.utterance
                    audio = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
                    delta = await asyncio.wait_for(
                        model.push(session, audio, final=final), 25
                    )
                    event = {
                        "delta": delta,
                        "audio_ms": round(session.samples * 1000 / SAMPLE_RATE),
                        "inference_ms": round(
                            (time.perf_counter() - started) * 1000, 1
                        ),
                        "done": final,
                        # This delta belongs to `utterance`; a pause or the end
                        # closes it at `audio_ms`.
                        "utterance": utterance,
                        "utterance_end": final or session.utterance != utterance,
                    }
                    if final:
                        event["text"] = session.text
                    await asyncio.wait_for(ws.send_json(event), 5)

                while (data := await queue.get()) is not None:
                    buffer.extend(data)
                    while len(buffer) >= session.next_samples * 2:
                        size = session.next_samples * 2
                        audio = bytes(buffer[:size])
                        del buffer[:size]
                        await decode(audio)
                # Always flush: even exact-sized input may have a withheld token.
                await decode(bytes(buffer), final=True)

            await supervise(receive(), consume())
        except WebSocketDisconnect:
            pass
        except Exception as error:
            if isinstance(error, StreamError):
                code, message = error.code, str(error)
            elif isinstance(error, (ValidationError, ValueError)):
                code, message = "invalid_config", "Invalid stream configuration"
            elif isinstance(error, asyncio.TimeoutError):
                code, message = (
                    "session_timeout",
                    "Session timed out or consumer is too slow",
                )
            else:
                logger.exception("R2T2 session failed")
                code, message = "inference_failed", "Realtime inference failed"
            with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
                await asyncio.wait_for(
                    ws.send_json({"code": code, "error": message}), 2
                )
        finally:
            try:
                await queue.abort()
                if session is not None:
                    await model.abort(session)
            finally:
                app.state.active -= 1
                with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
                    await asyncio.wait_for(ws.close(), 2)

    return app
