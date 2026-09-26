"""Public native WebSocket relay; no local models or text postprocessors."""

import asyncio
import logging
from contextlib import suppress

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.security import validate_websocket_token

from .client import open_stream, receive_event
from .protocol import (
    MAX_SECONDS,
    SAMPLE_RATE,
    StreamConfig,
    StreamError,
    supervise,
    validate_pcm,
)
from .speakers import SpeakerStream

logger = logging.getLogger(__name__)


async def handle_stream(websocket: WebSocket):
    valid, _ = validate_websocket_token(websocket)
    if not valid:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        config = StreamConfig.model_validate(
            await asyncio.wait_for(websocket.receive_json(), 10)
        )
        speakers = SpeakerStream()
        async with open_stream(config) as (upstream, ready):
            lock = asyncio.Lock()

            async def send(*events):
                async with lock:  # Audio and result tasks both emit labels.
                    for event in events:
                        await asyncio.wait_for(websocket.send_json(event), 5)

            await send(dict(ready, speaker_diarization=speakers.model is not None))

            async def send_audio():
                samples = 0
                ended = False
                while True:
                    message = await asyncio.wait_for(websocket.receive(), 30)
                    if message["type"] == "websocket.disconnect":
                        return
                    if ended:
                        raise StreamError(
                            "session_finished", "No messages are allowed after end"
                        )
                    data = message.get("bytes")
                    if data is not None:
                        samples += validate_pcm(data)
                        if (
                            samples
                            > min(MAX_SECONDS, ready["max_session_seconds"])
                            * SAMPLE_RATE
                        ):
                            raise StreamError(
                                "session_limit", "Session audio duration limit exceeded"
                            )
                        await asyncio.wait_for(upstream.send(data), 5)
                        speakers.feed(data)
                        await send(*await speakers.advance())
                    elif message.get("text") == "end":
                        ended = True
                        await asyncio.wait_for(upstream.send("end"), 5)
                    else:
                        raise StreamError(
                            "invalid_message", "Send int16 PCM frames or the text end"
                        )

            async def send_results():
                start_ms, spoken = 0, False
                while True:
                    event = await receive_event(upstream)
                    spoken = spoken or bool(event["delta"].strip())
                    if event["utterance_end"]:
                        if spoken:
                            speakers.utterance(
                                event["utterance"], start_ms, event["audio_ms"]
                            )
                        start_ms, spoken = event["audio_ms"], False
                    if event["done"]:
                        # Every label precedes done, so clients can close on it.
                        speakers.ended = True
                        await send(*await speakers.advance(), event)
                        return
                    await send(event, *speakers.ready())

            await supervise(send_audio(), send_results())
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
            logger.exception("Realtime gateway failed")
            code, message = "stream_failed", "Realtime session failed"
        with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
            await asyncio.wait_for(
                websocket.send_json({"error": message, "code": code}), 2
            )
    finally:
        with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
            await asyncio.wait_for(websocket.close(), 2)
