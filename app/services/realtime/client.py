"""A bounded client for the private R2T2 service."""

import asyncio
import json
from contextlib import asynccontextmanager
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from app.core.config import settings

from .protocol import (
    MODEL_ID,
    OFFLINE_MAX_SAMPLES,
    PROTOCOL_VERSION,
    SAMPLE_RATE,
    StreamError,
)


def endpoint(path, *, websocket=False):
    if not settings.R2T2_URL:
        raise StreamError(
            "realtime_unavailable", "Realtime service is not configured", 503
        )
    parts = urlsplit(settings.R2T2_URL)
    if (
        parts.scheme not in ("http", "https")
        or not parts.netloc
        or parts.query
        or parts.fragment
    ):
        raise StreamError(
            "realtime_unavailable", "Invalid realtime service configuration", 503
        )
    scheme = {"http": "ws", "https": "wss"}[parts.scheme] if websocket else parts.scheme
    return urlunsplit((scheme, parts.netloc, parts.path.rstrip("/") + path, "", ""))


def internal_headers():
    return (
        {"Authorization": f"Bearer {settings.R2T2_INTERNAL_TOKEN}"}
        if settings.R2T2_INTERNAL_TOKEN
        else {}
    )


def get_engine_capabilities() -> dict[str, object]:
    try:
        with urlopen(
            Request(endpoint("/v1/config"), headers=internal_headers()), timeout=5
        ) as response:
            data = response.read(65537)
        if len(data) > 65536:
            raise ValueError("Backend configuration exceeds response limit")
        result = json.loads(data)
        if (
            not isinstance(result, dict)
            or result.get("model") != MODEL_ID
            or result.get("sample_rate") != SAMPLE_RATE
            or result.get("protocol_version") != PROTOCOL_VERSION
            or result.get("offline_transcription") is not True
        ):
            raise ValueError("Backend protocol/model mismatch")
        return result
    except StreamError:
        raise
    except Exception as error:
        raise StreamError(
            "realtime_unavailable",
            "Shared R2T2 backend is unavailable or incompatible",
            503,
        ) from error


async def get_capabilities() -> dict[str, object]:
    return await asyncio.to_thread(get_engine_capabilities)


def transcribe_segment(audio: np.ndarray, context: str = "") -> str:
    if (
        audio.ndim != 1
        or not 0 < len(audio) <= OFFLINE_MAX_SAMPLES
        or not np.isfinite(audio).all()
    ):
        raise ValueError("Expected 1 sample to 60 seconds of finite mono 16 kHz audio")
    if len(context) > 2048:
        raise ValueError("Transcription context exceeds 2048 characters")
    request = Request(
        endpoint("/v1/transcribe") + "?" + urlencode({"context": context}),
        data=audio.astype("<f4").tobytes(),
        headers={**internal_headers(), "Content-Type": "application/octet-stream"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            payload = response.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise ValueError("Offline response exceeds response limit")
        result = json.loads(payload)
        if (
            not isinstance(result, dict)
            or result.get("error")
            or not isinstance(result.get("text"), str)
        ):
            raise ValueError("Invalid offline response")
        return result["text"]
    except HTTPError as error:
        error.close()
        raise StreamError(
            "upstream_error", "Shared R2T2 transcription request failed", error.code
        ) from error
    except Exception as error:
        raise StreamError(
            "upstream_error", "Shared R2T2 transcription failed", 502
        ) from error


@asynccontextmanager
async def open_stream(config):
    connection = None
    try:
        connection = await websockets.connect(
            endpoint("/v1/stream", websocket=True),
            additional_headers=internal_headers(),
            open_timeout=5,
            close_timeout=3,
            ping_interval=10,
            ping_timeout=20,
            max_size=1024 * 1024,
            max_queue=8,
        )
        try:
            await connection.send(config.model_dump_json())
        except ConnectionClosed:
            # Admission can reject before our config write; recv still drains
            # the buffered error frame before surfacing the close handshake.
            pass
        ready = json.loads(await asyncio.wait_for(connection.recv(), 10))
        if ready.get("error"):
            raise StreamError(ready.get("code", "upstream_error"), ready["error"], 503)
        if (
            not ready.get("ready")
            or ready.get("model") != MODEL_ID
            or ready.get("protocol_version") != PROTOCOL_VERSION
        ):
            raise StreamError(
                "realtime_unavailable", "Realtime backend protocol mismatch", 503
            )
    except StreamError:
        if connection:
            await connection.close()
        raise
    except asyncio.CancelledError:
        if connection is not None:
            await connection.close()
        raise
    except Exception as error:
        if connection:
            await connection.close()
        raise StreamError(
            "realtime_unavailable", "Could not establish realtime session", 503
        ) from error
    try:
        yield connection, ready
    finally:
        await connection.close()


async def receive_event(connection, timeout=30):
    try:
        result = json.loads(await asyncio.wait_for(connection.recv(), timeout))
    except Exception as error:
        raise StreamError(
            "upstream_disconnected", "Realtime backend stopped responding", 502
        ) from error
    if result.get("error"):
        raise StreamError(result.get("code", "upstream_error"), result["error"], 502)
    return result
