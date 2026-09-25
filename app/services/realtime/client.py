"""A bounded client for the private R2T2 service."""

import asyncio
import json
from contextlib import asynccontextmanager
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import websockets
from websockets.exceptions import ConnectionClosed

from app.core.config import settings

from .protocol import MODEL_ID, PROTOCOL_VERSION, SAMPLE_RATE, StreamError


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


async def get_capabilities():
    def fetch():
        with urlopen(
            Request(endpoint("/v1/config"), headers=internal_headers()), timeout=5
        ) as response:
            return json.loads(response.read(65536))

    try:
        result = await asyncio.to_thread(fetch)
        if (
            result.get("model") != MODEL_ID
            or result.get("sample_rate") != SAMPLE_RATE
            or result.get("protocol_version") != PROTOCOL_VERSION
        ):
            raise ValueError("Backend protocol/model mismatch")
        return result
    except StreamError:
        raise
    except Exception as error:
        raise StreamError(
            "realtime_unavailable",
            "Realtime backend is unavailable or incompatible",
            503,
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
