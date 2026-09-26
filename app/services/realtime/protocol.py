"""Shared, model-free wire validation for the gateway and inference server."""

import asyncio
from collections import deque

from pydantic import BaseModel, ConfigDict, Field

MODEL_ID = "confucius4-r2t2"
MODEL_REPOSITORY = "netease-youdao/Confucius4-R2T2"
MODEL_REVISION = "185ce639118ad1362d049ca0d8ed04b6ec5cd6c9"
UPSTREAM_REVISION = "26d55a54ce5670cff9947a167d8ed95d569fd4d9"
SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2
CHUNK_SAMPLES = 2560  # 160 ms; the first decode has 160 ms lookahead.
MAX_SECONDS = 3600
OFFLINE_TAIL_SAMPLES = 4 * CHUNK_SAMPLES
OFFLINE_MAX_SAMPLES = 60 * SAMPLE_RATE
OFFLINE_MAX_BYTES = OFFLINE_MAX_SAMPLES * 4
PROTOCOL_VERSION = 1


class StreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    context: str = Field(default="", max_length=2048)


class StreamError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def validate_pcm(data: bytes) -> int:
    if not data or len(data) % 2 or len(data) > BYTES_PER_SECOND:
        raise StreamError(
            "invalid_audio", "Expected 1 sample to 1 second of 16 kHz mono int16 LE PCM"
        )
    return len(data) // 2


class AudioQueue:
    """Bound retained PCM bytes; a slow consumer must not retain unlimited audio."""

    def __init__(self, max_bytes: int = BYTES_PER_SECOND * 10):
        self.max_bytes = max_bytes
        self.size = 0
        self.frames = deque()
        self.finished = False
        self.condition = asyncio.Condition()

    async def put(self, data: bytes):
        if len(data) > self.max_bytes:
            raise StreamError("invalid_audio", "Audio frame exceeds queue capacity")
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.finished or self.size + len(data) <= self.max_bytes
            )
            if self.finished:
                raise StreamError("session_finished", "Audio queue is closed")
            self.frames.append(data)
            self.size += len(data)
            self.condition.notify_all()

    async def get(self):
        async with self.condition:
            await self.condition.wait_for(lambda: self.frames or self.finished)
            if not self.frames:
                return None
            data = self.frames.popleft()
            self.size -= len(data)
            self.condition.notify_all()
            return data

    async def finish(self):
        async with self.condition:
            self.finished = True
            self.condition.notify_all()

    async def abort(self):
        async with self.condition:
            self.frames.clear()
            self.size = 0
            self.finished = True
            self.condition.notify_all()


async def supervise(*coroutines):
    """End a duplex session when either peer finishes; always reap every task."""
    tasks = [asyncio.create_task(coro) for coro in coroutines]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
