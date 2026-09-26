"""One Rust context shared by prioritized realtime and offline requests."""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
from transformers import AutoTokenizer

from app.core.executor import run_sync, wait_for_completion
from app.services.asr.rust_backend import RustBackend

AUDIO_PAD = 151676


class AudioData(TypedDict):
    audio: list[np.ndarray]


class Prompt(TypedDict):
    prompt: str
    multi_modal_data: AudioData


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int
    temperature: float = 0
    skip_special_tokens: bool = True

    def __post_init__(self) -> None:
        if (
            self.temperature != 0
            or type(self.max_tokens) is not int
            or not 1 <= self.max_tokens <= 4096
        ):
            raise ValueError(
                "Rust decoding requires temperature=0 and max_tokens between 1 and 4096"
            )


@dataclass(frozen=True)
class Output:
    text: str
    finish_reason: str


@dataclass(frozen=True)
class RequestOutput:
    outputs: list[Output]


@dataclass
class _Request:
    audio: np.ndarray
    before: list[int]
    after: list[int]
    sampling: SamplingParams
    future: asyncio.Future[RequestOutput]


class RustAsyncEngine:
    def __init__(self, max_sessions: int) -> None:
        if not 1 <= max_sessions <= 64:
            raise ValueError("max_sessions must be between 1 and 64")
        self.backend = RustBackend()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.backend.model_path, local_files_only=True, fix_mistral_regex=True
            )
            self.tokenizer.chat_template = json.loads(
                (Path(self.backend.model_path) / "chat_template.json").read_text()
            )["chat_template"]
        except BaseException:
            self.backend.close()
            raise
        self._queue: asyncio.PriorityQueue[tuple[int, int, _Request]] = (
            asyncio.PriorityQueue(maxsize=2 * (max_sessions + 1))
        )
        self._sequence = itertools.count()
        self._requests: dict[str, asyncio.Future[RequestOutput]] = {}
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    async def generate(
        self,
        prompt: Prompt,
        sampling: SamplingParams,
        request_id: str,
        priority: int = 0,
    ) -> AsyncIterator[RequestOutput]:
        if self._closed:
            raise RuntimeError("Rust inference engine is closed")
        if request_id in self._requests:
            raise ValueError("Duplicate inference request ID")
        ids = self.tokenizer.encode(prompt["prompt"], add_special_tokens=False)
        if ids.count(AUDIO_PAD) != 1:
            raise ValueError(
                "Rust audio prompt must contain exactly one audio pad token"
            )
        audio = prompt["multi_modal_data"]["audio"]
        if len(audio) != 1:
            raise ValueError("Rust inference accepts exactly one audio input")
        position = ids.index(AUDIO_PAD)
        future = asyncio.get_running_loop().create_future()
        request = _Request(
            audio[0], ids[:position], ids[position + 1 :], sampling, future
        )
        self._requests[request_id] = future
        try:
            try:
                self._queue.put_nowait((priority, next(self._sequence), request))
            except asyncio.QueueFull as exc:
                raise RuntimeError("Rust inference queue is full") from exc
            if self._worker is None:
                self._worker = asyncio.create_task(self._serve())
            yield await future
        finally:
            future.cancel()
            self._requests.pop(request_id, None)

    async def _serve(self) -> None:
        # ponytail: decode is nonpreemptive; reduce offline segments if latency matters.
        while True:
            _, _, request = await self._queue.get()
            try:
                if request.future.cancelled():
                    continue
                result = await run_sync(
                    self.backend.generate_pcm,
                    request.audio,
                    request.before,
                    request.after,
                    request.sampling.max_tokens,
                )
                if not request.future.done():
                    text = self.tokenizer.decode(
                        result.token_ids,
                        skip_special_tokens=request.sampling.skip_special_tokens,
                    )
                    request.future.set_result(
                        RequestOutput([Output(text, result.finish_reason)])
                    )
            except asyncio.CancelledError:
                request.future.cancel()
                raise
            except Exception as exc:
                if not request.future.done():
                    request.future.set_exception(exc)
            finally:
                self._queue.task_done()

    async def abort(self, request_id: str) -> None:
        future = self._requests.get(request_id)
        if future is not None:
            future.cancel()

    async def shutdown(self) -> None:
        self._closed = True
        for future in self._requests.values():
            future.cancel()
        try:
            if self._worker is not None:
                self._worker.cancel()
                await wait_for_completion(
                    asyncio.gather(self._worker, return_exceptions=True)
                )
        finally:
            self.backend.close()
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()
