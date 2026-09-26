"""One serialized offline pipeline with cancellation-safe ownership."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from app.core.executor import run_sync
from app.services.asr.engines import ASRFullResult
from app.services.asr.long_audio import OfflineASRRequest
from app.services.asr.manager import get_model_manager

if TYPE_CHECKING:
    from app.services.asr.r2t2_engine import R2T2Engine


class RuntimeEngineLease:
    def __init__(self, engine: R2T2Engine, release: Callable[[], None]) -> None:
        self.engine = engine
        self._release = release
        self._closed = False

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._release()

    async def __aenter__(self) -> R2T2Engine:
        return self.engine

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()


class RuntimeRouter:
    def __init__(self) -> None:
        self._manager = get_model_manager()
        self._engine: R2T2Engine | None = None
        self._init_lock = threading.Lock()
        # The synchronous forced aligner cannot overlap calls.
        self._inference_lock = asyncio.Lock()

    def resolve_model_id(self, model_id: str | None) -> str:
        return self._manager.get_declared_entry_config(model_id).model_id

    def _get_engine(self, model_id: str) -> R2T2Engine:
        with self._init_lock:
            if self._engine is None:
                self._engine = self._manager.create_engine(model_id)
            return self._engine

    def warmup_model(self, model_id: str | None = None) -> None:
        self._get_engine(self.resolve_model_id(model_id))

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def get_loaded_model_ids(self) -> list[str]:
        return [self._engine.model_id] if self._engine is not None else []

    def get_memory_usage(self) -> dict[str, object]:
        models = self.get_loaded_model_ids()
        memory: dict[str, object] = {
            "model_list": models,
            "loaded_count": len(models),
        }
        if torch.cuda.is_available():
            memory["gpu_memory"] = {
                "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                "cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB",
                "max_allocated": f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB",
            }
        return memory

    async def acquire_engine(self, model_id: str | None = None) -> RuntimeEngineLease:
        resolved = self.resolve_model_id(model_id)
        await self._inference_lock.acquire()
        try:
            engine = await run_sync(self._get_engine, resolved)
        except BaseException:
            self._inference_lock.release()
            raise
        return RuntimeEngineLease(engine, self._inference_lock.release)

    async def run_offline(self, request: OfflineASRRequest) -> ASRFullResult:
        async with await self.acquire_engine(request.model_id) as engine:
            return await run_sync(
                engine.transcribe_long_audio,
                audio_path=request.audio_path,
                hotwords=request.hotwords,
                enable_punctuation=request.enable_punctuation,
                enable_itn=request.enable_itn,
                sample_rate=request.sample_rate,
                enable_speaker_diarization=request.enable_speaker_diarization,
                word_timestamps=request.word_timestamps,
                timestamp_scale=request.timestamp_scale,
                task_id=request.task_id,
            )


_runtime_router: RuntimeRouter | None = None
_runtime_router_lock = threading.Lock()


def get_runtime_router() -> RuntimeRouter:
    global _runtime_router
    if _runtime_router is None:
        with _runtime_router_lock:
            if _runtime_router is None:
                _runtime_router = RuntimeRouter()
    return _runtime_router
