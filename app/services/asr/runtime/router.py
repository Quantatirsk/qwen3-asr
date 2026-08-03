"""Concurrency boundary for the single remote Ascend ASR engine."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from app.core.executor import run_sync
from app.services.asr.results import ASRFullResult
from app.services.asr.manager import ASCEND_MODEL_ID, get_model_manager

if TYPE_CHECKING:
    from app.services.asr.engines import BaseASREngine

_REMOTE_CONCURRENCY = 8


@dataclass(frozen=True)
class OfflineASRRequest:
    model_id: str
    audio_path: str
    hotwords: str = ""
    enable_punctuation: bool = True
    enable_itn: bool = True
    sample_rate: int = 16000
    enable_speaker_diarization: bool = True
    timestamp_scale: float = 1.0
    task_id: Optional[str] = None


class RuntimeEngineLease:
    def __init__(self, engine: BaseASREngine, semaphore: asyncio.Semaphore) -> None:
        self.engine = engine
        self._semaphore = semaphore

    async def __aenter__(self) -> BaseASREngine:
        return self.engine

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._semaphore.release()


class RuntimeRouter:
    """Own one stateless remote adapter and bound request concurrency."""

    def __init__(self) -> None:
        self._manager = get_model_manager()
        self._engine: Optional[BaseASREngine] = None
        self._engine_lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(_REMOTE_CONCURRENCY)

    def resolve_model_id(self, model_id: Optional[str]) -> str:
        return self._manager.get_declared_entry_config(model_id).model_id

    def _get_engine(self, model_id: Optional[str] = None) -> BaseASREngine:
        self.resolve_model_id(model_id)
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    self._engine = self._manager.create_engine(ASCEND_MODEL_ID)
        return self._engine

    def warmup_model(self, model_id: Optional[str] = None) -> None:
        self._get_engine(model_id)

    def get_loaded_model_ids(self) -> list[str]:
        return [ASCEND_MODEL_ID] if self._engine is not None else []

    def get_memory_usage(self) -> dict[str, object]:
        loaded = self.get_loaded_model_ids()
        return {"model_list": loaded, "loaded_count": len(loaded), "gpu_memory": None}

    async def acquire_engine(
        self, model_id: Optional[str] = None
    ) -> RuntimeEngineLease:
        engine = self._get_engine(model_id)
        await self._semaphore.acquire()
        return RuntimeEngineLease(engine, self._semaphore)

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
                timestamp_scale=request.timestamp_scale,
                task_id=request.task_id,
            )


_runtime_router: Optional[RuntimeRouter] = None
_runtime_router_lock = threading.Lock()


def get_runtime_router() -> RuntimeRouter:
    global _runtime_router
    if _runtime_router is None:
        with _runtime_router_lock:
            if _runtime_router is None:
                _runtime_router = RuntimeRouter()
    return _runtime_router
