# -*- coding: utf-8 -*-
"""Runtime router for pooled ASR execution."""

from __future__ import annotations
import asyncio
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

import torch

from app.core.config import settings
from app.core.device import detect_device
from app.core.executor import run_sync
from app.services.asr.engines import ASRFullResult, BaseASREngine
from app.services.asr.manager import get_model_manager
from app.services.asr.qwenasr_rust import is_qwenasr_rust_available
from .local_pool import LocalEnginePool

_VLLM_SHARED_CONCURRENCY = 8


class RuntimeFamily(str, Enum):
    QWEN_VLLM = "qwen_vllm"
    QWEN_RUST_CPU = "qwen_rust_cpu"
    FUNASR = "funasr"


@dataclass
class OfflineASRRequest:
    model_id: str
    audio_path: str
    hotwords: str = ""
    enable_punctuation: bool = True
    enable_itn: bool = True
    sample_rate: int = 16000
    enable_speaker_diarization: bool = True
    word_timestamps: bool = False
    task_id: Optional[str] = None


class RuntimeEngineLease:
    """Lifecycle wrapper around a pooled engine instance."""

    def __init__(self, engine: BaseASREngine, release_callback: Callable[[], None | Awaitable[None]]):
        self.engine = engine
        self._release_callback = release_callback
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        result = self._release_callback()
        if asyncio.iscoroutine(result):
            await result

    async def __aenter__(self) -> BaseASREngine:
        return self.engine

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


class RuntimeRouter:
    """Central backend router for all ASR entrypoints."""

    def __init__(self):
        self._manager = get_model_manager()
        self._pools: dict[tuple[RuntimeFamily, str], LocalEnginePool[BaseASREngine]] = {}
        self._shared_engines: dict[tuple[RuntimeFamily, str], BaseASREngine] = {}
        self._shared_limits: dict[tuple[RuntimeFamily, str], asyncio.Semaphore] = {}
        self._pool_lock = threading.Lock()
        self._loaded_model_ids: set[str] = set()

    def resolve_model_id(self, model_id: Optional[str]) -> str:
        if model_id:
            return model_id
        config = self._manager.get_declared_entry_config()
        return config.model_id

    def _resolve_family(self, model_id: str) -> RuntimeFamily:
        device = detect_device(settings.DEVICE)
        if model_id.startswith("qwen3-asr-"):
            if device.startswith("cuda"):
                return RuntimeFamily.QWEN_VLLM
            if device == "cpu" and is_qwenasr_rust_available():
                return RuntimeFamily.QWEN_RUST_CPU
            raise RuntimeError(f"Qwen3-ASR is not available on device '{device}'")
        return RuntimeFamily.FUNASR

    def _pool_size_for_family(self, family: RuntimeFamily) -> int:
        if family == RuntimeFamily.QWEN_VLLM:
            return 1
        if family == RuntimeFamily.QWEN_RUST_CPU:
            return settings.QWEN_RUST_CPU_WORKERS
        return settings.FUNASR_WORKERS

    def _create_pool(self, family: RuntimeFamily, model_id: str) -> LocalEnginePool[BaseASREngine]:
        pool_key = (family, model_id)
        existing = self._pools.get(pool_key)
        if existing is not None:
            return existing

        with self._pool_lock:
            existing = self._pools.get(pool_key)
            if existing is not None:
                return existing
            pool = LocalEnginePool(
                size=self._pool_size_for_family(family),
                factory=lambda: self._manager.create_engine(model_id),
            )
            self._pools[pool_key] = pool
            self._loaded_model_ids.add(model_id)
            return pool

    def _get_shared_engine(self, family: RuntimeFamily, model_id: str) -> tuple[BaseASREngine, asyncio.Semaphore]:
        runtime_key = (family, model_id)
        engine = self._shared_engines.get(runtime_key)
        semaphore = self._shared_limits.get(runtime_key)
        if engine is not None and semaphore is not None:
            return engine, semaphore

        with self._pool_lock:
            engine = self._shared_engines.get(runtime_key)
            semaphore = self._shared_limits.get(runtime_key)
            if engine is None:
                engine = self._manager.create_engine(model_id)
                self._shared_engines[runtime_key] = engine
                self._loaded_model_ids.add(model_id)
            if semaphore is None:
                semaphore = asyncio.Semaphore(_VLLM_SHARED_CONCURRENCY)
                self._shared_limits[runtime_key] = semaphore
            return engine, semaphore

    def warmup_model(self, model_id: Optional[str] = None) -> None:
        resolved_model_id = self.resolve_model_id(model_id)
        family = self._resolve_family(resolved_model_id)
        if family == RuntimeFamily.QWEN_VLLM:
            self._get_shared_engine(family, resolved_model_id)
            return
        pool = self._create_pool(family, resolved_model_id)
        pool.warmup()

    def get_loaded_model_ids(self) -> list[str]:
        return sorted(self._loaded_model_ids)

    def get_memory_usage(self) -> dict[str, object]:
        memory_info: dict[str, object] = {
            "model_list": self.get_loaded_model_ids(),
            "loaded_count": len(self._loaded_model_ids),
        }

        if torch.cuda.is_available():
            memory_info["gpu_memory"] = {
                "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                "cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB",
                "max_allocated": f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB",
            }

        return memory_info

    async def acquire_engine(self, model_id: Optional[str] = None) -> RuntimeEngineLease:
        resolved_model_id = self.resolve_model_id(model_id)
        family = self._resolve_family(resolved_model_id)
        if family == RuntimeFamily.QWEN_VLLM:
            engine, semaphore = self._get_shared_engine(family, resolved_model_id)
            await semaphore.acquire()
            return RuntimeEngineLease(
                engine=engine,
                release_callback=semaphore.release,
            )
        pool = self._create_pool(family, resolved_model_id)
        engine = await pool.acquire()
        return RuntimeEngineLease(
            engine=engine,
            release_callback=lambda: pool.release(engine),
        )

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
