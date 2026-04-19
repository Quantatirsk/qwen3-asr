# -*- coding: utf-8 -*-
"""Small async pool for per-request ASR engines."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class _PoolState(Generic[T]):
    queue: asyncio.Queue[T]


class LocalEnginePool(Generic[T]):
    """Fixed-size lazy engine pool backed by ``asyncio.Queue``."""

    def __init__(self, size: int, factory: Callable[[], T]):
        self._size = max(1, size)
        self._factory = factory
        self._state: Optional[_PoolState[T]] = None
        self._init_lock = threading.Lock()

    def _ensure_state(self) -> _PoolState[T]:
        if self._state is not None:
            return self._state

        with self._init_lock:
            if self._state is None:
                self._state = _PoolState(queue=asyncio.Queue(maxsize=self._size))
                for _ in range(self._size):
                    self._state.queue.put_nowait(self._factory())
        return self._state

    def warmup(self) -> None:
        self._ensure_state()

    async def acquire(self) -> T:
        state = self._ensure_state()
        return await state.queue.get()

    async def release(self, engine: T) -> None:
        state = self._ensure_state()
        await state.queue.put(engine)
