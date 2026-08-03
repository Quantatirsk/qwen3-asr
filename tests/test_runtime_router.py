from __future__ import annotations

import asyncio
import threading
import time
import unittest
from unittest.mock import patch

from app.core.config import settings
from app.services.asr.engines import ASRFullResult
from app.services.asr.runtime.router import (
    OfflineASRRequest,
    RuntimeFamily,
    RuntimeRouter,
)


class _StatefulEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.current_audio_path = ""

    def transcribe_long_audio(
        self, *, audio_path: str, **_kwargs: object
    ) -> ASRFullResult:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.current_audio_path = audio_path
        time.sleep(0.01)
        with self._lock:
            result = self.current_audio_path
            self.active -= 1
        return ASRFullResult(text=result, segments=[], duration=0.0)


class RuntimeRouterTest(unittest.IsolatedAsyncioTestCase):
    def test_remote_vllm_configuration_overrides_local_device(self) -> None:
        router = RuntimeRouter()

        with (
            patch.object(settings, "QWEN_VLLM_BASE_URL", "http://qwen-npu:8000"),
            patch.object(settings, "DEVICE", "cpu"),
        ):
            family = router._resolve_family("qwen3-asr-1.7b")

        self.assertEqual(family, RuntimeFamily.QWEN_REMOTE_VLLM)

    async def test_vllm_offline_requests_do_not_overlap(self) -> None:
        engine = _StatefulEngine()
        router = RuntimeRouter()
        semaphore = asyncio.Semaphore(8)
        router._resolve_family = lambda _model_id: RuntimeFamily.QWEN_VLLM  # type: ignore[method-assign]
        router._get_shared_engine = lambda _family, _model_id: (  # type: ignore[method-assign]
            engine,
            semaphore,
        )

        requests = [
            OfflineASRRequest(
                model_id="qwen3-asr-test",
                audio_path=f"request-{index}",
            )
            for index in range(8)
        ]
        results = await asyncio.gather(
            *(router.run_offline(request) for request in requests)
        )

        self.assertEqual(engine.max_active, 1)
        self.assertEqual(
            [result.text for result in results],
            [request.audio_path for request in requests],
        )
