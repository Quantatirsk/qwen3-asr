from __future__ import annotations

import asyncio
import threading
import time
import unittest

from app.services.asr.manager import ASCEND_MODEL_ID
from app.services.asr.results import ASRFullResult
from app.services.asr.runtime.router import OfflineASRRequest, RuntimeRouter


class _Config:
    model_id = ASCEND_MODEL_ID


class _ConcurrentEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def transcribe_long_audio(self, *, audio_path: str, **_kwargs) -> ASRFullResult:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)
        with self._lock:
            self.active -= 1
        return ASRFullResult(text=audio_path, segments=[], duration=0.0)


class _Manager:
    def __init__(self, engine: _ConcurrentEngine) -> None:
        self.engine = engine

    def get_declared_entry_config(self, _model_id=None):
        return _Config()

    def create_engine(self, _model_id=None):
        return self.engine


class RuntimeRouterTest(unittest.IsolatedAsyncioTestCase):
    async def test_remote_requests_are_bounded_and_keep_results_isolated(self) -> None:
        engine = _ConcurrentEngine()
        router = RuntimeRouter()
        router._manager = _Manager(engine)
        requests = [
            OfflineASRRequest(model_id=ASCEND_MODEL_ID, audio_path=f"request-{index}")
            for index in range(12)
        ]

        results = await asyncio.gather(*(router.run_offline(item) for item in requests))

        self.assertLessEqual(engine.max_active, 8)
        self.assertGreater(engine.max_active, 1)
        self.assertEqual(
            [result.text for result in results],
            [item.audio_path for item in requests],
        )


if __name__ == "__main__":
    unittest.main()
