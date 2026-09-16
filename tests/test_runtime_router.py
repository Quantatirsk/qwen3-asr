from __future__ import annotations

import asyncio
import threading
import time
import unittest
from unittest.mock import patch

from app.services.asr.manager import ASCEND_MODEL_ID
from app.services.asr.results import ASRFullResult
from app.services.asr.long_audio import OfflineASRRequest
from app.services.asr.runtime.router import RuntimeRouter


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
    async def test_cancelled_remote_call_keeps_concurrency_slot_until_finished(
        self,
    ) -> None:
        entered = threading.Event()
        release = threading.Event()
        router = RuntimeRouter()
        router._semaphore = asyncio.Semaphore(1)
        engine = _ConcurrentEngine()
        router._manager = _Manager(engine)

        def infer(**kwargs: object) -> ASRFullResult:
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Remote request was not released")
            return ASRFullResult(text="done", segments=[], duration=0.0)

        with patch.object(engine, "transcribe_long_audio", side_effect=infer):
            task = asyncio.create_task(
                router.run_offline(
                    OfflineASRRequest(model_id=ASCEND_MODEL_ID, audio_path="audio.wav")
                )
            )
            waiter = None
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                waiter = asyncio.create_task(router.acquire_engine())
                await asyncio.sleep(0.02)
                self.assertFalse(task.done())
                self.assertFalse(
                    waiter.done(), "Cancelled call released its slot early"
                )
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                if waiter is not None:
                    async with await asyncio.wait_for(waiter, 2):
                        pass
            self.assertTrue(task.cancelled())

    async def test_cold_readiness_is_offloaded_and_failed_engine_is_not_published(
        self,
    ) -> None:
        router = RuntimeRouter()
        engine = _ConcurrentEngine()
        router._manager = _Manager(engine)
        pulse = threading.Event()
        timer = asyncio.get_running_loop().call_later(0.01, pulse.set)

        def fail_readiness(model_id: str) -> None:
            if not pulse.wait(1):
                raise AssertionError("Readiness check blocked the event loop")
            raise RuntimeError("Remote server is not ready")

        try:
            with patch.object(
                router._manager, "create_engine", side_effect=fail_readiness
            ):
                with self.assertRaisesRegex(RuntimeError, "not ready"):
                    await router.acquire_engine()
            self.assertEqual(router.get_loaded_model_ids(), [])
            async with await router.acquire_engine() as loaded:
                self.assertIs(loaded, engine)
            self.assertEqual(router.get_loaded_model_ids(), [ASCEND_MODEL_ID])
        finally:
            timer.cancel()

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
