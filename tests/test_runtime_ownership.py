"""Resource ownership checks without loading model weights."""

import asyncio
import tempfile
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.config import settings
from app.services.asr.engines import ASRFullResult
from app.services.asr.qwen3_engine import Qwen3ASREngine
from app.services.asr.runtime.local_pool import LocalEnginePool
from app.services.asr.long_audio import OfflineASRRequest, PreparedLongAudio
from app.services.asr.runtime.router import (
    RuntimeFamily,
    RuntimeRouter,
)

from app.utils.audio_splitter import AudioSegment


class RuntimeOwnershipTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        manager = SimpleNamespace(create_engine=lambda _: object())
        patcher = patch(
            "app.services.asr.runtime.router.get_model_manager", return_value=manager
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_cancelled_inference_keeps_lease_until_worker_finishes(self) -> None:
        entered = asyncio.Event()
        second_entered = threading.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        class Engine:
            def transcribe_long_audio(
                self, *, audio_path: str, **kwargs: object
            ) -> ASRFullResult:
                if audio_path == "first":
                    loop.call_soon_threadsafe(entered.set)
                    if not release.wait(3):
                        raise TimeoutError("Worker was not released")
                else:
                    second_entered.set()
                return ASRFullResult(text=audio_path, segments=[], duration=1.0)

        router = RuntimeRouter()
        with (
            patch.object(
                router, "_resolve_family", return_value=RuntimeFamily.QWEN_VLLM
            ),
            patch.object(
                router,
                "_get_shared_engine",
                return_value=(Engine(), asyncio.Semaphore(8)),
            ),
        ):
            first = asyncio.create_task(
                router.run_offline(OfflineASRRequest("model", "first"))
            )
            second = None
            try:
                await asyncio.wait_for(entered.wait(), 1)
                first.cancel()
                await asyncio.sleep(0)
                first.cancel()
                second = asyncio.create_task(
                    router.run_offline(OfflineASRRequest("model", "second"))
                )
                await asyncio.sleep(0.03)
                self.assertFalse(
                    second_entered.is_set(), "Cancelled worker's engine was reused"
                )
                self.assertFalse(
                    first.done(), "Cancellation escaped before worker completion"
                )
            finally:
                release.set()
                await asyncio.gather(
                    first, *([second] if second else []), return_exceptions=True
                )
            self.assertTrue(first.cancelled())
            self.assertEqual(second.result().text, "second")

    async def test_failed_pool_initialization_is_retryable(self) -> None:
        calls = 0

        def factory() -> object:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("Model load failed")
            return object()

        pool = LocalEnginePool(2, factory)
        with self.assertRaisesRegex(ValueError, "Model load failed"):
            pool.warmup()
        pool.warmup()
        self.assertEqual(calls, 4, "A partially initialized pool was published")
        first = await asyncio.wait_for(pool.acquire(), 1)
        second = await asyncio.wait_for(pool.acquire(), 1)
        self.assertIsNot(first, second)
        await pool.release(first)
        await pool.release(second)

    async def test_failed_model_is_not_reported_loaded(self) -> None:
        router = RuntimeRouter()
        with (
            patch.object(
                router, "_resolve_family", return_value=RuntimeFamily.QWEN_RUST_CPU
            ),
            patch.object(
                router._manager, "create_engine", side_effect=ValueError("Load failed")
            ),
        ):
            with self.assertRaises(ValueError):
                router.warmup_model("model")
        self.assertEqual(router.get_loaded_model_ids(), [])

    async def test_cpu_runtime_count_stays_within_request_pool_budget(self) -> None:
        runtimes: list[object] = []
        barrier = threading.Barrier(2)

        class Runtime:
            def __init__(self, **kwargs: object) -> None:
                runtimes.append(self)

            def transcribe_file(self, path: str) -> str:
                return Path(path).name

        class Engine(Qwen3ASREngine):
            def transcribe_long_audio(
                self, *, audio_path: str, **kwargs: object
            ) -> ASRFullResult:
                barrier.wait(timeout=3)
                segments = [
                    SimpleNamespace(
                        temp_file=audio_path, start_sec=float(i), end_sec=float(i + 1)
                    )
                    for i in range(2)
                ]
                results = self.transcribe_segments(segments, enable_itn=False)
                return ASRFullResult(
                    text="\n".join(item.text for item in results),
                    segments=results,
                    duration=2.0,
                )

        router = RuntimeRouter()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(settings, "QWEN_RUST_CPU_WORKERS", 2),
            patch.object(
                router, "_resolve_family", return_value=RuntimeFamily.QWEN_RUST_CPU
            ),
            patch.object(
                router._manager,
                "create_engine",
                side_effect=lambda _: Engine(device="cpu"),
            ),
            patch.object(Qwen3ASREngine, "_select_backend", return_value="rust"),
            patch.object(Qwen3ASREngine, "_warmup_forced_aligner"),
            patch("app.services.asr.qwen3_engine.QwenASRRustRuntime", Runtime),
            patch(
                "app.services.asr.long_audio.prepare_long_audio",
                side_effect=lambda audio_path, *args: nullcontext(
                    PreparedLongAudio(
                        [
                            AudioSegment(i * 1000, (i + 1) * 1000, temp_file=audio_path)
                            for i in range(2)
                        ],
                        2.0,
                    )
                ),
            ),
        ):
            path = Path(directory) / "sample.wav"
            path.touch()
            results = await asyncio.gather(
                *(
                    router.run_offline(OfflineASRRequest("model", str(path)))
                    for _ in range(2)
                )
            )
        self.assertEqual(
            len(runtimes), 2, "Engine-local expansion multiplied the pool budget"
        )
        self.assertEqual(
            [result.text for result in results], ["sample.wav\nsample.wav"] * 2
        )


if __name__ == "__main__":
    unittest.main()
