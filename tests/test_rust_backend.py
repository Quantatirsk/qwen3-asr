import asyncio
import ctypes
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from app.services.asr.rust_backend import (
    NativeGeneration,
    RustBackend,
    RustForcedAligner,
)
from app.services.realtime.rust_engine import AUDIO_PAD, RustAsyncEngine, SamplingParams


class RustBackendTest(unittest.TestCase):
    def make_backend(
        self, payload: object, aligner: bool = False
    ) -> tuple[RustBackend, Mock]:
        self.buffer = ctypes.create_string_buffer(json.dumps(payload).encode())
        library = Mock()
        library.qwen_asr_load_model.return_value = 123
        library.qwen_asr_generate_pcm.return_value = ctypes.addressof(self.buffer)
        library.qwen_asr_force_align_file.return_value = ctypes.addressof(self.buffer)
        with (
            patch("app.services.asr.rust_backend._load_library", return_value=library),
            patch(
                "app.services.asr.rust_backend.resolve_huggingface_snapshot_dir",
                return_value=Path("/models/test"),
            ),
        ):
            backend = (RustForcedAligner if aligner else RustBackend)("/models/test")
        return backend, library

    def test_native_buffers_validation_and_single_owned_handle(self) -> None:
        backend, library = self.make_backend(
            {"token_ids": [1, 2], "finish_reason": "stop"}
        )
        try:
            result = backend.generate_pcm(np.zeros(10), [1], [2], 10)
            self.assertEqual(result, NativeGeneration([1, 2], "stop"))
            args = library.qwen_asr_generate_pcm.call_args.args
            self.assertEqual(
                (args[0], args[2], args[4], args[6], args[7]), (123, 10, 1, 1, 10)
            )
            library.qwen_asr_load_model.assert_called_once()
            library.qwen_asr_free_string.assert_called_once_with(
                ctypes.addressof(self.buffer)
            )
            for audio in (np.empty(0), np.zeros((2, 2)), np.array([np.nan])):
                with self.assertRaises(ValueError):
                    backend.generate_pcm(audio, [1], [2], 10)
            with self.assertRaises(ValueError):
                backend.generate_pcm(np.zeros(10), [-1], [2], 10)
            with self.assertRaises(ValueError):
                backend.generate_pcm(np.zeros(10), [1], [2], 0)
            library.qwen_asr_generate_pcm.assert_called_once()
        finally:
            backend.close()
        backend.close()
        library.qwen_asr_free.assert_called_once_with(123)
        with self.assertRaises(RuntimeError):
            backend.generate_pcm(np.zeros(10), [1], [2], 10)

    def test_invalid_native_payload_still_frees_string(self) -> None:
        backend, library = self.make_backend(
            {"token_ids": "bad", "finish_reason": "stop"}
        )
        try:
            with self.assertRaises(RuntimeError):
                backend.generate_pcm(np.zeros(10), [1], [2], 10)
            library.qwen_asr_free_string.assert_called_once()
        finally:
            backend.close()

    def test_alignment_preserves_words_and_repairs_bounds(self) -> None:
        backend, library = self.make_backend(
            [
                {"text": "Hello", "start_ms": -1, "end_ms": 80},
                {"text": "world", "start_ms": 20, "end_ms": 2000},
            ],
            aligner=True,
        )
        try:
            result = backend.align_transcript(
                "audio.wav", "Hello world", np.zeros(16000)
            )
            self.assertEqual([item["text"] for item in result], ["Hello", "world"])
            spans = [item[key] for item in result for key in ("start_ms", "end_ms")]
            self.assertEqual(spans, sorted(spans))
            self.assertEqual((spans[0], spans[-1]), (0, 1000))
            self.assertEqual(
                library.qwen_asr_force_align_file.call_args.args[-1], b"English"
            )
        finally:
            backend.close()

    def test_mixed_alignment_uses_shared_units_without_punctuation(self) -> None:
        units = ["\u4e2d", "\u6587", "Hello", "world"]
        backend, library = self.make_backend(
            [
                {"text": unit, "start_ms": index * 100, "end_ms": index * 100 + 50}
                for index, unit in enumerate(units)
            ],
            aligner=True,
        )
        try:
            text = "\u4e2d\u6587, Hello world!"
            result = backend.align_transcript("audio.wav", text, np.zeros(16000))
            self.assertEqual([item["text"] for item in result], units)
            self.assertEqual(
                library.qwen_asr_force_align_file.call_args.args[2:],
                (" ".join(units).encode(), b"English"),
            )
        finally:
            backend.close()


class RustSchedulingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls: list[int] = []
        self.backend = Mock(model_path="/models/test")
        self.backend.generate_pcm.side_effect = self.decode
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text, **kwargs: [int(text), AUDIO_PAD, 3]
        tokenizer.decode.side_effect = lambda ids, **kwargs: f"<asr_text>{ids[0]}"
        with (
            patch(
                "app.services.realtime.rust_engine.RustBackend",
                return_value=self.backend,
            ) as constructor,
            patch(
                "app.services.realtime.rust_engine.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ),
            patch("pathlib.Path.read_text", return_value='{"chat_template":"test"}'),
        ):
            self.engine = RustAsyncEngine(2)
            constructor.assert_called_once()

    def decode(
        self, audio: np.ndarray, before: list[int], after: list[int], maximum: int
    ) -> NativeGeneration:
        self.calls.append(before[0])
        if before[0] == 1:
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("Test did not release native inference")
        return NativeGeneration(before, "stop")

    async def collect(self, number: int, priority: int = 0) -> str:
        prompt = {"prompt": str(number), "multi_modal_data": {"audio": [np.zeros(10)]}}
        async for result in self.engine.generate(
            prompt, SamplingParams(16), str(number), priority
        ):
            return result.outputs[0].text
        raise AssertionError("Missing result")

    async def wait_for_requests(self, count: int) -> None:
        async def ready() -> None:
            while len(self.engine._requests) < count:
                await asyncio.sleep(0)

        await asyncio.wait_for(ready(), 2)

    async def asyncTearDown(self) -> None:
        self.release.set()
        await self.engine.shutdown()

    async def test_realtime_priority_and_aborted_queued_request(self) -> None:
        active = asyncio.create_task(self.collect(1))
        self.assertTrue(await asyncio.to_thread(self.entered.wait, 2))
        offline = asyncio.create_task(self.collect(2, 10))
        stream = asyncio.create_task(self.collect(3, 0))
        aborted = asyncio.create_task(self.collect(4, 0))
        await self.wait_for_requests(4)
        await self.engine.abort("4")
        with self.assertRaises(asyncio.CancelledError):
            await aborted
        self.release.set()
        self.assertEqual(
            await asyncio.gather(active, offline, stream),
            ["<asr_text>1", "<asr_text>2", "<asr_text>3"],
        )
        self.assertEqual(self.calls, [1, 3, 2])

    async def test_abort_and_shutdown_wait_for_native_call_before_free(self) -> None:
        active = asyncio.create_task(self.collect(1))
        self.assertTrue(await asyncio.to_thread(self.entered.wait, 2))
        await self.engine.abort("1")
        with self.assertRaises(asyncio.CancelledError):
            await active
        shutdown = asyncio.create_task(self.engine.shutdown())
        await asyncio.sleep(0.02)
        self.assertFalse(shutdown.done())
        self.backend.close.assert_not_called()
        self.release.set()
        await shutdown
        self.backend.close.assert_called_once()

    async def test_rejects_duplicate_audio_placeholder_and_sampling(self) -> None:
        self.engine.tokenizer.encode.return_value = [AUDIO_PAD, AUDIO_PAD]
        self.engine.tokenizer.encode.side_effect = None
        with self.assertRaises(ValueError):
            await self.collect(1)
        with self.assertRaises(ValueError):
            SamplingParams(16, temperature=1)
        self.backend.generate_pcm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
