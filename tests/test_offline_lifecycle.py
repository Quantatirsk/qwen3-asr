"""Exercise real preparation, runtime ownership, and protocol cancellation together."""

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from starlette.types import ASGIApp, Message, Scope

from app.core.config import settings
from app.services.asr.engines import ASRFullResult, ASRSegmentResult
from app.services.asr.offline_transcription_service import (
    OfflineTranscriptionOptions,
    OfflineTranscriptionService,
)
from app.services.asr.runtime.router import RuntimeRouter
from app.utils.audio import NormalizedAudio
from app.utils.audio import normalize_audio_for_asr

with (
    patch(
        "app.services.asr.model_selection.get_offline_model_ids",
        return_value=["confucius4-r2t2"],
    ),
    patch(
        "app.services.asr.model_selection.get_default_offline_model_id",
        return_value="confucius4-r2t2",
    ),
):
    from app.api.v1 import asr, openai_compatible


async def request(
    app: ASGIApp, path: str, body: bytes, content_type: str
) -> tuple[int, bytes]:
    messages: list[Message] = []
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "server": ("test", 80),
        "client": ("test", 1234),
        "headers": [
            (b"content-type", content_type.encode()),
            (b"content-length", str(len(body)).encode()),
        ],
    }

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    await asyncio.wait_for(app(scope, receive, send), 3)
    status = next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )
    payload = b"".join(message.get("body", b"") for message in messages)
    return status, payload


class OfflineLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths: list[str] = []
        self.cleaned: list[str] = []
        self.prepare_thread: int | None = None
        self.prepare_started = asyncio.Event()
        self.inference_started = asyncio.Event()
        self.prepare_release = threading.Event()
        self.prepare_release.set()
        self.inference_release = threading.Event()
        self.inference_release.set()
        self.inference_error = False
        self.file_survived_inference = False
        self.loop = asyncio.get_event_loop()

        def normalize(path: str, sample_rate: int) -> NormalizedAudio:
            self.prepare_thread = threading.get_ident()
            normalized = path + ".normalized.wav"
            Path(normalized).write_bytes(b"normalized")
            self.paths.extend([path, normalized])
            self.loop.call_soon_threadsafe(self.prepare_started.set)
            if not self.prepare_release.wait(3):
                raise TimeoutError("Preparation was not released")
            return NormalizedAudio(normalized, 1.25)

        def cleanup(path: str) -> None:
            self.cleaned.append(path)
            Path(path).unlink(missing_ok=True)

        def infer(
            *, audio_path: str, timestamp_scale: float, **kwargs: object
        ) -> ASRFullResult:
            self.loop.call_soon_threadsafe(self.inference_started.set)
            if not self.inference_release.wait(3):
                raise TimeoutError("Inference was not released")
            self.file_survived_inference = Path(audio_path).exists()
            if self.inference_error:
                raise ValueError("Inference failed")
            return ASRFullResult(
                text="recognized text",
                segments=[
                    ASRSegmentResult("recognized text", 0.0, 2.0 * timestamp_scale)
                ],
                duration=2.0 * timestamp_scale,
            )

        patches = [
            patch.object(settings, "TEMP_DIR", self.directory.name),
            patch.object(settings, "API_KEY", ""),
            patch(
                "app.services.audio.audio_service.normalize_audio_for_asr",
                side_effect=normalize,
            ),
            patch(
                "app.services.audio.audio_service.get_audio_duration", return_value=2.0
            ),
            patch(
                "app.services.audio.audio_service.cleanup_temp_file",
                side_effect=cleanup,
            ),
            patch(
                "app.services.asr.runtime.router.get_model_manager",
                return_value=SimpleNamespace(),
            ),
            patch(
                "app.services.asr.offline_transcription_service.get_default_offline_model_id",
                return_value="model",
            ),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        router = RuntimeRouter()
        for patcher in [
            patch.object(router, "resolve_model_id", return_value="confucius4-r2t2"),
            patch.object(
                router,
                "_get_engine",
                return_value=SimpleNamespace(transcribe_long_audio=infer),
            ),
            patch(
                "app.services.asr.offline_transcription_service.get_runtime_router",
                return_value=router,
            ),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.service = OfflineTranscriptionService()

    async def start(self, **kwargs: object) -> asyncio.Task[ASRFullResult]:
        return await self.service.start_transcription(
            audio_data=b"audio", options=OfflineTranscriptionOptions(), **kwargs
        )

    def assert_cleaned_once(self) -> None:
        self.assertEqual(sorted(self.cleaned), sorted(self.paths))
        self.assertFalse(list(Path(self.directory.name).iterdir()))

    async def test_success_upload_priority_and_timestamp_scale(self) -> None:
        with patch(
            "app.services.audio.audio_service.download_audio_from_url"
        ) as download:
            task = await self.start(audio_address="https://example.test/audio.wav")
            result = await task
        download.assert_not_called()
        self.assertEqual(result.duration, 2.5)
        self.assertEqual(result.segments[0].end_time, 2.5)
        self.assertTrue(
            self.file_survived_inference, "Input was deleted before inference completed"
        )
        self.assertNotEqual(self.prepare_thread, threading.get_ident())
        self.assert_cleaned_once()

    async def test_url_preparation(self) -> None:
        with patch(
            "app.services.audio.audio_service.download_audio_from_url",
            return_value=b"audio",
        ) as download:
            task = await self.service.start_transcription(
                audio_data=None,
                audio_address="https://example.test/audio.wav",
                options=OfflineTranscriptionOptions(),
            )
            await task
        download.assert_called_once_with("https://example.test/audio.wav")
        self.assert_cleaned_once()

    async def test_preparation_keeps_event_loop_responsive(self) -> None:
        pulse = threading.Event()
        observed: list[bool] = []

        def normalize(path: str, sample_rate: int) -> NormalizedAudio:
            self.paths.append(path)
            observed.append(pulse.wait(0.15))
            return NormalizedAudio(path)

        timer = self.loop.call_later(0.01, pulse.set)
        try:
            with patch(
                "app.services.audio.audio_service.normalize_audio_for_asr",
                side_effect=normalize,
            ):
                await (await self.start())
        finally:
            timer.cancel()
        self.assertEqual(observed, [True], "Audio preparation blocked the event loop")
        self.assert_cleaned_once()

    async def test_empty_upload_does_not_fall_back_to_url(self) -> None:
        from app.core.exceptions import InvalidMessageException

        with patch(
            "app.services.audio.audio_service.download_audio_from_url"
        ) as download:
            with self.assertRaises(InvalidMessageException):
                await self.service.start_transcription(
                    audio_data=b"",
                    audio_address="https://example.test/audio.wav",
                    options=OfflineTranscriptionOptions(),
                )
        download.assert_not_called()
        self.assert_cleaned_once()

    async def test_unchanged_audio_path_is_cleaned_once(self) -> None:
        def normalize(path: str, sample_rate: int) -> NormalizedAudio:
            self.paths.append(path)
            return NormalizedAudio(path)

        with patch(
            "app.services.audio.audio_service.normalize_audio_for_asr",
            side_effect=normalize,
        ):
            await (await self.start())
        self.assertEqual(len(self.cleaned), 1)
        self.assert_cleaned_once()

    async def test_preparation_failure_cleans_both_files(self) -> None:
        with patch(
            "app.services.audio.audio_service.get_audio_duration",
            side_effect=ValueError("Decode failed"),
        ):
            with self.assertRaisesRegex(ValueError, "Decode failed"):
                await self.start()
        self.assert_cleaned_once()

    async def test_inference_failure_cleans_files(self) -> None:
        self.inference_error = True
        task = await self.start()
        with self.assertRaisesRegex(ValueError, "Inference failed"):
            await task
        self.assert_cleaned_once()

    async def test_cancel_before_task_starts_cleans_files(self) -> None:
        task = await self.start()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.inference_started.is_set())
        self.assert_cleaned_once()

    async def test_cancel_during_preparation_cleans_files(self) -> None:
        self.prepare_release.clear()
        preparing = asyncio.create_task(self.start())
        try:
            await asyncio.wait_for(self.prepare_started.wait(), 1)
            preparing.cancel()
            await asyncio.sleep(0)
            preparing.cancel()
            await asyncio.sleep(0.01)
            self.assertFalse(preparing.done())
        finally:
            self.prepare_release.set()
            await asyncio.gather(preparing, return_exceptions=True)
        self.assertTrue(preparing.cancelled())
        self.assert_cleaned_once()

    async def test_cancel_during_inference_keeps_files_until_completion(self) -> None:
        self.inference_error = True
        self.inference_release.clear()
        task = await self.start()
        try:
            await asyncio.wait_for(self.inference_started.wait(), 1)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0.01)
            self.assertFalse(task.done())
            self.assertTrue(
                all(Path(path).exists() for path in self.paths),
                "Input was deleted while inference was running",
            )
        finally:
            self.inference_release.set()
            await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(task.cancelled())
        self.assertTrue(self.file_survived_inference)
        self.assert_cleaned_once()

    async def test_unconsumed_response_task_still_cleans_files(self) -> None:
        task = await self.start()
        openai_compatible.create_heartbeat_streaming_response(
            response_format=openai_compatible.ResponseFormat.JSON,
            inference_task=task,
            language=None,
        )
        await task
        self.assert_cleaned_once()

    async def test_cancel_waiting_for_engine_preserves_active_files(self) -> None:
        self.inference_release.clear()
        first = await self.start()
        second = None
        try:
            await asyncio.wait_for(self.inference_started.wait(), 1)
            active_paths = list(self.paths)
            second = await self.start()
            waiting_paths = self.paths[len(active_paths) :]
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)
            self.assertTrue(second.cancelled())
            self.assertTrue(all(Path(path).exists() for path in active_paths))
            self.assertTrue(all(not Path(path).exists() for path in waiting_paths))
        finally:
            self.inference_release.set()
            await asyncio.gather(
                first, *([second] if second else []), return_exceptions=True
            )
        self.assert_cleaned_once()

    async def test_stream_disconnect_drains_owned_task(self) -> None:
        self.inference_release.clear()
        task = await self.start()
        response = openai_compatible.create_heartbeat_streaming_response(
            response_format=openai_compatible.ResponseFormat.JSON,
            inference_task=task,
            language=None,
        )

        async def receive() -> dict[str, str]:
            await self.inference_started.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            pass

        response_task = asyncio.create_task(
            response({"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send)
        )
        try:
            await asyncio.wait_for(self.inference_started.wait(), 1)
            await asyncio.sleep(0.02)
            self.assertFalse(task.done())
            self.assertTrue(all(Path(path).exists() for path in self.paths))
        finally:
            self.inference_release.set()
            await asyncio.wait_for(response_task, 2)
        self.assertTrue(task.cancelled())
        self.assertTrue(self.file_survived_inference)
        self.assert_cleaned_once()

    async def test_protocol_response_formats(self) -> None:
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(openai_compatible.router)
        app.include_router(asr.router)
        with (
            patch.object(
                openai_compatible,
                "get_offline_transcription_service",
                return_value=self.service,
            ),
            patch.object(
                asr, "get_offline_transcription_service", return_value=self.service
            ),
        ):
            for fmt in ("json", "verbose_json", "text", "srt", "vtt"):
                with self.subTest(format=fmt):
                    body = (
                        '--audio-test\r\nContent-Disposition: form-data; name="response_format"\r\n\r\n'
                        + fmt
                        + '\r\n--audio-test\r\nContent-Disposition: form-data; name="file"; filename="sample.wav"'
                        + "\r\nContent-Type: audio/wav\r\n\r\naudio\r\n--audio-test--\r\n"
                    ).encode()
                    status, payload = await request(
                        app,
                        "/v1/audio/transcriptions",
                        body,
                        "multipart/form-data; boundary=audio-test",
                    )
                    self.assertEqual(status, 200)
                    self.assertIn(b"recognized text", payload)
                    if fmt == "verbose_json":
                        self.assertEqual(json.loads(payload)["duration"], 2.5)
                    if fmt == "srt":
                        self.assertIn(b"00:00:02,500", payload)
                    if fmt == "vtt":
                        self.assertTrue(payload.startswith(b"WEBVTT"))
            status, payload = await request(
                app, "/stream/v1/asr", b"audio", "application/octet-stream"
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(payload)["result"], "recognized text")
        self.assert_cleaned_once()

    async def test_stream_send_failure_drains_owned_task(self) -> None:
        from starlette.requests import ClientDisconnect

        self.inference_release.clear()
        task = await self.start()
        response = openai_compatible.create_heartbeat_streaming_response(
            response_format=openai_compatible.ResponseFormat.JSON,
            inference_task=task,
            language=None,
        )
        send_failed = asyncio.Event()

        async def receive() -> Message:
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            if message["type"] == "http.response.body":
                send_failed.set()
                raise OSError("Client disconnected")

        with patch.object(openai_compatible, "HEARTBEAT_INTERVAL_SECONDS", 0.001):
            response_task = asyncio.create_task(
                response(
                    {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
                )
            )
            try:
                await asyncio.wait_for(send_failed.wait(), 1)
                await asyncio.sleep(0.01)
                self.assertFalse(
                    response_task.done(), "Response abandoned its inference task"
                )
                self.assertTrue(all(Path(path).exists() for path in self.paths))
            finally:
                self.inference_release.set()
                outcomes = await asyncio.gather(
                    response_task, task, return_exceptions=True
                )
        self.assertIsInstance(outcomes[0], ClientDisconnect)
        self.assertTrue(task.cancelled())
        self.assert_cleaned_once()


class AudioNormalizationOwnershipTests(unittest.TestCase):
    def test_failed_conversion_removes_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "input.mp3"
            original.write_bytes(b"audio")

            def fail_conversion(
                input_path: str, output_path: str | None = None, target_sr: int = 16000
            ) -> str:
                output = Path(output_path or str(original.with_suffix(".wav")))
                output.write_bytes(b"partial")
                raise ValueError("Conversion failed")

            with patch(
                "app.utils.audio.convert_audio_to_wav", side_effect=fail_conversion
            ):
                with self.assertRaisesRegex(Exception, "Conversion failed"):
                    normalize_audio_for_asr(str(original))
            self.assertEqual(list(Path(directory).iterdir()), [original])


if __name__ == "__main__":
    unittest.main()
