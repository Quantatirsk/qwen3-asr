import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from app.core.exceptions import InvalidParameterException
from app.services.asr.offline_transcription_service import (
    OfflineTranscriptionOptions,
    OfflineTranscriptionService,
)
from app.services.asr.results import ASRFullResult, ASRSegmentResult


class _RuntimeRouter:
    def __init__(self) -> None:
        self.request = None

    async def run_offline(self, request):
        self.request = request
        return ASRFullResult(
            text="测试",
            segments=[ASRSegmentResult(text="测试", start_time=1.0, end_time=3.0)],
            duration=4.0,
        )


class OfflineTranscriptionServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_hotwords_fail_before_runtime_inference(self) -> None:
        router = _RuntimeRouter()
        service = OfflineTranscriptionService()

        with patch(
            "app.services.asr.offline_transcription_service.get_runtime_router",
            return_value=router,
        ):
            with self.assertRaisesRegex(InvalidParameterException, "vocabulary_id"):
                await service.start_transcription(
                    audio_data=b"audio",
                    options=OfflineTranscriptionOptions(hotwords="product"),
                )

        self.assertIsNone(router.request)

    async def test_word_timestamps_use_service_level_uniform_fallback(self) -> None:
        router = _RuntimeRouter()
        service = OfflineTranscriptionService()

        audio = SimpleNamespace(normalized_path="audio.wav", timestamp_scale=1.0)
        with (
            patch(
                "app.services.asr.offline_transcription_service.get_runtime_router",
                return_value=router,
            ),
            patch.object(
                service,
                "_get_audio_service",
                return_value=SimpleNamespace(
                    prepare=lambda **kwargs: nullcontext(audio)
                ),
            ),
        ):
            result = await (
                await service.start_transcription(
                    audio_data=b"audio",
                    options=OfflineTranscriptionOptions(word_timestamps=True),
                )
            )

        self.assertFalse(hasattr(router.request, "word_timestamps"))
        self.assertEqual(result.word_timestamp_method, "uniform_fallback")
        self.assertEqual(
            [
                (token.text, token.start_time, token.end_time)
                for token in result.segments[0].word_tokens or []
            ],
            [("测", 1.0, 2.0), ("试", 2.0, 3.0)],
        )


if __name__ == "__main__":
    unittest.main()
