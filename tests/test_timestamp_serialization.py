import unittest

from app.api.v1.asr import _serialize_segment
from app.api.v1.openai_compatible import ResponseFormat, build_transcription_payload
from app.services.asr.results import (
    ASRFullResult,
    ASRSegmentResult,
    WordToken,
)


def _aligned_result() -> ASRFullResult:
    start = 1.23456789
    midpoint = 1.23477777
    end = 1.23498765
    return ASRFullResult(
        text="测试",
        segments=[
            ASRSegmentResult(
                text="测试",
                start_time=start,
                end_time=end,
                word_tokens=[
                    WordToken(text="测", start_time=start, end_time=midpoint),
                    WordToken(text="试", start_time=midpoint, end_time=end),
                ],
            )
        ],
        duration=end,
        word_timestamp_method="uniform_fallback",
    )


class TimestampSerializationTest(unittest.TestCase):
    def test_rest_tokens_preserve_serialized_segment_boundaries(self) -> None:
        segment = _serialize_segment(_aligned_result().segments[0])
        tokens = segment["word_tokens"]

        self.assertEqual(tokens[0]["start_time"], segment["start_time"])
        self.assertEqual(tokens[-1]["end_time"], segment["end_time"])

    def test_openai_tokens_preserve_serialized_segment_boundaries(self) -> None:
        result = _aligned_result()
        payload, _, _ = build_transcription_payload(
            response_format=ResponseFormat.VERBOSE_JSON,
            asr_result=result,
            audio_duration=result.duration,
            language="zh",
        )

        self.assertEqual(payload["words"][0]["start"], payload["segments"][0]["start"])
        self.assertEqual(payload["words"][-1]["end"], payload["segments"][0]["end"])


if __name__ == "__main__":
    unittest.main()
