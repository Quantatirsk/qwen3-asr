import unittest

from app.services.asr.results import ASRFullResult, ASRSegmentResult
from app.services.asr.uniform_alignment import apply_uniform_word_timestamps


class UniformAlignmentTest(unittest.TestCase):
    def test_preserves_strict_boundaries_for_dense_short_segments(self) -> None:
        result = ASRFullResult(
            text="测试词",
            segments=[ASRSegmentResult(text="测试词", start_time=0.0, end_time=0.001)],
            duration=0.001,
        )

        apply_uniform_word_timestamps(result)

        tokens = result.segments[0].word_tokens or []
        self.assertTrue(all(token.start_time < token.end_time for token in tokens))
        self.assertTrue(
            all(left.end_time == right.start_time for left, right in zip(tokens, tokens[1:]))
        )

    def test_distributes_mixed_text_within_each_segment(self) -> None:
        result = ASRFullResult(
            text="你好，OpenAI 2026!",
            segments=[
                ASRSegmentResult(
                    text="你好，OpenAI 2026!",
                    start_time=2.0,
                    end_time=6.0,
                    speaker_id="speaker-1",
                )
            ],
            duration=8.0,
        )

        aligned = apply_uniform_word_timestamps(result)

        self.assertIs(aligned, result)
        self.assertEqual(aligned.word_timestamp_method, "uniform_fallback")
        self.assertEqual(
            [token.text for token in aligned.segments[0].word_tokens or []],
            ["你", "好，", "OpenAI", "2026!"],
        )
        self.assertEqual(
            [
                (token.start_time, token.end_time)
                for token in aligned.segments[0].word_tokens or []
            ],
            [(2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0)],
        )


if __name__ == "__main__":
    unittest.main()
