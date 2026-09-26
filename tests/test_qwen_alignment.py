import random
import unittest
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.services.asr.qwen3_alignment import repair_timestamps, split_alignment_units
from app.services.asr.r2t2_vllm import R2T2VLLMBackend


class TimestampRepairTest(unittest.TestCase):
    def test_clean_spans_are_unchanged(self):
        for values in ([], [0.0], [80.0, 160.0, 160.0, 320.0]):
            self.assertEqual(repair_timestamps(values, 1000), values)

    def test_cross_word_regression_and_reversed_pair(self):
        # Afternoon was assigned later than the following word. Repair times,
        # never the order of text or whole spans.
        self.assertEqual(
            repair_timestamps([100, 200, 700, 800, 300, 400, 500, 600], 1000),
            [100, 200, 200, 300, 300, 400, 500, 600],
        )
        self.assertEqual(
            repair_timestamps([100, 300, 200, 400], 1000), [100, 300, 300, 400]
        )

    def test_long_anomaly_interpolates_between_stable_anchors(self):
        self.assertEqual(
            repair_timestamps([0, 900, 800, 700, 400, 500, 600], 1000),
            [0, 100, 200, 300, 400, 500, 600],
        )

    def test_edges_overlap_and_audio_bounds(self):
        self.assertEqual(
            repair_timestamps([900, 100, 200, 300], 1000), [100, 100, 200, 300]
        )
        self.assertEqual(
            repair_timestamps([100, 200, 300, 0], 1000), [100, 200, 300, 300]
        )
        self.assertEqual(
            repair_timestamps([100, 300, 250, 400], 1000), [100, 300, 300, 400]
        )
        self.assertEqual(
            repair_timestamps([-80, 160, 800, 900], 550.25), [0, 160, 550.25, 550.25]
        )
        self.assertEqual(
            repair_timestamps([0, 100, 200, 0], 50.25), [0, 50.25, 50.25, 50.25]
        )

    def test_nonfinite_predictions_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                repair_timestamps([0, value], 1000)

    def test_arbitrary_predictions_remain_monotonic_and_bounded(self):
        rng = random.Random(47)
        for _ in range(200):
            values = [rng.uniform(-100, 1500) for _ in range(rng.randrange(1, 80))]
            fixed = repair_timestamps(values, 999.25)
            self.assertEqual(len(fixed), len(values))
            self.assertTrue(all(0 <= x <= 999.25 for x in fixed))
            self.assertTrue(all(a <= b for a, b in pairwise(fixed)))
            self.assertEqual(repair_timestamps(fixed, 999.25), fixed)


class AlignmentAdapterTest(unittest.TestCase):
    def make_backend(self, bins):
        backend = R2T2VLLMBackend.__new__(R2T2VLLMBackend)
        backend._timestamp_token_id = 42
        backend._timestamp_segment_time = 80
        # Non-timestamp tokens also have classifier outputs and must be ignored.
        output = SimpleNamespace(
            prompt_token_ids=[9] + [42] * len(bins) + [10],
            outputs=SimpleNamespace(data=np.eye(32)[[31] + bins + [31]]),
        )
        backend._get_forced_aligner = Mock(
            return_value=SimpleNamespace(encode=Mock(return_value=[output]))
        )
        return backend

    def test_units_follow_official_cleaning_for_chinese_and_english(self):
        self.assertEqual(
            split_alignment_units("我今天下午吃了鸡蛋。"), list("我今天下午吃了鸡蛋")
        )
        self.assertEqual(
            split_alignment_units("今天用Qwen3， it's good!"),
            ["今", "天", "用", "Qwen3", "it's", "good"],
        )
        self.assertEqual(
            split_alignment_units("café déjà vu 𠀀"), ["café", "déjà", "vu", "𠀀"]
        )
        self.assertEqual(split_alignment_units("，。！？"), [])

    def test_vllm_result_repairs_cross_character_order_and_preserves_text(self):
        backend = self.make_backend([1, 2, 7, 8, 3, 4, 5, 6])
        result = backend.align_transcript(
            "unused.wav", "下午吃蛋。", audio=np.zeros(16000)
        )
        self.assertEqual([x["text"] for x in result], list("下午吃蛋"))
        self.assertEqual([x["start_ms"] for x in result], [80, 160, 240, 400])
        self.assertTrue(all(a["end_ms"] <= b["start_ms"] for a, b in pairwise(result)))
        prompt = backend._get_forced_aligner().encode.call_args.args[0][0]["prompt"]
        self.assertNotIn("。", prompt)
        self.assertEqual(prompt.count("<timestamp>"), 8)

    def test_vllm_bounds_timestamps_to_actual_audio(self):
        backend = self.make_backend([1, 20])
        result = backend.align_transcript("unused.wav", "字", audio=np.zeros(8000))
        self.assertEqual(result, [{"text": "字", "start_ms": 80, "end_ms": 500}])

    def test_timestamp_count_mismatch_is_not_silently_mapped(self):
        for bins in ([1], [1, 2, 3]):
            backend = self.make_backend(bins)
            with self.assertRaisesRegex(RuntimeError, "count mismatch"):
                backend.align_transcript("unused.wav", "字", audio=np.zeros(16000))

    def test_punctuation_only_skips_inference(self):
        backend = self.make_backend([])
        self.assertEqual(backend.align_transcript("unused.wav", "。？！"), [])
        backend._get_forced_aligner.assert_not_called()

    def test_transcription_retains_punctuation_but_aligns_only_spoken_units(self):
        backend = self.make_backend([1, 2, 3, 4])
        backend._run_generate = Mock(return_value=[SimpleNamespace(text="你好。")])
        with patch(
            "app.services.asr.r2t2_vllm._load_audio", return_value=np.zeros(16000)
        ):
            backend._max_inference_batch_size = 4
            result = backend.transcribe_batch(["unused.wav"], word_timestamps=True)[0]
        self.assertEqual(result.text, "你好。")
        self.assertEqual([word.text for word in result.word_tokens], ["你", "好"])


if __name__ == "__main__":
    unittest.main()
