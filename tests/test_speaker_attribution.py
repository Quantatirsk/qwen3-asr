import unittest

import numpy as np

from app.services.asr.engines.base import ASRSegmentResult, WordToken
from app.services.asr.qwen3_alignment import split_alignment_units
from app.services.asr.speaker_attribution import assign_speakers
from app.utils.speaker_diarizer import DiarizationResult, SpeakerSegment


def diarization(
    spans: list[tuple[float, float, str]], duration: float = 5.0
) -> DiarizationResult:
    labels = list(dict.fromkeys(speaker for _, _, speaker in spans))
    probabilities = np.zeros((int(duration * 100), 8), dtype=np.float32)
    for start, end, speaker in spans:
        probabilities[int(start * 100) : int(end * 100), labels.index(speaker)] = 0.9
    return DiarizationResult(
        segments=[
            SpeakerSegment(start, end, speaker, 0.9) for start, end, speaker in spans
        ],
        probabilities=probabilities,
        frame_seconds=0.01,
        duration=duration,
        speaker_ids=tuple(labels + [None] * (8 - len(labels))),
    )


def transcript(
    text: str,
    timings: list[tuple[float, float]],
    start: float = 0.0,
    end: float = 5.0,
) -> ASRSegmentResult:
    units = split_alignment_units(text)
    assert len(units) == len(timings)
    return ASRSegmentResult(
        text=text,
        start_time=start,
        end_time=end,
        word_tokens=[
            WordToken(unit, lower, upper)
            for unit, (lower, upper) in zip(units, timings)
        ],
    )


class SpeakerAttributionTest(unittest.TestCase):
    def assert_preserved(
        self, original: ASRSegmentResult, output: list[ASRSegmentResult]
    ) -> None:
        self.assertEqual("".join(segment.text for segment in output), original.text)
        actual = [
            (
                word.text,
                segment.start_time + word.start_time,
                segment.start_time + word.end_time,
            )
            for segment in output
            for word in segment.word_tokens or []
        ]
        expected = [
            (
                word.text,
                original.start_time + word.start_time,
                original.start_time + word.end_time,
            )
            for word in original.word_tokens or []
        ]
        self.assertEqual([item[0] for item in actual], [item[0] for item in expected])
        np.testing.assert_allclose(
            [item[1:] for item in actual], [item[1:] for item in expected]
        )
        for segment in output:
            for word in segment.word_tokens or []:
                self.assertGreaterEqual(word.start_time, 0)
                self.assertLessEqual(
                    word.end_time, segment.end_time - segment.start_time + 1e-6
                )

    def test_single_speaker_preserves_punctuation_spaces_and_numbers(self) -> None:
        text = "  \u4eca\u5929\uff0c Qwen3 it's 12.5%: caf\u00e9!\n"
        units = split_alignment_units(text)
        original = transcript(
            text, [(i * 0.3, i * 0.3 + 0.2) for i in range(len(units))]
        )
        output = assign_speakers(original, diarization([(0, 5, "speaker-1")]))
        self.assert_preserved(original, output)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].speaker_id, "speaker-1")

    def test_short_interjection_survives_and_times_are_rebased(self) -> None:
        original = transcript(
            "Start. Yes! Continue.", [(0, 0.8), (1, 1.1), (1.2, 2)], 10, 13
        )
        output = assign_speakers(
            original,
            diarization(
                [
                    (10, 11, "speaker-1"),
                    (11, 11.1, "speaker-2"),
                    (11.1, 13, "speaker-1"),
                ],
                13,
            ),
        )
        self.assert_preserved(original, output)
        self.assertEqual(
            [segment.speaker_id for segment in output],
            ["speaker-1", "speaker-2", "speaker-1"],
        )
        self.assertEqual(
            [segment.text for segment in output], ["Start. ", "Yes! ", "Continue."]
        )
        self.assertAlmostEqual(output[1].word_tokens[0].start_time, 0)
        self.assertAlmostEqual(output[1].word_tokens[0].end_time, 0.1)

    def test_overlap_is_not_duplicated_or_assigned_by_probability(self) -> None:
        original = transcript("Mixed voices.", [(0, 1), (1, 2)])
        speakers = diarization([(0, 2, "speaker-1"), (0, 2, "speaker-2")])
        speakers.probabilities[:, 0] = 0.99
        speakers.probabilities[:, 1] = 0.55
        speakers.probabilities[100:, 0] = 0.55
        speakers.probabilities[100:, 1] = 0.99
        output = assign_speakers(original, speakers)
        self.assert_preserved(original, output)
        self.assertEqual(len(output), 1)
        self.assertIsNone(output[0].speaker_id)
        self.assertEqual(output[0].speaker_candidates, ["speaker-1", "speaker-2"])

    def test_switch_inside_word_is_uncertain(self) -> None:
        original = transcript("Across.", [(0, 1)])
        output = assign_speakers(
            original, diarization([(0, 0.6, "speaker-1"), (0.6, 1, "speaker-2")])
        )
        self.assertIsNone(output[0].speaker_id)
        self.assertEqual(output[0].speaker_candidates, ["speaker-1", "speaker-2"])

    def test_silence_and_insufficient_coverage_are_unknown(self) -> None:
        original = transcript("One. Two.", [(0, 1), (2, 3)])
        output = assign_speakers(original, diarization([(0, 0.3, "speaker-1")]))
        self.assert_preserved(original, output)
        self.assertTrue(all(segment.speaker_id is None for segment in output))
        self.assertEqual(output[0].speaker_candidates, ["speaker-1"])
        self.assertIsNone(output[1].speaker_candidates)

    def test_no_diarization_retains_transcript(self) -> None:
        original = transcript("Still here!", [(0, 1), (1, 2)])
        output = assign_speakers(original, diarization([]))
        self.assert_preserved(original, output)
        self.assertIsNone(output[0].speaker_id)

    def test_empty_and_unaligned_text(self) -> None:
        self.assertEqual(
            assign_speakers(ASRSegmentResult("", 0, 1), diarization([])), []
        )
        for text in ("! \n", "No alignment."):
            original = ASRSegmentResult(text, 0, 1)
            output = assign_speakers(original, diarization([(0, 1, "speaker-1")]))
            self.assertEqual(output[0].text, text)
            self.assertIsNone(output[0].speaker_id)

    def test_alignment_mismatch_retains_all_text_and_tokens_as_unknown(self) -> None:
        original = transcript("Keep every word.", [(0, 1), (1, 2), (2, 3)])
        original.word_tokens[1].text = "different"
        output = assign_speakers(original, diarization([(0, 5, "speaker-1")]))
        self.assert_preserved(original, output)
        self.assertIsNone(output[0].speaker_id)

    def test_zero_duration_word_at_boundary_and_recording_end(self) -> None:
        original = transcript(
            "Before. After. End.", [(0.5, 0.5), (1, 1), (2, 2)], end=2
        )
        output = assign_speakers(
            original, diarization([(0, 1, "speaker-1"), (1, 2, "speaker-2")], 2)
        )
        self.assert_preserved(original, output)
        self.assertEqual(
            [segment.speaker_id for segment in output], ["speaker-1", "speaker-2"]
        )

    def test_recording_labels_are_preserved_across_asr_chunks(self) -> None:
        speakers = diarization(
            [(0, 1, "speaker-1"), (1, 2, "speaker-2"), (90, 91, "speaker-1")], 91
        )
        early = transcript("Early.", [(0, 1)], end=1)
        late = transcript("Returned.", [(0, 1)], start=90, end=91)
        self.assertEqual(assign_speakers(early, speakers)[0].speaker_id, "speaker-1")
        self.assertEqual(assign_speakers(late, speakers)[0].speaker_id, "speaker-1")

    def test_invalid_timestamps_raise(self) -> None:
        for lower, upper in ((0, 6), (-1, 1), (2, 1), (float("nan"), 1)):
            with self.subTest(lower=lower, upper=upper), self.assertRaises(ValueError):
                assign_speakers(
                    transcript("Invalid.", [(lower, upper)]), diarization([])
                )


if __name__ == "__main__":
    unittest.main()
