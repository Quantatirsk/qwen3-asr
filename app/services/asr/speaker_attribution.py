"""Associate aligned words with independent diarization without duplicating text."""

from __future__ import annotations

import math
import unicodedata
from bisect import bisect_right
from dataclasses import replace

import numpy as np

from app.utils.speaker_diarizer import DiarizationResult

from .engines.base import ASRSegmentResult, WordToken

MIN_SPEAKER_COVERAGE = 0.5
MATERIAL_SPEAKER_COVERAGE = 0.2


def _text_positions(text: str) -> list[int]:
    # Match the forced aligner's punctuation removal, including contractions.
    return [
        index
        for index, char in enumerate(text)
        if char == "'" or unicodedata.category(char).startswith(("L", "N"))
    ]


def assign_speakers(
    result: ASRSegmentResult, diarization: DiarizationResult
) -> list[ASRSegmentResult]:
    """Keep each word once; mark competing or insufficient speech as uncertain.

    One speaker must cover at least half the word. Any other speaker covering
    at least one fifth makes attribution ambiguous, including speaker changes
    within a word. Activity probabilities only order candidates, never select
    a winner from overlapping voices. Input and output word times are relative
    to their respective segment starts; diarization uses recording time.
    """
    words = result.word_tokens or []
    if not words and not result.text:
        return []
    positions = _text_positions(result.text)
    normalized = "".join(result.text[index] for index in positions)
    units = [
        "".join(word.text[i] for i in _text_positions(word.text)) for word in words
    ]
    if not words or not all(units) or "".join(units) != normalized:
        # Never fabricate text spans when alignment and transcription disagree.
        return [replace(result, speaker_id=None, speaker_candidates=None)]

    intervals: dict[str, list[tuple[float, float]]] = {}
    for segment in sorted(diarization.segments, key=lambda item: item.start_sec):
        ranges = intervals.setdefault(segment.speaker_id, [])
        if ranges and segment.start_sec <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], segment.end_sec))
        else:
            ranges.append((segment.start_sec, segment.end_sec))
    interval_ends = {
        speaker: [end for _, end in spans] for speaker, spans in intervals.items()
    }

    def attribute(start: float, end: float) -> tuple[str | None, list[str] | None]:
        if start == end:
            if start < 0 or start > diarization.duration:
                return None, None
            # At EOF use the preceding instant; elsewhere use a half-open point.
            point = min(start, math.nextafter(diarization.duration, -math.inf))
            start, end = point, math.nextafter(point, math.inf)
        duration = end - start
        coverage: dict[str, float] = {}
        for speaker, spans in intervals.items():
            total = 0.0
            index = bisect_right(interval_ends[speaker], start)
            while index < len(spans) and spans[index][0] < end:
                lower, upper = spans[index]
                total += max(0.0, min(end, upper) - max(start, lower))
                index += 1
            if total > 0:
                coverage[speaker] = min(1.0, total / duration)
        candidates = [
            speaker
            for speaker, fraction in coverage.items()
            if fraction + 1e-9 >= MATERIAL_SPEAKER_COVERAGE
        ]
        if not candidates:
            return None, None
        if (
            len(candidates) == 1
            and coverage[candidates[0]] + 1e-9 >= MIN_SPEAKER_COVERAGE
        ):
            return candidates[0], None

        first = max(0, int(math.floor(start / diarization.frame_seconds)))
        last = min(
            len(diarization.probabilities),
            int(math.ceil(end / diarization.frame_seconds)),
        )
        scores: dict[str, float] = {}
        if last > first:
            frame_starts = np.arange(first, last) * diarization.frame_seconds
            weights = np.maximum(
                0.0,
                np.minimum(end, frame_starts + diarization.frame_seconds)
                - np.maximum(start, frame_starts),
            )
            probabilities = weights @ diarization.probabilities[first:last]
            scores = {
                speaker: float(probabilities[index])
                for index, speaker in enumerate(diarization.speaker_ids)
                if speaker is not None
            }
        candidates.sort(key=lambda speaker: -scores.get(speaker, 0.0))
        return None, candidates

    output: list[ASRSegmentResult] = []
    unit_offset = 0
    text_offset = 0
    for index, (word, unit) in enumerate(zip(words, units)):
        if not (
            math.isfinite(word.start_time)
            and math.isfinite(word.end_time)
            and 0
            <= word.start_time
            <= word.end_time
            <= result.end_time - result.start_time + 1e-6
        ):
            raise ValueError("Aligned word timestamps are outside their ASR segment")
        start = result.start_time + word.start_time
        end = result.start_time + word.end_time
        speaker, candidates = attribute(start, end)
        unit_offset += len(unit)
        text_end = (
            positions[unit_offset] if index + 1 < len(words) else len(result.text)
        )
        piece = result.text[text_offset:text_end]
        text_offset = text_end
        if (
            output
            and output[-1].speaker_id == speaker
            and set(output[-1].speaker_candidates or []) == set(candidates or [])
        ):
            group = output[-1]
            group.text += piece
            group.end_time = max(group.end_time, end)
        else:
            group = ASRSegmentResult(
                text=piece,
                start_time=start,
                end_time=end,
                speaker_id=speaker,
                speaker_candidates=candidates,
                word_tokens=[],
            )
            output.append(group)
        assert group.word_tokens is not None
        group.word_tokens.append(
            WordToken(word.text, start - group.start_time, end - group.start_time)
        )
    return output
