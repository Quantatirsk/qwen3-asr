"""Compatibility word timestamps for the Ascend offline runtime."""

from __future__ import annotations

import unicodedata

from .results import ASRFullResult, WordToken

UNIFORM_FALLBACK_METHOD = "uniform_fallback"


def _is_cjk_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _alignment_units(text: str) -> list[str]:
    units: list[str] = []
    current_word: list[str] = []
    leading_punctuation = ""

    def flush_word() -> None:
        nonlocal leading_punctuation
        if not current_word:
            return
        units.append(f"{leading_punctuation}{''.join(current_word)}")
        current_word.clear()
        leading_punctuation = ""

    for character in text:
        if character.isspace():
            flush_word()
            continue
        if _is_cjk_character(character):
            flush_word()
            units.append(f"{leading_punctuation}{character}")
            leading_punctuation = ""
            continue

        category = unicodedata.category(character)
        if category[0] in {"L", "M", "N"}:
            current_word.append(character)
            continue

        flush_word()
        if units:
            units[-1] += character
        else:
            leading_punctuation += character

    flush_word()
    if leading_punctuation and units:
        units[-1] += leading_punctuation
    return units


def apply_uniform_word_timestamps(result: ASRFullResult) -> ASRFullResult:
    """Populate estimated word tokens inside each recognized audio segment."""
    for segment in result.segments:
        units = _alignment_units(segment.text)
        duration = max(0.0, segment.end_time - segment.start_time)
        if not units or duration == 0.0:
            segment.word_tokens = []
            continue

        step = duration / len(units)
        segment.word_tokens = [
            WordToken(
                text=unit,
                start_time=round(segment.start_time + index * step, 3),
                end_time=(
                    round(segment.end_time, 3)
                    if index == len(units) - 1
                    else round(segment.start_time + (index + 1) * step, 3)
                ),
            )
            for index, unit in enumerate(units)
        ]

    result.word_timestamp_method = UNIFORM_FALLBACK_METHOD
    return result
