"""Offline transcript results; word timestamps are relative to their segment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.utils.speaker_diarizer import SpeakerSegment


@dataclass
class WordToken:
    text: str
    start_time: float
    end_time: float


@dataclass
class ASRSegmentResult:
    text: str
    start_time: float
    end_time: float
    speaker_id: str | None = None
    word_tokens: list[WordToken] | None = None
    speaker_candidates: list[str] | None = None


@dataclass
class ASRFullResult:
    text: str
    segments: list[ASRSegmentResult]
    duration: float
    speaker_segments: list[SpeakerSegment] | None = None
