"""Offline transcript results; word timestamps are relative to their segment."""

from dataclasses import dataclass


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


@dataclass
class ASRFullResult:
    text: str
    segments: list[ASRSegmentResult]
    duration: float
