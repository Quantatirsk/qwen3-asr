"""Result types shared by the offline ASR pipeline."""

from dataclasses import dataclass
from typing import Optional


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
    speaker_id: Optional[str] = None
    word_tokens: Optional[list[WordToken]] = None


@dataclass
class ASRFullResult:
    text: str
    segments: list[ASRSegmentResult]
    duration: float
    word_timestamp_method: Optional[str] = None
