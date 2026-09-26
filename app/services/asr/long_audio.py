"""Diarized audio preparation, result assembly, and temporary file ownership."""

from __future__ import annotations

import logging
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

from app.core.config import settings
from app.core.logging import log_inference_metrics
from app.services.asr.engines.base import ASRFullResult, ASRSegmentResult
from app.utils.audio import get_audio_duration

if TYPE_CHECKING:
    from app.utils.audio_splitter import AudioSegment
    from app.utils.speaker_diarizer import DiarizationResult

logger = logging.getLogger(__name__)


@dataclass
class OfflineASRRequest:
    model_id: str
    audio_path: str
    hotwords: str = ""
    enable_punctuation: bool = True
    enable_itn: bool = True
    sample_rate: int = 16000
    enable_speaker_diarization: bool = True
    word_timestamps: bool = False
    timestamp_scale: float = 1.0
    task_id: Optional[str] = None


@dataclass
class PreparedLongAudio:
    segments: Sequence[AudioSegment]
    duration: float
    diarization: DiarizationResult | None = None

    def finish(
        self,
        results: Sequence[ASRSegmentResult],
        timestamp_scale: float,
        *,
        word_timestamps: bool = True,
    ) -> ASRFullResult:
        from .speaker_attribution import assign_speakers

        output = []
        for segment, result in zip(self.segments, results, strict=True):
            if not result.text:
                continue
            absolute = replace(
                result, start_time=segment.start_sec, end_time=segment.end_sec
            )
            groups = (
                assign_speakers(absolute, self.diarization)
                if self.diarization is not None
                else [absolute]
            )
            for group in groups:
                words = (
                    [
                        replace(
                            word,
                            start_time=word.start_time * timestamp_scale,
                            end_time=word.end_time * timestamp_scale,
                        )
                        for word in group.word_tokens
                    ]
                    if word_timestamps and group.word_tokens
                    else None
                )
                output.append(
                    replace(
                        group,
                        start_time=group.start_time * timestamp_scale,
                        end_time=group.end_time * timestamp_scale,
                        word_tokens=words,
                    )
                )
        speaker_segments = None
        if self.diarization is not None:
            speaker_segments = [
                replace(
                    span,
                    start_sec=span.start_sec * timestamp_scale,
                    end_sec=span.end_sec * timestamp_scale,
                )
                for span in self.diarization.segments
            ]
        return ASRFullResult(
            text="\n".join(result.text for result in results if result.text),
            segments=output,
            duration=self.duration * timestamp_scale,
            speaker_segments=speaker_segments,
        )


@contextmanager
def prepare_long_audio(
    audio_path: str,
    device: str,
    enable_speaker_diarization: bool,
    model_id: str,
    task_id: str | None = None,
) -> Iterator[PreparedLongAudio]:
    from app.utils.audio_splitter import AudioSplitter

    started = time.perf_counter()
    duration = 0.0
    status = "error"
    Path(settings.TEMP_DIR).mkdir(parents=True, exist_ok=True)
    try:
        duration = get_audio_duration(audio_path)
        # Own the directory, including partial writes; never delete a borrowed input.
        with tempfile.TemporaryDirectory(
            prefix="asr-segments-", dir=settings.TEMP_DIR
        ) as directory:
            diarization = None
            if enable_speaker_diarization:
                from app.utils.speaker_diarizer import get_speaker_diarizer

                diarization = get_speaker_diarizer().diarize(audio_path)
            # Recognition never duplicates overlapping speaker intervals.
            segments = AudioSplitter(device=device).split_audio_file(
                audio_path, output_dir=directory
            )
            yield PreparedLongAudio(segments, duration, diarization)
            status = "success"
    finally:
        log_inference_metrics(
            logger=logger,
            message="Long audio transcription finished",
            task_id=task_id,
            duration_ms=(time.perf_counter() - started) * 1000,
            audio_duration_sec=duration,
            model_id=model_id,
            status=status,
        )
