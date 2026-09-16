"""Long audio preparation, assembly, and file ownership."""

from __future__ import annotations

import logging
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

from app.core.config import settings
from app.core.logging import log_inference_metrics
from app.services.asr.results import ASRFullResult, ASRSegmentResult
from app.utils.audio import get_audio_duration

if TYPE_CHECKING:
    from app.utils.audio_splitter import AudioSegment
    from app.utils.speaker_diarizer import SpeakerSegment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OfflineASRRequest:
    model_id: str
    audio_path: str
    enable_itn: bool = True
    sample_rate: int = 16000
    enable_speaker_diarization: bool = True
    timestamp_scale: float = 1.0
    task_id: Optional[str] = None


@dataclass
class PreparedLongAudio:
    segments: Sequence[AudioSegment | SpeakerSegment]
    duration: float

    def finish(
        self, results: Sequence[ASRSegmentResult], timestamp_scale: float
    ) -> ASRFullResult:
        output = []
        for segment, result in zip(self.segments, results, strict=True):
            if not result.text:
                continue
            words = result.word_tokens
            if words and timestamp_scale != 1.0:
                for word in words:
                    word.start_time *= timestamp_scale
                    word.end_time *= timestamp_scale
            output.append(
                ASRSegmentResult(
                    text=result.text,
                    start_time=segment.start_sec * timestamp_scale,
                    end_time=segment.end_sec * timestamp_scale,
                    speaker_id=segment.speaker_id,
                    word_tokens=words,
                )
            )
        return ASRFullResult(
            text="\n".join(item.text for item in output),
            segments=output,
            duration=self.duration * timestamp_scale,
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
            segments: Sequence[AudioSegment | SpeakerSegment] = []
            if enable_speaker_diarization:
                from app.utils.speaker_diarizer import SpeakerDiarizer

                segments = SpeakerDiarizer().split_audio_by_speakers(
                    audio_path, output_dir=directory
                )
            if not segments:
                segments = AudioSplitter(device=device).split_audio_file(
                    audio_path, output_dir=directory
                )
            if not segments:
                raise ValueError("Audio preparation produced no segments")
            yield PreparedLongAudio(segments, duration)
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
