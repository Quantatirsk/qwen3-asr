"""Long audio preparation, pooled execution, assembly, and file ownership."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    AsyncContextManager,
    Awaitable,
    Callable,
    Iterator,
    Optional,
    Sequence,
)

from app.core.config import settings
from app.core.executor import run_sync, wait_for_completion
from app.core.logging import log_inference_metrics
from app.services.asr.engines.base import ASRFullResult, ASRSegmentResult, BaseASREngine
from app.utils.audio import get_audio_duration

if TYPE_CHECKING:
    from app.utils.audio_splitter import AudioSegment
    from app.utils.speaker_diarizer import SpeakerSegment

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


async def transcribe_with_pool(
    request: OfflineASRRequest,
    acquire_engine: Callable[[], Awaitable[AsyncContextManager[BaseASREngine]]],
    model_id: str,
) -> ASRFullResult:
    """Lease one engine per segment and drain all owned work before cleanup."""
    with ExitStack() as resources:
        audio = await run_sync(
            resources.enter_context,
            prepare_long_audio(
                request.audio_path,
                "cpu",
                request.enable_speaker_diarization,
                model_id,
                request.task_id,
            ),
        )

        async def transcribe(
            segment: AudioSegment | SpeakerSegment,
        ) -> ASRSegmentResult:
            async with await acquire_engine() as engine:
                results = await run_sync(
                    engine.transcribe_segments,
                    segments=[segment],
                    hotwords=request.hotwords,
                    enable_punctuation=request.enable_punctuation,
                    enable_itn=request.enable_itn,
                    sample_rate=request.sample_rate,
                    word_timestamps=request.word_timestamps,
                )
                if len(results) != 1:
                    raise ValueError("A segment must produce exactly one result")
                return results[0]

        results = []
        for start in range(0, len(audio.segments), settings.ASR_BATCH_SIZE):
            # Keep waiting tasks bounded per request; leases are shared with realtime.
            tasks = [
                asyncio.create_task(transcribe(segment))
                for segment in audio.segments[start : start + settings.ASR_BATCH_SIZE]
            ]
            try:
                results.extend(await asyncio.gather(*tasks))
            except BaseException:
                for task in tasks:
                    task.cancel()
                await wait_for_completion(
                    asyncio.gather(*tasks, return_exceptions=True)
                )
                raise
        return audio.finish(results, request.timestamp_scale)
