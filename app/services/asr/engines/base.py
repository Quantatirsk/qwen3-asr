"""Offline ASR engine contract and long-audio orchestration."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from app.core.config import settings
from app.core.exceptions import DefaultServerErrorException
from app.core.logging import log_inference_metrics
from app.services.asr.results import ASRFullResult, ASRSegmentResult
from app.utils.audio import get_audio_duration

logger = logging.getLogger(__name__)


class BaseASREngine(ABC):
    """Small interface for the single offline Qwen runtime."""

    @abstractmethod
    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
    ) -> str:
        """Transcribe one normalized audio file."""

    @abstractmethod
    def _transcribe_batch(
        self,
        segments: list[Any],
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
    ) -> list[ASRSegmentResult]:
        """Transcribe prepared VAD or diarization segments."""

    def transcribe_long_audio(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
        enable_speaker_diarization: bool = True,
        timestamp_scale: float = 1.0,
        task_id: Optional[str] = None,
    ) -> ASRFullResult:
        from app.utils.audio_splitter import AudioSplitter

        started_at = time.time()
        task_prefix = f"[{task_id}] " if task_id else ""
        duration = 0.0
        speaker_segments = None
        audio_segments = None

        try:
            duration = get_audio_duration(audio_path)
            if enable_speaker_diarization:
                from app.utils.speaker_diarizer import SpeakerDiarizer

                speaker_segments = SpeakerDiarizer().split_audio_by_speakers(audio_path)
                if not speaker_segments:
                    logger.warning("%sspeaker diarization found no segments; using VAD", task_prefix)

            if not speaker_segments:
                audio_segments = AudioSplitter(device=self.device).split_audio_file(audio_path)

            segments_to_process = speaker_segments or audio_segments
            if not segments_to_process:
                raise DefaultServerErrorException("音频分割失败：未生成任何片段")

            results: list[ASRSegmentResult] = []
            for batch_start in range(0, len(segments_to_process), settings.ASR_BATCH_SIZE):
                batch = segments_to_process[
                    batch_start : batch_start + settings.ASR_BATCH_SIZE
                ]
                batch_results = self._transcribe_batch(
                    batch,
                    hotwords=hotwords,
                    enable_punctuation=enable_punctuation,
                    enable_itn=enable_itn,
                    sample_rate=sample_rate,
                )
                for source, transcribed in zip(batch, batch_results):
                    if not transcribed.text:
                        continue
                    results.append(
                        ASRSegmentResult(
                            text=transcribed.text,
                            start_time=float(getattr(source, "start_sec", 0.0)),
                            end_time=float(getattr(source, "end_sec", 0.0)),
                            speaker_id=getattr(source, "speaker_id", None),
                        )
                    )

            if timestamp_scale != 1.0:
                for segment in results:
                    segment.start_time *= timestamp_scale
                    segment.end_time *= timestamp_scale
                duration *= timestamp_scale

            result = ASRFullResult(
                text="\n".join(segment.text for segment in results),
                segments=results,
                duration=duration,
            )
            log_inference_metrics(
                logger=logger,
                message="离线长音频识别完成",
                task_id=task_id,
                duration_ms=(time.time() - started_at) * 1000,
                audio_duration_sec=duration,
                model_id=self.model_id,
                status="success",
                segments_count=len(results),
                batch_size=settings.ASR_BATCH_SIZE,
                enable_speaker_diarization=enable_speaker_diarization,
            )
            return result
        except Exception as exc:
            log_inference_metrics(
                logger=logger,
                message="离线长音频识别失败",
                task_id=task_id,
                duration_ms=(time.time() - started_at) * 1000,
                audio_duration_sec=duration,
                model_id=getattr(self, "model_id", "unknown"),
                status="error",
                error=str(exc),
            )
            if isinstance(exc, DefaultServerErrorException):
                raise
            raise DefaultServerErrorException(f"长音频识别失败: {exc}") from exc
        finally:
            try:
                if speaker_segments:
                    from app.utils.speaker_diarizer import SpeakerDiarizer

                    SpeakerDiarizer.cleanup_segments(speaker_segments)
                if audio_segments:
                    AudioSplitter.cleanup_segments(audio_segments)
            except Exception as exc:
                logger.warning("cleanup of temporary audio segments failed: %s", exc)

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the public model identifier."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the logical inference device."""

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Return whether the remote model is ready."""
