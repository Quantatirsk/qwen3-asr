"""Offline ASR engine contract and long-audio orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from app.core.config import settings
from app.core.exceptions import DefaultServerErrorException
from app.services.asr.results import ASRFullResult, ASRSegmentResult


class BaseASREngine(ABC):
    """Small interface for the single offline Qwen runtime."""

    @abstractmethod
    def transcribe_file(
        self,
        audio_path: str,
        enable_itn: bool = True,
        sample_rate: int = 16000,
    ) -> str:
        """Transcribe one normalized audio file."""

    @abstractmethod
    def transcribe_segments(
        self,
        segments: list[Any],
        enable_itn: bool = True,
        sample_rate: int = 16000,
    ) -> list[ASRSegmentResult]:
        """Transcribe prepared VAD or diarization segments."""

    def transcribe_long_audio(
        self,
        audio_path: str,
        enable_itn: bool = True,
        sample_rate: int = 16000,
        enable_speaker_diarization: bool = True,
        timestamp_scale: float = 1.0,
        task_id: Optional[str] = None,
    ) -> ASRFullResult:
        from app.services.asr.long_audio import prepare_long_audio

        try:
            with prepare_long_audio(
                audio_path,
                self.device,
                enable_speaker_diarization,
                self.model_id,
                task_id,
            ) as audio:
                results = []
                for start in range(0, len(audio.segments), settings.ASR_BATCH_SIZE):
                    results.extend(
                        self.transcribe_segments(
                            list(
                                audio.segments[start : start + settings.ASR_BATCH_SIZE]
                            ),
                            enable_itn=enable_itn,
                            sample_rate=sample_rate,
                        )
                    )
                return audio.finish(results, timestamp_scale)
        except DefaultServerErrorException:
            raise
        except Exception as exc:
            raise DefaultServerErrorException(
                f"Long audio transcription failed: {exc}"
            ) from exc

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
