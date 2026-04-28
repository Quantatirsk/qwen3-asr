# -*- coding: utf-8 -*-
"""Shared offline transcription workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Request

from app.models.common import SampleRate
from app.services.asr.engines import ASRFullResult
from app.services.asr.model_selection import get_default_offline_model_id
from app.services.asr.runtime import OfflineASRRequest, get_runtime_router
from app.services.audio import get_audio_service


@dataclass(frozen=True)
class PreparedAudio:
    normalized_path: str
    duration: float
    original_path: str
    timestamp_scale: float = 1.0


@dataclass(frozen=True)
class OfflineTranscriptionOptions:
    sample_rate: int = 16000
    hotwords: str = ""
    enable_speaker_diarization: bool = True
    word_timestamps: bool = False
    task_id: Optional[str] = None


class OfflineTranscriptionService:
    """Prepare audio and run the active offline ASR model."""

    def __init__(self) -> None:
        self._audio_service = get_audio_service()

    async def prepare_from_request(
        self,
        *,
        request: Request,
        audio_address: Optional[str],
        task_id: str,
        sample_rate: int,
    ) -> PreparedAudio:
        audio = await self._audio_service.process_from_request(
            request=request,
            audio_address=audio_address,
            task_id=task_id,
            sample_rate=sample_rate,
        )
        return PreparedAudio(
            normalized_path=audio.normalized_path,
            duration=audio.duration,
            original_path=audio.original_path,
            timestamp_scale=audio.timestamp_scale,
        )

    async def prepare_upload(
        self,
        *,
        audio_data: bytes,
        filename: Optional[str],
        task_id: str,
        sample_rate: int,
    ) -> PreparedAudio:
        audio = await self._audio_service.process_upload_file(
            audio_data=audio_data,
            filename=filename,
            task_id=task_id,
            sample_rate=sample_rate,
        )
        return PreparedAudio(
            normalized_path=audio.normalized_path,
            duration=audio.duration,
            original_path=audio.original_path,
            timestamp_scale=audio.timestamp_scale,
        )

    async def transcribe(
        self,
        prepared_audio: PreparedAudio,
        options: OfflineTranscriptionOptions,
    ) -> ASRFullResult:
        model_id = get_default_offline_model_id()
        return await get_runtime_router().run_offline(
            OfflineASRRequest(
                model_id=model_id,
                audio_path=prepared_audio.normalized_path,
                hotwords=options.hotwords,
                enable_punctuation=True,
                enable_itn=True,
                sample_rate=options.sample_rate or int(SampleRate.RATE_16000),
                enable_speaker_diarization=options.enable_speaker_diarization,
                word_timestamps=options.word_timestamps,
                timestamp_scale=prepared_audio.timestamp_scale,
                task_id=options.task_id,
            )
        )

    def cleanup(self, prepared_audio: Optional[PreparedAudio]) -> None:
        if prepared_audio is None:
            return
        self._audio_service.cleanup(
            prepared_audio.original_path,
            prepared_audio.normalized_path,
        )


_offline_transcription_service: Optional[OfflineTranscriptionService] = None


def get_offline_transcription_service() -> OfflineTranscriptionService:
    global _offline_transcription_service
    if _offline_transcription_service is None:
        _offline_transcription_service = OfflineTranscriptionService()
    return _offline_transcription_service
