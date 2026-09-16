"""Shared offline transcription workflow and task resource ownership."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Optional

from app.core.executor import run_sync
from app.services.asr.engines import ASRFullResult
from app.services.asr.model_selection import get_default_offline_model_id
from app.services.asr.long_audio import OfflineASRRequest
from app.services.asr.runtime import get_runtime_router
from app.services.audio import get_audio_service


@dataclass(frozen=True)
class OfflineTranscriptionOptions:
    sample_rate: int = 16000
    hotwords: str = ""
    enable_speaker_diarization: bool = True
    word_timestamps: bool = False
    task_id: Optional[str] = None


class OfflineTranscriptionService:
    """Prepare audio before returning a task that owns its files until completion."""

    def __init__(self) -> None:
        self._audio_service = get_audio_service()

    async def start_transcription(
        self,
        *,
        audio_data: Optional[bytes],
        options: OfflineTranscriptionOptions,
        filename: Optional[str] = None,
        audio_address: Optional[str] = None,
    ) -> asyncio.Task[ASRFullResult]:
        resources = ExitStack()
        try:
            # Register ownership in the worker before returning across a cancellation point.
            audio = await run_sync(
                resources.enter_context,
                self._audio_service.prepare(
                    audio_data=audio_data,
                    audio_address=audio_address,
                    filename=filename,
                    task_id=options.task_id,
                    sample_rate=options.sample_rate,
                ),
            )
            request = OfflineASRRequest(
                model_id=get_default_offline_model_id(),
                audio_path=audio.normalized_path,
                hotwords=options.hotwords,
                enable_punctuation=True,
                enable_itn=True,
                sample_rate=options.sample_rate,
                enable_speaker_diarization=options.enable_speaker_diarization,
                word_timestamps=options.word_timestamps,
                timestamp_scale=audio.timestamp_scale,
                task_id=options.task_id,
            )
            task = asyncio.create_task(get_runtime_router().run_offline(request))
        except BaseException:
            resources.close()
            raise

        def finish(completed: asyncio.Task[ASRFullResult]) -> None:
            resources.close()
            # Observe failures even if a response never starts consuming the task.
            if not completed.cancelled():
                completed.exception()

        # A callback also handles cancellation before the coroutine starts running.
        task.add_done_callback(finish)
        return task


_offline_transcription_service: Optional[OfflineTranscriptionService] = None


def get_offline_transcription_service() -> OfflineTranscriptionService:
    global _offline_transcription_service
    if _offline_transcription_service is None:
        _offline_transcription_service = OfflineTranscriptionService()
    return _offline_transcription_service
