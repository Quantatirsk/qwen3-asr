"""Shared offline transcription workflow and task resource ownership."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from app.core.executor import run_sync
from app.core.exceptions import InvalidParameterException
from app.services.asr.results import ASRFullResult
from app.services.asr.manager import ASCEND_MODEL_ID
from app.services.asr.uniform_alignment import apply_uniform_word_timestamps
from app.services.asr.long_audio import OfflineASRRequest
from app.services.asr.runtime import get_runtime_router

if TYPE_CHECKING:
    from app.services.audio.audio_service import AudioProcessingService


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
        self._audio_service: Optional[AudioProcessingService] = None

    def _get_audio_service(self) -> AudioProcessingService:
        if self._audio_service is None:
            from app.services.audio import get_audio_service

            self._audio_service = get_audio_service()
        return self._audio_service

    async def start_transcription(
        self,
        *,
        audio_data: Optional[bytes],
        options: OfflineTranscriptionOptions,
        filename: Optional[str] = None,
        audio_address: Optional[str] = None,
    ) -> asyncio.Task[ASRFullResult]:
        if options.hotwords.strip():
            raise InvalidParameterException(
                "vocabulary_id is not supported by the Ascend offline runtime"
            )
        resources = ExitStack()
        try:
            # Register ownership in the worker before returning across a cancellation point.
            audio = await run_sync(
                resources.enter_context,
                self._get_audio_service().prepare(
                    audio_data=audio_data,
                    audio_address=audio_address,
                    filename=filename,
                    task_id=options.task_id,
                    sample_rate=options.sample_rate,
                ),
            )
            request = OfflineASRRequest(
                model_id=ASCEND_MODEL_ID,
                audio_path=audio.normalized_path,
                enable_itn=True,
                sample_rate=options.sample_rate,
                enable_speaker_diarization=options.enable_speaker_diarization,
                timestamp_scale=audio.timestamp_scale,
                task_id=options.task_id,
            )

            async def transcribe() -> ASRFullResult:
                result = await get_runtime_router().run_offline(request)
                if options.word_timestamps:
                    apply_uniform_word_timestamps(result)
                return result

            task = asyncio.create_task(transcribe())
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
