"""Synchronous audio preparation with scoped ownership of temporary files."""

import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Optional

from ...core.config import settings
from ...core.exceptions import InvalidMessageException
from ...utils.audio import (
    cleanup_temp_file,
    download_audio_from_url,
    get_audio_duration,
    get_audio_file_suffix,
    normalize_audio_for_asr,
    save_audio_to_temp_file,
)


@dataclass(frozen=True)
class AudioProcessingResult:
    normalized_path: str
    duration: float
    original_path: str
    timestamp_scale: float = 1.0


class AudioProcessingService:
    """Prepare uploaded bytes or a URL without knowing the transport protocol."""

    @contextmanager
    def prepare(
        self,
        *,
        audio_data: Optional[bytes],
        audio_address: Optional[str] = None,
        filename: Optional[str] = None,
        task_id: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> Iterator[AudioProcessingResult]:
        task_id = task_id or "unknown"
        # An explicitly supplied upload wins, including an invalid empty upload.
        if audio_data is None and audio_address:
            audio_data = download_audio_from_url(audio_address)
            filename = audio_address
        if not audio_data:
            raise InvalidMessageException("Audio data is empty", task_id)
        if len(audio_data) > settings.MAX_AUDIO_SIZE:
            raise InvalidMessageException(
                f"Audio exceeds the {settings.MAX_AUDIO_SIZE} byte limit", task_id
            )

        with ExitStack() as files:
            suffix = get_audio_file_suffix(
                audio_address=filename, audio_data=audio_data
            )
            original_path = save_audio_to_temp_file(audio_data, suffix)
            files.callback(cleanup_temp_file, original_path)
            normalized = normalize_audio_for_asr(original_path, sample_rate)
            if normalized.path != original_path:
                files.callback(cleanup_temp_file, normalized.path)
            yield AudioProcessingResult(
                normalized_path=normalized.path,
                duration=get_audio_duration(normalized.path)
                * normalized.timestamp_scale,
                original_path=original_path,
                timestamp_scale=normalized.timestamp_scale,
            )


_audio_service: Optional[AudioProcessingService] = None
_audio_service_lock = threading.Lock()


def get_audio_service() -> AudioProcessingService:
    global _audio_service
    if _audio_service is None:
        with _audio_service_lock:
            if _audio_service is None:
                _audio_service = AudioProcessingService()
    return _audio_service
