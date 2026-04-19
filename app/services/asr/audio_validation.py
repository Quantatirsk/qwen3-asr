# -*- coding: utf-8 -*-
"""Audio-related request validation helpers."""

from __future__ import annotations

from typing import Optional

from ...core.exceptions import InvalidParameterException
from ...models.common import AudioFormat, SampleRate


SUPPORTED_AUDIO_FORMATS = AudioFormat.get_enums()
SUPPORTED_SAMPLE_RATES = SampleRate.get_enums()
SUPPORTED_LANGUAGE_CODES = [
    "zh",
    "en",
    "ja",
    "ko",
    "fr",
    "de",
    "es",
    "it",
    "pt",
    "ru",
    "ar",
    "hi",
    "th",
    "vi",
    "id",
    "ms",
    "tr",
    "pl",
    "nl",
    "sv",
]


def validate_audio_format(format_value: Optional[str]) -> str:
    """Validate input audio format."""
    if not format_value:
        return "wav"

    normalized = format_value.lower()
    if normalized not in SUPPORTED_AUDIO_FORMATS:
        raise InvalidParameterException(
            f"不支持的音频格式: {format_value}。支持的格式: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )
    return normalized


def validate_sample_rate(rate: Optional[int]) -> int:
    """Validate input sample rate."""
    if not rate:
        return 16000

    if rate not in SUPPORTED_SAMPLE_RATES:
        raise InvalidParameterException(
            f"不支持的采样率: {rate}。支持的采样率: {', '.join(map(str, SUPPORTED_SAMPLE_RATES))}"
        )
    return rate


def validate_audio_size(size_bytes: int, max_size: int, task_id: str = "") -> None:
    """Validate uploaded audio size."""
    if size_bytes > max_size:
        max_mb = max_size // 1024 // 1024
        raise InvalidParameterException(
            f"音频文件过大，最大支持 {max_mb}MB", task_id=task_id
        )


def validate_language(language: Optional[str]) -> Optional[str]:
    """Validate language code but keep permissive fallback behaviour."""
    if not language:
        return None

    normalized = language.lower()
    if normalized not in SUPPORTED_LANGUAGE_CODES:
        return normalized
    return normalized
