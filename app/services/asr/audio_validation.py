# -*- coding: utf-8 -*-
"""Audio-related request validation helpers."""

from __future__ import annotations

from typing import Optional

from ...core.exceptions import InvalidParameterException
from ...models.common import SampleRate


SUPPORTED_SAMPLE_RATES = SampleRate.get_enums()


def validate_sample_rate(rate: Optional[int]) -> int:
    """Validate input sample rate."""
    if not rate:
        return 16000

    if rate not in SUPPORTED_SAMPLE_RATES:
        raise InvalidParameterException(
            f"不支持的采样率: {rate}。支持的采样率: {', '.join(map(str, SUPPORTED_SAMPLE_RATES))}"
        )
    return rate
