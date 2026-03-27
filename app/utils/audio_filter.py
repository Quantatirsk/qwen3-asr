# -*- coding: utf-8 -*-
"""
音频过滤工具 - 用于流式ASR的近场/远场声音检测
"""

import numpy as np
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


def calculate_rms_energy(audio_array: np.ndarray) -> float:
    """计算音频RMS能量

    Args:
        audio_array: float32音频数组，范围-1.0到1.0

    Returns:
        RMS能量值
    """
    if len(audio_array) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_array ** 2)))


def is_nearfield_voice(
    audio_array: np.ndarray,
    sample_rate: int = 16000,  # noqa: ARG001
    rms_threshold: float = 0.01,
    enable_filter: bool = True,
) -> Tuple[bool, Dict]:
    """判断是否为近场有效声音（仅基于RMS能量）

    Args:
        audio_array: float32音频数组，范围-1.0到1.0
        sample_rate: 采样率（保留用于兼容性）
        rms_threshold: RMS能量阈值
        enable_filter: 是否启用过滤（开关）

    Returns:
        (is_nearfield, metrics): 是否近场声音 + 检测指标详情
    """
    if not enable_filter:
        return True, {'enabled': False}

    if len(audio_array) == 0:
        return False, {'error': 'empty_array'}

    # 计算RMS能量
    rms_energy = calculate_rms_energy(audio_array)

    # 仅使用RMS能量判断
    is_nearfield = rms_energy >= rms_threshold

    metrics = {
        'rms_energy': round(rms_energy, 6),
        'is_nearfield': is_nearfield,
        'thresholds': {
            'rms': rms_threshold,
        }
    }

    return is_nearfield, metrics
