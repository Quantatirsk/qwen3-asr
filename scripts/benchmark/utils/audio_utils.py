# -*- coding: utf-8 -*-
"""
音频处理工具
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf


def load_audio_file(
    audio_path: Path,
    target_sample_rate: int = 16000,
) -> Tuple[bytes, float]:
    """
    加载音频文件并转换为单声道 int16 LE PCM

    Args:
        audio_path: 音频文件路径
        target_sample_rate: 目标采样率

    Returns:
        (pcm_bytes, duration_seconds): PCM 字节数据和音频时长(秒)
    """
    # 读取音频文件
    audio_data, sample_rate = sf.read(audio_path, dtype="float32")

    # 如果是立体声，转换为单声道
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # 重采样到目标采样率
    if sample_rate != target_sample_rate:
        audio_data = resample_audio(audio_data, sample_rate, target_sample_rate)

    # 计算时长
    duration_seconds = len(audio_data) / target_sample_rate

    pcm_data = (np.clip(audio_data, -1, 1) * 32767).astype("<i2")
    pcm_bytes = pcm_data.tobytes()

    return pcm_bytes, duration_seconds


def resample_audio(
    audio_data: np.ndarray,
    orig_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """
    重采样音频

    Args:
        audio_data: 音频数据
        orig_sample_rate: 原始采样率
        target_sample_rate: 目标采样率

    Returns:
        重采样后的音频数据
    """
    if orig_sample_rate == target_sample_rate:
        return audio_data

    from math import gcd
    from scipy.signal import resample_poly

    divisor = gcd(orig_sample_rate, target_sample_rate)
    resampled = resample_poly(audio_data, target_sample_rate // divisor, orig_sample_rate // divisor)

    return resampled.astype(np.float32)


def get_audio_duration(audio_path: Path) -> float:
    """
    获取音频文件时长

    Args:
        audio_path: 音频文件路径

    Returns:
        时长(秒)
    """
    info = sf.info(audio_path)
    return info.duration


def split_audio_into_chunks(
    pcm_bytes: bytes,
    chunk_size: int,
) -> list:
    """
    将 PCM 数据分割成块

    Args:
        pcm_bytes: PCM 字节数据
        chunk_size: 每块的采样数

    Returns:
        字节块列表
    """
    chunk_bytes = chunk_size * 4
    chunks = []

    for i in range(0, len(pcm_bytes), chunk_bytes):
        chunk = pcm_bytes[i : i + chunk_bytes]
        chunks.append(chunk)

    return chunks
