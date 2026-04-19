# -*- coding: utf-8 -*-
"""
统一音频处理工具
ASR音频处理功能
"""

import os
import tempfile
import requests
import librosa
import soundfile as sf
import torchaudio
import torch
import numpy as np
import subprocess
import logging
from typing import Tuple, Optional, Any, cast
from io import BytesIO

from ..core.config import settings
from ..core.exceptions import (
    InvalidParameterException,
    InvalidMessageException,
    DefaultServerErrorException,
)
from ..models.common import SampleRate, AudioFormat

logger = logging.getLogger(__name__)


def validate_audio_format(format_str: Optional[str]) -> bool:
    """验证音频格式是否支持"""
    if not format_str:
        return True  # 如果未指定格式，允许通过

    # 统一转换为小写进行比较
    format_lower = format_str.lower()
    supported_formats = [fmt.lower() for fmt in AudioFormat.get_enums()]

    return format_lower in supported_formats


def validate_sample_rate(sample_rate: Optional[int]) -> bool:
    """验证采样率是否支持"""
    if not sample_rate:
        return True  # 如果未指定采样率，允许通过

    return sample_rate in SampleRate.get_enums()


def download_audio_from_url(url: str, max_size: Optional[int] = None) -> bytes:
    """从URL下载音频文件

    Args:
        url: 音频文件URL
        max_size: 最大文件大小限制

    Returns:
        音频文件的二进制数据

    Raises:
        InvalidParameterException: URL无效或下载失败
        InvalidMessageException: 文件太大
    """
    if not url:
        raise InvalidParameterException("URL不能为空")

    max_file_size = max_size or settings.MAX_AUDIO_SIZE

    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        # 检查Content-Length头
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > max_file_size:
            max_size_mb = max_file_size // 1024 // 1024
            raise InvalidMessageException(f"音频文件太大，最大支持{max_size_mb}MB")

        # 分块下载并检查大小
        audio_data = BytesIO()
        downloaded_size = 0

        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > max_file_size:
                max_size_mb = max_file_size // 1024 // 1024
                raise InvalidMessageException(f"音频文件太大，最大支持{max_size_mb}MB")
            audio_data.write(chunk)

        return audio_data.getvalue()

    except requests.RequestException as e:
        raise InvalidParameterException(f"下载音频文件失败: {str(e)}")


def save_audio_to_temp_file(audio_data: bytes, suffix: str = ".wav") -> str:
    """保存音频数据到临时文件

    Args:
        audio_data: 音频二进制数据
        suffix: 文件后缀

    Returns:
        临时文件路径

    Raises:
        AudioProcessingException: 保存失败
    """
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=settings.TEMP_DIR
        ) as temp_file:
            temp_file.write(audio_data)
            return temp_file.name
    except Exception as e:
        raise DefaultServerErrorException(f"保存音频文件失败: {str(e)}")


def cleanup_temp_file(file_path: str) -> None:
    """清理临时文件

    Args:
        file_path: 文件路径
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            # Clean up sidecar timestamp scale file if exists
            tsscale_path = file_path + ".tsscale"
            if os.path.exists(tsscale_path):
                os.remove(tsscale_path)
    except Exception:
        # 静默忽略清理错误
        pass


def load_audio_file(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """加载音频文件并转换为指定采样率

    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率

    Returns:
        (audio_data, sample_rate): 音频数据和采样率

    Raises:
        AudioProcessingException: 加载失败
    """
    try:
        # 使用librosa加载音频
        audio_data, sr = librosa.load(audio_path, sr=target_sr)
        return audio_data, int(sr)
    except Exception as e:
        raise DefaultServerErrorException(f"加载音频文件失败: {str(e)}")


def get_audio_duration(audio_path: str) -> float:
    """获取音频文件时长

    Args:
        audio_path: 音频文件路径

    Returns:
        音频时长（秒）

    Raises:
        AudioProcessingException: 获取时长失败
    """
    try:
        # Load audio and get duration
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except Exception as e:
        raise DefaultServerErrorException(f"获取音频时长失败: {str(e)}")


def get_container_duration(audio_path: str) -> Optional[float]:
    """通过 ffprobe 获取音频容器的 metadata 时长

    对于 m4a/AAC 等压缩格式，容器记录的时长可能与实际解码样本数不一致
    （常见于 m3u8/ts 分片合并的音频）。返回 None 表示获取失败。
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        logger.debug(f"ffprobe 获取容器时长失败: {e}")
    return None


def get_timestamp_scale(original_audio_path: str, decoded_duration: float) -> float:
    """计算时间戳缩放系数

    对比容器 metadata 时长与解码后实际时长，返回缩放系数。
    用于修正 m4a/AAC 等格式中容器时长与解码时长不一致的问题。

    Args:
        original_audio_path: 原始音频文件路径（转换前）
        decoded_duration: 解码后的实际音频时长（秒）

    Returns:
        缩放系数（容器时长 / 解码时长），无差异时返回 1.0
    """
    container_duration = get_container_duration(original_audio_path)
    if container_duration is None or decoded_duration <= 0:
        return 1.0

    scale = container_duration / decoded_duration
    if abs(scale - 1.0) < 0.001:
        # 差异 < 0.1%，忽略
        return 1.0

    logger.info(
        f"检测到容器/解码时长不一致: container={container_duration:.3f}s, "
        f"decoded={decoded_duration:.3f}s, scale={scale:.6f}"
    )
    return scale


def resample_audio_array(
    audio_array: np.ndarray,
    original_sr: int,
    target_sr: int,
) -> np.ndarray:
    """重采样音频数组

    Args:
        audio_array: 原始音频数据
        original_sr: 原始采样率
        target_sr: 目标采样率

    Returns:
        重采样后的音频数据
    """
    if original_sr == target_sr:
        return audio_array

    try:
        # 确保是1D数组用于librosa重采样
        if audio_array.ndim > 1:
            # 如果是多声道，取第一个声道
            if audio_array.shape[0] > audio_array.shape[1]:
                audio_1d = audio_array[0, :]
            else:
                audio_1d = (
                    audio_array[:, 0]
                    if audio_array.shape[1] > 1
                    else audio_array.flatten()
                )
        else:
            audio_1d = audio_array

        # 使用librosa进行重采样
        resampled = librosa.resample(audio_1d, orig_sr=original_sr, target_sr=target_sr)

        logger.info(f"音频重采样: {original_sr}Hz -> {target_sr}Hz")
        return resampled

    except Exception as e:
        logger.warning(f"音频重采样失败: {str(e)}，使用原始音频")
        return audio_array


def adjust_audio_volume(audio_array: np.ndarray, volume: int) -> np.ndarray:
    """调节音频音量

    Args:
        audio_array: 音频数据数组
        volume: 音量值，范围0~100，50为原始音量

    Returns:
        调节后的音频数据
    """
    if int(volume) == 50:
        return audio_array

    if volume < 0 or volume > 100:
        logger.warning(f"音量值{volume}超出范围[0,100]，使用默认值50")
        volume = 50

    # 将音量值转换为倍数 (0-100 -> 0-2.0)
    volume_factor = volume / 50.0

    # 应用音量调节
    adjusted_audio = audio_array * volume_factor

    # 防止削波，如果音量过大导致超过范围，进行归一化
    max_val = np.max(np.abs(adjusted_audio))
    if max_val > 1.0:
        adjusted_audio = adjusted_audio / max_val
        logger.info(f"音量调节后进行归一化，最大值: {max_val:.3f}")

    logger.info(f"音频音量已调节: {volume}/100 (倍数: {volume_factor:.2f})")
    return adjusted_audio


def save_audio_array(
    audio_array: np.ndarray,
    output_path: str,
    sample_rate: int = 22050,
    format: str = "wav",
    original_sr: Optional[int] = None,
    volume: int = 50,
) -> str:
    """保存音频数组到文件

    Args:
        audio_array: 音频数据数组
        output_path: 输出文件路径
        sample_rate: 目标采样率
        format: 音频格式
        original_sr: 原始采样率（用于重采样）
        volume: 音量值，范围0~100，默认50

    Returns:
        保存的文件路径

    Raises:
        AudioProcessingException: 保存失败
    """
    try:
        # 如果指定了原始采样率且与目标采样率不同，进行重采样
        if original_sr and original_sr != sample_rate:
            audio_array = resample_audio_array(audio_array, original_sr, sample_rate)

        # 调节音频音量
        audio_array = adjust_audio_volume(audio_array, volume)

        # 确保音频数据是float32格式
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # 确保音频数据在正确的范围内
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))

        # 确保是2D张量 (channels, samples)
        if audio_array.ndim == 1:
            audio_array = audio_array[np.newaxis, :]  # 添加通道维度
        elif audio_array.ndim > 2:
            audio_array = audio_array.squeeze()
            if audio_array.ndim == 1:
                audio_array = audio_array[np.newaxis, :]

        # 根据格式选择保存方法
        if format.lower() == "wav":
            # 使用torchaudio保存WAV格式
            audio_tensor = cast(Any, torch).as_tensor(audio_array)
            torchaudio.save(output_path, audio_tensor, sample_rate)
        else:
            # 使用soundfile保存其他格式
            # 确保音频数据是单声道
            if audio_array.shape[0] > 1:
                audio_array = np.mean(audio_array, axis=0)

            sf.write(output_path, audio_array.T, sample_rate, format=format.upper())

        return output_path

    except Exception as e:
        raise DefaultServerErrorException(f"保存音频文件失败: {str(e)}")


def convert_audio_to_wav(
    input_path: str, output_path: Optional[str] = None, target_sr: int = 16000
) -> str:
    """转换音频文件为WAV格式

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（可选）
        target_sr: 目标采样率，默认16000Hz

    Returns:
        转换后的文件路径

    Raises:
        AudioProcessingException: 转换失败
    """
    if not output_path:
        output_path = input_path.rsplit(".", 1)[0] + ".wav"

    try:
        # 使用librosa加载并重采样
        audio_data, _ = librosa.load(input_path, sr=target_sr)
        sf.write(output_path, audio_data, target_sr, format="WAV")
        return output_path

    except Exception as e:
        # 尝试使用ffmpeg转换
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-f", "s16le",
                    "-ar", str(target_sr),
                    "-ac", "1",
                    "-i", input_path,
                    "-acodec", "pcm_s16le",
                    output_path,
                    "-y",
                ],
                check=True,
                capture_output=True,
            )
            return output_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise DefaultServerErrorException(f"音频格式转换失败: {str(e)}")


def normalize_audio_for_asr(audio_path: str, target_sr: int = 16000) -> str:
    """将音频文件标准化为ASR模型所需的格式

    转换后会自动检测容器/解码时长不一致（常见于 m3u8/ts 合并的 m4a/AAC），
    若检测到差异，将缩放系数写入 sidecar 文件 ``{normalized_path}.tsscale``，
    下游 ``transcribe_long_audio`` 会自动读取并修正时间戳。

    Args:
        audio_path: 输入音频文件路径
        target_sr: 目标采样率，默认16000Hz

    Returns:
        标准化后的WAV文件路径
    """
    try:
        # 检查文件扩展名
        file_ext = os.path.splitext(audio_path)[1].lower()

        # 如果已经是WAV格式且采样率正确，直接返回
        if file_ext == ".wav":
            # 检查采样率
            _, sr = librosa.load(audio_path, sr=None)
            if sr == target_sr:
                return audio_path

        # 转换为标准WAV格式
        normalized_path = convert_audio_to_wav(audio_path, target_sr=target_sr)
        logger.debug(f"音频文件已标准化: {audio_path} -> {normalized_path}")

        # Detect container/decoded duration mismatch and write sidecar
        if normalized_path != audio_path:
            decoded_duration = get_audio_duration(normalized_path)
            ts_scale = get_timestamp_scale(audio_path, decoded_duration)
            if ts_scale != 1.0:
                tsscale_path = normalized_path + ".tsscale"
                with open(tsscale_path, "w") as f:
                    f.write(str(ts_scale))

        return normalized_path

    except Exception as e:
        raise DefaultServerErrorException(f"音频标准化失败: {str(e)}")


def generate_temp_audio_path(prefix: str = "audio", suffix: str = ".wav") -> str:
    """生成临时音频文件路径

    Args:
        prefix: 文件名前缀
        suffix: 文件后缀

    Returns:
        临时文件路径
    """
    import time

    timestamp = int(time.time())
    filename = f"{prefix}_{timestamp}_{os.getpid()}{suffix}"
    return os.path.join(settings.TEMP_DIR, filename)


def detect_audio_format_from_bytes(data: bytes) -> str:
    """通过文件头（magic bytes）检测音频格式

    Args:
        data: 音频文件的前几个字节

    Returns:
        文件后缀（包含点号）
    """
    if len(data) < 12:
        return ".wav"

    # 检查常见音频格式的文件头
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    elif data[:3] == b"ID3" or (data[0:2] == b"\xff\xfb") or (data[0:2] == b"\xff\xfa"):
        return ".mp3"
    elif data[:4] == b"fLaC":
        return ".flac"
    elif data[:4] == b"OggS":
        return ".ogg"
    elif data[4:8] == b"ftyp":
        # M4A/AAC/MP4 容器
        return ".m4a"
    elif data[:4] == b"\x1aE\xdf\xa3":
        # WebM/MKV
        return ".webm"

    # 默认为 wav，librosa 会自动处理
    return ".wav"


def get_audio_file_suffix(
    audio_address: Optional[str] = None, audio_data: Optional[bytes] = None
) -> str:
    """自动识别音频文件后缀

    Args:
        audio_address: 音频文件URL（可选）
        audio_data: 音频二进制数据（可选，用于检测文件头）

    Returns:
        文件后缀（包含点号）
    """
    if audio_address:
        # 从URL中提取扩展名
        from urllib.parse import urlparse, unquote

        parsed = urlparse(audio_address)
        path = unquote(parsed.path)

        # 获取扩展名
        ext = os.path.splitext(path)[1].lower()
        if ext and ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".pcm", ".webm"]:
            return ext

        # 无法识别扩展名，默认为 .wav
        return ".wav"

    elif audio_data:
        # 通过文件头检测格式
        return detect_audio_format_from_bytes(audio_data[:12])

    else:
        # 默认为 .wav
        return ".wav"
