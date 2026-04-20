# -*- coding: utf-8 -*-
"""
音频处理服务

封装音频处理逻辑，提供统一的音频下载、格式转换、归一化等服务。
API层应该通过此服务层处理音频，而不是直接调用 utils/audio.py 中的函数。
"""

import logging
import threading
from typing import Optional, Tuple
from fastapi import Request

from ...core.config import settings
from ...core.exceptions import InvalidMessageException
from ...utils.audio import (
    download_audio_from_url,
    save_audio_to_temp_file,
    normalize_audio_for_asr,
    get_audio_duration,
    cleanup_temp_file,
    get_audio_file_suffix,
)

logger = logging.getLogger(__name__)


class AudioProcessingService:
    """音频处理服务

    提供统一的音频处理接口，包括：
    1. 从URL下载音频
    2. 处理上传的音频文件
    3. 音频格式转换和归一化
    4. 临时文件管理
    """

    async def process_from_request(
        self,
        request: Request,
        audio_address: Optional[str] = None,
        task_id: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ) -> Tuple[str, float, str]:
        """从请求中处理音频

        支持两种方式：
        1. 请求体上传：从请求体读取二进制音频/视频数据
        2. URL下载：通过 audio_address 参数指定音频/视频 URL

        当请求体和 audio_address 同时存在时，优先使用请求体，
        并忽略 audio_address。

        Args:
            request: FastAPI请求对象
            audio_address: 音频文件URL（可选）
            task_id: 任务ID，用于日志记录（可选）
            sample_rate: 目标采样率（可选，默认16000）

        Returns:
            Tuple[str, float, str]: (normalized_audio_path, duration_seconds, original_audio_path)

        Raises:
            InvalidMessageException: 音频数据为空或文件太大
            InvalidParameterException: URL无效或下载失败
        """
        task_id = task_id or "unknown"
        target_sr = sample_rate or 16000
        audio_path = None
        normalized_audio_path = None

        try:
            # 优先读取请求体；若请求体为空，再回退到 audio_address。
            # 注意：对于 FastAPI 已经解析过 form/multipart 的请求，
            # 再次读取 body 可能抛出 "Stream consumed"。
            try:
                uploaded_data = await request.body()
            except RuntimeError as exc:
                if "Stream consumed" in str(exc):
                    logger.info(f"[{task_id}] 请求体已被上游读取，跳过 request.body() 回退逻辑")
                    uploaded_data = b""
                else:
                    raise

            if uploaded_data:
                if audio_address:
                    logger.info(f"[{task_id}] 检测到同时提供上传内容和 audio_address，已忽略 audio_address")

                logger.info(f"[{task_id}] 开始接收上传音频...")

                if len(uploaded_data) > settings.MAX_AUDIO_SIZE:
                    max_size_mb = settings.MAX_AUDIO_SIZE // 1024 // 1024
                    raise InvalidMessageException(
                        f"音频文件太大，最大支持{max_size_mb}MB", task_id
                    )

                logger.info(
                    f"[{task_id}] 音频接收完成，大小: {len(uploaded_data) / 1024 / 1024:.2f}MB"
                )

                file_suffix = get_audio_file_suffix(audio_data=uploaded_data)
                logger.info(f"[{task_id}] 识别文件格式: {file_suffix}")
                audio_path = save_audio_to_temp_file(uploaded_data, file_suffix)

            elif audio_address:
                # 方式2: 从URL下载音频
                logger.info(f"[{task_id}] 开始从URL下载音频: {audio_address}")
                audio_data = download_audio_from_url(audio_address)
                logger.info(
                    f"[{task_id}] 音频下载完成，大小: {len(audio_data) / 1024 / 1024:.2f}MB"
                )

                # 自动从URL识别文件格式
                file_suffix = get_audio_file_suffix(audio_address)
                logger.info(f"[{task_id}] 识别文件格式: {file_suffix}")
                audio_path = save_audio_to_temp_file(audio_data, file_suffix)

            else:
                raise InvalidMessageException("音频数据为空", task_id)

            logger.info(f"[{task_id}] 临时文件已保存: {audio_path}")

            # 将音频标准化为ASR模型所需的格式
            logger.info(f"[{task_id}] 开始音频格式转换...")
            normalized_audio_path = normalize_audio_for_asr(audio_path, target_sr)
            logger.info(f"[{task_id}] 音频格式转换完成: {normalized_audio_path}")

            # 获取音频时长
            audio_duration = get_audio_duration(normalized_audio_path)
            logger.info(f"[{task_id}] 音频时长: {audio_duration:.1f}秒")

            return normalized_audio_path, audio_duration, audio_path

        except Exception:
            # 清理已创建的临时文件
            if audio_path:
                cleanup_temp_file(audio_path)
            if normalized_audio_path and normalized_audio_path != audio_path:
                cleanup_temp_file(normalized_audio_path)
            raise

    async def process_upload_file(
        self,
        audio_data: bytes,
        filename: Optional[str] = None,
        task_id: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ) -> Tuple[str, float, str]:
        """处理上传的音频文件

        Args:
            audio_data: 音频二进制数据
            filename: 原始文件名（用于检测格式，可选）
            task_id: 任务ID，用于日志记录（可选）
            sample_rate: 目标采样率（可选，默认16000）

        Returns:
            Tuple[str, float, str]: (normalized_audio_path, duration_seconds, original_audio_path)

        Raises:
            InvalidMessageException: 音频数据为空或文件太大
        """
        task_id = task_id or "unknown"
        target_sr = sample_rate or 16000
        audio_path = None
        normalized_audio_path = None

        try:
            if not audio_data:
                raise InvalidMessageException("音频数据为空", task_id)

            file_size = len(audio_data)
            logger.info(f"[{task_id}] 音频文件大小: {file_size / 1024 / 1024:.2f}MB")

            # 检查文件大小
            if file_size > settings.MAX_AUDIO_SIZE:
                max_mb = settings.MAX_AUDIO_SIZE // 1024 // 1024
                raise InvalidMessageException(
                    f"音频文件太大，最大支持{max_mb}MB", task_id
                )

            # 检测音频格式并保存临时文件
            file_suffix = get_audio_file_suffix(
                audio_address=filename, audio_data=audio_data
            )
            audio_path = save_audio_to_temp_file(audio_data, file_suffix)
            logger.info(f"[{task_id}] 临时文件: {audio_path}")

            # 标准化音频格式
            normalized_audio_path = normalize_audio_for_asr(audio_path, target_sr)

            # 获取音频时长
            audio_duration = get_audio_duration(normalized_audio_path)
            logger.info(f"[{task_id}] 音频时长: {audio_duration:.1f}s")

            return normalized_audio_path, audio_duration, audio_path

        except Exception:
            # 清理已创建的临时文件
            if audio_path:
                cleanup_temp_file(audio_path)
            if normalized_audio_path and normalized_audio_path != audio_path:
                cleanup_temp_file(normalized_audio_path)
            raise

    def cleanup(
        self, audio_path: Optional[str], normalized_path: Optional[str] = None
    ) -> None:
        """清理临时文件

        Args:
            audio_path: 原始音频文件路径
            normalized_path: 归一化后的音频文件路径（可选）
        """
        if audio_path:
            cleanup_temp_file(audio_path)
        if normalized_path and normalized_path != audio_path:
            cleanup_temp_file(normalized_path)


# 全局服务实例（单例模式）
_audio_service: Optional[AudioProcessingService] = None
_audio_service_lock = threading.Lock()


def get_audio_service() -> AudioProcessingService:
    """获取音频处理服务实例（线程安全的单例）

    Returns:
        AudioProcessingService: 音频处理服务实例
    """
    global _audio_service
    if _audio_service is None:
        with _audio_service_lock:
            if _audio_service is None:
                _audio_service = AudioProcessingService()
    return _audio_service
