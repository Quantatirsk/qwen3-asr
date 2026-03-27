# -*- coding: utf-8 -*-
"""
音频分割模块
基于 VAD 的智能音频分割，支持长音频分段识别
"""

import logging
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..core.config import settings
from ..core.exceptions import DefaultServerErrorException

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """音频片段信息"""

    start_ms: int  # 开始时间（毫秒）
    end_ms: int  # 结束时间（毫秒）
    audio_data: Optional[np.ndarray] = None  # 音频数据
    temp_file: Optional[str] = None  # 临时文件路径
    speaker_id: Optional[str] = None  # 说话人ID（多说话人模式）

    @property
    def start_sec(self) -> float:
        """开始时间（秒）"""
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        """结束时间（秒）"""
        return self.end_ms / 1000.0

    @property
    def duration_ms(self) -> int:
        """时长（毫秒）"""
        return self.end_ms - self.start_ms

    @property
    def duration_sec(self) -> float:
        """时长（秒）"""
        return self.duration_ms / 1000.0


class AudioSplitter:
    """音频分割器

    使用 VAD 模型检测语音边界，智能分割长音频
    """

    # 默认配置
    DEFAULT_MIN_SEGMENT_SEC = 1.0  # 每段最小时长（秒）
    DEFAULT_SAMPLE_RATE = 16000  # 默认采样率

    def __init__(
        self,
        min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
        device: str = "auto",
    ):
        """初始化音频分割器

        Args:
            min_segment_sec: 每段最小时长（秒）
            device: 计算设备（"cuda", "cpu", "auto"）
        """
        split_trigger_sec = settings.MAX_SEGMENT_SEC

        self.split_trigger_sec = split_trigger_sec
        self.min_segment_sec = min_segment_sec
        self.split_trigger_ms = int(split_trigger_sec * 1000)
        self.min_segment_ms = int(min_segment_sec * 1000)
        self.device = device

    def get_vad_segments(
        self, audio_path: str
    ) -> List[Tuple[int, int]]:
        """使用 VAD 模型获取语音段

        Args:
            audio_path: 音频文件路径

        Returns:
            语音段列表，每个元素为 (start_ms, end_ms)
        """
        try:
            from ..services.asr.engines import get_global_vad_model

            logger.info("开始 VAD 语音段检测...")
            vad_model = get_global_vad_model(self.device)
            if vad_model is None:
                raise DefaultServerErrorException("VAD 模型未加载")

            # 调用 VAD 模型
            result = vad_model.generate(input=audio_path, cache={})

            if not result or len(result) == 0:
                logger.warning("VAD 未检测到语音段")
                return []

            # 解析 VAD 结果
            # FunASR VAD 返回格式: [[start_ms, end_ms], [start_ms, end_ms], ...]
            vad_segments = result[0].get("value", [])

            if not vad_segments:
                logger.warning("VAD 结果为空")
                return []

            logger.info(f"VAD 检测到 {len(vad_segments)} 个语音段")
            logger.info(
                "开始按 VAD 边界重分段 "
                f"(split_trigger={self.split_trigger_sec}s, min_segment={self.min_segment_sec}s)..."
            )
            return [(int(seg[0]), int(seg[1])) for seg in vad_segments]

        except Exception as e:
            logger.error(f"VAD 检测失败: {e}")
            raise DefaultServerErrorException(f"VAD 检测失败: {str(e)}")

    def merge_segments_greedy(
        self, vad_segments: List[Tuple[int, int]], total_duration_ms: int
    ) -> List[Tuple[int, int]]:
        """按 VAD 结果重分段

        策略：
        1. 默认保留 VAD 原始边界，避免将整段连续语音合并成超长片段
        2. 仅对短片段（< min_segment_ms）做邻段合并
        3. 对重叠片段进行边界修正，避免重复音频

        Args:
            vad_segments: VAD 检测到的语音段列表 [(start_ms, end_ms), ...]
            total_duration_ms: 音频总时长（毫秒）

        Returns:
            合并后的段列表 [(start_ms, end_ms), ...]
        """
        if not vad_segments:
            # 没有 VAD 段，返回整个音频（按最大时长切分）
            return self._split_by_fixed_duration(total_duration_ms)

        # 按时间排序并修正边界（防止越界、重叠）
        sorted_vad = sorted(vad_segments, key=lambda x: x[0])
        normalized: List[Tuple[int, int]] = []
        for raw_start, raw_end in sorted_vad:
            start_ms = max(0, int(raw_start))
            end_ms = min(total_duration_ms, int(raw_end))
            if end_ms <= start_ms:
                continue

            if not normalized:
                normalized.append((start_ms, end_ms))
                continue

            last_end = normalized[-1][1]
            # 有重叠时，优先保持边界，避免与上一段重复采样
            if start_ms < last_end:
                start_ms = last_end

            if end_ms > start_ms:
                normalized.append((start_ms, end_ms))

        if not normalized:
            return self._split_by_fixed_duration(total_duration_ms)

        merged = list(normalized)

        # 只处理短片段：与相邻片段合并（不基于静音间隙）
        idx = 0
        while idx < len(merged):
            start_ms, end_ms = merged[idx]
            duration = end_ms - start_ms

            if duration >= self.min_segment_ms or len(merged) == 1:
                idx += 1
                continue

            if idx == 0:
                # 首段过短：并入后段
                next_end = merged[idx + 1][1]
                merged[idx + 1] = (start_ms, next_end)
                del merged[idx]
                continue

            if idx == len(merged) - 1:
                # 尾段过短：并入前段
                prev_start, _ = merged[idx - 1]
                merged[idx - 1] = (prev_start, end_ms)
                del merged[idx]
                idx = max(0, idx - 1)
                continue

            # 中间短段：优先并入时长更短的一侧，避免单段过长
            prev_start = merged[idx - 1][0]
            next_end = merged[idx + 1][1]
            merged_with_prev_duration = end_ms - prev_start
            merged_with_next_duration = next_end - start_ms

            if merged_with_prev_duration <= merged_with_next_duration:
                merged[idx - 1] = (prev_start, end_ms)
                del merged[idx]
                idx = max(0, idx - 1)
            else:
                merged[idx + 1] = (start_ms, next_end)
                del merged[idx]

        return merged

    def _split_by_fixed_duration(self, total_duration_ms: int) -> List[Tuple[int, int]]:
        """按固定时长切分（无 VAD 时的 fallback）

        Args:
            total_duration_ms: 音频总时长（毫秒）

        Returns:
            切分后的段列表
        """
        segments = []
        current = 0
        while current < total_duration_ms:
            end = min(current + self.split_trigger_ms, total_duration_ms)
            if end - current >= self.min_segment_ms:
                segments.append((current, end))
            current = end
        return segments

    def split_audio_file(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
    ) -> List[AudioSegment]:
        """分割音频文件

        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录（可选，默认使用临时目录）

        Returns:
            音频片段列表
        """
        try:
            # 加载音频
            audio_data, sr = librosa.load(audio_path, sr=self.DEFAULT_SAMPLE_RATE)
            total_duration_ms = int(len(audio_data) / sr * 1000)

            logger.info(f"音频总时长: {total_duration_ms / 1000:.2f}秒")

            # 检查是否需要分割
            if total_duration_ms <= self.split_trigger_ms:
                logger.info("音频时长在限制内，无需分割")
                return [
                    AudioSegment(
                        start_ms=0,
                        end_ms=total_duration_ms,
                        audio_data=audio_data,
                        temp_file=audio_path,
                    )
                ]

            # 获取 VAD 段
            vad_segments = self.get_vad_segments(audio_path)

            # 贪婪合并
            merged_segments = self.merge_segments_greedy(vad_segments, total_duration_ms)
            logger.info(f"重分段完成: 原始VAD={len(vad_segments)}, 输出={len(merged_segments)}")

            # 切分音频并保存到临时文件
            logger.info("开始切分音频并保存临时文件...")
            output_dir = output_dir or settings.TEMP_DIR
            os.makedirs(output_dir, exist_ok=True)

            audio_segments = []
            for idx, (start_ms, end_ms) in enumerate(merged_segments):
                # 计算采样点范围
                start_sample = int(start_ms / 1000 * sr)
                end_sample = int(end_ms / 1000 * sr)

                # 提取音频片段
                segment_data = audio_data[start_sample:end_sample]

                # 保存到临时文件
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav",
                    dir=output_dir,
                    prefix=f"segment_{idx:03d}_",
                )
                temp_path = temp_file.name
                temp_file.close()

                sf.write(temp_path, segment_data, sr)

                segment = AudioSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    audio_data=segment_data,
                    temp_file=temp_path,
                )
                audio_segments.append(segment)

                logger.debug(
                    f"分段 {idx + 1}/{len(merged_segments)}: "
                    f"{start_ms / 1000:.2f}s - {end_ms / 1000:.2f}s "
                    f"(时长: {segment.duration_sec:.2f}s)"
                )

            logger.info(f"音频切分完成，共 {len(audio_segments)} 个分段")
            return audio_segments

        except Exception as e:
            logger.error(f"音频分割失败: {e}")
            raise DefaultServerErrorException(f"音频分割失败: {str(e)}")

    @staticmethod
    def cleanup_segments(segments: List[AudioSegment]) -> None:
        """清理临时文件

        Args:
            segments: 音频片段列表
        """
        for segment in segments:
            if segment.temp_file and os.path.exists(segment.temp_file):
                try:
                    os.remove(segment.temp_file)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {segment.temp_file}, {e}")


def split_long_audio(
    audio_path: str,
    device: str = "auto",
) -> List[AudioSegment]:
    """分割长音频的便捷函数

    Args:
        audio_path: 音频文件路径
        device: 计算设备

    Returns:
        音频片段列表
    """
    splitter = AudioSplitter(device=device)
    return splitter.split_audio_file(audio_path)
