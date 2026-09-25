# -*- coding: utf-8 -*-
"""
ASR引擎基础模块
包含抽象基类和数据类定义
"""

import logging
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.config import settings


logger = logging.getLogger(__name__)


@dataclass
class WordToken:
    """字词级时间戳信息"""

    text: str  # 字词文本
    start_time: float  # 开始时间（秒）
    end_time: float  # 结束时间（秒）


@dataclass
class ASRSegmentResult:
    """ASR 分段识别结果"""

    text: str  # 该段识别文本
    start_time: float  # 开始时间（秒）
    end_time: float  # 结束时间（秒）
    speaker_id: Optional[str] = None  # 说话人ID（多说话人模式）
    word_tokens: Optional[List[WordToken]] = None  # 字词级时间戳（可选）


@dataclass
class ASRFullResult:
    """ASR 完整识别结果（支持长音频）"""

    text: str  # 完整识别文本
    segments: List[ASRSegmentResult]  # 分段结果
    duration: float  # 音频总时长（秒）


@dataclass
class ASRRawResult:
    """ASR 原始识别结果（包含时间戳）"""

    text: str  # 完整识别文本
    segments: List[ASRSegmentResult]  # 分段结果（从 VAD 时间戳解析）


class BaseASREngine(ABC):
    """基础ASR引擎抽象基类"""

    @abstractmethod
    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        enable_vad: bool = False,
        sample_rate: int = 16000,
    ) -> str:
        """转录音频文件"""
        pass

    @abstractmethod
    def transcribe_file_with_vad(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
        **kwargs,
    ) -> ASRRawResult:
        """使用 VAD 转录音频文件，返回带时间戳分段的结果

        Args:
            audio_path: 音频文件路径
            hotwords: 热词/上下文提示
            enable_punctuation: 是否启用标点
            enable_itn: 是否启用 ITN
            sample_rate: 采样率
            **kwargs: 额外参数（如 word_timestamps 字词级时间戳）

        Returns:
            ASRRawResult 包含文本和分段信息
        """
        pass

    def transcribe_long_audio(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        sample_rate: int = 16000,
        enable_speaker_diarization: bool = True,
        word_timestamps: bool = False,
        timestamp_scale: float = 1.0,
        task_id: Optional[str] = None,
    ) -> ASRFullResult:
        """Transcribe prepared batches while owning every generated segment file."""
        from app.services.asr.long_audio import prepare_long_audio

        with prepare_long_audio(
            audio_path,
            self.device,
            enable_speaker_diarization,
            getattr(self, "model_id", "unknown"),
            task_id,
        ) as audio:
            results = []
            for start in range(0, len(audio.segments), settings.ASR_BATCH_SIZE):
                results.extend(
                    self.transcribe_segments(
                        segments=list(
                            audio.segments[start : start + settings.ASR_BATCH_SIZE]
                        ),
                        hotwords=hotwords,
                        enable_punctuation=enable_punctuation,
                        enable_itn=enable_itn,
                        sample_rate=sample_rate,
                        word_timestamps=word_timestamps,
                    )
                )
            return audio.finish(results, timestamp_scale)

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """获取设备信息"""
        pass

    @property
    def supports_realtime(self) -> bool:
        """Local engines are offline only; realtime uses the remote service."""
        return False

    def transcribe_segments(
        self,
        segments: List[Any],
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        sample_rate: int = 16000,
        word_timestamps: bool = False,
    ) -> List[ASRSegmentResult]:
        """批量推理多个音频片段

        Args:
            segments: 音频片段列表（每个片段需要有 temp_file 属性）
            hotwords: 热词
            enable_punctuation: 是否启用标点
            enable_itn: 是否启用 ITN
            sample_rate: 采样率
            word_timestamps: 是否返回字词级时间戳

        Returns:
            ASRSegmentResult 列表，与输入片段一一对应
        """
        # 默认实现：逐个推理（子类可以重写实现真正的批处理）
        results = []
        for idx, seg in enumerate(segments):
            try:
                if not seg.temp_file:
                    logger.warning(f"批处理片段 {idx + 1} 临时文件不存在，跳过")
                    results.append(ASRSegmentResult(text="", start_time=0.0, end_time=0.0))
                    continue

                if word_timestamps:
                    # 需要时间戳：使用 transcribe_file_with_vad
                    raw_result = self.transcribe_file_with_vad(
                        audio_path=seg.temp_file,
                        hotwords=hotwords,
                        enable_punctuation=enable_punctuation,
                        enable_itn=enable_itn,
                        sample_rate=sample_rate,
                        word_timestamps=True,
                    )
                    if raw_result.segments:
                        result_seg = raw_result.segments[0]
                        results.append(
                            ASRSegmentResult(
                                text=result_seg.text,
                                start_time=seg.start_sec,
                                end_time=seg.end_sec,
                                speaker_id=getattr(seg, 'speaker_id', None),
                                word_tokens=result_seg.word_tokens,
                            )
                        )
                    else:
                        results.append(
                            ASRSegmentResult(
                                text=raw_result.text,
                                start_time=seg.start_sec,
                                end_time=seg.end_sec,
                                speaker_id=getattr(seg, 'speaker_id', None),
                            )
                        )
                else:
                    # 不需要时间戳：使用 transcribe_file
                    text = self.transcribe_file(
                        audio_path=seg.temp_file,
                        hotwords=hotwords,
                        enable_punctuation=enable_punctuation,
                        enable_itn=enable_itn,
                        enable_vad=False,
                        sample_rate=sample_rate,
                    )
                    results.append(
                        ASRSegmentResult(
                            text=text or "",
                            start_time=seg.start_sec,
                            end_time=seg.end_sec,
                            speaker_id=getattr(seg, 'speaker_id', None),
                        )
                    )
            except Exception as e:
                logger.error(f"批处理片段 {idx + 1} 推理失败: {e}")
                results.append(
                    ASRSegmentResult(
                        text="",
                        start_time=getattr(seg, 'start_sec', 0.0),
                        end_time=getattr(seg, 'end_sec', 0.0),
                        speaker_id=getattr(seg, 'speaker_id', None),
                    )
                )

        return results

    @staticmethod
    def _detect_device(device: str = "auto") -> str:
        """检测可用设备"""
        from app.core.device import detect_device

        return detect_device(device)
