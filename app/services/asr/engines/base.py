# -*- coding: utf-8 -*-
"""
ASR引擎基础模块
包含抽象基类和数据类定义
"""

import time
import logging
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.config import settings
from app.core.exceptions import DefaultServerErrorException
from app.core.logging import log_inference_metrics
from app.utils.audio import get_audio_duration


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
        """转录长音频文件（自动分段）

        Args:
            audio_path: 音频文件路径
            hotwords: 热词
            enable_punctuation: 是否启用标点
            enable_itn: 是否启用 ITN
            sample_rate: 采样率
            enable_speaker_diarization: 是否启用说话人分离
            word_timestamps: 是否返回字词级时间戳（仅部分模型支持）
            timestamp_scale: Timestamp correction factor from audio normalization.
            task_id: 任务ID（用于日志追踪）

        Returns:
            ASRFullResult: 包含完整文本、分段结果和时长的结果
        """
        from app.utils.audio_splitter import AudioSplitter

        # 开始性能计时
        start_time = time.time()
        model_id = getattr(self, 'model_id', 'unknown')

        task_prefix = f"[{task_id}] " if task_id else ""

        logger.info(
            f"{task_prefix}[transcribe_long_audio] 音频: {audio_path}, "
            f"speaker_diarization={enable_speaker_diarization}, word_level={word_timestamps}"
        )

        try:
            # 获取音频时长
            duration = get_audio_duration(audio_path)
            logger.info(f"{task_prefix}[transcribe_long_audio] 音频时长: {duration:.2f}秒")

            # 统一使用分段处理
            speaker_segments = None
            audio_segments = None

            if enable_speaker_diarization:
                # 多说话人：使用说话人分离
                from app.utils.speaker_diarizer import SpeakerDiarizer

                logger.info(f"{task_prefix}使用说话人分离模式")
                diarizer = SpeakerDiarizer()
                speaker_segments = diarizer.split_audio_by_speakers(audio_path)

                if not speaker_segments:
                    logger.warning(f"{task_prefix}说话人分离未检测到片段，fallback 到 VAD 分割")

            if not speaker_segments:
                # 单说话人：使用 VAD 分割
                logger.info(f"{task_prefix}使用 VAD 分割模式")
                splitter = AudioSplitter(device=self.device)
                audio_segments = splitter.split_audio_file(audio_path)

            # 选择要处理的片段
            segments_to_process = speaker_segments if speaker_segments else audio_segments
            if not segments_to_process:
                raise DefaultServerErrorException("音频分割失败：未生成任何片段")

            logger.info(f"{task_prefix}音频已分割为 {len(segments_to_process)} 段")

            results: List[ASRSegmentResult] = []
            all_texts: List[str] = []

            # 使用批处理推理
            batch_size = settings.ASR_BATCH_SIZE
            logger.info(
                f"{task_prefix}使用批处理推理，batch_size={batch_size}, "
                f"word_timestamps={word_timestamps}"
            )

            for batch_start in range(0, len(segments_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(segments_to_process))
                batch_segments = segments_to_process[batch_start:batch_end]

                logger.info(
                    f"{task_prefix}推理批次 "
                    f"{batch_start//batch_size + 1}/{(len(segments_to_process) + batch_size - 1)//batch_size}: "
                    f"片段 {batch_start+1}-{batch_end}/{len(segments_to_process)}"
                )

                try:
                    # 批量推理，支持时间戳
                    batch_results = self._transcribe_batch(
                        segments=batch_segments,
                        hotwords=hotwords,
                        enable_punctuation=enable_punctuation,
                        enable_itn=enable_itn,
                        sample_rate=sample_rate,
                        word_timestamps=word_timestamps,
                    )

                    for seg, result in zip(batch_segments, batch_results):
                        if result and result.text:
                            start_sec = float(getattr(seg, "start_sec", 0.0))
                            end_sec = float(getattr(seg, "end_sec", start_sec))
                            results.append(
                                ASRSegmentResult(
                                    text=result.text,
                                    start_time=start_sec,
                                    end_time=end_sec,
                                    speaker_id=getattr(seg, "speaker_id", None),
                                    word_tokens=result.word_tokens if word_timestamps else None,
                                )
                            )
                            all_texts.append(result.text)

                    logger.info(
                        f"{task_prefix}批次推理完成，有效片段: "
                        f"{len([r for r in batch_results if r and r.text])}"
                    )

                except Exception as e:
                    logger.error(f"{task_prefix}批次推理失败: {e}, 跳过该批次")

            # 清理临时文件（独立清理，避免条件遗漏）
            try:
                if speaker_segments:
                    from app.utils.speaker_diarizer import SpeakerDiarizer
                    SpeakerDiarizer.cleanup_segments(speaker_segments)
                if audio_segments:
                    AudioSplitter.cleanup_segments(audio_segments)
            except Exception as e:
                logger.warning(f"清理临时文件时出错: {e}")

            full_text = "\n".join(all_texts)

            logger.info(
                f"长音频识别完成，共 {len(results)} 个有效分段，"
                f"总字符数: {len(full_text)}"
            )

            # 计算性能指标
            total_duration_ms = (time.time() - start_time) * 1000

            if timestamp_scale != 1.0:
                for seg in results:
                    seg.start_time *= timestamp_scale
                    seg.end_time *= timestamp_scale
                    if seg.word_tokens:
                        for word_token in seg.word_tokens:
                            word_token.start_time *= timestamp_scale
                            word_token.end_time *= timestamp_scale
                duration *= timestamp_scale
                logger.info(
                    f"{task_prefix}Timestamp scaling applied: scale={timestamp_scale:.6f}"
                )

            log_inference_metrics(
                logger=logger,
                message="长音频识别完成",
                task_id=task_id,
                duration_ms=total_duration_ms,
                audio_duration_sec=duration,
                model_id=model_id,
                status="success",
                segments_count=len(results),
                batch_size=settings.ASR_BATCH_SIZE,
                enable_speaker_diarization=enable_speaker_diarization,
                word_timestamps=word_timestamps,
            )

            return ASRFullResult(
                text=full_text,
                segments=results,
                duration=duration,
            )

        except Exception as e:
            # 计算失败时的性能指标
            total_duration_ms = (time.time() - start_time) * 1000
            try:
                duration = get_audio_duration(audio_path)
            except Exception:
                duration = 0

            log_inference_metrics(
                logger=logger,
                message="长音频识别失败",
                task_id=task_id,
                duration_ms=total_duration_ms,
                audio_duration_sec=duration,
                model_id=model_id,
                status="error",
                error=str(e),
            )

            logger.error(f"长音频识别失败: {e}")
            raise DefaultServerErrorException(f"长音频识别失败: {str(e)}")

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
    @abstractmethod
    def supports_realtime(self) -> bool:
        """是否支持实时识别"""
        pass

    def _transcribe_batch(
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


class RealTimeASREngine(BaseASREngine):
    """实时ASR引擎抽象基类"""

    @property
    def supports_realtime(self) -> bool:
        """支持实时识别"""
        return True

    @abstractmethod
    def transcribe_websocket(
        self,
        audio_chunk: bytes,
        cache: Optional[Dict] = None,
        is_final: bool = False,
        **kwargs,
    ) -> str:
        """WebSocket流式语音识别"""
        pass
