# -*- coding: utf-8 -*-
"""
说话人分离模块
基于 CAM++ 的说话人分离，用于多说话人音频分割
"""

from loguru import logger
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import threading
from typing import Any, List, Mapping, Optional, Sequence, cast
from dataclasses import dataclass

import torch

from ..core.config import settings
from ..core.exceptions import DefaultServerErrorException

# 全局 CAM++ pipeline 缓存（单例）
_global_diarization_pipeline: Any | None = None
_diarization_pipeline_lock = threading.Lock()
_diarization_inference_semaphore = threading.BoundedSemaphore(1)


@dataclass
class SpeakerSegment:
    """说话人分段信息"""

    start_ms: int
    end_ms: int
    speaker_id: str
    audio_data: Optional[np.ndarray] = None
    temp_file: Optional[str] = None

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0


def _resolve_modelscope_device() -> str:
    """根据配置和硬件自动选择 modelscope pipeline 设备
    """
    from ..core.device import detect_device

    return detect_device(settings.DEVICE)


def _move_pipeline_model_to_device(pipeline_instance: Any, modelscope_device: str) -> None:
    """将 pipeline 的底层模型迁移到目标设备。"""
    if hasattr(pipeline_instance, "device_name"):
        pipeline_instance.device_name = modelscope_device
    model = getattr(pipeline_instance, "model", None)
    if model is not None and hasattr(model, "to"):
        pipeline_instance.model = model.to(modelscope_device)


def _create_modelscope_pipeline(
    *,
    task: Any,
    model: str,
    modelscope_device: str,
    model_revision: Optional[str] = None,
) -> Any:
    """创建 modelscope pipeline，并在需要时把底层模型迁移到目标设备。"""
    from modelscope.pipelines import pipeline

    pipeline_kwargs: dict[str, Any] = {
        "task": task,
        "model": model,
        "device": modelscope_device,
    }
    if model_revision is not None:
        pipeline_kwargs["model_revision"] = model_revision

    pipeline_instance = pipeline(**pipeline_kwargs)
    _move_pipeline_model_to_device(pipeline_instance, modelscope_device)
    return pipeline_instance


def _enable_batched_sv(
    pipeline_instance: Any,
    modelscope_device: str,
    max_batch_size: int = 16,
) -> Any:
    """
    对说话人分离 pipeline 启用 batched SV 推理。

    原始 pipeline 的 forward 方法逐个 segment 调用 sv_pipeline 提取 embedding，
    这里改为将所有 segment 拼成一个 batch 一次性推理，大幅减少 GPU 调用次数。
    同时将子 pipeline（sv / vad / change_locator）绑定到指定 device。

    Args:
        pipeline_instance: CAM++ diarization pipeline 实例
        modelscope_device: 设备名称
        max_batch_size: 最大批处理大小，防止 OOM
    """
    if getattr(pipeline_instance, "_batched_sv_enabled", False):
        return pipeline_instance

    from modelscope.utils.constant import Tasks

    config = getattr(pipeline_instance, "config", None)
    if not isinstance(config, Mapping):
        logger.warning("CAM++ pipeline 缺少可读取的 config，跳过 batched SV 优化")
        return pipeline_instance

    sv_model = config.get("speaker_model")
    vad_model = config.get("vad_model")
    change_locator = config.get("change_locator")

    if isinstance(sv_model, str) and sv_model:
        pipeline_instance.sv_pipeline = _create_modelscope_pipeline(
            task=Tasks.speaker_verification,
            model=sv_model,
            modelscope_device=modelscope_device,
        )

    if isinstance(vad_model, str) and vad_model:
        pipeline_instance.vad_pipeline = _create_modelscope_pipeline(
            task=Tasks.voice_activity_detection,
            model=vad_model,
            modelscope_device=modelscope_device,
            model_revision="v2.0.2",
        )

    if isinstance(change_locator, str) and change_locator:
        pipeline_instance.change_locator_pipeline = _create_modelscope_pipeline(
            task=Tasks.speaker_diarization,
            model=change_locator,
            modelscope_device=modelscope_device,
        )

    def batched_forward(self: Any, segments: Sequence[Sequence[Any]]) -> np.ndarray:
        """批量提取说话人 embedding，替代逐段串行推理"""
        sv_model_instance = getattr(getattr(self, "sv_pipeline", None), "model", None)
        emb_size = int(getattr(sv_model_instance, "emb_size", 192))

        if not segments:
            return np.empty((0, emb_size), dtype=np.float32)

        if sv_model_instance is None:
            raise RuntimeError("CAM++ sv_pipeline.model 未初始化")

        all_embeddings: list[np.ndarray] = []
        total_segments = len(segments)
        start_idx = 0

        while start_idx < total_segments:
            end_idx = min(start_idx + max_batch_size, total_segments)
            batch_segments = segments[start_idx:end_idx]

            batch_items: list[np.ndarray] = []
            for segment in batch_segments:
                if len(segment) < 3:
                    continue
                batch_items.append(np.asarray(segment[2], dtype=np.float32))

            if not batch_items:
                start_idx = end_idx
                continue

            batch = np.stack(batch_items, axis=0)

            with torch.no_grad():
                embeddings = sv_model_instance(
                    cast(Any, torch).as_tensor(batch).to(modelscope_device)
                )

            if isinstance(embeddings, torch.Tensor):
                all_embeddings.append(embeddings.detach().cpu().numpy())
            else:
                all_embeddings.append(np.asarray(embeddings, dtype=np.float32))

            start_idx = end_idx

        if not all_embeddings:
            return np.empty((0, emb_size), dtype=np.float32)

        return (
            np.concatenate(all_embeddings, axis=0)
            if len(all_embeddings) > 1
            else all_embeddings[0]
        )

    import types

    pipeline_instance.forward = types.MethodType(batched_forward, pipeline_instance)
    pipeline_instance._batched_sv_enabled = True

    logger.info(
        "CAM++ 说话人分离启用 batched SV: device={}, sv_device={}, vad_device={}",
        modelscope_device,
        getattr(getattr(pipeline_instance, "sv_pipeline", None), "device_name", "unknown"),
        getattr(getattr(pipeline_instance, "vad_pipeline", None), "device_name", "unknown"),
    )
    return pipeline_instance


def get_global_diarization_pipeline() -> Any:
    """获取全局说话人分离 pipeline（懒加载单例）"""
    global _global_diarization_pipeline

    with _diarization_pipeline_lock:
        if _global_diarization_pipeline is None:
            try:
                from modelscope.utils.constant import Tasks
                from ..infrastructure.model_utils import resolve_model_path

                model_id = 'iic/speech_campplus_speaker-diarization_common'
                model_path = resolve_model_path(model_id)
                modelscope_device = _resolve_modelscope_device()

                logger.info(
                    "正在加载 CAM++ 说话人分离模型: {}, device={}",
                    model_path,
                    modelscope_device,
                )
                _global_diarization_pipeline = _create_modelscope_pipeline(
                    task=Tasks.speaker_diarization,
                    model=model_path,
                    modelscope_device=modelscope_device,
                )
                _global_diarization_pipeline = _enable_batched_sv(
                    _global_diarization_pipeline, modelscope_device
                )
                logger.info("CAM++ 模型加载成功（已启用 batched SV）")
            except Exception as e:
                logger.error(f"CAM++ 模型加载失败: {e}")
                raise DefaultServerErrorException(f"说话人分离模型加载失败: {str(e)}")

    return _global_diarization_pipeline


class SpeakerDiarizer:
    """基于 CAM++ 的说话人分离器"""

    DEFAULT_MIN_SEGMENT_SEC = 1.0
    DEFAULT_SAMPLE_RATE = 16000

    def __init__(
        self,
        min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
    ):
        self.min_segment_sec = min_segment_sec
        self.min_segment_ms = int(min_segment_sec * 1000)

    def diarize(
        self, audio_path: str
    ) -> List[SpeakerSegment]:
        """执行说话人分离

        Args:
            audio_path: 音频文件路径

        Returns:
            原始分段列表（未合并）
        """
        try:
            pipeline = get_global_diarization_pipeline()

            logger.info(f"开始说话人分离: {audio_path}")
            with _diarization_inference_semaphore:
                result = pipeline(audio_path)

            # 解析结果: {'text': [[start, end, speaker_id], ...]}
            # pipeline 返回类型不确定，需要安全地获取 'text' 字段
            if isinstance(result, dict):
                raw_output = result.get('text', [])
            else:
                raw_output = getattr(result, 'text', []) or []

            segments = []
            for seg in raw_output:
                if isinstance(seg, list) and len(seg) == 3:
                    try:
                        start_ms = int(float(seg[0]) * 1000)
                        end_ms = int(float(seg[1]) * 1000)
                        speaker_id = f"说话人{int(seg[2]) + 1}"
                        segments.append(SpeakerSegment(
                            start_ms=start_ms,
                            end_ms=end_ms,
                            speaker_id=speaker_id,
                        ))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"跳过格式错误的片段: {seg}, 错误: {e}")

            logger.info(f"说话人分离完成，原始片段数: {len(segments)}")
            # 诊断日志：打印前20个原始片段
            for i, seg in enumerate(segments[:20]):
                logger.debug(
                    f"[CAM++原始] #{i}: {seg.start_sec:.2f}-{seg.end_sec:.2f}s "
                    f"({seg.duration_sec:.2f}s) {seg.speaker_id}"
                )
            return segments

        except Exception as e:
            error_msg = str(e).lower()

            # 音频太短时，返回默认的单说话人片段
            if "too short" in error_msg:
                logger.warning(f"音频时长过短，CAM++ 无法处理，返回单说话人片段: {e}")

                # 获取音频时长
                try:
                    audio_duration_ms = int(librosa.get_duration(path=audio_path) * 1000)
                except Exception:
                    # 无法获取时长时使用默认值
                    audio_duration_ms = 5000

                return [
                    SpeakerSegment(
                        start_ms=0,
                        end_ms=audio_duration_ms,
                        speaker_id="说话人1",
                    )
                ]

            # 其他异常正常抛出
            logger.error(f"说话人分离失败: {e}")
            raise DefaultServerErrorException(f"说话人分离失败: {str(e)}")

    def merge_consecutive_segments(
        self, segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        """合并同一说话人的连续片段"""
        if not segments:
            return []

        # 按开始时间排序
        sorted_segments = sorted(segments, key=lambda x: x.start_ms)

        merged = []
        current = SpeakerSegment(
            start_ms=sorted_segments[0].start_ms,
            end_ms=sorted_segments[0].end_ms,
            speaker_id=sorted_segments[0].speaker_id,
        )

        for seg in sorted_segments[1:]:
            if seg.speaker_id == current.speaker_id:
                # 同一说话人，扩展结束时间
                current.end_ms = max(current.end_ms, seg.end_ms)
            else:
                # 不同说话人，保存当前段，开始新段
                logger.debug(
                    f"[合并中断] 说话人切换: {current.speaker_id} → {seg.speaker_id} "
                    f"在 {seg.start_sec:.2f}s，保存片段 {current.start_sec:.2f}-{current.end_sec:.2f}s"
                )
                merged.append(current)
                current = SpeakerSegment(
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    speaker_id=seg.speaker_id,
                )

        # 保存最后一段
        merged.append(current)

        logger.info(f"合并同一说话人连续片段: {len(segments)} → {len(merged)}")
        # 诊断日志：打印合并后的前20个片段
        for i, seg in enumerate(merged[:20]):
            logger.debug(
                f"[合并后] #{i}: {seg.start_sec:.2f}-{seg.end_sec:.2f}s "
                f"({seg.duration_sec:.2f}s) {seg.speaker_id}"
            )
        return merged

    def merge_short_segments(
        self, segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        """智能合并短片段

        策略：
        1. 第一层：<10s的片段向后合并（避免孤立短片段）
        2. 第二层：60s累积合并（合并连续片段）
        """
        if not segments:
            return []

        # 按开始时间排序
        sorted_segments = sorted(segments, key=lambda x: x.start_ms)

        # 第一层：<10s累积向后合并（循环计算直到>=10s或超过60s）
        merged = []
        i = 0
        while i < len(sorted_segments):
            seg = sorted_segments[i]

            # 如果>=10s，直接添加
            if seg.duration_sec >= 10.0:
                merged.append(seg)
                i += 1
                continue

            # <10s，开始累积合并
            current_start_ms = seg.start_ms
            current_end_ms = seg.end_ms
            current_duration_sec = seg.duration_sec
            j = i + 1

            # 累积合并，只要<10s且同说话人且不超过60s
            while j < len(sorted_segments) and current_duration_sec < 10.0:
                next_seg = sorted_segments[j]
                if next_seg.speaker_id != seg.speaker_id:
                    break
                new_duration = (next_seg.end_ms - current_start_ms) / 1000.0
                if new_duration > 60.0:  # 不能超过60s上限
                    break
                current_end_ms = next_seg.end_ms
                current_duration_sec = new_duration
                j += 1

            # 创建合并后的片段
            merged_seg = SpeakerSegment(
                start_ms=current_start_ms,
                end_ms=current_end_ms,
                speaker_id=seg.speaker_id,
            )
            merged.append(merged_seg)

            if j > i + 1:
                logger.debug(
                    f"[第一层] {seg.speaker_id}: "
                    f"累积合并了 {j - i} 个片段，结果 {merged_seg.duration_sec:.1f}s"
                )
            i = j

        # 第二层：60s累积合并
        final_merged = []
        i = 0
        while i < len(merged):
            seg = merged[i]
            current_start_ms = seg.start_ms
            current_end_ms = seg.end_ms
            j = i + 1

            # 累积合并，只要 <= 60s 且同说话人
            while j < len(merged):
                next_seg = merged[j]
                if next_seg.speaker_id != seg.speaker_id:
                    break
                new_duration = (next_seg.end_ms - current_start_ms) / 1000.0
                if new_duration > 60.0:
                    break
                current_end_ms = next_seg.end_ms
                j += 1

            merged_seg = SpeakerSegment(
                start_ms=current_start_ms,
                end_ms=current_end_ms,
                speaker_id=seg.speaker_id,
            )
            final_merged.append(merged_seg)

            if j > i + 1:
                logger.debug(
                    f"[第二层] {seg.speaker_id}: "
                    f"合并了 {j - i} 个片段"
                )
            i = j

        return final_merged

    def split_audio_by_speakers(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
    ) -> List[SpeakerSegment]:
        """完整的说话人分离流程

        流程：
        1. 执行CAM++说话人分离
        2. 智能合并短片段（两层合并策略）
           - 第一层：<10s片段累积合并
           - 第二层：60s累积合并
        3. 提取音频数据，保存临时文件

        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录

        Returns:
            SpeakerSegment 列表
        """
        try:
            # 1. 执行说话人分离
            raw_segments = self.diarize(audio_path)

            if not raw_segments:
                logger.warning("说话人分离未检测到任何片段")
                return []

            # 2. 智能合并短片段（第一个<10s的同说话人片段向后合并）
            final_segments = self.merge_short_segments(raw_segments)
            logger.info(f"智能合并完成: {len(raw_segments)} → {len(final_segments)} 个片段")

            # 4. 加载音频并提取片段
            logger.info("加载音频并提取片段...")
            audio_data, sr = librosa.load(audio_path, sr=self.DEFAULT_SAMPLE_RATE)

            output_dir = output_dir or settings.TEMP_DIR
            os.makedirs(output_dir, exist_ok=True)

            for idx, seg in enumerate(final_segments):
                start_sample = int(seg.start_ms / 1000 * sr)
                end_sample = int(seg.end_ms / 1000 * sr)

                seg.audio_data = audio_data[start_sample:end_sample]

                # 保存临时文件
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav",
                    dir=output_dir,
                    prefix=f"{seg.speaker_id}_{idx:03d}_",
                )
                temp_path = temp_file.name
                temp_file.close()

                sf.write(temp_path, seg.audio_data, sr)
                seg.temp_file = temp_path

            # 统计
            unique_speakers = sorted(set(seg.speaker_id for seg in final_segments))
            logger.info(
                f"音频分割完成: {len(final_segments)} 个片段, "
                f"{len(unique_speakers)} 个说话人"
            )
            for spk in unique_speakers:
                spk_segs = [s for s in final_segments if s.speaker_id == spk]
                total_time = sum(s.duration_sec for s in spk_segs)
                logger.info(f"  {spk}: {len(spk_segs)} 片段, {total_time:.2f}s")

            return final_segments

        except Exception as e:
            logger.error(f"说话人分离流程失败: {e}")
            raise DefaultServerErrorException(f"说话人分离失败: {str(e)}")

    @staticmethod
    def cleanup_segments(segments: List[SpeakerSegment]) -> None:
        """清理临时文件"""
        for seg in segments:
            if seg.temp_file and os.path.exists(seg.temp_file):
                try:
                    os.remove(seg.temp_file)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {seg.temp_file}, {e}")
