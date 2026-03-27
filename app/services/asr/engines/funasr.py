# -*- coding: utf-8 -*-
"""
FunASR语音识别引擎模块
支持 Paraformer ASR模型架构
"""

import time
import logging
from typing import Optional, Dict, List, Any, cast
from funasr import AutoModel

from app.core.config import settings
from app.core.exceptions import DefaultServerErrorException
from app.core.logging import log_inference_metrics
from app.utils.text_processing import apply_itn_to_text
from app.infrastructure import resolve_model_path
from app.services.asr.loaders import ModelLoaderFactory, BaseModelLoader
from app.services.asr.engines.base import RealTimeASREngine, ASRRawResult, ASRSegmentResult
from app.services.asr.engines.global_models import (
    get_global_vad_model,
    get_global_punc_model,
    get_vad_inference_lock,
    get_punc_inference_lock,
    get_main_asr_inference_lock,
)


logger = logging.getLogger(__name__)

SENTENCE_END_CHARS = frozenset({"。", "！", "？", "；", "!", "?", ";"})
NO_TIMESTAMP_CHARS = frozenset(
    {
        "。",
        "！",
        "？",
        "；",
        "!",
        "?",
        ";",
        "，",
        ",",
        "、",
        "：",
        ":",
        "“",
        "”",
        "\"",
        "'",
        "（",
        "）",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
    }
)


def _char_uses_timestamp(ch: str) -> bool:
    return bool(ch.strip()) and ch not in NO_TIMESTAMP_CHARS


def _segments_from_text_timestamps(
    text: str,
    timestamps: list[Any],
) -> list[ASRSegmentResult]:
    """Build sentence-level segments from punctuated text and token timestamps."""
    if not text or not timestamps:
        return []

    segments: list[ASRSegmentResult] = []
    current_chars: list[str] = []
    current_start: float | None = None
    current_end: float | None = None
    last_known_time = 0.0
    timestamp_index = 0

    def flush_segment() -> None:
        nonlocal current_chars, current_start, current_end
        segment_text = "".join(current_chars).strip()
        if not segment_text:
            current_chars = []
            current_start = None
            current_end = None
            return

        start_time = current_start if current_start is not None else last_known_time
        end_time = current_end if current_end is not None else start_time
        segments.append(
            ASRSegmentResult(
                text=segment_text,
                start_time=start_time,
                end_time=max(end_time, start_time),
            )
        )
        current_chars = []
        current_start = None
        current_end = None

    for ch in text:
        current_chars.append(ch)

        if _char_uses_timestamp(ch) and timestamp_index < len(timestamps):
            raw_timestamp = timestamps[timestamp_index]
            timestamp_index += 1
            if (
                isinstance(raw_timestamp, (list, tuple))
                and len(raw_timestamp) >= 2
            ):
                char_start = float(raw_timestamp[0]) / 1000.0
                char_end = float(raw_timestamp[1]) / 1000.0
                if current_start is None:
                    current_start = char_start
                current_end = char_end
                last_known_time = char_end

        if ch in SENTENCE_END_CHARS:
            flush_segment()

    flush_segment()
    return [
        segment
        for segment in segments
        if segment.text.strip()
    ]


class TempAutoModelWrapper:
    """临时AutoModel包装器，用于动态组合VAD/PUNC模型"""

    def __init__(self) -> None:
        self.model: Any = None
        self.kwargs: Any = {}
        self.model_path: Any = ""
        self.spk_model: Any = None
        self.vad_model: Any = None
        self.vad_kwargs: Any = {}
        self.punc_model: Any = None
        self.punc_kwargs: Any = {}
        self._base_kwargs_map: Any = None

    def _reset_runtime_configs(self) -> None:
        """Compat shim for newer FunASR AutoModel.generate()."""
        return None

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        """调用AutoModel.inference"""
        return AutoModel.inference(cast(Any, self), *args, **kwargs)

    def inference_with_vad(self, *args: Any, **kwargs: Any) -> Any:
        """调用AutoModel.inference_with_vad"""
        return AutoModel.inference_with_vad(cast(Any, self), *args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """调用AutoModel.generate"""
        return AutoModel.generate(cast(Any, self), *args, **kwargs)


class FunASREngine(RealTimeASREngine):
    """FunASR语音识别引擎 - 使用模块化加载器架构"""

    def __init__(
        self,
        offline_model_path: Optional[str] = None,
        realtime_model_path: Optional[str] = None,
        device: str = "auto",
        vad_model: Optional[str] = None,
        punc_model: Optional[str] = None,
        punc_realtime_model: Optional[str] = None,
        enable_lm: bool = True,
        extra_model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.offline_model: Optional[AutoModel] = None
        self.realtime_model: Optional[AutoModel] = None
        self.punc_model_instance: Optional[AutoModel] = None
        self.punc_realtime_model_instance: Optional[AutoModel] = None
        self._device: str = self._detect_device(device)

        # 模型路径配置
        self.offline_model_path = offline_model_path
        self.realtime_model_path = realtime_model_path

        # 辅助模型配置
        self.vad_model = vad_model or settings.VAD_MODEL
        self.punc_model = punc_model or settings.PUNC_MODEL
        self.punc_realtime_model = punc_realtime_model or settings.PUNC_REALTIME_MODEL

        # 语言模型配置
        self.enable_lm = enable_lm and settings.ASR_ENABLE_LM
        self.lm_model = settings.LM_MODEL if self.enable_lm else None
        self.lm_weight = settings.LM_WEIGHT
        self.lm_beam_size = settings.LM_BEAM_SIZE

        # 额外的模型加载参数
        self.extra_model_kwargs = extra_model_kwargs or {}

        # 模型加载器（由 _load_offline_model 创建）
        self._offline_loader: Optional[BaseModelLoader] = None

        self._load_models_based_on_mode()

    def _load_models_based_on_mode(self) -> None:
        """加载所有可用的模型（离线和实时）"""
        if self.offline_model_path:
            self._load_offline_model()
        if self.realtime_model_path:
            self._load_realtime_model()

    def _load_offline_model(self) -> None:
        """加载离线模型 - 使用模块化加载器"""
        if not self.offline_model_path:
            raise DefaultServerErrorException("未提供离线模型路径")

        try:
            # 使用工厂创建对应的加载器
            self._offline_loader = ModelLoaderFactory.create_loader(
                model_path=self.offline_model_path,
                device=self._device,
                extra_kwargs=self.extra_model_kwargs,
                enable_lm=self.enable_lm,
                lm_model=self.lm_model,
                lm_weight=self.lm_weight,
                lm_beam_size=self.lm_beam_size,
            )

            # 使用加载器加载模型
            self.offline_model = self._offline_loader.load()
            logger.info(f"离线模型加载成功（类型: {self._offline_loader.model_type}）")

        except Exception as e:
            raise DefaultServerErrorException(f"离线FunASR模型加载失败: {str(e)}")

    def _load_realtime_model(self) -> None:
        """加载实时FunASR模型（不再内嵌PUNC，改用全局实例）"""
        try:
            # 解析模型路径：优先使用本地缓存
            resolved_model_path = resolve_model_path(self.realtime_model_path)
            logger.info(f"正在加载实时FunASR模型: {resolved_model_path}")

            model_kwargs = {
                "model": resolved_model_path,
                "device": self._device,
                **settings.FUNASR_AUTOMODEL_KWARGS,
            }

            self.realtime_model = AutoModel(**model_kwargs)
            logger.info("实时FunASR模型加载成功（PUNC将按需使用全局实例）")

        except Exception as e:
            raise DefaultServerErrorException(f"实时FunASR模型加载失败: {str(e)}")

    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        enable_vad: bool = False,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """使用FunASR转录音频文件

        使用加载器处理推理逻辑：
        - Paraformer（传统模型）：支持动态组合 VAD/PUNC/LM
        """
        _ = sample_rate  # 当前未使用
        if not self.offline_model or not self._offline_loader:
            raise DefaultServerErrorException(
                "离线模型未加载，无法进行文件识别。"
                "请将 ASR_MODEL_MODE 设置为 offline 或 all"
            )

        # 性能计时
        start_time = time.time()
        model_id = getattr(self._offline_loader, 'model_type', 'unknown')

        try:
            # 使用加载器准备推理参数
            generate_kwargs = self._offline_loader.prepare_generate_kwargs(
                audio_path=audio_path,
                hotwords=hotwords,
                enable_punctuation=enable_punctuation,
                enable_itn=enable_itn,
                language=language,
            )

            # 使用外部 VAD + PUNC
            if enable_vad:
                result = self._transcribe_with_vad(
                    audio_path, generate_kwargs, enable_punctuation
                )
            else:
                # 主ASR推理加全局锁，避免并发请求串音
                with get_main_asr_inference_lock():
                    result = self.offline_model.generate(**generate_kwargs)

                # 如果需要 PUNC，手动添加
                if enable_punctuation:
                    result = self._apply_punc_to_result(result)

            # 提取识别结果
            if result and len(result) > 0:
                text = result[0].get("text", "")
                text = text.strip()

                # 应用ITN处理
                if enable_itn and text:
                    logger.debug(f"应用ITN处理前: {text}")
                    text = apply_itn_to_text(text)
                    logger.debug(f"应用ITN处理后: {text}")

                # 记录性能指标
                duration_ms = (time.time() - start_time) * 1000
                try:
                    from app.utils.audio import get_audio_duration
                    audio_duration = get_audio_duration(audio_path)
                except Exception:
                    audio_duration = 0

                log_inference_metrics(
                    logger=logger,
                    message="单文件识别完成",
                    task_id=task_id,
                    duration_ms=duration_ms,
                    audio_duration_sec=audio_duration,
                    model_id=model_id,
                    status="success",
                    enable_punctuation=enable_punctuation,
                    enable_itn=enable_itn,
                    enable_vad=enable_vad,
                )

                return text
            else:
                return ""

        except Exception as e:
            # 记录失败指标
            duration_ms = (time.time() - start_time) * 1000
            try:
                from app.utils.audio import get_audio_duration
                audio_duration = get_audio_duration(audio_path)
            except Exception:
                audio_duration = 0

            log_inference_metrics(
                logger=logger,
                message="单文件识别失败",
                task_id=task_id,
                duration_ms=duration_ms,
                audio_duration_sec=audio_duration,
                model_id=model_id,
                status="error",
                error=str(e),
            )

            raise DefaultServerErrorException(f"语音识别失败: {str(e)}")

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
            hotwords: 热词
            enable_punctuation: 是否启用标点
            enable_itn: 是否启用 ITN
            sample_rate: 采样率
            **kwargs: 额外参数（如 word_timestamps 字词级时间戳）

        Returns:
            ASRRawResult: 包含文本和分段时间戳的结果
        """
        _ = sample_rate  # 当前未使用
        _ = kwargs  # FunASR 暂不支持字词级时间戳，忽略此参数

        if not self.offline_model or not self._offline_loader:
            raise DefaultServerErrorException(
                "离线模型未加载，无法进行文件识别。"
            )

        try:
            # 使用加载器准备推理参数
            generate_kwargs = self._offline_loader.prepare_generate_kwargs(
                audio_path=audio_path,
                hotwords=hotwords,
                enable_punctuation=enable_punctuation,
                enable_itn=enable_itn,
            )

            # 使用外部 VAD
            result = self._transcribe_with_vad(
                audio_path, generate_kwargs, enable_punctuation
            )

            # 解析结果
            segments: List[ASRSegmentResult] = []
            full_text = ""

            if result and len(result) > 0:
                full_text = result[0].get("text", "").strip()

                # 解析时间戳
                # FunASR 返回格式可能是:
                # 1. {"sentence_info": [[start_ms, end_ms, "text"], ...]}
                # 2. {"timestamp": [[start_ms, end_ms], ...]}
                sentence_info = result[0].get("sentence_info", [])

                if sentence_info and isinstance(sentence_info, list):
                    for sent in sentence_info:
                        try:
                            if isinstance(sent, dict):
                                # 格式: {"start": ms, "end": ms, "text": "..."}
                                start_ms = sent.get("start", 0)
                                end_ms = sent.get("end", 0)
                                text = sent.get("text", "")
                            elif isinstance(sent, (list, tuple)) and len(sent) >= 3:
                                # 格式: [start_ms, end_ms, "text"]
                                start_ms = sent[0]
                                end_ms = sent[1]
                                text = sent[2] if len(sent) > 2 else ""
                            else:
                                continue

                            segments.append(ASRSegmentResult(
                                text=str(text),
                                start_time=start_ms / 1000.0,
                                end_time=end_ms / 1000.0,
                            ))
                        except (IndexError, TypeError, KeyError) as e:
                            logger.warning(f"解析 sentence_info 项失败: {e}")

                # 如果没有解析到分段信息，尝试从 timestamp 字段解析
                if not segments:
                    timestamp = result[0].get("timestamp", [])
                    if (
                        timestamp
                        and isinstance(timestamp, list)
                        and len(timestamp) > 0
                        and full_text
                    ):
                        try:
                            segments = _segments_from_text_timestamps(
                                full_text,
                                timestamp,
                            )
                        except (IndexError, TypeError, ValueError) as e:
                            logger.warning(f"解析 timestamp 失败: {e}")

                # 应用 ITN 处理
                if enable_itn and full_text:
                    full_text = apply_itn_to_text(full_text)
                    # 同时对分段文本应用 ITN
                    for seg in segments:
                        seg.text = apply_itn_to_text(seg.text)

            return ASRRawResult(text=full_text, segments=segments)

        except Exception as e:
            raise DefaultServerErrorException(f"语音识别失败: {str(e)}")

    def transcribe_websocket(
        self,
        audio_chunk: bytes,
        cache: Optional[Dict] = None,
        is_final: bool = False,
        **kwargs: Any,
    ) -> str:
        """WebSocket流式语音识别（未实现）"""
        # 忽略未使用的参数（功能尚未实现）
        _ = (audio_chunk, cache, is_final, kwargs)
        if not self.realtime_model:
            raise DefaultServerErrorException(
                "实时模型未加载，无法进行WebSocket流式识别。"
                "请将 ASR_MODEL_MODE 设置为 realtime 或 all"
            )

        logger.warning("WebSocket流式识别功能尚未实现")
        return ""

    def _transcribe_with_vad(
        self,
        audio_path: str,
        generate_kwargs: Dict[str, Any],
        enable_punctuation: bool,
    ) -> List[Dict[str, Any]]:
        """使用 VAD 进行转录（传统模型专用）

        注意：使用全局VAD/PUNC模型时加锁，防止并发状态混乱（issue #18）
        """
        logger.debug("使用VAD进行分段识别")
        vad_model_instance = get_global_vad_model(self._device)

        punc_model_instance = None
        if enable_punctuation:
            logger.debug("预加载全局PUNC模型")
            punc_model_instance = get_global_punc_model(self._device)

        # 创建临时AutoModel包装器
        if self.offline_model is None:
            raise DefaultServerErrorException("离线模型未加载")
        temp_automodel = TempAutoModelWrapper()
        temp_automodel.model = self.offline_model.model
        temp_automodel.kwargs = self.offline_model.kwargs
        temp_automodel.model_path = self.offline_model.model_path

        # 设置VAD（加锁保护，防止并发状态混乱）
        with get_vad_inference_lock():
            temp_automodel.vad_model = vad_model_instance.model
            temp_automodel.vad_kwargs = vad_model_instance.kwargs

            # 设置PUNC（加锁保护）
            if punc_model_instance:
                temp_automodel.punc_model = punc_model_instance.model
                temp_automodel.punc_kwargs = punc_model_instance.kwargs

            # 在锁内执行推理，确保VAD/PUNC模型状态不被其他请求干扰
            return temp_automodel.generate(**generate_kwargs)

    def _apply_punc_to_text(self, text: str) -> str:
        """手动应用标点符号到文本

        Args:
            text: 无标点的识别文本

        Returns:
            添加标点后的文本
        """
        if not text:
            return text

        try:
            logger.debug(f"手动应用PUNC模型: {text[:50]}...")
            punc_model_instance = get_global_punc_model(self._device)

            # 加锁保护，防止并发状态混乱（issue #18）
            with get_punc_inference_lock():
                punc_result = punc_model_instance.generate(
                    input=text,
                    cache={},
                )

            if punc_result and len(punc_result) > 0:
                text_with_punc = punc_result[0].get("text", text)
                logger.debug(f"标点添加完成: {text_with_punc[:50]}...")
                return text_with_punc
        except Exception as e:
            logger.warning(f"PUNC模型应用失败: {e}, 返回原文本")

        return text

    def _apply_punc_to_result(
        self, result: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """手动应用标点符号到识别结果"""
        if not result or len(result) == 0:
            return result

        text = result[0].get("text", "").strip()
        if not text:
            return result

        text_with_punc = self._apply_punc_to_text(text)
        result[0]["text"] = text_with_punc

        return result

    def _transcribe_batch(
        self,
        segments: List[Any],
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        sample_rate: int = 16000,
        word_timestamps: bool = False,
    ) -> List[ASRSegmentResult]:
        """批量推理多个音频片段（FunASR 优化版）

        利用 FunASR 的批量推理能力，比逐个推理快 2-3 倍
        注意：FunASR 不支持字词级时间戳，word_timestamps 参数会被忽略
        """
        if not self.offline_model or not self._offline_loader:
            logger.warning("离线模型未加载，使用默认批处理实现")
            return super()._transcribe_batch(segments, hotwords, enable_punctuation, enable_itn, sample_rate, word_timestamps)

        # 过滤有效片段
        valid_segments = [(idx, seg) for idx, seg in enumerate(segments) if seg.temp_file]
        if not valid_segments:
            return [ASRSegmentResult(text="", start_time=0.0, end_time=0.0) for _ in segments]

        try:
            import librosa

            # 加载音频数据（批量输入）
            logger.info(f"加载 {len(valid_segments)} 个音频片段到内存...")
            audio_inputs = []
            for idx, seg in valid_segments:
                try:
                    # 加载音频数据为numpy数组
                    audio_data, _sr = librosa.load(seg.temp_file, sr=sample_rate)
                    audio_inputs.append(audio_data)
                except Exception as e:
                    logger.error(f"加载音频片段 {idx + 1} 失败: {e}")
                    audio_inputs.append(None)

            # 过滤加载成功的音频，保留原始索引和片段信息
            valid_inputs = [
                (idx, seg, audio) for (idx, seg), audio in zip(valid_segments, audio_inputs) if audio is not None
            ]

            if not valid_inputs:
                logger.warning("没有成功加载的音频片段")
                return [ASRSegmentResult(text="", start_time=0.0, end_time=0.0) for _ in segments]

            logger.info(f"FunASR 批量推理: {len(valid_inputs)} 个片段")

            # 准备批量推理参数
            batch_audio_data = [audio for _, _, audio in valid_inputs]

            # 使用加载器准备推理参数（注意：传入音频数据而不是路径）
            generate_kwargs = self._offline_loader.prepare_generate_kwargs(
                audio_path=None,  # 不使用路径
                hotwords=hotwords,
                enable_punctuation=enable_punctuation,
                enable_itn=enable_itn,
            )

            # 覆盖input参数为批量音频数据
            generate_kwargs['input'] = batch_audio_data
            generate_kwargs['batch_size'] = len(batch_audio_data)

            # 主ASR批量推理加全局锁，避免并发请求串音
            with get_main_asr_inference_lock():
                batch_results = self.offline_model.generate(**generate_kwargs)

            # 解析批量结果
            batch_results_parsed = []
            if batch_results and isinstance(batch_results, list):
                for res in batch_results:
                    if isinstance(res, dict):
                        text = res.get("text", "").strip()

                        # 应用PUNC模型（Paraformer需要手动添加标点）
                        if enable_punctuation and text:
                            text = self._apply_punc_to_text(text)

                        # 应用ITN处理
                        if enable_itn and text:
                            text = apply_itn_to_text(text)

                        batch_results_parsed.append(text)
                    else:
                        batch_results_parsed.append("")
            else:
                logger.warning("批量推理返回结果格式异常")
                batch_results_parsed = [""] * len(valid_inputs)

            # 将结果映射回原始顺序（包括跳过的片段）
            results: List[ASRSegmentResult] = [
                ASRSegmentResult(text="", start_time=0.0, end_time=0.0) for _ in segments
            ]
            for (idx, seg, _), text in zip(valid_inputs, batch_results_parsed):
                results[idx] = ASRSegmentResult(
                    text=text,
                    start_time=seg.start_sec,
                    end_time=seg.end_sec,
                    speaker_id=getattr(seg, 'speaker_id', None),
                )

            logger.info(f"FunASR 批量推理完成，有效结果: {sum(1 for r in results if r.text)}/{len(results)}")
            return results

        except Exception as e:
            logger.error(f"FunASR 批量推理失败: {e}，fallback 到逐个推理")
            # fallback 到父类的逐个推理实现
            return super()._transcribe_batch(segments, hotwords, enable_punctuation, enable_itn, sample_rate, word_timestamps)

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.offline_model is not None or self.realtime_model is not None

    @property
    def device(self) -> str:
        """获取设备信息"""
        return self._device

    @property
    def model_id(self) -> str:
        """获取模型ID"""
        if self._offline_loader:
            return self._offline_loader.model_type
        return "funasr-unknown"


# 自动注册 FunASR 引擎（由 manager.py 显式触发）
def _register_funasr_engine(register_func, _model_config_cls):
    """注册 FunASR 引擎到引擎注册表

    Args:
        register_func: register_engine 函数
        model_config_cls: ModelConfig 类
    """
    from app.core.config import settings

    def _create_funasr_engine(config) -> "FunASREngine":
        return FunASREngine(
            offline_model_path=config.models.get("offline"),
            realtime_model_path=config.models.get("realtime"),
            device=settings.DEVICE,
            vad_model=settings.VAD_MODEL,
            punc_model=settings.PUNC_MODEL,
            punc_realtime_model=settings.PUNC_REALTIME_MODEL,
            extra_model_kwargs=config.extra_kwargs,
        )

    register_func("funasr", _create_funasr_engine)
