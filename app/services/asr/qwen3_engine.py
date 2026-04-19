# -*- coding: utf-8 -*-
"""Qwen3-ASR engine with official vLLM, CPU Rust, and Apple MLX backends."""

import logging
import os
from typing import Optional, List, Any
from dataclasses import dataclass

import torch
import numpy as np

from .engines import BaseASREngine, ASRRawResult, ASRSegmentResult, WordToken
from .qwenasr_rust import (
    QwenASRRustRuntime,
    is_qwenasr_rust_available,
    resolve_qwenasr_num_threads,
)
from .qwen3_mlx import Qwen3MLXBackend, is_mlx_qwen_available
from .qwen3_vllm import Qwen3VLLMBackend, is_vllm_available
from ...core.exceptions import DefaultServerErrorException
from ...core.config import settings

logger = logging.getLogger(__name__)


def calculate_gpu_memory_utilization(model_path: str) -> float:
    """Calculate optimal gpu_memory_utilization based on model size and available VRAM

    Model memory requirements (observed, including KV cache):
    - 0.6B: ~8GB (model + KV cache)
    - 1.7B: ~12GB (model + KV cache)

    Examples:
    - 8GB VRAM + 0.6B: 8/8 = 1.0 → clamped to 0.95
    - 24GB VRAM + 1.7B: 12/24 = 0.5
    - 80GB VRAM + 1.7B: 12/80 = 0.15

    Args:
        model_path: Path to model (used to detect model size)

    Returns:
        gpu_memory_utilization ratio (0.0 to 1.0)
    """
    # Check environment variable override first
    env_override = os.getenv("QWEN_GPU_MEMORY_UTILIZATION")
    if env_override:
        try:
            value = float(env_override)
            if 0.0 < value <= 1.0:
                logger.info(f"Using environment override: gpu_memory_utilization={value}")
                return value
            else:
                logger.warning(f"Invalid QWEN_GPU_MEMORY_UTILIZATION={env_override}, must be 0.0-1.0")
        except ValueError:
            logger.warning(f"Invalid QWEN_GPU_MEMORY_UTILIZATION={env_override}, not a float")

    # Model memory requirements (GB) - includes model + KV cache
    MODEL_MEMORY_REQUIREMENTS = {
        "0.6B": 8.0,
        "1.7B": 12.0,
    }

    # Detect model size from path
    if "0.6B" in model_path:
        required_memory_gb = MODEL_MEMORY_REQUIREMENTS["0.6B"]
        model_size = "0.6B"
    else:
        required_memory_gb = MODEL_MEMORY_REQUIREMENTS["1.7B"]
        model_size = "1.7B"

    # Get total VRAM
    try:
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using fallback gpu_memory_utilization=0.5")
            return 0.5

        # Use first GPU for memory detection
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Calculate utilization ratio
        utilization = required_memory_gb / total_vram_gb

        # Clamp to safe maximum (0.95)
        utilization = min(utilization, 0.95)

        logger.info(
            f"GPU memory calculation: model={model_size}, "
            f"requires={required_memory_gb:.1f}GB, total_vram={total_vram_gb:.1f}GB, "
            f"utilization={utilization:.2f}"
        )

        # Warn if VRAM is insufficient
        if utilization >= 0.90:
            logger.warning(
                f"VRAM may be insufficient: {total_vram_gb:.1f}GB available, "
                f"{required_memory_gb:.1f}GB required. Consider using smaller model."
            )

        return round(utilization, 2)

    except Exception as e:
        logger.error(f"Failed to detect VRAM: {e}, using fallback gpu_memory_utilization=0.5")
        return 0.5


def _handle_asr_error(operation: str):
    """统一错误处理装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{operation} 失败: {e}")
                raise DefaultServerErrorException(f"{operation} 失败: {e}")
        return wrapper
    return decorator


def _get_word_tokens(result, word_level: bool) -> Optional[List[WordToken]]:
    """提取字词级时间戳"""
    if not word_level:
        return None
    ts = getattr(result, "time_stamps", None)
    items = getattr(ts, "items", None)
    if not items:
        return None
    return [
        WordToken(text=item.text, start_time=round(item.start_time, 3), end_time=round(item.end_time, 3))
        for item in items
    ]


@dataclass
class Qwen3StreamingState:
    internal_state: Any
    chunk_size_sec: float = 2.0
    unfixed_chunk_num: int = 2
    unfixed_token_num: int = 5
    max_new_tokens: int = 32
    language: Optional[str] = None
    chunk_count: int = 0
    last_text: str = ""
    last_language: str = ""


class Qwen3ASREngine(BaseASREngine):
    model: Any

    @property
    def supports_realtime(self) -> bool:
        return self._backend in {"vllm", "rust", "mlx"}

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "auto",
        forced_aligner_path: Optional[str] = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 1024,
        max_model_len: Optional[int] = None,
        **_kwargs,
    ):
        """Initialize Qwen3-ASR engine

        CUDA -> official vLLM backend
        CPU  -> QwenASR Rust backend
        MPS  -> MLX backend
        """
        from app.core.device import detect_device

        self._device = detect_device(device)
        self.model_id = model_path
        self.model_path = model_path
        self._backend = self._select_backend()
        self._forced_aligner_path = forced_aligner_path

        try:
            if self._backend == "vllm":
                self.model = self._load_vllm(
                    model_path, forced_aligner_path,
                    max_inference_batch_size, max_new_tokens, max_model_len,
                )
            elif self._backend == "rust":
                self.model = self._load_rust_backend(model_path, forced_aligner_path)
            elif self._backend == "mlx":
                self.model = self._load_mlx_backend(
                    model_path=model_path,
                    forced_aligner_path=forced_aligner_path,
                    max_new_tokens=max_new_tokens,
                )
            else:
                raise DefaultServerErrorException(
                    f"Qwen3-ASR is not available on device '{self._device}'"
                )
            self._warmup_forced_aligner()
            logger.info("Qwen3-ASR model loaded successfully with backend=%s", self._backend)
        except Exception as e:
            logger.error(f"Failed to load Qwen3-ASR model: {e}")
            raise DefaultServerErrorException(f"Failed to load Qwen3-ASR model: {e}")

    def _select_backend(self) -> str:
        if self._device.startswith("cuda"):
            if not is_vllm_available():
                raise DefaultServerErrorException(
                    "Current Python environment is missing official vLLM nightly. "
                    "Install it with: pip install --pre 'vllm[audio]'"
                )
            return "vllm"
        if self._device == "mps":
            if not is_mlx_qwen_available():
                raise DefaultServerErrorException(
                    "Current Python environment is missing mlx-qwen3-asr. "
                    "Apple Silicon Qwen3-ASR now uses the MLX backend. "
                    "Install it with: pip install 'mlx-qwen3-asr[aligner]'"
                )
            return "mlx"
        if self._device == "cpu" and is_qwenasr_rust_available():
            return "rust"
        return "transformers"

    def _load_rust_backend(
        self,
        model_path: str,
        forced_aligner_path: Optional[str],
    ) -> QwenASRRustRuntime:
        logger.info("Loading Qwen3-ASR (QwenASR Rust): %s, device=%s", model_path, self._device)

        num_threads = resolve_qwenasr_num_threads(settings.QWEN_RUST_CPU_WORKERS)
        if not (os.getenv("QWENASR_CPU_NUM_THREADS") or "").strip() and settings.QWEN_RUST_CPU_WORKERS > 1:
            logger.info(
                "Using safe QwenASR CPU thread count: num_threads=%s workers=%s",
                num_threads,
                settings.QWEN_RUST_CPU_WORKERS,
            )
        return QwenASRRustRuntime(
            model_path=model_path,
            forced_aligner_path=forced_aligner_path,
            num_threads=num_threads,
            verbosity=0,
        )

    def _load_mlx_backend(
        self,
        model_path: str,
        forced_aligner_path: Optional[str],
        max_new_tokens: int,
    ) -> Qwen3MLXBackend:
        logger.info("Loading Qwen3-ASR (MLX): %s, device=%s", model_path, self._device)
        return Qwen3MLXBackend(
            model_path=model_path,
            forced_aligner_path=forced_aligner_path,
            max_new_tokens=max_new_tokens,
        )

    def _warmup_forced_aligner(self) -> None:
        if not self._forced_aligner_path:
            return
        if self._backend == "vllm":
            self.model.ensure_forced_aligner_loaded()
        if self._backend == "mlx":
            self.model.ensure_forced_aligner_loaded()

    def _load_vllm(
        self, model_path: str, forced_aligner_path: Optional[str],
        max_inference_batch_size: int, max_new_tokens: int,
        max_model_len: Optional[int],
    ) -> Qwen3VLLMBackend:
        """Load model via official vLLM backend (CUDA only)."""
        gpu_memory_utilization = calculate_gpu_memory_utilization(model_path)
        logger.info(
            f"Loading Qwen3-ASR (official vLLM): {model_path}, "
            f"device={self._device}, gpu_memory_utilization={gpu_memory_utilization}"
        )
        return Qwen3VLLMBackend(
            model_path=model_path,
            forced_aligner_path=forced_aligner_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
        )

    @_handle_asr_error("转写")
    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        enable_vad: bool = False,
        sample_rate: int = 16000,
    ) -> str:
        if self._backend == "rust":
            return self.model.transcribe_file(audio_path)
        if self._backend == "mlx":
            return self.model.transcribe_text(audio_path, context=hotwords or "")
        if self._backend == "vllm":
            return self.model.transcribe_text(
                audio_path,
                context=hotwords or "",
            )
        raise DefaultServerErrorException(f"Qwen3 backend={self._backend} does not support offline transcription")

    @_handle_asr_error("VAD 转写")
    def transcribe_file_with_vad(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
        **kwargs,
    ) -> ASRRawResult:
        if self._backend == "rust":
            text = self.transcribe_file(
                audio_path=audio_path,
                hotwords=hotwords,
                enable_punctuation=enable_punctuation,
                enable_itn=enable_itn,
                sample_rate=sample_rate,
            )
            if kwargs.get("word_timestamps", False):
                word_tokens = [
                    WordToken(
                        text=str(item["text"]),
                        start_time=round(float(item["start_ms"]) / 1000.0, 3),
                        end_time=round(float(item["end_ms"]) / 1000.0, 3),
                    )
                    for item in self.model.align_transcript(
                        audio_path=audio_path,
                        text=text,
                        language=kwargs.get("language"),
                    )
                ]
                if word_tokens:
                    return ASRRawResult(
                        text=text,
                        segments=[
                            ASRSegmentResult(
                                text=text,
                                start_time=word_tokens[0].start_time,
                                end_time=word_tokens[-1].end_time,
                                word_tokens=word_tokens,
                            )
                        ],
                    )
            return ASRRawResult(
                text=text,
                segments=[ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)] if text else [],
            )
        if self._backend == "mlx":
            word_timestamps = kwargs.get("word_timestamps", False)
            return self.model.transcribe_raw(
                audio_path=audio_path,
                context=hotwords or "",
                word_timestamps=word_timestamps,
            )
        if self._backend == "vllm":
            return self.model.transcribe_raw(
                audio_path=audio_path,
                context=hotwords or "",
                language=kwargs.get("language"),
                word_timestamps=kwargs.get("word_timestamps", False),
            )

        return ASRRawResult(
            text="",
            segments=[],
        )

    def _to_segments(self, text: str, time_stamps: Any, word_level: bool) -> List[ASRSegmentResult]:
        """转换时间戳为分段"""
        items: List[Any] = list(
            getattr(getattr(time_stamps, "items", None), "__iter__", lambda: [])()
        )

        if not items:
            return [ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)] if text else []

        segments = []
        current, start, words = "", items[0].start_time, []
        breaks = set("。！？；\n")

        for i, item in enumerate(items):
            current += item.text
            words.append(WordToken(item.text, round(item.start_time, 3), round(item.end_time, 3))) if word_level else None

            if item.text in breaks or i == len(items) - 1:
                if current.strip():
                    segments.append(ASRSegmentResult(
                        text=current.strip(),
                        start_time=round(start, 2),
                        end_time=round(item.end_time, 2),
                        word_tokens=words if word_level else None,
                    ))
                current, words = "", []
                if i < len(items) - 1:
                    start = items[i + 1].start_time

        return segments

    @_handle_asr_error("批量推理")
    def _transcribe_batch(
        self,
        segments: List[Any],
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        sample_rate: int = 16000,
        word_timestamps: bool = False,
    ) -> List[ASRSegmentResult]:
        if self._backend in {"rust", "mlx"}:
            return super()._transcribe_batch(
                segments=segments,
                hotwords=hotwords,
                enable_punctuation=enable_punctuation,
                enable_itn=enable_itn,
                sample_rate=sample_rate,
                word_timestamps=word_timestamps,
            )

        output = [ASRSegmentResult(text="", start_time=0.0, end_time=0.0) for _ in segments]

        valid: List[tuple[int, Any]] = []
        for idx, seg in enumerate(segments):
            temp_file = getattr(seg, "temp_file", None)
            if temp_file and os.path.exists(temp_file):
                valid.append((idx, seg))
            else:
                logger.warning(f"Qwen3 批处理片段无效或文件不存在: segment={idx + 1}, file={temp_file}")

        if not valid:
            return output

        if self._backend == "vllm":
            vllm_results = self.model.transcribe_batch(
                [seg.temp_file for _, seg in valid],
                context=hotwords or "",
                word_timestamps=word_timestamps,
            )
            for (idx, seg), result in zip(valid, vllm_results):
                output[idx] = ASRSegmentResult(
                    text=result.text,
                    start_time=round(seg.start_sec, 2),
                    end_time=round(seg.end_sec, 2),
                    speaker_id=getattr(seg, "speaker_id", None),
                    word_tokens=result.word_tokens if word_timestamps else None,
                )
            return output

        def _build_result(seg: Any, result: Any) -> ASRSegmentResult:
            return ASRSegmentResult(
                text=(getattr(result, "text", "") or ""),
                start_time=round(seg.start_sec, 2),
                end_time=round(seg.end_sec, 2),
                speaker_id=getattr(seg, "speaker_id", None),
                word_tokens=_get_word_tokens(result, word_timestamps),
            )

        indices, segs = zip(*valid)

        try:
            results = self.model.transcribe(
                audio=[seg.temp_file for seg in segs],
                context=hotwords or "",
                return_time_stamps=word_timestamps,
            )
            if len(results) != len(segs):
                raise DefaultServerErrorException(
                    "Qwen3 批量结果数量不匹配: "
                    f"expected={len(segs)}, got={len(results)}"
                )

            for idx, seg, result in zip(indices, segs, results):
                output[idx] = _build_result(seg, result)
            return output
        except Exception as batch_error:
            logger.warning(
                "Qwen3 批量推理失败，降级为逐段推理: "
                f"batch_size={len(segs)}, error={batch_error}"
            )

        for idx, seg in valid:
            try:
                single_results = self.model.transcribe(
                    audio=seg.temp_file,
                    context=hotwords or "",
                    return_time_stamps=word_timestamps,
                )
                if single_results:
                    output[idx] = _build_result(seg, single_results[0])
                else:
                    logger.warning(f"Qwen3 逐段推理返回空结果: segment={idx + 1}")
            except Exception as single_error:
                logger.error(f"Qwen3 逐段推理失败: segment={idx + 1}, error={single_error}")

        return output

    @_handle_asr_error("初始化流式状态")
    def init_streaming_state(self, context: str = "", language: Optional[str] = None, **kwargs) -> Qwen3StreamingState:
        if self._backend not in {"vllm", "rust", "mlx"}:
            raise DefaultServerErrorException(
                f"Qwen3 backend={self._backend} does not support realtime streaming"
            )
        if self._backend == "rust":
            if context:
                logger.debug("QwenASR Rust backend ignores streaming context hints")
            chunk_size_sec = float(kwargs.get("chunk_size_sec", 2.0))
            unfixed_chunk_num = int(kwargs.get("unfixed_chunk_num", 2))
            unfixed_token_num = int(kwargs.get("unfixed_token_num", 5))
            max_new_tokens = int(kwargs.get("max_new_tokens", 32))
            stream_handle = self.model.create_stream(
                chunk_size_sec=chunk_size_sec,
                unfixed_chunk_num=unfixed_chunk_num,
                rollback_tokens=unfixed_token_num,
                max_new_tokens=max_new_tokens,
                language=language,
            )
            return Qwen3StreamingState(
                internal_state=stream_handle,
                chunk_size_sec=chunk_size_sec,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                max_new_tokens=max_new_tokens,
                language=language,
                chunk_count=0,
                last_text="",
                last_language=language or "",
            )
        if self._backend == "mlx":
            chunk_size_sec = float(kwargs.get("chunk_size_sec", 2.0))
            unfixed_chunk_num = int(kwargs.get("unfixed_chunk_num", 2))
            unfixed_token_num = int(kwargs.get("unfixed_token_num", 5))
            max_new_tokens = int(kwargs.get("max_new_tokens", 32))
            streaming_state = self.model.init_streaming_state(
                context=context,
                language=language,
                chunk_size_sec=chunk_size_sec,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                max_new_tokens=max_new_tokens,
                max_context_sec=float(kwargs.get("max_context_sec", 30.0)),
                finalization_mode=str(kwargs.get("finalization_mode", "accuracy")),
                endpointing_mode=str(kwargs.get("endpointing_mode", "fixed")),
            )
            return Qwen3StreamingState(
                internal_state=streaming_state,
                chunk_size_sec=chunk_size_sec,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                max_new_tokens=max_new_tokens,
                language=language,
                chunk_count=int(getattr(streaming_state, "chunk_id", 0)),
                last_text=str(getattr(streaming_state, "text", "") or ""),
                last_language=str(getattr(streaming_state, "language", "") or ""),
            )

        if self._backend == "vllm":
            streaming_state = self.model.init_streaming_state(context=context, language=language, **kwargs)
            return Qwen3StreamingState(
                internal_state=streaming_state,
                chunk_size_sec=float(kwargs.get("chunk_size_sec", 2.0)),
                unfixed_chunk_num=int(kwargs.get("unfixed_chunk_num", 2)),
                unfixed_token_num=int(kwargs.get("unfixed_token_num", 5)),
                max_new_tokens=int(kwargs.get("max_new_tokens", 32)),
                language=language,
                chunk_count=int(getattr(streaming_state, "chunk_id", 0)),
                last_text=str(getattr(streaming_state, "text", "") or ""),
                last_language=str(getattr(streaming_state, "language", "") or ""),
            )

        raise DefaultServerErrorException(
            f"Qwen3 backend={self._backend} does not support realtime streaming"
        )

    @_handle_asr_error("流式识别")
    def streaming_transcribe(self, pcm16k: np.ndarray, state: Qwen3StreamingState) -> Qwen3StreamingState:
        if self._backend not in {"vllm", "rust", "mlx"}:
            raise DefaultServerErrorException(
                f"Qwen3 backend={self._backend} does not support realtime streaming"
            )
        pcm = pcm16k.astype(np.float32) / (32768.0 if pcm16k.dtype == np.int16 else 1.0)
        if self._backend == "rust":
            text = self.model.push_stream(
                stream=state.internal_state,
                samples=pcm,
                chunk_size_sec=state.chunk_size_sec,
                unfixed_chunk_num=state.unfixed_chunk_num,
                rollback_tokens=state.unfixed_token_num,
                max_new_tokens=state.max_new_tokens,
                language=state.language,
            )
            state.chunk_count += 1
            state.last_text = text
            state.last_language = state.language or ""
            return state
        if self._backend == "mlx":
            streaming_state = self.model.feed_stream(pcm, state.internal_state)
            state.internal_state = streaming_state
            state.chunk_count = int(getattr(streaming_state, "chunk_id", state.chunk_count))
            state.last_text = str(getattr(streaming_state, "text", "") or "")
            state.last_language = str(getattr(streaming_state, "language", "") or "")
            return state

        streaming_state = self.model.feed_stream(pcm, state.internal_state)
        state.internal_state = streaming_state
        state.chunk_count = int(getattr(streaming_state, "chunk_id", state.chunk_count))
        state.last_text = str(getattr(streaming_state, "text", "") or "")
        state.last_language = str(getattr(streaming_state, "language", "") or "")
        return state

    @_handle_asr_error("结束流式识别")
    def finish_streaming_transcribe(self, state: Qwen3StreamingState) -> Qwen3StreamingState:
        if self._backend not in {"vllm", "rust", "mlx"}:
            raise DefaultServerErrorException(
                f"Qwen3 backend={self._backend} does not support realtime streaming"
            )
        if self._backend == "rust":
            text = self.model.finish_stream(
                stream=state.internal_state,
                chunk_size_sec=state.chunk_size_sec,
                unfixed_chunk_num=state.unfixed_chunk_num,
                rollback_tokens=state.unfixed_token_num,
                max_new_tokens=state.max_new_tokens,
                language=state.language,
            )
            state.last_text = text
            state.last_language = state.language or ""
            return state
        if self._backend == "mlx":
            streaming_state = self.model.finish_stream(state.internal_state)
            state.internal_state = streaming_state
            state.chunk_count = int(getattr(streaming_state, "chunk_id", state.chunk_count))
            state.last_text = str(getattr(streaming_state, "text", "") or "")
            state.last_language = str(getattr(streaming_state, "language", "") or "")
            return state

        streaming_state = self.model.finish_stream(state.internal_state)
        state.internal_state = streaming_state
        state.chunk_count = int(getattr(streaming_state, "chunk_id", state.chunk_count))
        state.last_text = str(getattr(streaming_state, "text", "") or "")
        state.last_language = str(getattr(streaming_state, "language", "") or "")
        return state

    def is_model_loaded(self) -> bool:
        return self.model is not None

    @property
    def device(self) -> str:
        return self._device


def _register_qwen3_engine(register_func, _declared_entry_cls):
    from app.core.config import settings

    def _create(config):
        extra = {k: v for k, v in config.extra_kwargs.items() if v is not None}
        model_id = config.models.get("offline")
        return Qwen3ASREngine(model_path=model_id, device=settings.DEVICE, **extra)

    register_func("qwen3", _create)
