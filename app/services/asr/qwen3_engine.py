# -*- coding: utf-8 -*-
"""Qwen3-ASR engine with official vLLM and vendored Rust backends."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Any
from dataclasses import dataclass

import torch
import numpy as np

from .engines import BaseASREngine, ASRRawResult, ASRSegmentResult, WordToken
from .qwenasr_rust import (
    QwenASRRustRuntime,
    is_qwenasr_rust_available,
)
from .qwen3_vllm import Qwen3VLLMBackend, is_vllm_available
from ...core.exceptions import DefaultServerErrorException
from ...core.config import settings

logger = logging.getLogger(__name__)


def calculate_gpu_memory_utilization(model_path: str) -> float:
    """Calculate vLLM GPU memory utilization for the active model.

    vLLM uses this ratio as an allocation budget, not just model weights.
    Keep the observed requirement slightly above the bare minimum so KV cache
    and profiling have enough room.
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

    model_memory_profiles = {
        "0.6B": 8,
        "1.7B": 12.0,
    }

    if "0.6B" in model_path:
        model_size = "0.6B"
    else:
        model_size = "1.7B"
    required_memory_gb = model_memory_profiles[model_size]

    try:
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using fallback gpu_memory_utilization=0.5")
            return 0.5

        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        utilization = required_memory_gb / total_vram_gb
        utilization = min(utilization, 0.95)

        logger.info(
            "GPU memory calculation: model=%s, requires=%.1fGB, total_vram=%.1fGB, utilization=%.2f",
            model_size,
            required_memory_gb,
            total_vram_gb,
            utilization,
        )

        if utilization >= 0.90:
            logger.warning(
                "VRAM may be insufficient: %.1fGB available, %.1fGB required. Consider using smaller model.",
                total_vram_gb,
                required_memory_gb,
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
        return self._backend in {"vllm", "rust"}

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
        CPU/macOS -> QwenASR Rust backend
        """
        from app.core.device import detect_device

        model_id = _kwargs.pop("model_id", None)
        if model_id:
            model_path = model_id

        self._device = detect_device(device)
        self.model_id = model_path
        self.model_path = model_path
        self._backend = self._select_backend()
        self._forced_aligner_path = forced_aligner_path
        self._rust_num_threads = 0
        self._rust_verbosity = 0
        self._rust_batch_runtimes: list[QwenASRRustRuntime] = []

        try:
            if self._backend == "vllm":
                self.model = self._load_vllm(
                    model_path, forced_aligner_path,
                    max_inference_batch_size, max_new_tokens, max_model_len,
                )
            elif self._backend == "rust":
                self.model = self._load_rust_backend(model_path, forced_aligner_path)
            self._warmup_forced_aligner()
            logger.info("Qwen3-ASR model loaded successfully with backend=%s", self._backend)
        except Exception as e:
            logger.error(f"Failed to load Qwen3-ASR model: {e}")
            raise DefaultServerErrorException(f"Failed to load Qwen3-ASR model: {e}")

    def _select_backend(self) -> str:
        if self._device.startswith("cuda"):
            if not is_vllm_available():
                raise DefaultServerErrorException(
                    "Current Python environment is missing official vLLM with Qwen3 forced aligner support. "
                    "Install it with: pip install 'vllm[audio]==0.19.0'"
                )
            return "vllm"
        if self._device == "cpu" and is_qwenasr_rust_available():
            return "rust"
        raise DefaultServerErrorException(
            f"Qwen3-ASR is not available on device '{self._device}'. "
            "Supported backends are CUDA vLLM and CPU QwenASR Rust."
        )

    def _load_rust_backend(
        self,
        model_path: str,
        forced_aligner_path: Optional[str],
    ) -> QwenASRRustRuntime:
        logger.info("Loading Qwen3-ASR (QwenASR Rust): %s, device=%s", model_path, self._device)

        num_threads = 0 if settings.QWEN_RUST_CPU_WORKERS <= 1 else 1
        if settings.QWEN_RUST_CPU_WORKERS > 1:
            logger.info(
                "Using fixed QwenASR CPU thread count for multi-runtime mode: num_threads=%s workers=%s",
                num_threads,
                settings.QWEN_RUST_CPU_WORKERS,
            )
        self._rust_num_threads = num_threads
        self._rust_verbosity = 0
        return QwenASRRustRuntime(
            model_path=model_path,
            forced_aligner_path=forced_aligner_path,
            num_threads=num_threads,
            verbosity=0,
        )

    def _get_rust_batch_runtimes(self, worker_count: int) -> list[QwenASRRustRuntime]:
        if worker_count <= 1:
            return [self.model]

        if not self._rust_batch_runtimes:
            self._rust_batch_runtimes = [self.model]

        while len(self._rust_batch_runtimes) < worker_count:
            self._rust_batch_runtimes.append(
                QwenASRRustRuntime(
                    model_path=self.model_path,
                    forced_aligner_path=self._forced_aligner_path,
                    num_threads=self._rust_num_threads,
                    verbosity=self._rust_verbosity,
                )
            )

        return self._rust_batch_runtimes[:worker_count]

    def _get_rust_stage_concurrency(self, configured: int, segment_count: int) -> int:
        target = configured if configured > 0 else settings.QWEN_RUST_CPU_WORKERS
        return max(1, min(target, segment_count))

    def _get_rust_asr_concurrency(self, segment_count: int) -> int:
        return self._get_rust_stage_concurrency(
            settings.QWEN_RUST_ASR_CONCURRENCY,
            segment_count,
        )

    def _get_rust_align_concurrency(self, segment_count: int) -> int:
        return self._get_rust_stage_concurrency(
            settings.QWEN_RUST_ALIGN_CONCURRENCY,
            segment_count,
        )

    def _rust_transcribe_text_segment(
        self,
        runtime: QwenASRRustRuntime,
        seg: Any,
        hotwords: str,
        enable_punctuation: bool,
        enable_itn: bool,
        sample_rate: int,
    ) -> str:
        _ = (hotwords, enable_punctuation, enable_itn, sample_rate)
        return runtime.transcribe_file(seg.temp_file) or ""

    def _rust_align_word_tokens(
        self,
        runtime: QwenASRRustRuntime,
        seg: Any,
        text: str,
        language: Optional[str] = None,
    ) -> list[WordToken]:
        return [
            WordToken(
                text=str(item["text"]),
                start_time=round(float(item["start_ms"]) / 1000.0, 3),
                end_time=round(float(item["end_ms"]) / 1000.0, 3),
            )
            for item in runtime.align_transcript(
                audio_path=seg.temp_file,
                text=text,
                language=language,
            )
        ]

    def _run_rust_asr_stage(
        self,
        valid_segments: List[tuple[int, Any]],
        hotwords: str,
        enable_punctuation: bool,
        enable_itn: bool,
        sample_rate: int,
    ) -> dict[int, str]:
        if not valid_segments:
            return {}

        worker_count = self._get_rust_asr_concurrency(len(valid_segments))
        runtimes = self._get_rust_batch_runtimes(worker_count)
        output: dict[int, str] = {}
        for batch_start in range(0, len(valid_segments), worker_count):
            chunk = valid_segments[batch_start:batch_start + worker_count]
            chunk_runtimes = runtimes[:len(chunk)]
            with ThreadPoolExecutor(max_workers=len(chunk)) as executor:
                futures = [
                    executor.submit(
                        self._rust_transcribe_text_segment,
                        runtime,
                        seg,
                        hotwords,
                        enable_punctuation,
                        enable_itn,
                        sample_rate,
                    )
                    for runtime, (_idx, seg) in zip(chunk_runtimes, chunk)
                ]
                for (idx, _seg), future in zip(chunk, futures):
                    output[idx] = future.result()

        return output

    def _run_rust_align_stage(
        self,
        valid_segments: List[tuple[int, Any]],
        texts: dict[int, str],
        language: Optional[str] = None,
    ) -> dict[int, list[WordToken]]:
        if not valid_segments:
            return {}

        align_inputs = [(idx, seg, texts.get(idx, "")) for idx, seg in valid_segments if texts.get(idx, "").strip()]
        worker_count = self._get_rust_align_concurrency(len(valid_segments))
        runtimes = self._get_rust_batch_runtimes(worker_count)
        output: dict[int, list[WordToken]] = {}

        if not align_inputs:
            return output

        for batch_start in range(0, len(align_inputs), worker_count):
            chunk = align_inputs[batch_start:batch_start + worker_count]
            chunk_runtimes = runtimes[:len(chunk)]
            with ThreadPoolExecutor(max_workers=len(chunk)) as executor:
                futures = [
                    executor.submit(
                        self._rust_align_word_tokens,
                        runtime,
                        seg,
                        text,
                        language,
                    )
                    for runtime, (_idx, seg, text) in zip(chunk_runtimes, chunk)
                ]
                for (idx, _seg, _text), future in zip(chunk, futures):
                    output[idx] = future.result()

        return output

    def _warmup_forced_aligner(self) -> None:
        if not self._forced_aligner_path:
            return
        if self._backend == "vllm":
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
        if self._backend == "vllm":
            return self.model.transcribe_raw(
                audio_path=audio_path,
                context=hotwords or "",
                language=kwargs.get("language"),
                word_timestamps=kwargs.get("word_timestamps", False),
            )

        raise DefaultServerErrorException(
            f"Qwen3 backend={self._backend} does not support VAD transcription"
        )

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

        if self._backend == "rust":
            texts = self._run_rust_asr_stage(
                valid_segments=valid,
                hotwords=hotwords,
                enable_punctuation=enable_punctuation,
                enable_itn=enable_itn,
                sample_rate=sample_rate,
            )

            word_tokens_by_idx: dict[int, list[WordToken]] = {}
            if word_timestamps:
                word_tokens_by_idx = self._run_rust_align_stage(
                    valid_segments=valid,
                    texts=texts,
                )

            for idx, seg in valid:
                text = texts.get(idx, "")
                output[idx] = ASRSegmentResult(
                    text=text,
                    start_time=seg.start_sec,
                    end_time=seg.end_sec,
                    speaker_id=getattr(seg, "speaker_id", None),
                    word_tokens=word_tokens_by_idx.get(idx) or None,
                )
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

        raise DefaultServerErrorException(
            f"Qwen3 backend={self._backend} does not support batch transcription"
        )

    @_handle_asr_error("初始化流式状态")
    def init_streaming_state(self, context: str = "", language: Optional[str] = None, **kwargs) -> Qwen3StreamingState:
        if self._backend not in {"vllm", "rust"}:
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
        if self._backend not in {"vllm", "rust"}:
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

        streaming_state = self.model.feed_stream(pcm, state.internal_state)
        state.internal_state = streaming_state
        state.chunk_count = int(getattr(streaming_state, "chunk_id", state.chunk_count))
        state.last_text = str(getattr(streaming_state, "text", "") or "")
        state.last_language = str(getattr(streaming_state, "language", "") or "")
        return state

    @_handle_asr_error("结束流式识别")
    def finish_streaming_transcribe(self, state: Qwen3StreamingState) -> Qwen3StreamingState:
        if self._backend not in {"vllm", "rust"}:
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
