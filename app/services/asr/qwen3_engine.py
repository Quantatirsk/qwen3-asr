"""Ascend-only Qwen3-ASR offline engine."""

from __future__ import annotations

import os
from typing import Any

from app.core.config import settings
from app.core.exceptions import DefaultServerErrorException
from app.services.asr.engines.base import BaseASREngine
from app.services.asr.qwen3_remote_vllm import Qwen3RemoteVLLMBackend
from app.services.asr.results import ASRSegmentResult

class Qwen3ASREngine(BaseASREngine):
    """Offline Qwen3-ASR backed exclusively by remote Ascend vLLM."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        max_inference_batch_size: int = 16,
        **_kwargs: Any,
    ) -> None:
        if not settings.QWEN_VLLM_BASE_URL:
            raise DefaultServerErrorException(
                "QWEN_VLLM_BASE_URL is required for the Ascend-only runtime"
            )
        self._model_id = "qwen3-asr-1.7b"
        self.model_path = model_path
        self.model = Qwen3RemoteVLLMBackend(
            base_url=settings.QWEN_VLLM_BASE_URL,
            model=settings.QWEN_VLLM_SERVED_MODEL,
            api_key=settings.QWEN_VLLM_API_KEY,
            timeout_sec=settings.QWEN_VLLM_TIMEOUT_SEC,
            max_inference_batch_size=max_inference_batch_size,
        )
        self.model.ensure_ready()

    def transcribe_file(
        self,
        audio_path: str,
        enable_itn: bool = True,
        sample_rate: int = 16000,
    ) -> str:
        _ = sample_rate
        return self.model.transcribe_text(
            audio_path,
            enable_itn=enable_itn,
        )

    def _transcribe_batch(
        self,
        segments: list[Any],
        enable_itn: bool = True,
        sample_rate: int = 16000,
    ) -> list[ASRSegmentResult]:
        _ = sample_rate
        output = [ASRSegmentResult(text="", start_time=0.0, end_time=0.0) for _ in segments]
        valid = [
            (index, segment)
            for index, segment in enumerate(segments)
            if getattr(segment, "temp_file", None)
            and os.path.exists(segment.temp_file)
        ]
        if not valid:
            return output

        transcribed = self.model.transcribe_batch(
            [segment.temp_file for _, segment in valid],
            enable_itn=enable_itn,
        )
        for (index, _segment), result in zip(valid, transcribed):
            output[index] = result
        return output

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        from app.core.device import detect_device

        return detect_device(settings.DEVICE)

    def is_model_loaded(self) -> bool:
        return self.model.is_ready()
