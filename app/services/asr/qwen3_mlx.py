# -*- coding: utf-8 -*-
"""MLX runtime adapter for Qwen3-ASR on Apple Silicon."""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import Any, Optional

from .engines import ASRRawResult, ASRSegmentResult, WordToken

logger = logging.getLogger(__name__)


def is_mlx_qwen_available() -> bool:
    """Return True when the MLX Qwen3-ASR runtime is installed."""
    return importlib.util.find_spec("mlx_qwen3_asr") is not None


class Qwen3MLXBackend:
    """Thin adapter around ``mlx_qwen3_asr`` session APIs."""

    def __init__(
        self,
        model_path: str,
        forced_aligner_path: Optional[str],
        max_new_tokens: int,
    ) -> None:
        try:
            module = importlib.import_module("mlx_qwen3_asr")
            ForcedAligner = getattr(module, "ForcedAligner")
            Session = getattr(module, "Session")
        except ImportError as exc:
            raise RuntimeError(
                "Apple Silicon Qwen3-ASR now requires mlx-qwen3-asr. "
                "Install it with: pip install 'mlx-qwen3-asr[aligner]'"
            ) from exc

        self._session = Session(model=model_path)
        self._forced_aligner_cls = ForcedAligner
        self._forced_aligner_path = forced_aligner_path
        self._forced_aligner: Any | None = None
        self._max_new_tokens = max_new_tokens

    def _get_forced_aligner(self) -> Any:
        if not self._forced_aligner_path:
            raise RuntimeError(
                "word_timestamps requires a configured forced aligner model "
                "for the MLX backend"
            )

        if self._forced_aligner is None:
            logger.info("Loading Qwen3 forced aligner via MLX: %s", self._forced_aligner_path)
            self._forced_aligner = self._forced_aligner_cls(
                model_path=self._forced_aligner_path,
            )
        return self._forced_aligner

    def ensure_forced_aligner_loaded(self) -> None:
        """Preload the forced aligner for first-request readiness."""
        if self._forced_aligner_path:
            self._get_forced_aligner()

    def transcribe_text(
        self,
        audio_path: str,
        context: str = "",
    ) -> str:
        result = self._session.transcribe(
            audio_path,
            context=context,
            return_timestamps=False,
            max_new_tokens=self._max_new_tokens,
        )
        return str(getattr(result, "text", "") or "")

    def transcribe_raw(
        self,
        audio_path: str,
        context: str = "",
        word_timestamps: bool = False,
    ) -> ASRRawResult:
        kwargs: dict[str, Any] = {
            "context": context,
            "return_timestamps": word_timestamps,
            "max_new_tokens": self._max_new_tokens,
        }
        if word_timestamps:
            kwargs["forced_aligner"] = self._get_forced_aligner()

        result = self._session.transcribe(audio_path, **kwargs)
        text = str(getattr(result, "text", "") or "")
        if not word_timestamps:
            return ASRRawResult(
                text=text,
                segments=[ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)] if text else [],
            )

        aligned_items = list(getattr(result, "segments", None) or [])
        word_tokens = [
            WordToken(
                text=str(item.get("text", "")),
                start_time=round(float(item.get("start", 0.0)), 3),
                end_time=round(float(item.get("end", 0.0)), 3),
            )
            for item in aligned_items
            if str(item.get("text", "")).strip()
        ]
        if not word_tokens:
            return ASRRawResult(
                text=text,
                segments=[ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)] if text else [],
            )

        return ASRRawResult(
            text=text,
            segments=[
                ASRSegmentResult(
                    text=text or " ".join(token.text for token in word_tokens).strip(),
                    start_time=word_tokens[0].start_time,
                    end_time=word_tokens[-1].end_time,
                    word_tokens=word_tokens,
                )
            ],
        )

    def init_streaming_state(
        self,
        *,
        context: str = "",
        language: Optional[str] = None,
        chunk_size_sec: float = 2.0,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        max_new_tokens: int = 32,
        max_context_sec: float = 30.0,
        finalization_mode: str = "accuracy",
        endpointing_mode: str = "fixed",
    ) -> Any:
        return self._session.init_streaming(
            context=context,
            language=language,
            chunk_size_sec=chunk_size_sec,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            max_new_tokens=max_new_tokens,
            max_context_sec=max_context_sec,
            finalization_mode=finalization_mode,
            endpointing_mode=endpointing_mode,
        )

    def feed_stream(self, pcm: Any, state: Any) -> Any:
        return self._session.feed_audio(pcm, state)

    def finish_stream(self, state: Any) -> Any:
        return self._session.finish_streaming(state)
