# -*- coding: utf-8 -*-
"""Official vLLM adapter for CUDA Qwen3-ASR."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import librosa
import numpy as np

from app.infrastructure import resolve_huggingface_snapshot_dir
from app.utils.text_processing import normalize_asr_text

from .engines import ASRRawResult, ASRSegmentResult, WordToken
from .qwen3_alignment import repair_timestamps, split_alignment_units

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_RATE = 16000
_LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",
    "cn": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "ja": "Japanese",
    "jp": "Japanese",
    "ko": "Korean",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
}


def is_vllm_available() -> bool:
    """Return True when the official vLLM runtime is installed."""
    return importlib.util.find_spec("vllm") is not None


def _normalize_language_name(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    normalized = language.strip()
    if not normalized:
        return None
    alias = _LANGUAGE_ALIASES.get(normalized.lower())
    if alias:
        return alias
    if " " in normalized:
        return " ".join(part.capitalize() for part in normalized.split())
    return normalized.capitalize()


def _load_audio(audio_path: str) -> np.ndarray:
    audio, _sample_rate = librosa.load(audio_path, sr=_DEFAULT_SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def _build_chat_prompt(context: str = "", language: Optional[str] = None) -> str:
    instructions: list[str] = []
    if language:
        instructions.append(f"Transcribe the speech in {language}.")
    else:
        instructions.append("Transcribe the speech accurately.")
    if context.strip():
        instructions.append(f"Use this context when resolving named entities: {context.strip()}")
    system_text = " ".join(instructions).strip()
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _build_alignment_prompt(tokens: list[str]) -> str:
    body = "<timestamp><timestamp>".join(tokens) + "<timestamp><timestamp>"
    return f"<|audio_start|><|audio_pad|><|audio_end|>{body}"


def _parse_asr_output(raw_text: str, language: Optional[str]) -> tuple[str, str]:
    text = (raw_text or "").strip()
    if "<asr_text>" in text:
        left, right = text.split("<asr_text>", 1)
        detected = left.strip()
        if detected.lower().startswith("language "):
            detected = detected[9:].strip()
        return detected or (language or ""), right.strip()
    return (language or ""), text


def _resolve_forced_aligner_gpu_memory_utilization(primary_utilization: float) -> float:
    override = (os.getenv("QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION") or "").strip()
    if override:
        try:
            value = float(override)
            if 0.0 < value <= 1.0:
                return value
        except ValueError:
            logger.warning(
                "Invalid QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION=%s, ignoring override",
                override,
            )

    return primary_utilization


def _shared_gpu_engine_options() -> dict[str, Any]:
    options: dict[str, Any] = {}
    max_seqs = os.getenv("QWEN_VLLM_MAX_NUM_SEQS")
    if max_seqs:
        count = int(max_seqs)
        if count < 1:
            raise ValueError("QWEN_VLLM_MAX_NUM_SEQS must be positive")
        options.update(max_num_seqs=count, limit_mm_per_prompt={"audio": 1})
    if os.getenv("QWEN_VLLM_ENFORCE_EAGER") == "1":
        options["enforce_eager"] = True
    return options


@dataclass
class _GeneratedTranscript:
    text: str
    language: str


class Qwen3VLLMBackend:
    """Thin adapter over official vLLM APIs for Qwen3-ASR."""

    def __init__(
        self,
        model_path: str,
        forced_aligner_path: Optional[str],
        gpu_memory_utilization: float,
        max_inference_batch_size: int,
        max_new_tokens: int,
        max_model_len: Optional[int] = None,
    ) -> None:
        try:
            vllm_module = importlib.import_module("vllm")
            transformers_module = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "CUDA Qwen3-ASR now requires official vLLM with Qwen3 forced aligner support. "
                "Install it with: pip install 'vllm[audio]==0.19.0'"
            ) from exc

        local_model_path = str(resolve_huggingface_snapshot_dir(model_path))
        local_forced_aligner_path = (
            str(resolve_huggingface_snapshot_dir(forced_aligner_path))
            if forced_aligner_path
            else None
        )

        self._llm_cls = getattr(vllm_module, "LLM")
        self._sampling_params_cls = getattr(vllm_module, "SamplingParams")
        self._tokenizer = getattr(transformers_module, "AutoTokenizer").from_pretrained(
            local_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        llm_kwargs: dict[str, Any] = {
            "model": local_model_path,
            "gpu_memory_utilization": gpu_memory_utilization,
            **_shared_gpu_engine_options(),
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self._llm = self._llm_cls(**llm_kwargs)
        self._sampling_params = self._sampling_params_cls(
            temperature=0.01,
            max_tokens=max_new_tokens,
        )
        self._max_inference_batch_size = max_inference_batch_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._forced_aligner_path = local_forced_aligner_path
        self._forced_aligner: Any | None = None
        self._timestamp_token_id: int | None = None
        self._timestamp_segment_time: float | None = None

    def _get_forced_aligner_gpu_memory_utilization(self) -> float:
        configured = _resolve_forced_aligner_gpu_memory_utilization(self._gpu_memory_utilization)
        logger.info(
            "Resolved forced aligner gpu_memory_utilization=%s (primary=%s)",
            configured,
            self._gpu_memory_utilization,
        )
        return configured

    def _get_forced_aligner(self) -> Any:
        if not self._forced_aligner_path:
            raise RuntimeError("word_timestamps requires a configured forced aligner model")

        if self._forced_aligner is None:
            forced_aligner_gpu_memory_utilization = self._get_forced_aligner_gpu_memory_utilization()
            logger.info(
                "Loading Qwen3 forced aligner via official vLLM: %s (gpu_memory_utilization=%s)",
                self._forced_aligner_path,
                forced_aligner_gpu_memory_utilization,
            )
            aligner_options = _shared_gpu_engine_options()
            aligner_options["enforce_eager"] = True
            self._forced_aligner = self._llm_cls(
                model=self._forced_aligner_path,
                runner="pooling",
                **aligner_options,
                gpu_memory_utilization=forced_aligner_gpu_memory_utilization,
                hf_overrides={
                    "architectures": ["Qwen3ASRForcedAlignerForTokenClassification"],
                },
            )
            llm_engine = getattr(self._forced_aligner, "llm_engine", None)
            if llm_engine is None:
                raise RuntimeError("Forced aligner did not expose a vLLM engine instance")
            config = llm_engine.vllm_config.model_config.hf_config
            self._timestamp_token_id = int(config.timestamp_token_id)
            self._timestamp_segment_time = float(config.timestamp_segment_time)

        return self._forced_aligner

    def ensure_forced_aligner_loaded(self) -> None:
        if self._forced_aligner_path:
            self._get_forced_aligner()

    def _run_generate(
        self,
        audio_items: list[tuple[np.ndarray, str, Optional[str]]],
    ) -> list[_GeneratedTranscript]:
        prompts: list[dict[str, Any]] = []
        for audio, context, language in audio_items:
            prompts.append(
                {
                    "prompt": _build_chat_prompt(context=context, language=_normalize_language_name(language)),
                    "multi_modal_data": {"audio": [audio]},
                }
            )

        outputs = self._llm.generate(
            prompts,
            sampling_params=self._sampling_params,
            use_tqdm=False,
        )

        transcripts: list[_GeneratedTranscript] = []
        for output, (_audio, _context, language) in zip(outputs, audio_items):
            raw_text = str(output.outputs[0].text if output.outputs else "")
            parsed_language, parsed_text = _parse_asr_output(raw_text, _normalize_language_name(language))
            transcripts.append(_GeneratedTranscript(text=parsed_text, language=parsed_language))
        return transcripts

    def transcribe_text(
        self,
        audio_path: str,
        context: str = "",
        language: Optional[str] = None,
        enable_itn: bool = False,
    ) -> str:
        transcript = self._run_generate([(_load_audio(audio_path), context, language)])[0]
        return normalize_asr_text(transcript.text, enable_itn=enable_itn)

    def transcribe_raw(
        self,
        audio_path: str,
        context: str = "",
        language: Optional[str] = None,
        word_timestamps: bool = False,
        enable_itn: bool = False,
    ) -> ASRRawResult:
        audio = _load_audio(audio_path)
        transcript = self._run_generate([(audio, context, language)])[0]
        text = normalize_asr_text(transcript.text, enable_itn=enable_itn)
        if not word_timestamps:
            return ASRRawResult(
                text=text,
                segments=[ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)] if text else [],
            )

        aligned = self.align_transcript(audio_path=audio_path, text=text, language=language, audio=audio)
        word_tokens = [
            WordToken(
                text=str(item["text"]),
                start_time=round(float(item["start_ms"]) / 1000.0, 3),
                end_time=round(float(item["end_ms"]) / 1000.0, 3),
            )
            for item in aligned
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
                    text=text,
                    start_time=word_tokens[0].start_time,
                    end_time=word_tokens[-1].end_time,
                    word_tokens=word_tokens,
                )
            ],
        )

    def transcribe_batch(
        self,
        audio_paths: list[str],
        context: str = "",
        language: Optional[str] = None,
        word_timestamps: bool = False,
        enable_itn: bool = False,
    ) -> list[ASRSegmentResult]:
        audios = [_load_audio(path) for path in audio_paths]
        results: list[ASRSegmentResult] = []
        for start in range(0, len(audios), self._max_inference_batch_size):
            chunk = audios[start:start + self._max_inference_batch_size]
            stage_started = time.monotonic()
            logger.info("Qwen GPU ASR batch started: segments=%s audio_seconds=%.2f", len(chunk), sum(len(a) for a in chunk) / _DEFAULT_SAMPLE_RATE)
            transcripts = self._run_generate([(audio, context, language) for audio in chunk])
            logger.info("Qwen GPU ASR batch finished: segments=%s elapsed_seconds=%.2f", len(chunk), time.monotonic() - stage_started)
            for audio_path, audio, transcript in zip(audio_paths[start:start + len(chunk)], chunk, transcripts):
                text = normalize_asr_text(transcript.text, enable_itn=enable_itn)
                if not word_timestamps:
                    results.append(ASRSegmentResult(text=text, start_time=0.0, end_time=0.0))
                    continue
                aligned = self.align_transcript(
                    audio_path=audio_path,
                    text=text,
                    language=language,
                    audio=audio,
                )
                word_tokens = [
                    WordToken(
                        text=str(item["text"]),
                        start_time=round(float(item["start_ms"]) / 1000.0, 3),
                        end_time=round(float(item["end_ms"]) / 1000.0, 3),
                    )
                    for item in aligned
                ]
                results.append(
                    ASRSegmentResult(
                        text=text,
                        start_time=word_tokens[0].start_time if word_tokens else 0.0,
                        end_time=word_tokens[-1].end_time if word_tokens else 0.0,
                        word_tokens=word_tokens or None,
                    )
                )
        return results

    def align_transcript(
        self,
        audio_path: str,
        text: str,
        language: Optional[str] = None,
        audio: Optional[np.ndarray] = None,
    ) -> list[dict[str, float | str]]:
        tokens = split_alignment_units(text)
        if not tokens:
            return []

        aligner = self._get_forced_aligner()
        stage_started = time.monotonic()
        logger.info("Qwen GPU alignment started: file=%s units=%s", os.path.basename(audio_path), len(tokens))
        prompt = _build_alignment_prompt(tokens)
        audio_array = audio if audio is not None else _load_audio(audio_path)
        outputs = aligner.encode(
            [{"prompt": prompt, "multi_modal_data": {"audio": audio_array}}],
            pooling_task="token_classify",
        )
        output = outputs[0]
        logits = output.outputs.data
        predictions = logits.argmax(-1) if hasattr(logits, "argmax") else np.argmax(logits, axis=-1)
        ts_predictions = [
            float(pred.item() if hasattr(pred, "item") else pred) * float(self._timestamp_segment_time or 0.0)
            for tid, pred in zip(output.prompt_token_ids, predictions)
            if int(tid) == int(self._timestamp_token_id or -1)
        ]

        expected_timestamps = len(tokens) * 2
        if len(ts_predictions) != expected_timestamps:
            raise RuntimeError(
                "Forced aligner timestamp count mismatch: "
                f"expected={expected_timestamps}, got={len(ts_predictions)}, tokens={len(tokens)}"
            )

        fixed_timestamps = repair_timestamps(
            ts_predictions, len(audio_array) * 1000.0 / _DEFAULT_SAMPLE_RATE
        )
        repaired = sum(a != b for a, b in zip(ts_predictions, fixed_timestamps))
        if repaired:
            logger.warning(
                "Repaired forced alignment timestamps: file=%s changed=%s total=%s",
                os.path.basename(audio_path), repaired, expected_timestamps,
            )
        aligned: list[dict[str, float | str]] = []
        for index, token in enumerate(tokens):
            start_ms = fixed_timestamps[index * 2]
            end_ms = fixed_timestamps[index * 2 + 1]
            aligned.append({"text": token, "start_ms": start_ms, "end_ms": end_ms})
        logger.info("Qwen GPU alignment finished: units=%s elapsed_seconds=%.2f", len(aligned), time.monotonic() - stage_started)
        return aligned
