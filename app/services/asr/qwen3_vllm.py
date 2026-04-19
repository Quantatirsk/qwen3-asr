# -*- coding: utf-8 -*-
"""Official vLLM adapter for CUDA Qwen3-ASR."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import librosa
import numpy as np

from .engines import ASRRawResult, ASRSegmentResult, WordToken

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


def _split_alignment_units(text: str) -> list[str]:
    compact = "".join(ch for ch in text if not ch.isspace())
    if not compact:
        return []
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in compact)
    if has_cjk and " " not in text:
        return [ch for ch in compact]
    return re.findall(r"\S+", text)


@dataclass
class _GeneratedTranscript:
    text: str
    language: str


@dataclass
class VLLMRealtimeState:
    prompt_raw: str
    language: str
    chunk_size_sec: float
    unfixed_chunk_num: int
    unfixed_token_num: int
    max_new_tokens: int
    chunk_id: int = 0
    text: str = ""
    raw_decoded: str = ""
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    audio_accum: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))


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
                "CUDA Qwen3-ASR now requires official vLLM nightly. "
                "Install it with: pip install --pre 'vllm[audio]'"
            ) from exc

        self._llm_cls = getattr(vllm_module, "LLM")
        self._sampling_params_cls = getattr(vllm_module, "SamplingParams")
        self._tokenizer = getattr(transformers_module, "AutoTokenizer").from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self._llm = self._llm_cls(**llm_kwargs)
        self._sampling_params = self._sampling_params_cls(
            temperature=0.01,
            max_tokens=max_new_tokens,
        )
        self._max_inference_batch_size = max_inference_batch_size
        self._forced_aligner_path = forced_aligner_path
        self._forced_aligner: Any | None = None
        self._timestamp_token_id: int | None = None
        self._timestamp_segment_time: float | None = None

    def _get_forced_aligner(self) -> Any:
        if not self._forced_aligner_path:
            raise RuntimeError("word_timestamps requires a configured forced aligner model")

        if self._forced_aligner is None:
            logger.info("Loading Qwen3 forced aligner via official vLLM: %s", self._forced_aligner_path)
            self._forced_aligner = self._llm_cls(
                model=self._forced_aligner_path,
                runner="pooling",
                enforce_eager=True,
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
    ) -> str:
        transcript = self._run_generate([(_load_audio(audio_path), context, language)])[0]
        return transcript.text

    def transcribe_raw(
        self,
        audio_path: str,
        context: str = "",
        language: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> ASRRawResult:
        audio = _load_audio(audio_path)
        transcript = self._run_generate([(audio, context, language)])[0]
        if not word_timestamps:
            return ASRRawResult(
                text=transcript.text,
                segments=[ASRSegmentResult(text=transcript.text, start_time=0.0, end_time=0.0)] if transcript.text else [],
            )

        aligned = self.align_transcript(audio_path=audio_path, text=transcript.text, language=language, audio=audio)
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
                text=transcript.text,
                segments=[ASRSegmentResult(text=transcript.text, start_time=0.0, end_time=0.0)] if transcript.text else [],
            )
        return ASRRawResult(
            text=transcript.text,
            segments=[
                ASRSegmentResult(
                    text=transcript.text,
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
    ) -> list[ASRSegmentResult]:
        audios = [_load_audio(path) for path in audio_paths]
        results: list[ASRSegmentResult] = []
        for start in range(0, len(audios), self._max_inference_batch_size):
            chunk = audios[start:start + self._max_inference_batch_size]
            transcripts = self._run_generate([(audio, context, language) for audio in chunk])
            for audio_path, audio, transcript in zip(audio_paths[start:start + len(chunk)], chunk, transcripts):
                if not word_timestamps:
                    results.append(ASRSegmentResult(text=transcript.text, start_time=0.0, end_time=0.0))
                    continue
                aligned = self.align_transcript(
                    audio_path=audio_path,
                    text=transcript.text,
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
                        text=transcript.text,
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
        tokens = _split_alignment_units(text)
        if not tokens:
            return []

        aligner = self._get_forced_aligner()
        prompt = _build_alignment_prompt(tokens)
        audio_array = audio if audio is not None else _load_audio(audio_path)
        outputs = aligner.encode(
            [{"prompt": prompt, "multi_modal_data": {"audio": audio_array}}],
            pooling_task="token_classify",
        )
        output = outputs[0]
        logits = output.outputs.data
        predictions = logits.argmax(dim=-1) if hasattr(logits, "argmax") else np.argmax(logits, axis=-1)
        ts_predictions = [
            float(pred.item() if hasattr(pred, "item") else pred) * float(self._timestamp_segment_time or 0.0)
            for tid, pred in zip(output.prompt_token_ids, predictions)
            if int(tid) == int(self._timestamp_token_id or -1)
        ]

        aligned: list[dict[str, float | str]] = []
        for index, token in enumerate(tokens):
            start_ms = ts_predictions[index * 2]
            end_ms = ts_predictions[index * 2 + 1]
            aligned.append({"text": token, "start_ms": start_ms, "end_ms": end_ms})
        return aligned

    def init_streaming_state(
        self,
        *,
        context: str = "",
        language: Optional[str] = None,
        chunk_size_sec: float = 2.0,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        max_new_tokens: int = 32,
    ) -> VLLMRealtimeState:
        normalized_language = _normalize_language_name(language) or ""
        return VLLMRealtimeState(
            prompt_raw=_build_chat_prompt(context=context, language=normalized_language or None),
            language=normalized_language,
            chunk_size_sec=chunk_size_sec,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            max_new_tokens=max_new_tokens,
            audio_buffer=np.array([], dtype=np.float32),
            audio_accum=np.array([], dtype=np.float32),
        )

    def _decode_stream(self, state: VLLMRealtimeState) -> VLLMRealtimeState:
        prefix = ""
        if state.chunk_id >= state.unfixed_chunk_num and state.raw_decoded:
            token_ids = self._tokenizer.encode(state.raw_decoded, add_special_tokens=False)
            rollback = token_ids[-state.unfixed_token_num:] if state.unfixed_token_num > 0 else []
            if rollback:
                prefix = self._tokenizer.decode(rollback, skip_special_tokens=False).replace("\ufffd", "")

        output = self._llm.generate(
            [
                {
                    "prompt": state.prompt_raw + prefix,
                    "multi_modal_data": {"audio": [state.audio_accum]},
                }
            ],
            sampling_params=self._sampling_params_cls(
                temperature=0.01,
                max_tokens=state.max_new_tokens,
            ),
            use_tqdm=False,
        )[0]
        generated = str(output.outputs[0].text if output.outputs else "")
        parsed_language, parsed_text = _parse_asr_output(prefix + generated, state.language or None)
        state.raw_decoded = prefix + generated
        state.text = parsed_text
        state.language = parsed_language or state.language
        state.chunk_id += 1
        return state

    def feed_stream(self, pcm: np.ndarray, state: VLLMRealtimeState) -> VLLMRealtimeState:
        state.audio_buffer = np.concatenate([state.audio_buffer, pcm.astype(np.float32)])
        segment_size = int(max(state.chunk_size_sec, 0.1) * _DEFAULT_SAMPLE_RATE)
        while len(state.audio_buffer) >= segment_size:
            segment = state.audio_buffer[:segment_size].copy()
            state.audio_buffer = state.audio_buffer[segment_size:]
            state.audio_accum = np.concatenate([state.audio_accum, segment])
            state = self._decode_stream(state)
        return state

    def finish_stream(self, state: VLLMRealtimeState) -> VLLMRealtimeState:
        if len(state.audio_buffer) > 0:
            state.audio_accum = np.concatenate([state.audio_accum, state.audio_buffer])
            state.audio_buffer = np.array([], dtype=np.float32)
            state = self._decode_stream(state)
        elif state.chunk_id == 0 and len(state.audio_accum) > 0:
            state = self._decode_stream(state)
        return state
