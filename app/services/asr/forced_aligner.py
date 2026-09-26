"""Independent Qwen forced alignment; recognition runs in the shared R2T2 service."""

from __future__ import annotations

import importlib
import logging
import os
import time

import librosa
import numpy as np

from app.infrastructure import resolve_huggingface_snapshot_dir

from .qwen3_alignment import repair_timestamps, split_alignment_units

logger = logging.getLogger(__name__)
_DEFAULT_SAMPLE_RATE = 16000


def _load_audio(audio_path: str) -> np.ndarray:
    audio, _sample_rate = librosa.load(audio_path, sr=_DEFAULT_SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def _build_alignment_prompt(tokens: list[str]) -> str:
    body = "<timestamp><timestamp>".join(tokens) + "<timestamp><timestamp>"
    return f"<|audio_start|><|audio_pad|><|audio_end|>{body}"


def _gpu_memory_utilization() -> float:
    value = float(os.getenv("FORCED_ALIGNER_GPU_MEMORY_UTILIZATION", "0.15"))
    if not 0.0 < value <= 1.0:
        raise ValueError(
            "FORCED_ALIGNER_GPU_MEMORY_UTILIZATION must be greater than zero and at most one"
        )
    return value


class ForcedAligner:
    def __init__(self, model_path: str) -> None:
        try:
            vllm_module = importlib.import_module("vllm")
        except ImportError as exc:
            raise RuntimeError(
                "CUDA forced alignment requires vllm[audio]==0.30.0"
            ) from exc
        model_path = str(resolve_huggingface_snapshot_dir(model_path))
        memory = _gpu_memory_utilization()
        logger.info(
            "Loading Qwen3 forced aligner: %s (gpu_memory_utilization=%s)",
            model_path,
            memory,
        )
        self._forced_aligner = vllm_module.LLM(
            model=model_path,
            runner="pooling",
            max_num_seqs=1,
            limit_mm_per_prompt={"audio": 1},
            enforce_eager=True,
            gpu_memory_utilization=memory,
            hf_overrides={
                "architectures": ["Qwen3ASRForcedAlignerForTokenClassification"],
            },
        )
        config = self._forced_aligner.llm_engine.vllm_config.model_config.hf_config
        self._timestamp_token_id = int(config.timestamp_token_id)
        self._timestamp_segment_time = float(config.timestamp_segment_time)

    def align_transcript(
        self,
        audio_path: str,
        text: str,
        audio: np.ndarray | None = None,
    ) -> list[dict[str, float | str]]:
        tokens = split_alignment_units(text)
        if not tokens:
            return []

        aligner = self._forced_aligner
        stage_started = time.monotonic()
        logger.info(
            "Qwen GPU alignment started: file=%s units=%s",
            os.path.basename(audio_path),
            len(tokens),
        )
        prompt = _build_alignment_prompt(tokens)
        audio_array = audio if audio is not None else _load_audio(audio_path)
        outputs = aligner.encode(
            [{"prompt": prompt, "multi_modal_data": {"audio": audio_array}}],
            pooling_task="token_classify",
        )
        output = outputs[0]
        logits = output.outputs.data
        predictions = (
            logits.argmax(-1)
            if hasattr(logits, "argmax")
            else np.argmax(logits, axis=-1)
        )
        ts_predictions = [
            float(pred.item() if hasattr(pred, "item") else pred)
            * float(self._timestamp_segment_time or 0.0)
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
                os.path.basename(audio_path),
                repaired,
                expected_timestamps,
            )
        aligned: list[dict[str, float | str]] = []
        for index, token in enumerate(tokens):
            start_ms = fixed_timestamps[index * 2]
            end_ms = fixed_timestamps[index * 2 + 1]
            aligned.append({"text": token, "start_ms": start_ms, "end_ms": end_ms})
        logger.info(
            "Qwen GPU alignment finished: units=%s elapsed_seconds=%.2f",
            len(aligned),
            time.monotonic() - stage_started,
        )
        return aligned
