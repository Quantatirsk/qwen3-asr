"""Typed Rust FFI for the R2T2 decoder and independent forced aligner."""

from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download

from app.core.config import settings
from app.infrastructure import resolve_huggingface_snapshot_dir
from app.services.realtime.protocol import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    OFFLINE_MAX_SAMPLES,
    OFFLINE_TAIL_SAMPLES,
)

from .qwen3_alignment import repair_timestamps, split_alignment_units

_INT_MAX = 2**31 - 1
_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_INT_PTR = ctypes.POINTER(ctypes.c_int32)


@dataclass(frozen=True)
class NativeGeneration:
    token_ids: list[int]
    finish_reason: Literal["stop", "length"]


def _load_library() -> ctypes.CDLL:
    suffix = "dylib" if sys.platform == "darwin" else "so"
    default = (
        Path(__file__).resolve().parents[3]
        / "vendor/qwenasr/target/release"
        / f"libqwen_asr.{suffix}"
    )
    path = Path(os.getenv("R2T2_CPU_LIBRARY_PATH", str(default))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Rust library not found: {path}; run scripts/build-rust.sh or set "
            "R2T2_CPU_LIBRARY_PATH"
        )
    lib = ctypes.CDLL(str(path))
    signatures = {
        "qwen_asr_load_model": (
            [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32],
            ctypes.c_void_p,
        ),
        "qwen_asr_generate_pcm": (
            [
                ctypes.c_void_p,
                _FLOAT_PTR,
                ctypes.c_int32,
                _INT_PTR,
                ctypes.c_int32,
                _INT_PTR,
                ctypes.c_int32,
                ctypes.c_int32,
            ],
            ctypes.c_void_p,
        ),
        "qwen_asr_force_align_file": (
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p],
            ctypes.c_void_p,
        ),
        "qwen_asr_free_string": ([ctypes.c_void_p], None),
        "qwen_asr_free": ([ctypes.c_void_p], None),
    }
    try:
        for name, (arguments, result) in signatures.items():
            function = getattr(lib, name)
            function.argtypes = arguments
            function.restype = result
    except AttributeError as exc:
        raise RuntimeError(
            "Rust library ABI is outdated; run scripts/build-rust.sh"
        ) from exc
    return lib


class RustBackend:
    def __init__(self, model_path: str | None = None) -> None:
        configured = model_path or os.getenv("R2T2_MODEL_PATH")
        self.model_path = str(
            resolve_huggingface_snapshot_dir(configured)
            if configured
            else snapshot_download(
                MODEL_REPOSITORY, revision=MODEL_REVISION, local_files_only=True
            )
        )
        self._lib = _load_library()
        self._lock = threading.Lock()
        self._handle = self._lib.qwen_asr_load_model(
            self.model_path.encode(), settings.R2T2_CPU_THREADS, 0
        )
        if not self._handle:
            raise RuntimeError(f"Rust could not load model: {self.model_path}")

    def _read_json(self, pointer: int) -> object:
        if not pointer:
            raise RuntimeError("Rust inference failed; see native stderr")
        try:
            return json.loads(ctypes.string_at(pointer).decode("utf-8"))
        finally:
            self._lib.qwen_asr_free_string(pointer)

    def generate_pcm(
        self,
        audio: np.ndarray,
        before_ids: list[int],
        after_ids: list[int],
        max_tokens: int,
    ) -> NativeGeneration:
        samples = np.ascontiguousarray(audio, dtype=np.float32)
        if (
            samples.ndim != 1
            or not 0 < samples.size <= OFFLINE_MAX_SAMPLES + OFFLINE_TAIL_SAMPLES
        ):
            raise ValueError("Audio must be nonempty mono 16 kHz PCM")
        if not np.isfinite(samples).all():
            raise ValueError("Audio samples must be finite")
        if type(max_tokens) is not int or not 0 < max_tokens <= 4096:
            raise ValueError("max_tokens must be an integer between 1 and 4096")
        for ids in (before_ids, after_ids):
            if len(ids) > _INT_MAX or any(
                type(token) is not int or not 0 <= token <= _INT_MAX for token in ids
            ):
                raise ValueError("Prompt token IDs must be nonnegative int32 values")
        before = np.asarray(before_ids, dtype=np.int32)
        after = np.asarray(after_ids, dtype=np.int32)
        with self._lock:
            if not self._handle:
                raise RuntimeError("Rust model is closed")
            result = self._read_json(
                self._lib.qwen_asr_generate_pcm(
                    self._handle,
                    samples.ctypes.data_as(_FLOAT_PTR),
                    samples.size,
                    before.ctypes.data_as(_INT_PTR),
                    before.size,
                    after.ctypes.data_as(_INT_PTR),
                    after.size,
                    max_tokens,
                )
            )
        if not isinstance(result, dict) or result.get("finish_reason") not in (
            "stop",
            "length",
        ):
            raise RuntimeError("Invalid Rust generation result")
        ids = result.get("token_ids")
        if not isinstance(ids, list) or any(
            type(token) is not int or token < 0 for token in ids
        ):
            raise RuntimeError("Invalid Rust generation token IDs")
        return NativeGeneration(ids, result["finish_reason"])

    def close(self) -> None:
        with self._lock:
            if self._handle:
                self._lib.qwen_asr_free(self._handle)
                self._handle = None


class RustForcedAligner(RustBackend):
    def align_transcript(
        self, audio_path: str, text: str, audio: np.ndarray | None = None
    ) -> list[dict[str, float | str]]:
        units = split_alignment_units(text)
        if not units:
            return []
        # Native language only selects splitting; whitespace preserves shared units.
        with self._lock:
            if not self._handle:
                raise RuntimeError("Rust model is closed")
            result = self._read_json(
                self._lib.qwen_asr_force_align_file(
                    self._handle,
                    os.fsencode(audio_path),
                    " ".join(units).encode(),
                    b"English",
                )
            )
        if not isinstance(result, list) or not result:
            raise RuntimeError("Rust forced aligner returned no timestamps")
        duration_ms = (
            len(audio) * 1000 / 16000
            if audio is not None
            else sf.info(audio_path).duration * 1000
        )
        try:
            timestamps = [
                float(item[key]) for item in result for key in ("start_ms", "end_ms")
            ]
            if any(
                not isinstance(item["text"], str) or not item["text"] for item in result
            ):
                raise ValueError("Empty alignment unit")
            if [item["text"] for item in result] != units:
                raise ValueError("Forced alignment units differ from transcript")
            fixed = repair_timestamps(timestamps, duration_ms)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Invalid Rust alignment result") from exc
        return [
            {
                "text": item["text"],
                "start_ms": fixed[index * 2],
                "end_ms": fixed[index * 2 + 1],
            }
            for index, item in enumerate(result)
        ]
