# -*- coding: utf-8 -*-
"""QwenASR Rust FFI wrapper for CPU inference."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import re
import sys
from pathlib import Path
from typing import Optional


from app.infrastructure import resolve_huggingface_snapshot_dir

logger = logging.getLogger(__name__)

_SHARED_LIBRARY: Optional[ctypes.CDLL] = None

_LANGUAGE_MAP = {
    "": "",
    "auto": "",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "yue": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
}


def _shared_library_filename() -> str:
    if sys.platform == "darwin":
        return "libqwen_asr.dylib"
    if sys.platform == "win32":
        return "qwen_asr.dll"
    return "libqwen_asr.so"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _candidate_library_paths() -> list[Path]:
    filename = _shared_library_filename()
    candidates: list[Path] = []

    env_path = (os.getenv("QWENASR_LIBRARY_PATH") or "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_dir():
            candidates.append(candidate / filename)
        else:
            candidates.append(candidate)

    repo_root = _repo_root()
    candidates.extend(
        [
            repo_root / "vendor" / "qwenasr" / "target" / "release" / filename,
            repo_root / "vendor" / "qwenasr" / "target" / "debug" / filename,
            Path("/opt/qwenasr/lib") / filename,
            Path("/usr/local/lib") / filename,
        ]
    )

    return candidates


def resolve_qwenasr_library_path() -> Optional[Path]:
    for candidate in _candidate_library_paths():
        if candidate.exists():
            return candidate.resolve()
    return None


def is_qwenasr_rust_available() -> bool:
    return resolve_qwenasr_library_path() is not None


def validate_qwenasr_cpu_features() -> None:
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        return

    flags = _read_linux_cpu_flags()
    if not flags:
        return

    missing = [flag for flag in ("avx2", "fma") if flag not in flags]
    if missing:
        raise RuntimeError(
            "QwenASR Rust backend requires x86_64 CPU features: avx2, fma. "
            f"Missing: {', '.join(missing)}. Use a newer CPU host or rebuild the "
            "Rust backend with scalar x86 kernels."
        )


def pick_cpu_qwen_model(all_available_models: list[str]) -> Optional[str]:
    for model_id in ["qwen3-asr-0.6b", "qwen3-asr-1.7b"]:
        if model_id in all_available_models:
            return model_id
    return None


def _read_linux_cpu_flags() -> set[str]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return set()

    flags: set[str] = set()
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() in {"flags", "features"}:
            flags.update(value.strip().lower().split())
            if flags:
                break
    return flags


def resolve_qwenasr_model_path(model_ref_or_path: str) -> Path:
    return resolve_huggingface_snapshot_dir(model_ref_or_path)


def _bind_ffi_signatures(lib: ctypes.CDLL) -> None:
    lib.qwen_asr_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.qwen_asr_load_model.restype = ctypes.c_void_p

    lib.qwen_asr_transcribe_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.qwen_asr_transcribe_file.restype = ctypes.c_void_p

    lib.qwen_asr_force_align_file.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    lib.qwen_asr_force_align_file.restype = ctypes.c_void_p

    lib.qwen_asr_set_language.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.qwen_asr_set_language.restype = ctypes.c_int

    lib.qwen_asr_free_string.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_free_string.restype = None

    lib.qwen_asr_free.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_free.restype = None


def load_qwenasr_library() -> ctypes.CDLL:
    global _SHARED_LIBRARY

    if _SHARED_LIBRARY is not None:
        return _SHARED_LIBRARY

    library_path = resolve_qwenasr_library_path()
    if library_path is None:
        searched = ", ".join(str(path) for path in _candidate_library_paths())
        raise FileNotFoundError(
            "QwenASR shared library not found. "
            f"Checked: {searched}"
        )

    logger.info("Loading QwenASR Rust library from %s", library_path)
    library = ctypes.CDLL(str(library_path))
    _bind_ffi_signatures(library)
    _SHARED_LIBRARY = library
    return library


def normalize_qwen_language(language: Optional[str]) -> str:
    if language is None:
        return ""
    return _LANGUAGE_MAP.get(language.strip().lower(), language.strip())


def guess_alignment_language(text: str, language: Optional[str] = None) -> str:
    normalized = normalize_qwen_language(language)
    if normalized:
        return normalized
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"
    if re.search(r"[\u3040-\u30ff]", text):
        return "Japanese"
    if re.search(r"[\uac00-\ud7af]", text):
        return "Korean"
    return "English"


def _decode_and_free_string(lib: ctypes.CDLL, raw_ptr: ctypes.c_void_p) -> Optional[str]:
    if not raw_ptr:
        return None

    try:
        value = ctypes.cast(raw_ptr, ctypes.c_char_p).value
        if value is None:
            return None
        return value.decode("utf-8")
    finally:
        lib.qwen_asr_free_string(raw_ptr)


class QwenASRRustBackend:
    """Thin Python wrapper around the QwenASR C API."""

    def __init__(self, model_path: str, num_threads: int = 0, verbosity: int = 0):
        validate_qwenasr_cpu_features()
        self._lib = load_qwenasr_library()
        self.model_dir = resolve_qwenasr_model_path(model_path)
        self._engine = self._lib.qwen_asr_load_model(
            str(self.model_dir).encode("utf-8"),
            num_threads,
            verbosity,
        )
        if not self._engine:
            raise RuntimeError(f"Failed to load QwenASR model from '{self.model_dir}'")

    def close(self) -> None:
        if self._engine:
            self._lib.qwen_asr_free(self._engine)
            self._engine = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _set_language(self, language: Optional[str]) -> None:
        normalized = normalize_qwen_language(language)
        status = self._lib.qwen_asr_set_language(
            self._engine,
            normalized.encode("utf-8"),
        )
        if status != 0 and normalized:
            logger.warning("QwenASR rejected language hint: %s", normalized)


    def transcribe_file(self, audio_path: str, language: Optional[str] = None) -> str:
        self._set_language(language)
        raw_ptr = self._lib.qwen_asr_transcribe_file(
            self._engine,
            audio_path.encode("utf-8"),
        )
        text = _decode_and_free_string(self._lib, raw_ptr)
        if text is None:
            raise RuntimeError(f"QwenASR failed to transcribe '{audio_path}'")
        return text

    def force_align_file(
        self,
        audio_path: str,
        text: str,
        language: Optional[str] = None,
    ) -> list[dict[str, float | str]]:
        normalized_language = normalize_qwen_language(language) or "English"
        raw_ptr = self._lib.qwen_asr_force_align_file(
            self._engine,
            audio_path.encode("utf-8"),
            text.encode("utf-8"),
            normalized_language.encode("utf-8"),
        )
        payload = _decode_and_free_string(self._lib, raw_ptr)
        if payload is None:
            raise RuntimeError(f"QwenASR failed to force align '{audio_path}'")

        items = json.loads(payload)
        if not isinstance(items, list):
            raise RuntimeError("QwenASR force alignment returned invalid payload")
        return [
            {
                "text": str(item.get("text", "")),
                "start_ms": float(item.get("start_ms", 0.0)),
                "end_ms": float(item.get("end_ms", 0.0)),
            }
            for item in items
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        ]


class QwenASRRustRuntime:
    """Higher-level Rust runtime bundle for offline ASR + aligner."""

    def __init__(
        self,
        model_path: str,
        *,
        forced_aligner_path: Optional[str] = None,
        num_threads: int = 0,
        verbosity: int = 0,
    ) -> None:
        self._asr = QwenASRRustBackend(
            model_path=model_path,
            num_threads=num_threads,
            verbosity=verbosity,
        )
        self._aligner: Optional[QwenASRRustBackend] = None
        if forced_aligner_path:
            self._aligner = QwenASRRustBackend(
                model_path=forced_aligner_path,
                num_threads=num_threads,
                verbosity=verbosity,
            )

    def transcribe_file(self, audio_path: str, language: Optional[str] = None) -> str:
        return self._asr.transcribe_file(audio_path=audio_path, language=language)

    def align_transcript(
        self,
        audio_path: str,
        text: str,
        language: Optional[str] = None,
    ) -> list[dict[str, float | str]]:
        transcript = text.strip()
        if not transcript:
            return []
        if self._aligner is None:
            raise RuntimeError("Forced alignment requires a configured aligner model")
        return self._aligner.force_align_file(
            audio_path=audio_path,
            text=transcript,
            language=guess_alignment_language(transcript, language),
        )
