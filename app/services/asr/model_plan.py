# -*- coding: utf-8 -*-
"""Single-source deployment model planning."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Optional

from app.core.config import settings

QWEN_MODEL_OVERRIDE_ENV = "QWEN3_ASR_MODEL"
_QWEN_MODEL_ALIASES = {
    "qwen3-asr-0.6b": "qwen3-asr-0.6b",
    "0.6b": "qwen3-asr-0.6b",
    "0.6": "qwen3-asr-0.6b",
    "qwen/qwen3-asr-0.6b": "qwen3-asr-0.6b",
    "qwen3-asr-1.7b": "qwen3-asr-1.7b",
    "1.7b": "qwen3-asr-1.7b",
    "1.7": "qwen3-asr-1.7b",
    "qwen/qwen3-asr-1.7b": "qwen3-asr-1.7b",
}


def load_supported_model_ids() -> list[str]:
    """Load declared model ids from models.json."""
    models_file = Path(settings.models_config_path)
    if not models_file.exists():
        return []

    with open(models_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    return list(config.get("models", {}).keys())


def get_qwen_model_override() -> Optional[str]:
    """Return the explicit Qwen model override from the environment."""
    raw_value = (os.getenv(QWEN_MODEL_OVERRIDE_ENV) or "").strip()
    if not raw_value:
        return None

    normalized = _QWEN_MODEL_ALIASES.get(raw_value.lower(), raw_value)
    return normalized


def detect_qwen_model_by_vram(all_model_ids: Optional[list[str]] = None) -> Optional[str]:
    """Pick the active Qwen model for the current machine."""
    from app.core.device import detect_device, get_vram_gb
    from app.services.asr.qwenasr_rust import is_qwenasr_rust_available

    model_ids = all_model_ids or load_supported_model_ids()
    override_model = get_qwen_model_override()
    if override_model:
        return override_model if override_model in model_ids else None

    resolved_device = detect_device(settings.DEVICE)

    # macOS defaults to the lighter Rust CPU path unless QWEN3_ASR_MODEL is set.
    if platform.system() == "Darwin":
        return "qwen3-asr-0.6b" if is_qwenasr_rust_available() and "qwen3-asr-0.6b" in model_ids else None

    if resolved_device == "cpu":
        return "qwen3-asr-0.6b" if is_qwenasr_rust_available() and "qwen3-asr-0.6b" in model_ids else None

    vram = get_vram_gb()
    preferred = "qwen3-asr-1.7b" if vram >= 32 else "qwen3-asr-0.6b"
    if preferred in model_ids:
        return preferred

    fallback = "qwen3-asr-0.6b" if preferred == "qwen3-asr-1.7b" else "qwen3-asr-1.7b"
    return fallback if fallback in model_ids else None


def get_active_qwen_model(all_model_ids: Optional[list[str]] = None) -> str:
    """Return the required Qwen model for the current machine."""
    model_ids = all_model_ids or load_supported_model_ids()
    qwen_model = detect_qwen_model_by_vram(model_ids)
    if not qwen_model:
        override_model = get_qwen_model_override()
        if override_model:
            available_qwen_models = ", ".join(
                model_id for model_id in model_ids if model_id.startswith("qwen")
            )
            raise RuntimeError(
                f"{QWEN_MODEL_OVERRIDE_ENV}={override_model} 不在可用 Qwen3-ASR 模型中: "
                f"{available_qwen_models}"
            )
        raise RuntimeError("当前环境未找到可运行的 Qwen3-ASR 模型")
    return qwen_model


def get_runtime_model_ids(all_model_ids: Optional[list[str]] = None) -> list[str]:
    """Return the runtime model/capability plan for the current machine."""
    model_ids = all_model_ids or load_supported_model_ids()
    runtime_models = [get_active_qwen_model(model_ids)]
    if "paraformer-large" in model_ids:
        runtime_models.append("paraformer-large")
    return runtime_models


def get_default_model_id(all_model_ids: Optional[list[str]] = None) -> str:
    """Return the single default offline model for API/UI selection."""
    return get_active_qwen_model(all_model_ids)
