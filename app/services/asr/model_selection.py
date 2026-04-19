# -*- coding: utf-8 -*-
"""Offline/realtime model selection helpers."""

from __future__ import annotations

from typing import List, Optional

from ...core.exceptions import InvalidParameterException
from .manager import get_model_manager
from .model_plan import (
    get_active_qwen_model,
    get_default_model_id,
    get_runtime_model_ids,
)


def get_active_qwen_model_id() -> str:
    """Return the currently active Qwen model id."""
    active_qwen_model = get_active_qwen_model()
    return active_qwen_model or "qwen3-asr-0.6b"


def load_supported_model_list() -> List[str]:
    """Load supported model ids declared by static config."""
    return get_runtime_model_ids()


def get_offline_model_ids() -> List[str]:
    """Return enabled offline-capable models for docs and compatibility APIs."""
    manager = get_model_manager()
    runtime_models = get_runtime_model_ids()

    def sort_key(model_id: str) -> tuple[int, str]:
        if model_id.startswith("qwen"):
            return (0, model_id)
        return (1, model_id)

    offline_models = [
        model_id
        for model_id in runtime_models
        if manager.get_declared_entry_config(model_id).has_offline_model
    ]
    return sorted(offline_models, key=sort_key)


def get_default_offline_model_id() -> str:
    """Return the default offline-capable model."""
    default_model = get_default_model_id()
    if default_model:
        try:
            if get_model_manager().get_declared_entry_config(default_model).has_offline_model:
                return default_model
        except InvalidParameterException:
            pass
    return get_active_qwen_model_id()


def resolve_offline_model_id(_ignored_model_id: Optional[str] = None) -> str:
    """Resolve the single active offline model for all offline API calls."""
    return get_default_offline_model_id()


def validate_realtime_model_id(model_id: Optional[str]) -> str:
    """Validate realtime-capable model ids for websocket protocols."""
    available_models = get_offline_model_ids()

    if not model_id:
        return get_default_offline_model_id()

    if model_id.lower() == "qwen3-asr":
        active_qwen_model = get_active_qwen_model_id()
        if active_qwen_model.startswith("qwen") and active_qwen_model in available_models:
            return active_qwen_model
        raise InvalidParameterException("当前环境未启用 Qwen3-ASR 模型")

    if model_id not in available_models:
        raise InvalidParameterException(
            f"不支持的模型ID: {model_id}。可用模型: {', '.join(available_models)}"
        )

    return model_id
