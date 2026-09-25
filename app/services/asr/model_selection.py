# -*- coding: utf-8 -*-
"""Offline model selection helpers."""

from __future__ import annotations

from typing import List

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


def get_offline_model_ids() -> List[str]:
    """Return enabled offline-capable models for docs and APIs."""
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
