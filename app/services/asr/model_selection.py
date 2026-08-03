"""Offline model selection for the fixed Ascend deployment."""

from app.services.asr.manager import ASCEND_MODEL_ID


def get_active_qwen_model_id() -> str:
    return ASCEND_MODEL_ID


def get_offline_model_ids() -> list[str]:
    return [ASCEND_MODEL_ID]


def get_default_offline_model_id() -> str:
    return ASCEND_MODEL_ID
