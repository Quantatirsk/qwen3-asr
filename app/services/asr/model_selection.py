"""Offline and realtime use the same R2T2 checkpoint."""

from app.services.realtime.protocol import MODEL_ID


def get_offline_model_ids() -> list[str]:
    return [MODEL_ID]


def get_default_offline_model_id() -> str:
    return MODEL_ID
