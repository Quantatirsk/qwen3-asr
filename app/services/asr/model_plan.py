"""The single supported transcription model."""

from app.services.realtime.protocol import MODEL_ID


def get_runtime_model_ids() -> list[str]:
    return [MODEL_ID]


def get_default_model_id() -> str:
    return MODEL_ID
