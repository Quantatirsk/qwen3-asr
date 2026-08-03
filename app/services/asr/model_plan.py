"""Fixed deployment model plan for Ascend 910B."""

from app.services.asr.manager import ASCEND_MODEL_ID


def load_supported_model_ids() -> list[str]:
    return [ASCEND_MODEL_ID]


def get_active_qwen_model(all_model_ids: list[str] | None = None) -> str:
    if all_model_ids is not None and ASCEND_MODEL_ID not in all_model_ids:
        raise RuntimeError(f"required model is not declared: {ASCEND_MODEL_ID}")
    return ASCEND_MODEL_ID


def get_runtime_model_ids(all_model_ids: list[str] | None = None) -> list[str]:
    return [get_active_qwen_model(all_model_ids)]


def get_default_model_id(all_model_ids: list[str] | None = None) -> str:
    return get_active_qwen_model(all_model_ids)
