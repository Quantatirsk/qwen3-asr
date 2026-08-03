"""Model metadata and construction for the Ascend-only runtime."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings
from app.core.exceptions import DefaultServerErrorException, InvalidParameterException

ASCEND_MODEL_ID = "qwen3-asr-1.7b"


class DeclaredEntryConfig:
    def __init__(self, model_id: str, config: dict[str, Any]) -> None:
        self.model_id = model_id
        self.name = config["name"]
        self.engine = config["engine"]
        self.description = config.get("description", "")
        self.languages = config.get("languages", [])
        self.offline_model_path = config.get("models", {}).get("offline")
        self.extra_kwargs = config.get("extra_kwargs", {})

    @property
    def has_offline_model(self) -> bool:
        return bool(self.offline_model_path)


class ModelManager:
    def __init__(self) -> None:
        models_file = Path(settings.models_config_path)
        try:
            config = json.loads(models_file.read_text(encoding="utf-8"))
            raw_config = config["models"][ASCEND_MODEL_ID]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise DefaultServerErrorException(
                f"无法加载 Ascend 模型配置: {exc}"
            ) from exc
        self._config = DeclaredEntryConfig(ASCEND_MODEL_ID, raw_config)

    def get_declared_entry_config(
        self, model_id: Optional[str] = None
    ) -> DeclaredEntryConfig:
        if model_id not in {None, ASCEND_MODEL_ID, "qwen3-asr"}:
            raise InvalidParameterException(
                f"未知的模型: {model_id}，可用模型: {ASCEND_MODEL_ID}"
            )
        return self._config

    def list_declared_entries(self) -> list[dict[str, Any]]:
        config = self._config
        return [
            {
                "id": config.model_id,
                "name": config.name,
                "engine": config.engine,
                "description": config.description,
                "languages": config.languages,
                "default": True,
                "offline_model": {
                    "path": config.offline_model_path,
                    "exists": bool(settings.QWEN_VLLM_BASE_URL),
                    "remote": True,
                },
            }
        ]

    def create_engine(self, model_id: Optional[str] = None):
        config = self.get_declared_entry_config(model_id)
        from app.services.asr.qwen3_engine import Qwen3ASREngine

        return Qwen3ASREngine(
            model_path=config.offline_model_path,
            **config.extra_kwargs,
        )


_model_manager: Optional[ModelManager] = None
_model_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        with _model_manager_lock:
            if _model_manager is None:
                _model_manager = ModelManager()
    return _model_manager
