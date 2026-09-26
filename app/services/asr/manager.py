"""R2T2 metadata and CUDA engine construction."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.exceptions import InvalidParameterException
from app.infrastructure import get_huggingface_model_cache_dir
from app.services.realtime.protocol import MODEL_ID, MODEL_REPOSITORY, MODEL_REVISION

if TYPE_CHECKING:
    from .r2t2_engine import R2T2Engine


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = MODEL_ID
    name: str = "Confucius4-R2T2"
    offline_model_path: str = MODEL_REPOSITORY


class ModelManager:
    def get_declared_entry_config(self, model_id: str | None = None) -> ModelConfig:
        if model_id is not None and model_id != MODEL_ID:
            raise InvalidParameterException(f"Unsupported model: {model_id}")
        return ModelConfig()

    def list_declared_entries(self) -> list[dict[str, object]]:
        snapshot = (
            get_huggingface_model_cache_dir(MODEL_REPOSITORY)
            / "snapshots"
            / MODEL_REVISION
        )
        return [
            {
                "id": MODEL_ID,
                "kind": "model",
                "name": "Confucius4-R2T2",
                "engine": "r2t2",
                "description": "CUDA R2T2 offline and realtime transcription",
                "languages": ["zh", "en"],
                "default": True,
                "supports_realtime": True,
                "offline_model": {
                    "path": MODEL_REPOSITORY,
                    "exists": snapshot.is_dir(),
                },
                "realtime_model": {
                    "path": MODEL_REPOSITORY,
                    "exists": snapshot.is_dir(),
                },
            }
        ]

    def create_engine(self, model_id: str | None = None) -> "R2T2Engine":
        self.get_declared_entry_config(model_id)
        from .r2t2_engine import R2T2Engine

        return R2T2Engine()


_model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    return _model_manager
