# -*- coding: utf-8 -*-
"""Shared capability-to-model asset definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from app.core.config import settings
from app.services.realtime.protocol import MODEL_REPOSITORY, MODEL_REVISION

ModelSource = Literal["modelscope", "huggingface"]


@dataclass(frozen=True)
class ModelAsset:
    source: ModelSource
    model_id: str
    description: str
    revision: Optional[str] = None
    required_patterns: tuple[str, ...] = ()
    alternative_required_patterns: tuple[tuple[str, ...], ...] = ()
    min_total_size_bytes: int = 0
    local_dir: str | None = None


_VAD_ASSETS = (
    ModelAsset(
        source="modelscope",
        model_id=settings.VAD_MODEL,
        description="VAD",
        revision="v2.0.2",
        required_patterns=("configuration.json", "config.yaml", "model.pb"),
        min_total_size_bytes=1_000_000,
    ),
)

_PUNCTUATION_ASSETS = (
    ModelAsset(
        source="modelscope",
        model_id="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        description="Sentence-ending Punctuation",
        required_patterns=(
            "configuration.json",
            "config.yaml",
            "model.pt",
            "tokens.json",
        ),
        min_total_size_bytes=100_000_000,
    ),
)


def get_download_modelscope_assets() -> list[ModelAsset]:
    """Return the full static ModelScope export set used by predownload/export."""
    return [
        *_VAD_ASSETS,
        *_PUNCTUATION_ASSETS,
    ]


def get_runtime_required_modelscope_assets() -> list[ModelAsset]:
    """Return the VAD and punctuation assets used by offline recognition."""
    return [*_VAD_ASSETS, *_PUNCTUATION_ASSETS]


def get_huggingface_assets() -> list[ModelAsset]:
    """Return the diarizer, shared ASR checkpoint, and timestamp aligner."""
    return [
        ModelAsset(
            source="huggingface",
            model_id="nvidia/Nemotron-3-Diarization",
            revision="f667ed73aee57d40cc39428eb768b4fd87a0a29e",
            description="Nemotron Speaker Diarization",
            required_patterns=(
                "config.json",
                "processor_config.json",
                "model.safetensors",
            ),
            min_total_size_bytes=100_000_000,
            local_dir=settings.NEMOTRON_MODEL_PATH,
        ),
        ModelAsset(
            source="huggingface",
            model_id=MODEL_REPOSITORY,
            description="Confucius4-R2T2",
            revision=MODEL_REVISION,
            required_patterns=(
                "config.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
            alternative_required_patterns=(
                ("model.safetensors",),
                ("model.safetensors.index.json", "model-*.safetensors"),
            ),
            min_total_size_bytes=500_000_000,
        ),
        ModelAsset(
            source="huggingface",
            model_id="Qwen/Qwen3-ForcedAligner-0.6B",
            description="Forced Aligner",
            required_patterns=("config.json", "model.safetensors"),
            min_total_size_bytes=500_000_000,
        ),
    ]
