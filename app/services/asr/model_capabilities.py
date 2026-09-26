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

_DIARIZATION_ASSETS = (
    ModelAsset(
        source="modelscope",
        model_id="iic/speech_campplus_speaker-diarization_common",
        description="CAM++ Diarization",
        required_patterns=(
            "configuration.json",
            "config.yaml",
            "onnx/asd.onnx",
            "onnx/face_recog_ir101.onnx",
            "onnx/fqa.onnx",
            "onnx/version-RFB-320.onnx",
        ),
        min_total_size_bytes=50_000_000,
    ),
    ModelAsset(
        source="modelscope",
        model_id="damo/speech_campplus_sv_zh-cn_16k-common",
        description="CAM++ Speaker Verification",
        required_patterns=(
            "configuration.json",
            "config.yaml",
            "campplus_cn_common.bin",
        ),
        min_total_size_bytes=10_000_000,
    ),
    ModelAsset(
        source="modelscope",
        model_id="damo/speech_campplus-transformer_scl_zh-cn_16k-common",
        description="CAM++ Transformer",
        required_patterns=(
            "configuration.json",
            "campplus_cn_encoder.pt",
            "transformer_backend.pt",
        ),
        min_total_size_bytes=10_000_000,
    ),
)


def get_download_modelscope_assets() -> list[ModelAsset]:
    """Return the full static ModelScope export set used by predownload/export."""
    return [
        *_VAD_ASSETS,
        *_PUNCTUATION_ASSETS,
        *_DIARIZATION_ASSETS,
    ]


def get_runtime_required_modelscope_assets() -> list[ModelAsset]:
    """Return the offline VAD and speaker assets; realtime runs remotely."""
    return [*_VAD_ASSETS, *_PUNCTUATION_ASSETS, *_DIARIZATION_ASSETS]


def get_huggingface_assets() -> list[ModelAsset]:
    """One shared ASR checkpoint and an independent timestamp aligner."""
    return [
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


def get_camplusplus_replacement_paths(cache_dir: str) -> dict[str, str]:
    """Return the CAM++ offline replacement map for local cache paths."""
    return {
        "damo/speech_campplus_sv_zh-cn_16k-common": f"{cache_dir}/damo/speech_campplus_sv_zh-cn_16k-common",
        "damo/speech_campplus-transformer_scl_zh-cn_16k-common": f"{cache_dir}/damo/speech_campplus-transformer_scl_zh-cn_16k-common",
        "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch": f"{cache_dir}/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    }
