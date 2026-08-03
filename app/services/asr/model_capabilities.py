"""Model assets required by the Ascend offline deployment."""

from dataclasses import dataclass
from typing import Literal, Optional

from app.core.config import settings

ModelSource = Literal["modelscope", "huggingface"]
QWEN_ASCEND_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"


@dataclass(frozen=True)
class ModelAsset:
    source: ModelSource
    model_id: str
    description: str
    revision: Optional[str] = None
    required_patterns: tuple[str, ...] = ()
    alternative_required_patterns: tuple[tuple[str, ...], ...] = ()
    min_total_size_bytes: int = 0


_OFFLINE_MODELSCOPE_ASSETS = (
    ModelAsset(
        source="modelscope",
        model_id=settings.VAD_MODEL,
        description="FSMN VAD",
        revision="v2.0.2",
        required_patterns=("configuration.json", "config.yaml", "model.pb"),
        min_total_size_bytes=1_000_000,
    ),
    ModelAsset(
        source="modelscope",
        model_id="iic/speech_campplus_speaker-diarization_common",
        description="CAM++ Diarization",
        required_patterns=("configuration.json", "config.yaml"),
        min_total_size_bytes=50_000_000,
    ),
    ModelAsset(
        source="modelscope",
        model_id="damo/speech_campplus_sv_zh-cn_16k-common",
        description="CAM++ Speaker Verification",
        required_patterns=("configuration.json", "config.yaml", "campplus_cn_common.bin"),
        min_total_size_bytes=10_000_000,
    ),
    ModelAsset(
        source="modelscope",
        model_id="damo/speech_campplus-transformer_scl_zh-cn_16k-common",
        description="CAM++ Change Locator",
        required_patterns=(
            "configuration.json",
            "campplus_cn_encoder.pt",
            "transformer_backend.pt",
        ),
        min_total_size_bytes=10_000_000,
    ),
)


def get_download_modelscope_assets() -> list[ModelAsset]:
    return list(_OFFLINE_MODELSCOPE_ASSETS)


def get_runtime_required_modelscope_assets() -> list[ModelAsset]:
    return list(_OFFLINE_MODELSCOPE_ASSETS)


def get_enabled_qwen_huggingface_assets() -> list[ModelAsset]:
    return [
        ModelAsset(
            source="huggingface",
            model_id="Qwen/Qwen3-ASR-1.7B",
            description="Qwen3-ASR-1.7B for Ascend vLLM",
            revision=QWEN_ASCEND_REVISION,
            required_patterns=("snapshots/*/config.json",),
            alternative_required_patterns=(
                ("snapshots/*/model.safetensors",),
                (
                    "snapshots/*/model.safetensors.index.json",
                    "snapshots/*/model-*.safetensors",
                ),
            ),
            min_total_size_bytes=500_000_000,
        )
    ]


def get_camplusplus_replacement_paths(cache_dir: str) -> dict[str, str]:
    return {
        "damo/speech_campplus_sv_zh-cn_16k-common": f"{cache_dir}/damo/speech_campplus_sv_zh-cn_16k-common",
        "damo/speech_campplus-transformer_scl_zh-cn_16k-common": f"{cache_dir}/damo/speech_campplus-transformer_scl_zh-cn_16k-common",
        settings.VAD_MODEL: f"{cache_dir}/{settings.VAD_MODEL}",
    }
