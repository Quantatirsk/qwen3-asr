# -*- coding: utf-8 -*-
"""
模型预加载工具
在应用启动时预加载所有需要的模型,避免首次请求时的延迟
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelIntegritySpec:
    description: str
    path: Path
    required_patterns: tuple[str, ...]
    alternative_required_patterns: tuple[tuple[str, ...], ...] = ()
    min_total_size_bytes: int = 0
    expected_revision: str | None = None


def _find_pattern_matches(root: Path, pattern: str) -> list[Path]:
    return [path for path in root.glob(pattern) if path.is_file()]


def _find_missing_patterns(root: Path, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if not _find_pattern_matches(root, pattern)]


def _format_alternative_patterns(pattern_groups: tuple[tuple[str, ...], ...]) -> str:
    return " OR ".join(" + ".join(group) for group in pattern_groups)


def _check_model_integrity_spec(spec: ModelIntegritySpec) -> dict[str, Any]:
    if not spec.path.exists() or not spec.path.is_dir():
        return {
            "description": spec.description,
            "path": str(spec.path),
            "ok": False,
            "missing_patterns": [
                *spec.required_patterns,
                *(
                    [_format_alternative_patterns(spec.alternative_required_patterns)]
                    if spec.alternative_required_patterns
                    else []
                ),
            ],
            "total_size_bytes": 0,
            "reason": "directory_missing",
        }

    files = [path for path in spec.path.rglob("*") if path.is_file()]
    total_size_bytes = sum(path.stat().st_size for path in files)

    missing_patterns = _find_missing_patterns(spec.path, spec.required_patterns)
    if not missing_patterns and spec.alternative_required_patterns:
        alternative_missing_patterns = [
            _find_missing_patterns(spec.path, group)
            for group in spec.alternative_required_patterns
        ]
        if all(alternative_missing_patterns):
            missing_patterns = [
                _format_alternative_patterns(spec.alternative_required_patterns)
            ]

    for index_path in spec.path.glob("model.safetensors.index.json"):
        import json

        try:
            weights = json.loads(index_path.read_text())["weight_map"]
            if not isinstance(weights, dict) or not weights:
                raise ValueError("Empty weight map")
            for name in set(weights.values()):
                weight = index_path.parent / name
                if not weight.is_file() or not weight.stat().st_size:
                    missing_patterns.append(name)
        except (OSError, ValueError, KeyError, TypeError):
            missing_patterns.append("valid model.safetensors.index.json")

    if spec.expected_revision:
        for name in spec.required_patterns:
            metadata = spec.path / ".cache/huggingface/download" / (name + ".metadata")
            if not metadata.is_file() or metadata.read_text().splitlines()[:1] != [
                spec.expected_revision
            ]:
                missing_patterns.append(
                    f"{name}: expected revision {spec.expected_revision}"
                )

    if missing_patterns:
        return {
            "description": spec.description,
            "path": str(spec.path),
            "ok": False,
            "missing_patterns": missing_patterns,
            "total_size_bytes": total_size_bytes,
            "reason": "required_files_missing",
        }

    if total_size_bytes < spec.min_total_size_bytes:
        return {
            "description": spec.description,
            "path": str(spec.path),
            "ok": False,
            "missing_patterns": [],
            "total_size_bytes": total_size_bytes,
            "reason": "directory_too_small",
        }

    return {
        "description": spec.description,
        "path": str(spec.path),
        "ok": True,
        "missing_patterns": [],
        "total_size_bytes": total_size_bytes,
        "reason": "ok",
    }


def _build_modelscope_spec(
    model_id: str,
    description: str,
    required_patterns: tuple[str, ...],
    *,
    min_total_size_bytes: int,
    alternative_required_patterns: tuple[tuple[str, ...], ...] = (),
) -> ModelIntegritySpec:
    from ..core.config import settings

    return ModelIntegritySpec(
        description=description,
        path=Path(settings.MODELSCOPE_PATH) / model_id,
        required_patterns=required_patterns,
        alternative_required_patterns=alternative_required_patterns,
        min_total_size_bytes=min_total_size_bytes,
    )


def _build_required_model_integrity_specs() -> list[ModelIntegritySpec]:
    from app.infrastructure import (
        get_huggingface_model_cache_dir,
        find_huggingface_snapshot_dir,
    )
    from app.services.asr.model_capabilities import (
        get_huggingface_assets,
        get_runtime_required_modelscope_assets,
    )

    specs = [
        _build_modelscope_spec(
            asset.model_id,
            asset.description,
            asset.required_patterns,
            alternative_required_patterns=asset.alternative_required_patterns,
            min_total_size_bytes=asset.min_total_size_bytes,
        )
        for asset in get_runtime_required_modelscope_assets()
    ]
    for asset in get_huggingface_assets():
        cache = get_huggingface_model_cache_dir(asset.model_id)
        snapshot = (
            Path(asset.local_dir)
            if asset.local_dir
            else (
                cache / "snapshots" / asset.revision
                if asset.revision
                else find_huggingface_snapshot_dir(asset.model_id)
            )
        )
        specs.append(
            ModelIntegritySpec(
                description=asset.description,
                path=snapshot or cache / "snapshots" / "missing",
                required_patterns=asset.required_patterns,
                alternative_required_patterns=asset.alternative_required_patterns,
                min_total_size_bytes=asset.min_total_size_bytes,
                expected_revision=asset.revision if asset.local_dir else None,
            )
        )
    return specs


def verify_required_models_integrity(use_logger: bool = True) -> dict[str, Any]:
    results = [
        _check_model_integrity_spec(spec)
        for spec in _build_required_model_integrity_specs()
    ]
    invalid = [result for result in results if not result["ok"]]
    for result in results:
        message = f"Model integrity: {result['description']} {result['reason']} {result['path']}"
        if use_logger:
            logger.log(logging.INFO if result["ok"] else logging.ERROR, message)
        else:
            print(message)
    return {"total": len(results), "results": results, "invalid_models": invalid}


def preload_models() -> dict[str, Any]:
    """Load every required component; startup must not silently degrade."""
    from app.core.config import settings
    from app.core.device import detect_device
    from app.services.asr.engines import get_global_vad_model
    from app.services.asr.runtime import get_runtime_router
    from app.services.asr.punctuation import get_punctuation_model
    from app.services.realtime.protocol import MODEL_ID
    from app.services.realtime.client import get_engine_capabilities
    from app.utils.speaker_diarizer import get_speaker_diarizer

    if not get_engine_capabilities().get("ready"):
        raise RuntimeError("Shared R2T2 engine is not ready")
    device = detect_device(settings.DEVICE)
    get_runtime_router().warmup_model(MODEL_ID)
    get_global_vad_model(device)
    get_speaker_diarizer().warmup()
    get_punctuation_model()
    return {
        "asr_models": {MODEL_ID: {"loaded": True}},
        "vad_model": {"loaded": True},
        "speaker_diarization_model": {"loaded": True},
        "punctuation_model": {"loaded": True},
    }
