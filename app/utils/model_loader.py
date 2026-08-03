"""Startup integrity checks and preload for the Ascend offline stack."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .boot_events import emit_boot_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelIntegritySpec:
    description: str
    path: Path
    required_patterns: tuple[str, ...]
    alternative_required_patterns: tuple[tuple[str, ...], ...] = ()
    min_total_size_bytes: int = 0


def _find_pattern_matches(root: Path, pattern: str) -> list[Path]:
    return [path for path in root.glob(pattern) if path.is_file()]


def _find_missing_patterns(root: Path, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if not _find_pattern_matches(root, pattern)]


def _format_alternative_patterns(groups: tuple[tuple[str, ...], ...]) -> str:
    return " OR ".join(" + ".join(group) for group in groups)


def _check_model_integrity_spec(spec: ModelIntegritySpec) -> dict[str, Any]:
    if not spec.path.is_dir():
        return {
            "description": spec.description,
            "path": str(spec.path),
            "ok": False,
            "missing_patterns": list(spec.required_patterns),
            "total_size_bytes": 0,
            "reason": "directory_missing",
        }

    files = [path for path in spec.path.rglob("*") if path.is_file()]
    total_size = sum(path.stat().st_size for path in files)
    missing = _find_missing_patterns(spec.path, spec.required_patterns)
    if not missing and spec.alternative_required_patterns:
        if not any(
            not _find_missing_patterns(spec.path, group)
            for group in spec.alternative_required_patterns
        ):
            missing = [_format_alternative_patterns(spec.alternative_required_patterns)]

    reason = "ok"
    if missing:
        reason = "required_files_missing"
    elif total_size < spec.min_total_size_bytes:
        reason = "directory_too_small"
    return {
        "description": spec.description,
        "path": str(spec.path),
        "ok": reason == "ok",
        "missing_patterns": missing,
        "total_size_bytes": total_size,
        "reason": reason,
    }


def _build_required_model_integrity_specs() -> list[ModelIntegritySpec]:
    from app.core.config import settings
    from app.services.asr.model_capabilities import get_runtime_required_modelscope_assets

    return [
        ModelIntegritySpec(
            description=asset.description,
            path=Path(settings.MODELSCOPE_PATH) / asset.model_id,
            required_patterns=asset.required_patterns,
            alternative_required_patterns=asset.alternative_required_patterns,
            min_total_size_bytes=asset.min_total_size_bytes,
        )
        for asset in get_runtime_required_modelscope_assets()
    ]


def verify_required_models_integrity(use_logger: bool = True) -> dict[str, Any]:
    results = [
        _check_model_integrity_spec(spec)
        for spec in _build_required_model_integrity_specs()
    ]
    invalid = [result for result in results if not result["ok"]]
    if use_logger:
        logger.info(
            "model integrity: total=%s ok=%s failed=%s",
            len(results),
            len(results) - len(invalid),
            len(invalid),
        )
    else:
        print(
            f"model integrity: total={len(results)} "
            f"ok={len(results) - len(invalid)} failed={len(invalid)}"
        )
    return {"total": len(results), "results": results, "invalid_models": invalid}


def preload_models() -> dict[str, Any]:
    from app.core.config import settings
    from app.services.asr.engines import get_global_vad_model
    from app.services.asr.manager import ASCEND_MODEL_ID
    from app.services.asr.runtime import get_runtime_router
    from app.utils.download_models import fix_camplusplus_config
    from app.utils.speaker_diarizer import get_global_diarization_pipeline

    fix_camplusplus_config()
    result: dict[str, Any] = {
        "asr_models": {ASCEND_MODEL_ID: {"loaded": False, "error": None}},
        "vad_model": {"loaded": False, "error": None},
        "speaker_diarization_model": {"loaded": False, "error": None},
    }
    steps = (
        ("asr_models", lambda: get_runtime_router().warmup_model(ASCEND_MODEL_ID)),
        ("vad_model", lambda: get_global_vad_model(settings.DEVICE)),
        ("speaker_diarization_model", get_global_diarization_pipeline),
    )
    for key, loader in steps:
        emit_boot_event("step_start", phase="preload", message=key)
        try:
            loader()
            target = result[key][ASCEND_MODEL_ID] if key == "asr_models" else result[key]
            target["loaded"] = True
        except Exception as exc:
            target = result[key][ASCEND_MODEL_ID] if key == "asr_models" else result[key]
            target["error"] = str(exc)
            logger.error("preload failed for %s: %s", key, exc)
        emit_boot_event("step_done", phase="preload", message=key)
    return result
