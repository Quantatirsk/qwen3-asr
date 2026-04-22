# -*- coding: utf-8 -*-
"""
模型预加载工具
在应用启动时预加载所有需要的模型,避免首次请求时的延迟
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
except ImportError:
    Console = None

from .boot_events import emit_boot_event

logger = logging.getLogger(__name__)

_PRELOAD_QUIET_LOGGERS = (
    "root",
    "vllm",
    "app.infrastructure.model_utils",
    "app.services.asr.engines.funasr",
    "app.services.asr.engines.global_models",
    "app.services.asr.qwen3_engine",
    "app.utils.speaker_diarizer",
)


class _ProgressNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return not any(
            record.name == prefix or record.name.startswith(f"{prefix}.")
            for prefix in _PRELOAD_QUIET_LOGGERS
        )


class _StartupProgress:
    def __init__(self, title: str, total: int):
        self._title = title
        self._total = max(total, 1)
        self._enabled = bool(
            Console is not None
            and sys.stderr.isatty()
            and os.getenv("FUNASR_TUI_CHILD") != "1"
        )
        self._console: Console | None = None
        self._filter = _ProgressNoiseFilter()
        self._handlers: list[logging.Handler] = []
        self._current_step = 1
        self._last_description: str | None = None

    def __enter__(self) -> "_StartupProgress":
        emit_boot_event(
            "phase_start",
            phase=self._title,
            total=self._total,
            message=self._title,
        )
        if not self._enabled:
            return self
        self._console = Console(stderr=True)
        root_logger = logging.getLogger()
        self._handlers = list(root_logger.handlers)
        for handler in self._handlers:
            handler.addFilter(self._filter)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handler in self._handlers:
            handler.removeFilter(self._filter)
        self._handlers.clear()

    def update(self, description: str) -> None:
        emit_boot_event(
            "step_start",
            phase=self._title,
            step=self._current_step,
            total=self._total,
            message=description,
        )
        if self._console is None:
            return
        if description == self._last_description:
            return
        self._last_description = description
        self._console.print(
            f"[bold cyan][startup {self._current_step}/{self._total}][/bold cyan] {description}",
            highlight=False,
        )

    def advance(self, description: str) -> None:
        emit_boot_event(
            "step_done",
            phase=self._title,
            step=self._current_step,
            total=self._total,
            message=description,
        )
        self._last_description = description
        self._current_step = min(self._current_step + 1, self._total)


@dataclass(frozen=True)
class ModelIntegritySpec:
    description: str
    path: Path
    required_patterns: tuple[str, ...]
    min_total_size_bytes: int = 0


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _find_pattern_matches(root: Path, pattern: str) -> list[Path]:
    return [path for path in root.glob(pattern) if path.is_file()]


def _check_model_integrity_spec(spec: ModelIntegritySpec) -> dict[str, Any]:
    if not spec.path.exists() or not spec.path.is_dir():
        return {
            "description": spec.description,
            "path": str(spec.path),
            "ok": False,
            "missing_patterns": list(spec.required_patterns),
            "total_size_bytes": 0,
            "reason": "directory_missing",
        }

    files = [path for path in spec.path.rglob("*") if path.is_file()]
    total_size_bytes = sum(path.stat().st_size for path in files)

    missing_patterns = [
        pattern for pattern in spec.required_patterns
        if not _find_pattern_matches(spec.path, pattern)
    ]
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
) -> ModelIntegritySpec:
    from ..core.config import settings

    return ModelIntegritySpec(
        description=description,
        path=Path(settings.MODELSCOPE_PATH) / model_id,
        required_patterns=required_patterns,
        min_total_size_bytes=min_total_size_bytes,
    )


def _build_huggingface_spec(
    model_id: str,
    description: str,
    required_patterns: tuple[str, ...],
    *,
    min_total_size_bytes: int,
) -> ModelIntegritySpec:
    org, model = model_id.split("/", 1)
    return ModelIntegritySpec(
        description=description,
        path=Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{model}",
        required_patterns=required_patterns,
        min_total_size_bytes=min_total_size_bytes,
    )
def print_model_statistics(result: dict[str, Any], use_logger: bool = True) -> None:
    """打印模型加载统计信息 - KISS版本：只显示已加载的模型"""
    output = logger.info if use_logger else print

    loaded_models = []

    # 收集已加载的ASR模型
    for model_id, status in result["asr_models"].items():
        if status["loaded"]:
            loaded_models.append(f"ASR模型({model_id})")

    # 收集已加载的其他模型
    other_models = [
        ("vad_model", "语音活动检测模型(VAD)"),
        ("speaker_diarization_model", "说话人分离模型(CAM++)"),
    ]
    for key, name in other_models:
        if result[key]["loaded"]:
            loaded_models.append(name)

    # 简洁输出
    output("=" * 50)
    if loaded_models:
        output(f"✅ 已加载 {len(loaded_models)} 个模型:")
        for i, name in enumerate(loaded_models, 1):
            output(f"   {i}. {name}")
    else:
        output("⚠️  没有模型被加载")
    output("=" * 50)


def _should_check_qwen_forced_aligner(
    resolved_device: str,
    using_cpu_qwen_rust: bool,
) -> bool:
    """Return True when startup integrity should require Qwen forced aligner files."""
    _ = (resolved_device, using_cpu_qwen_rust)
    return True

def _build_required_model_integrity_specs() -> list[ModelIntegritySpec]:
    from ..core.config import settings
    from ..core.device import detect_device
    from ..services.asr.manager import get_model_manager
    from ..services.asr.model_capabilities import (
        get_enabled_qwen_huggingface_assets,
        get_runtime_required_modelscope_assets,
    )
    from ..services.asr.model_plan import get_runtime_model_ids
    from ..services.asr.qwenasr_rust import is_qwenasr_rust_available
    manager = get_model_manager()
    model_ids = [item["id"] for item in manager.list_declared_entries()]
    runtime_models = get_runtime_model_ids(model_ids)
    resolved_device = detect_device(settings.DEVICE)
    using_cpu_qwen_rust = (
        resolved_device == "cpu" and is_qwenasr_rust_available()
    )
    specs: list[ModelIntegritySpec] = []

    for asset in get_runtime_required_modelscope_assets(
        include_realtime_punc=settings.ASR_ENABLE_REALTIME_PUNC,
    ):
        specs.append(
            _build_modelscope_spec(
                asset.model_id,
                asset.description,
                asset.required_patterns,
                min_total_size_bytes=asset.min_total_size_bytes,
            )
        )

    for asset in get_enabled_qwen_huggingface_assets(
        include_forced_aligner=_should_check_qwen_forced_aligner(
            resolved_device=resolved_device,
            using_cpu_qwen_rust=using_cpu_qwen_rust,
        ),
    ):
        specs.append(
            _build_huggingface_spec(
                asset.model_id,
                asset.description,
                asset.required_patterns,
                min_total_size_bytes=asset.min_total_size_bytes,
            )
        )

    return specs


def verify_required_models_integrity(use_logger: bool = True) -> dict[str, Any]:
    output = logger.info if use_logger else print
    specs = _build_required_model_integrity_specs()
    total = len(specs)
    results: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    if not use_logger:
        output("=" * 60)
        output(f"🔍 开始检查运行时模型完整性，共 {total} 个")
        output("=" * 60)
        for index, spec in enumerate(specs, start=1):
            output(f"[{index}/{total}] 检查 {spec.description}")
            result = _check_model_integrity_spec(spec)
            results.append(result)
            if result["ok"]:
                output(
                    f"  ✅ OK  size={_format_bytes(result['total_size_bytes'])} "
                    f"path={result['path']}"
                )
                continue
            invalid.append(result)
            if result["reason"] == "directory_missing":
                output(f"  ❌ FAIL directory_missing path={result['path']}")
            elif result["reason"] == "required_files_missing":
                output(
                    f"  ❌ FAIL missing={', '.join(result['missing_patterns'])} "
                    f"size={_format_bytes(result['total_size_bytes'])} path={result['path']}"
                )
            else:
                output(
                    f"  ❌ FAIL size_too_small size={_format_bytes(result['total_size_bytes'])} "
                    f"path={result['path']}"
                )
        output("=" * 60)
        output(f"模型完整性检查完成: total={total} ok={total - len(invalid)} failed={len(invalid)}")
        output("=" * 60)
        return {
            "total": total,
            "results": results,
            "invalid_models": invalid,
        }

    logger.info("开始检查运行时模型完整性: total=%s", total)
    with _StartupProgress("检查运行时模型完整性", total) as progress:
        for spec in specs:
            progress.update(f"检查 {spec.description}")
            result = _check_model_integrity_spec(spec)
            results.append(result)
            if not result["ok"]:
                invalid.append(result)
                if result["reason"] == "directory_missing":
                    logger.error("模型完整性检查失败: %s, reason=directory_missing, path=%s", spec.description, result["path"])
                elif result["reason"] == "required_files_missing":
                    logger.error(
                        "模型完整性检查失败: %s, reason=required_files_missing, missing=%s, size=%s, path=%s",
                        spec.description,
                        ", ".join(result["missing_patterns"]),
                        _format_bytes(result["total_size_bytes"]),
                        result["path"],
                    )
                else:
                    logger.error(
                        "模型完整性检查失败: %s, reason=directory_too_small, size=%s, path=%s",
                        spec.description,
                        _format_bytes(result["total_size_bytes"]),
                        result["path"],
                    )
            progress.advance(f"检查完成 {spec.description}")

    logger.info(
        "模型完整性检查完成: total=%s ok=%s failed=%s",
        total,
        total - len(invalid),
        len(invalid),
    )

    return {
        "total": total,
        "results": results,
        "invalid_models": invalid,
    }
def preload_models() -> dict[str, Any]:
    """
    预加载所有需要的模型（根据 ENABLE_* 配置过滤）

    Returns:
        dict: 包含加载状态的字典
    """
    # 修复 CAM++ 配置文件（用于离线环境）
    try:
        from .download_models import fix_camplusplus_config
        fix_camplusplus_config()
    except Exception:
        pass  # 修复失败不影响启动

    result: dict[str, Any] = {
        "asr_models": {},  # 所有ASR模型加载状态
        "vad_model": {"loaded": False, "error": None},
        "punc_model": {"loaded": False, "error": None},
        "punc_realtime_model": {"loaded": False, "error": None},
        "speaker_diarization_model": {"loaded": False, "error": None},
    }

    from ..core.config import settings
    from ..core.device import detect_device

    # 初始化变量，避免未绑定错误
    asr_device = detect_device(settings.DEVICE)
    model_manager = None

    # 1. 预加载所有配置的ASR模型（根据 ENABLE_* 配置过滤）
    try:
        from ..services.asr.manager import get_model_manager
        from ..services.asr.model_plan import get_runtime_model_ids
        from ..services.asr.runtime import get_runtime_router

        model_manager = get_model_manager()
        runtime_router = get_runtime_router()

        # 获取所有模型配置
        all_models = model_manager.list_declared_entries()
        model_ids = [m["id"] for m in all_models]

        models_to_load = get_runtime_model_ids(model_ids)

        if not models_to_load:
            logger.warning("⚠️  当前环境未解析出可运行的 ASR 模型")

    except Exception as e:
        logger.error(f"❌ 获取模型管理器失败: {e}")
        models_to_load = []
        runtime_router = None

    # 辅助函数：检查是否要加载 paraformer
    paraformer_enabled = "paraformer-large" in models_to_load

    total_steps = len(models_to_load) + 2
    if paraformer_enabled:
        total_steps += 1
        if settings.ASR_ENABLE_REALTIME_PUNC:
            total_steps += 1

    logger.info(
        "开始预加载模型: declared=%s runtime=%s models=%s",
        len(model_ids) if model_manager else 0,
        len(models_to_load),
        ", ".join(models_to_load) if models_to_load else "（无）",
    )

    with _StartupProgress("预加载模型", total_steps) as progress:
        for model_id in models_to_load:
            result["asr_models"][model_id] = {"loaded": False, "error": None}
            progress.update(f"加载 ASR 模型 {model_id}")
            try:
                if runtime_router is None:
                    raise RuntimeError("runtime router unavailable")
                runtime_router.warmup_model(model_id)
                result["asr_models"][model_id]["loaded"] = True
            except Exception as e:
                result["asr_models"][model_id]["error"] = str(e)
                logger.error("ASR模型预加载失败: %s, error=%s", model_id, e)
            progress.advance(f"已完成 ASR 模型 {model_id}")

        # 2. 预加载语音活动检测模型(VAD)
        progress.update("加载语音活动检测模型(VAD)")
        try:
            from ..services.asr.engines import get_global_vad_model

            vad_model = get_global_vad_model(asr_device)
            if vad_model:
                result["vad_model"]["loaded"] = True
            else:
                result["vad_model"]["error"] = "语音活动检测模型(VAD)加载后返回None"
        except Exception as e:
            result["vad_model"]["error"] = str(e)
            logger.error("语音活动检测模型(VAD)加载失败: %s", e)
        progress.advance("已完成语音活动检测模型(VAD)")

        # 3. 预加载标点符号模型 (离线版)
        if paraformer_enabled:
            progress.update("加载标点符号模型(离线)")
            try:
                from ..services.asr.engines import get_global_punc_model

                punc_model = get_global_punc_model(asr_device)
                if punc_model:
                    result["punc_model"]["loaded"] = True
                else:
                    result["punc_model"]["error"] = "标点符号模型加载后返回None"
            except Exception as e:
                result["punc_model"]["error"] = str(e)
                logger.error("标点符号模型(离线)加载失败: %s", e)
            progress.advance("已完成标点符号模型(离线)")

        # 4. 预加载实时标点符号模型 (如果启用)
        if paraformer_enabled and settings.ASR_ENABLE_REALTIME_PUNC:
            progress.update("加载标点符号模型(实时)")
            try:
                from ..services.asr.engines import get_global_punc_realtime_model

                punc_realtime_model = get_global_punc_realtime_model(asr_device)
                if punc_realtime_model:
                    result["punc_realtime_model"]["loaded"] = True
                else:
                    result["punc_realtime_model"]["error"] = "实时标点符号模型加载后返回None"
            except Exception as e:
                result["punc_realtime_model"]["error"] = str(e)
                logger.error("实时标点符号模型加载失败: %s", e)
            progress.advance("已完成标点符号模型(实时)")

        # 5. 预加载说话人分离模型 (CAM++) - 必需模型，始终加载
        progress.update("加载说话人分离模型(CAM++)")
        try:
            from ..utils.speaker_diarizer import get_global_diarization_pipeline

            diarization_pipeline = get_global_diarization_pipeline()
            if diarization_pipeline:
                result["speaker_diarization_model"]["loaded"] = True
            else:
                result["speaker_diarization_model"]["error"] = "说话人分离模型加载后返回None"
        except Exception as e:
            result["speaker_diarization_model"]["error"] = str(e)
            logger.error("说话人分离模型(CAM++)加载失败: %s", e)
        progress.advance("已完成说话人分离模型(CAM++)")

    loaded_asr_count = sum(1 for status in result["asr_models"].values() if status["loaded"])
    total_asr_count = len(result["asr_models"])
    extra_loaded = sum(
        1
        for key in ("vad_model", "punc_model", "punc_realtime_model", "speaker_diarization_model")
        if result[key]["loaded"]
    )
    extra_failed = sum(
        1
        for key in ("vad_model", "punc_model", "punc_realtime_model", "speaker_diarization_model")
        if result[key]["error"]
    )
    logger.info(
        "模型预加载完成: asr=%s/%s extra_loaded=%s extra_failed=%s",
        loaded_asr_count,
        total_asr_count,
        extra_loaded,
        extra_failed,
    )

    return result
