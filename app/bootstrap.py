# -*- coding: utf-8 -*-
"""Shared bootstrap helpers for process startup."""

from __future__ import annotations

import importlib.util
import sys


def validate_apple_silicon_runtime() -> bool:
    """Validate Apple Silicon MLX runtime before model warmup."""
    try:
        from app.core.config import settings
        from app.core.device import detect_device

        if detect_device(settings.DEVICE) != "mps":
            return True

        if importlib.util.find_spec("mlx_qwen3_asr") is not None:
            return True

        print("❌ 当前 Python 环境未安装 mlx-qwen3-asr。")
        print("Apple Silicon 上的 Qwen3-ASR 现已使用 MLX backend。")
        print("请安装 Apple Silicon 依赖后重试：")
        print("  uv sync --group apple-silicon")
        return False
    except Exception as exc:
        print(f"⚠️  Apple Silicon 运行时检查失败: {exc}")
        return False


def ensure_models_downloaded(interactive: bool) -> bool:
    """Ensure declared deployment models already exist locally."""
    try:
        from app.utils.download_models import check_all_models, download_models

        missing = check_all_models()
        if not missing:
            return True

        print(f"\n⚠️  检测到 {len(missing)} 个模型未下载")
        for model_id, *_ in missing:
            print(f"  - {model_id}")

        if not interactive:
            print("\n非交互式终端，自动下载模型...")
            return download_models(auto_mode=True)

        response = input("\n自动下载? [Y/n] ").strip().lower()
        if response in ("", "y", "yes"):
            success = download_models(auto_mode=True)
            print("✅ 下载完成" if success else "❌ 下载失败")
            return success

        print("⚠️  跳过下载，将在启动完整性检查时失败")
        return False
    except Exception as exc:
        print(f"⚠️  模型检查失败: {exc}")
        return False


def run_cli_preflight() -> bool:
    """Preflight checks for the CLI entrypoint."""
    return validate_apple_silicon_runtime() and ensure_models_downloaded(
        interactive=sys.stdin.isatty(),
    )
