# -*- coding: utf-8 -*-
"""Shared bootstrap helpers for process startup."""

from __future__ import annotations

import os
import sys


def _is_truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def ensure_models_downloaded(interactive: bool) -> bool:
    """Ensure declared deployment models exist locally, downloading if needed."""
    try:
        from app.utils.download_models import check_all_models, download_models

        missing = check_all_models()
        if not missing:
            return True

        print(f"\n⚠️  检测到 {len(missing)} 个模型未下载")
        for model_id, *_ in missing:
            print(f"  - {model_id}")

        print("\n将自动下载缺失模型后继续启动。")
        if download_models(auto_mode=True):
            return True

        print("\n模型自动下载失败。")
        if _is_truthy_env(os.getenv("HF_HUB_LOCAL_FILES_ONLY")):
            print("当前设置了 HF_HUB_LOCAL_FILES_ONLY=1，HuggingFace 模型不会联网下载。")
        if interactive:
            print("可手动运行以下命令排查：")
            print("  uv run python -m app.utils.download_models")
            print("  ./scripts/prepare-models.sh")
        else:
            print("非交互式终端下请确认网络可用，或预先准备模型缓存。")
        return False
    except Exception as exc:
        print(f"⚠️  模型检查失败: {exc}")
        return False


def run_cli_preflight() -> bool:
    """Preflight checks for the CLI entrypoint."""
    return ensure_models_downloaded(interactive=sys.stdin.isatty())
