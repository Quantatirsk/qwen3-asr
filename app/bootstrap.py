# -*- coding: utf-8 -*-
"""Shared bootstrap helpers for process startup."""

from __future__ import annotations

import sys


def ensure_models_downloaded(interactive: bool) -> bool:
    """Ensure declared deployment models exist locally, downloading if needed."""
    try:
        from app.infrastructure import is_huggingface_offline
        from app.utils.download_models import check_all_models, download_models

        missing = check_all_models(include_qwen=False)
        if not missing:
            return True

        print(f"\n⚠️  检测到 {len(missing)} 个模型未下载")
        for model_id, *_ in missing:
            print(f"  - {model_id}")

        if is_huggingface_offline():
            print("\nHF_HUB_OFFLINE=1, startup will not download models.")
            print("The image is missing its embedded support models; rebuild it.")
            return False

        print("\n将自动下载缺失模型后继续启动。")
        if download_models(auto_mode=True, include_qwen=False):
            return True

        print("\n模型自动下载失败。")
        if interactive:
            print("Run this command to diagnose the support model download:")
            print("  uv run python -m app.utils.download_models --exclude-qwen")
        else:
            print("非交互式终端下请确认网络可用，或预先准备模型缓存。")
        return False
    except Exception as exc:
        print(f"⚠️  模型检查失败: {exc}")
        return False


def run_cli_preflight() -> bool:
    """Preflight checks for the CLI entrypoint."""
    return ensure_models_downloaded(interactive=sys.stdin.isatty())
