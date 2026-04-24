#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FunASR-API Server CLI entrypoint."""

import sys
import os

# 强制离线模式，必须在任何 HF/transformers 导入前设置
# 注意：不要设置 HF_HUB_OFFLINE=1，否则 vLLM 会把 model_id 替换为绝对路径
os.environ.setdefault("HF_HUB_LOCAL_FILES_ONLY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("DISABLE_TQDM", "1")

# docker-compose commonly injects HF_ENDPOINT="" when the variable is unset.
# huggingface_hub treats the empty string as an explicit endpoint and later
# builds invalid relative URLs such as "/api/models/...". Normalize blank
# values back to "unset" before any HF/transformers imports happen.
if not (os.getenv("HF_ENDPOINT") or "").strip():
    os.environ.pop("HF_ENDPOINT", None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def _disable_third_party_progress_bars() -> None:
    try:
        from huggingface_hub.utils.tqdm import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
    except Exception:
        pass


def _should_use_tui(workers: int) -> bool:
    mode = (os.getenv("FUNASR_STARTUP_UI") or "auto").strip().lower()
    if os.getenv("FUNASR_TUI_CHILD") == "1":
        return False
    if mode in {"0", "false", "plain", "off"}:
        return False
    if workers != 1:
        return False
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    return mode in {"auto", "1", "true", "tui"}


def _run_server(workers: int) -> None:
    from app.core.config import settings
    import uvicorn

    print(f"🚀 FunASR-API | http://{settings.HOST}:{settings.PORT} | {settings.DEVICE}")

    if workers == 1 and os.getenv("FUNASR_TUI_CHILD") != "1":
        from app.bootstrap import run_cli_preflight

        if not run_cli_preflight():
            sys.exit(1)
    elif workers > 1:
        print(f"多Worker模式({workers})，启动前仅进行最小 preflight")

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=workers,
        reload=settings.DEBUG if workers == 1 else False,
        log_level="debug" if settings.DEBUG else settings.LOG_LEVEL.lower(),
        access_log=True,
    )


def main() -> None:
    """主入口"""
    workers = int(os.getenv("WORKERS", "1"))
    _disable_third_party_progress_bars()

    if _should_use_tui(workers):
        try:
            from app.cli.startup_tui import run_tui
        except ImportError as exc:
            print(f"TUI 启动界面不可用，回退普通模式: {exc}")
        else:
            sys.exit(run_tui(sys.argv[1:]))

    try:
        _run_server(workers)
    except KeyboardInterrupt:
        print("\n已停止")
        sys.exit(0)
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
