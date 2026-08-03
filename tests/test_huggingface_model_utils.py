from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.infrastructure.model_utils import (
    find_huggingface_snapshot_dir,
    get_huggingface_cache_root,
    get_huggingface_model_cache_dir,
    is_huggingface_offline,
    resolve_huggingface_snapshot_dir,
)


@contextmanager
def patched_env(**updates: str | None) -> Iterator[None]:
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_huggingface_cache_priority_and_snapshot_resolution() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_root = Path(temp_dir) / "hub"
        model_dir = cache_root / "models--Qwen--Qwen3-ASR-1.7B"
        snapshot_dir = model_dir / "snapshots" / "abc123"
        (model_dir / "refs").mkdir(parents=True)
        snapshot_dir.mkdir(parents=True)
        (model_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
        (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

        with patched_env(
            FUNASR_TEST_HF_CACHE=str(cache_root),
            HF_HUB_CACHE="$FUNASR_TEST_HF_CACHE",
            HF_HOME=str(Path(temp_dir) / "ignored-home"),
            XDG_CACHE_HOME=str(Path(temp_dir) / "ignored-xdg"),
        ):
            assert get_huggingface_cache_root() == cache_root
            assert (
                get_huggingface_model_cache_dir("Qwen/Qwen3-ASR-1.7B")
                == model_dir
            )
            assert (
                find_huggingface_snapshot_dir("Qwen/Qwen3-ASR-1.7B")
                == snapshot_dir.resolve()
            )
            assert (
                resolve_huggingface_snapshot_dir("Qwen/Qwen3-ASR-1.7B")
                == snapshot_dir.resolve()
            )


def test_huggingface_offline_flag() -> None:
    with patched_env(HF_HUB_OFFLINE="1"):
        assert is_huggingface_offline()

    with patched_env(HF_HUB_OFFLINE="0"):
        assert not is_huggingface_offline()


if __name__ == "__main__":
    test_huggingface_cache_priority_and_snapshot_resolution()
    test_huggingface_offline_flag()
