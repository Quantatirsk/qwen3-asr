#!/usr/bin/env python3
"""Validate a staged local Qwen safetensors directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _load_weight_map(index_path: Path) -> dict[str, str]:
    try:
        payload: Any = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors index {index_path.name}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"invalid safetensors index object: {index_path.name}")
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"missing weight_map in safetensors index: {index_path.name}")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise ValueError(f"invalid weight_map entries: {index_path.name}")
    return weight_map


def validate_model_directory(model_dir: Path) -> list[str]:
    errors: list[str] = []
    resolved_root = model_dir.resolve(strict=False)
    config_path = model_dir / "config.json"
    if not config_path.is_file() or config_path.stat().st_size == 0:
        errors.append("config.json is missing or empty")

    index_paths = sorted(model_dir.glob("*.safetensors.index.json"))
    if not index_paths:
        weight_paths = [
            path
            for path in model_dir.glob("*.safetensors")
            if path.is_file() and path.stat().st_size > 0
        ]
        if not weight_paths:
            errors.append("no non-empty safetensors weight file was found")
        return errors

    referenced_shards: set[Path] = set()
    for index_path in index_paths:
        try:
            weight_map = _load_weight_map(index_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for shard_name in set(weight_map.values()):
            shard_path = (model_dir / shard_name).resolve(strict=False)
            try:
                shard_path.relative_to(resolved_root)
            except ValueError:
                errors.append(f"weight shard escapes model directory: {shard_name}")
                continue
            referenced_shards.add(shard_path)

    for shard_path in sorted(referenced_shards):
        if not shard_path.is_file() or shard_path.stat().st_size == 0:
            errors.append(f"weight shard is missing or empty: {shard_path.name}")
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} MODEL_DIRECTORY", file=sys.stderr)
        return 2

    model_dir = Path(sys.argv[1])
    if not model_dir.is_dir():
        print(f"ERROR: model directory does not exist: {model_dir}", file=sys.stderr)
        return 1

    errors = validate_model_directory(model_dir)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
