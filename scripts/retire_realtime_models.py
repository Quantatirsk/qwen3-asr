"""Dry-run or remove ONLY retired realtime weights from a project-owned cache."""

import argparse
import json
import shutil
from pathlib import Path

RETIRED_MODELS = (
    "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
    "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
)


def candidates(root):
    root = Path(root).absolute()
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("Refusing a cache root with symlinked ancestors")
    paths = []
    for model in RETIRED_MODELS:
        path = root / model
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"Refusing a symlinked model directory: {path}")
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"Not a model directory: {path}")
            paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models/modelscope/hub/models",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the exact allowlisted directories after offline regression",
    )
    args = parser.parse_args()
    paths = candidates(args.cache_root)
    print(
        json.dumps(
            {"apply": args.apply, "directories": [str(p) for p in paths]}, indent=2
        )
    )
    if args.apply:
        for path in paths:
            shutil.rmtree(path)
