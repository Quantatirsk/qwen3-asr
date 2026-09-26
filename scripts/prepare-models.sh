#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HOME="${HF_HOME:-$PWD/models/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/models/modelscope/hub}"
exec uv run python -m app.utils.download_models "$@"
