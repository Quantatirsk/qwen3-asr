#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export NEMOTRON_MODEL_PATH="${NEMOTRON_MODEL_PATH:-$PWD/models/nemotron-3-diarization}"
export HF_HOME="${HF_HOME:-$PWD/models/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/models/modelscope/hub}"
exec uv run python -m app.utils.download_models "$@"
