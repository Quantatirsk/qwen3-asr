#!/usr/bin/env bash
set -euo pipefail
hf download netease-youdao/Confucius4-R2T2 --revision 185ce639118ad1362d049ca0d8ed04b6ec5cd6c9
# Offline Qwen, forced aligner, VAD and speaker models use the existing exporter.
# To prepare an entirely new machine: uv run python -m app.utils.download_models --scope all
docker compose -f docker-compose.realtime.yml build --builder default asr
