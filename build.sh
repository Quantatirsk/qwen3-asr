#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec docker buildx build --platform linux/amd64 --load \
  -f Dockerfile.gpu -t "${IMAGE_TAG:-local/r2t2-asr:dev}" "$@" .
