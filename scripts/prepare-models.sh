#!/bin/bash
#
# Model Export Tool for Offline Deployment
#

set -euo pipefail

SCOPE="${1:-offline}"
case "$SCOPE" in offline|realtime|all) ;; *) printf 'Usage: %s [offline|realtime|all]\n' "$0" >&2; exit 1 ;; esac
EXPORT_ROOT="${EXPORT_ROOT:-./model_exports/$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${EXPORT_ROOT}/models"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

info() { echo "[INFO] $1"; }
die() { echo "[ERROR] $1" >&2; exit 1; }

# Check if running from project root or scripts directory
if [ -f "$PROJECT_ROOT/app/utils/download_models.py" ]; then
    cd "$PROJECT_ROOT"
elif [ -f "$SCRIPT_DIR/../app/utils/download_models.py" ]; then
    cd "$SCRIPT_DIR/.."
else
    die "Cannot find app/utils/download_models.py"
fi

command -v uv >/dev/null 2>&1 || die "uv not found; install uv first"

# Confirm
info "Export settings:"
info "  Models: ${SCOPE} (Qwen offline / pinned R2T2 realtime)"
info "  Output: ${OUTPUT_DIR}"
read -p "Start export? [Y/n]: " confirm
if [[ $confirm =~ ^[Nn]$ ]]; then
    echo "Cancelled."
    exit 0
fi

info "Exporting models..."

# Never clear an active model cache during export.
[ ! -e "${OUTPUT_DIR}" ] || die "Output already exists: ${OUTPUT_DIR}"
mkdir -p "${EXPORT_ROOT}"

# Run Python export
uv run python -m app.utils.download_models --scope "${SCOPE}" --export-dir "${OUTPUT_DIR}"

info "Packaging..."

PACKAGE="qwen3-asr-models-$(date +%Y%m%d-%H%M).tar.gz"

# Use pigz for multi-threaded compression if available
if command -v pigz &> /dev/null; then
    info "Using pigz for multi-threaded compression..."
    # Get CPU cores (cross-platform: Linux and macOS)
    if command -v nproc &> /dev/null; then
        CPU_CORES=$(nproc)
    elif command -v sysctl &> /dev/null; then
        CPU_CORES=$(sysctl -n hw.ncpu)
    else
        CPU_CORES=4
    fi
    tar -C "${EXPORT_ROOT}" -cf - models | pigz -p "${CPU_CORES}" > "${PACKAGE}"
else
    info "pigz not found, using standard gzip..."
    tar -C "${EXPORT_ROOT}" -czf "${PACKAGE}" models
fi

SIZE=$(du -sh "${PACKAGE}" | cut -f1)

# Done
info "Export complete"
info "Package: ${PACKAGE}"
info "Size: ${SIZE}"
echo "To deploy on an offline server:"
echo "1. Copy package:  scp ${PACKAGE} user@server:/opt/qwen3-asr/"
echo "2. Extract:       tar -xzvf ${PACKAGE}"
echo "3. Start service: docker-compose up -d"
