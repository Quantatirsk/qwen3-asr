#!/usr/bin/env bash

set -Eeuo pipefail

DEFAULT_REVISION="7278e1e70fe206f11671096ffdd38061171dd6e5"
DEFAULT_SOURCE="/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/${DEFAULT_REVISION}"
DEFAULT_DESTINATION="/workspace/hf_models/Qwen3-ASR-1.7B"

SOURCE_DIR="${1:-${QWEN_MODEL_SOURCE:-${DEFAULT_SOURCE}}}"
DESTINATION_DIR="${2:-${QWEN_ASCEND_MODEL_PATH:-${DEFAULT_DESTINATION}}}"
DESTINATION_PARENT="$(dirname "${DESTINATION_DIR}")"
STAGING_DIR="${DESTINATION_DIR}.partial.$$"
COPY_PID=""

die() {
    echo "ERROR: $*" >&2
    exit 1
}

directory_size_bytes() {
    local target="$1"
    local size

    if size=$(du -sbL "${target}" 2>/dev/null | awk '{print $1}'); then
        echo "${size:-0}"
        return
    fi

    size=$(du -skL "${target}" | awk '{print $1 * 1024}')
    echo "${size:-0}"
}

cleanup() {
    if [[ -n "${COPY_PID}" ]] && kill -0 "${COPY_PID}" 2>/dev/null; then
        kill "${COPY_PID}" 2>/dev/null || true
        wait "${COPY_PID}" 2>/dev/null || true
    fi
    if [[ -d "${STAGING_DIR}" ]]; then
        rm -rf -- "${STAGING_DIR}"
    fi
}

handle_signal() {
    exit 130
}

[[ -d "${SOURCE_DIR}" ]] || die "model source does not exist: ${SOURCE_DIR}"
[[ -f "${SOURCE_DIR}/config.json" ]] || die "config.json is missing: ${SOURCE_DIR}"

if [[ -f "${DESTINATION_DIR}/config.json" ]]; then
    echo "Model already staged: ${DESTINATION_DIR}"
    exit 0
fi
[[ ! -e "${DESTINATION_DIR}" ]] || die "destination exists but is incomplete: ${DESTINATION_DIR}"

mkdir -p "${DESTINATION_PARENT}" "${STAGING_DIR}"
trap cleanup EXIT
trap handle_signal INT TERM

TOTAL_BYTES=$(directory_size_bytes "${SOURCE_DIR}")
echo "Source: ${SOURCE_DIR}"
echo "Destination: ${DESTINATION_DIR}"
echo "Total bytes: ${TOTAL_BYTES}"

cp -aL "${SOURCE_DIR}/." "${STAGING_DIR}/" &
COPY_PID=$!
PREVIOUS_BYTES=0
PREVIOUS_SECONDS=${SECONDS}

while kill -0 "${COPY_PID}" 2>/dev/null; do
    COPIED_BYTES=$(directory_size_bytes "${STAGING_DIR}")
    CURRENT_SECONDS=${SECONDS}
    ELAPSED=$((CURRENT_SECONDS - PREVIOUS_SECONDS))
    ((ELAPSED > 0)) || ELAPSED=1
    SPEED_MB=$(awk "BEGIN { printf \"%.2f\", ((${COPIED_BYTES} - ${PREVIOUS_BYTES}) / ${ELAPSED}) / 1024 / 1024 }")
    PERCENT=$(awk "BEGIN { if (${TOTAL_BYTES} > 0) printf \"%.2f\", (${COPIED_BYTES} / ${TOTAL_BYTES}) * 100; else print \"0.00\" }")
    printf "\r%s copied=%'d/%'d bytes (%6.2f%%) %8s MB/s" \
        "$(date '+%H:%M:%S')" \
        "${COPIED_BYTES}" \
        "${TOTAL_BYTES}" \
        "${PERCENT}" \
        "${SPEED_MB}"
    PREVIOUS_BYTES=${COPIED_BYTES}
    PREVIOUS_SECONDS=${CURRENT_SECONDS}
    sleep "${QWEN_COPY_PROGRESS_INTERVAL_SEC:-0.2}"
done

wait "${COPY_PID}"
COPY_PID=""
printf "\n"

[[ -f "${STAGING_DIR}/config.json" ]] || die "copied model is missing config.json"
shopt -s nullglob
WEIGHT_FILES=("${STAGING_DIR}"/*.safetensors "${STAGING_DIR}"/*.safetensors.index.json)
shopt -u nullglob
(( ${#WEIGHT_FILES[@]} > 0 )) || die "copied model is missing safetensors weights"

mv "${STAGING_DIR}" "${DESTINATION_DIR}"
trap - EXIT INT TERM

echo "Copy finished."
du -sh "${DESTINATION_DIR}"
