#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${QWEN3_ASR_PROJECT_ROOT:-/workspace/qwen3-asr}"
API_PYTHON="${QWEN3_ASR_API_PYTHON:-/opt/qwen3-asr-venv/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_VALIDATOR="${SCRIPT_DIR}/validate-qwen-model.py"
MODEL_PATH="${QWEN_ASCEND_MODEL_PATH:-/workspace/hf_models/Qwen3-ASR-1.7B}"
SERVED_MODEL="${QWEN_VLLM_SERVED_MODEL:-qwen3-asr}"
VLLM_HOST="${QWEN_VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${QWEN_VLLM_PORT:-17004}"
API_HOST="${QWEN3_ASR_API_HOST:-0.0.0.0}"
API_PORT="${QWEN3_ASR_API_PORT:-17003}"
TENSOR_PARALLEL_SIZE="${QWEN_ASCEND_TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${QWEN_ASCEND_MAX_MODEL_LEN:-4096}"
MEMORY_UTILIZATION="${QWEN_ASCEND_MEMORY_UTILIZATION:-0.9}"
VLLM_LOG="${QWEN_VLLM_LOG:-${PROJECT_ROOT}/logs/vllm.log}"
PROCESS_POLL_INTERVAL="${QWEN_PROCESS_POLL_INTERVAL_SEC:-1}"
STARTUP_TIMEOUT_SEC="${QWEN_VLLM_STARTUP_TIMEOUT_SEC:-600}"

VLLM_PID=""
API_PID=""

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

cleanup() {
    local process_id
    trap - INT TERM EXIT
    for process_id in "${API_PID}" "${VLLM_PID}"; do
        if [[ -n "${process_id}" ]] && kill -0 "${process_id}" 2>/dev/null; then
            kill "${process_id}" 2>/dev/null || true
        fi
    done
    for process_id in "${API_PID}" "${VLLM_PID}"; do
        if [[ -n "${process_id}" ]]; then
            wait "${process_id}" 2>/dev/null || true
        fi
    done
}

handle_signal() {
    exit 143
}

command -v vllm >/dev/null 2>&1 || die "vllm executable is unavailable"
require_positive_integer "QWEN_VLLM_PORT" "${VLLM_PORT}"
require_positive_integer "QWEN3_ASR_API_PORT" "${API_PORT}"
require_positive_integer "QWEN_ASCEND_TENSOR_PARALLEL_SIZE" "${TENSOR_PARALLEL_SIZE}"
require_positive_integer "QWEN_ASCEND_MAX_MODEL_LEN" "${MAX_MODEL_LEN}"
[[ -x "${API_PYTHON}" ]] || die "API Python is unavailable: ${API_PYTHON}"
[[ -f "${PROJECT_ROOT}/start.py" ]] || die "API entrypoint is unavailable: ${PROJECT_ROOT}/start.py"
[[ -f "${MODEL_VALIDATOR}" ]] || die "model validator is unavailable: ${MODEL_VALIDATOR}"
python3 "${MODEL_VALIDATOR}" "${MODEL_PATH}" >/dev/null 2>&1 \
    || die "staged model is incomplete: ${MODEL_PATH}"

mkdir -p "$(dirname "${VLLM_LOG}")" "${PROJECT_ROOT}/temp" "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/logs"

trap cleanup EXIT
trap handle_signal INT TERM

VLLM_COMMAND=(
    vllm serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${MEMORY_UTILIZATION}"
    --enforce-eager
)
VLLM_COMMAND+=("$@")

echo "Starting Ascend vLLM: ${VLLM_HOST}:${VLLM_PORT}"
"${VLLM_COMMAND[@]}" > >(tee -a "${VLLM_LOG}") 2>&1 &
VLLM_PID=$!

VLLM_READY_URL="http://${VLLM_HOST}:${VLLM_PORT}/health"
VLLM_DEADLINE_EPOCH=$(( $(date +%s) + STARTUP_TIMEOUT_SEC ))
echo "Waiting for vLLM to become ready at ${VLLM_READY_URL} (timeout ${STARTUP_TIMEOUT_SEC}s)"
while true; do
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
        if curl -fsS --max-time 5 "${VLLM_READY_URL}" >/dev/null 2>&1; then
            echo "vLLM is ready at ${VLLM_READY_URL}"
            break
        fi
        if (( $(date +%s) >= VLLM_DEADLINE_EPOCH )); then
            echo "ERROR: timed out waiting ${STARTUP_TIMEOUT_SEC}s for vLLM at ${VLLM_READY_URL}" >&2
            exit 1
        fi
        sleep "${PROCESS_POLL_INTERVAL}"
    else
        echo "ERROR: vLLM process exited before becoming ready" >&2
        exit 1
    fi
done



export HOST="${API_HOST}"
export PORT="${API_PORT}"
export DEVICE="cpu"
export SPEAKER_DIARIZATION_DEVICE="cpu"
export FUNASR_STARTUP_UI="plain"
export QWEN_VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
export QWEN_VLLM_SERVED_MODEL="${SERVED_MODEL}"

echo "Starting Qwen3-ASR API: ${API_HOST}:${API_PORT}"
cd "${PROJECT_ROOT}"
"${API_PYTHON}" "${PROJECT_ROOT}/start.py" &
API_PID=$!

while kill -0 "${API_PID}" 2>/dev/null && kill -0 "${VLLM_PID}" 2>/dev/null; do
    sleep "${PROCESS_POLL_INTERVAL}"
done

if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    set +e
    wait "${VLLM_PID}"
    VLLM_STATUS=$?
    set -e
    VLLM_PID=""
    ((VLLM_STATUS != 0)) || VLLM_STATUS=1
    echo "ERROR: vLLM exited after startup" >&2
    exit "${VLLM_STATUS}"
fi

set +e
wait "${API_PID}"
API_STATUS=$?
set -e
API_PID=""
exit "${API_STATUS}"
