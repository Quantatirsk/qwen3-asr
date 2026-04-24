#!/usr/bin/env bash
set -euo pipefail

DEFAULT_CMD_1="/opt/venv/bin/python"
DEFAULT_CMD_2="start.py"

normalize_cuda_devices() {
  local raw="${CUDA_VISIBLE_DEVICES:-}"
  raw="${raw// /}"

  if [[ -z "$raw" || "$raw" == "none" || "$raw" == "void" ]]; then
    echo ""
    return 0
  fi

  if [[ "$raw" == "all" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      mapfile -t gpu_indexes < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed '/^$/d')
      if [[ ${#gpu_indexes[@]} -eq 0 ]]; then
        echo ""
        return 0
      fi
      local joined
      joined=$(IFS=,; echo "${gpu_indexes[*]}")
      echo "$joined"
      return 0
    fi
    echo ""
    return 0
  fi

  echo "$raw"
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout_sec="$3"
  local deadline=$((SECONDS + timeout_sec))

  while (( SECONDS < deadline )); do
    if (echo >/dev/tcp/"$host"/"$port") >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

is_positive_integer() {
  [[ "$1" =~ ^[0-9]+$ ]] && (( "$1" > 0 ))
}

start_internal_nginx_mode() {
  local devices_csv="$1"
  local bind_host="${MULTI_GPU_BIND_HOST:-127.0.0.1}"
  local base_port="18000"
  local public_port="${PORT:-8000}"
  local ready_timeout="${MULTI_GPU_READY_TIMEOUT:-180}"
  local rate_limit_rps="${NGINX_RATE_LIMIT_RPS:-0}"
  local rate_limit_burst="${NGINX_RATE_LIMIT_BURST:-0}"

  local valid_devices=()
  local devices=()
  if [[ -n "$devices_csv" ]]; then
    IFS=',' read -r -a devices <<< "$devices_csv"
    for dev in "${devices[@]}"; do
      if [[ -n "$dev" ]]; then
        valid_devices+=("$dev")
      fi
    done
  fi

  if [[ "$rate_limit_rps" != "0" ]] && ! is_positive_integer "$rate_limit_rps"; then
    echo "[entrypoint] Invalid NGINX_RATE_LIMIT_RPS=${rate_limit_rps}, fallback to 0 (disabled)"
    rate_limit_rps="0"
  fi
  if [[ "$rate_limit_burst" != "0" ]] && ! is_positive_integer "$rate_limit_burst"; then
    echo "[entrypoint] Invalid NGINX_RATE_LIMIT_BURST=${rate_limit_burst}, fallback to 0"
    rate_limit_burst="0"
  fi
  if [[ "$rate_limit_rps" != "0" && "$rate_limit_burst" == "0" ]]; then
    # By default, give short burst headroom equal to rate limit.
    rate_limit_burst="$rate_limit_rps"
  fi

  if ! command -v nginx >/dev/null 2>&1; then
    echo "[entrypoint] nginx not found in image; cannot start internal proxy mode"
    exit 1
  fi

  local backend_ports=()
  local backend_pids=()
  local idx=0
  local default_log_file="${LOG_FILE:-/app/logs/qwen3-asr.log}"

  if [[ ${#valid_devices[@]} -eq 0 ]]; then
    local single_port="$base_port"
    local single_log_file="${default_log_file%.log}-gpu0.log"
    backend_ports+=("$single_port")

    echo "[entrypoint] No explicit multi-GPU list detected, starting single backend instance"
    HOST="$bind_host" \
    PORT="$single_port" \
    LOG_FILE="$single_log_file" \
    "$DEFAULT_CMD_1" "$DEFAULT_CMD_2" &
    backend_pids+=("$!")
  else
    if [[ ${#valid_devices[@]} -eq 1 ]]; then
      echo "[entrypoint] Single GPU detected (${valid_devices[0]}), starting one backend instance"
      local single_gpu_port="$base_port"
      local single_gpu_log_file="${default_log_file%.log}-gpu0.log"
      backend_ports+=("$single_gpu_port")

      CUDA_VISIBLE_DEVICES="${valid_devices[0]}" \
      DEVICE="cuda:0" \
      HOST="$bind_host" \
      PORT="$single_gpu_port" \
      LOG_FILE="$single_gpu_log_file" \
      "$DEFAULT_CMD_1" "$DEFAULT_CMD_2" &
      backend_pids+=("$!")
    else
      echo "[entrypoint] Multi-GPU detected, starting one instance per device: ${valid_devices[*]}"
      for dev in "${valid_devices[@]}"; do
        local port=$((base_port + idx))
        local instance_log_file="${default_log_file%.log}-gpu${idx}.log"
        backend_ports+=("$port")

        echo "[entrypoint] Starting ASR instance #${idx} on GPU ${dev}, bind ${bind_host}:${port}"
        CUDA_VISIBLE_DEVICES="$dev" \
        DEVICE="cuda:0" \
        WORKERS="1" \
        HOST="$bind_host" \
        PORT="$port" \
        LOG_FILE="$instance_log_file" \
        "$DEFAULT_CMD_1" "$DEFAULT_CMD_2" &

        backend_pids+=("$!")
        idx=$((idx + 1))
      done
    fi
  fi

  local port
  for port in "${backend_ports[@]}"; do
    if ! wait_for_port "$bind_host" "$port" "$ready_timeout"; then
      echo "[entrypoint] Backend instance on ${bind_host}:${port} failed to become ready in ${ready_timeout}s"
      for pid in "${backend_pids[@]}"; do
        kill "$pid" >/dev/null 2>&1 || true
      done
      wait >/dev/null 2>&1 || true
      exit 1
    fi
  done

  local nginx_conf="/tmp/qwen3-asr-internal-nginx.conf"
  {
    echo "worker_processes auto;"
    echo "events { worker_connections 1024; }"
    echo "http {"
    echo "    limit_req_status 429;"
    if is_positive_integer "$rate_limit_rps"; then
      # Global token bucket for the whole service, not per-client-IP.
      echo "    limit_req_zone \$server_name zone=api_rps:10m rate=${rate_limit_rps}r/s;"
    fi
    echo
    echo "    upstream qwen3_asr_upstream {"
    echo "        least_conn;"
    for port in "${backend_ports[@]}"; do
      echo "        server ${bind_host}:${port} max_fails=3 fail_timeout=10s;"
    done
    echo "        keepalive 128;"
    echo "    }"
    echo
    echo "    map \$http_upgrade \$connection_upgrade {"
    echo "        default upgrade;"
    echo "        '' close;"
    echo "    }"
    echo
    echo "    server {"
    echo "        listen ${public_port};"
    echo "        server_name _;"
    echo "        client_max_body_size 2048m;"
    echo
    echo "        location / {"
    if is_positive_integer "$rate_limit_rps"; then
      echo "            limit_req zone=api_rps burst=${rate_limit_burst} nodelay;"
    fi
    echo "            proxy_pass http://qwen3_asr_upstream;"
    echo "            proxy_http_version 1.1;"
    echo "            proxy_set_header Host \$host;"
    echo "            proxy_set_header X-Real-IP \$remote_addr;"
    echo "            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;"
    echo "            proxy_set_header X-Forwarded-Proto \$scheme;"
    echo "            proxy_set_header Upgrade \$http_upgrade;"
    echo "            proxy_set_header Connection \$connection_upgrade;"
    echo "            proxy_connect_timeout 10s;"
    echo "            proxy_send_timeout 3600s;"
    echo "            proxy_read_timeout 3600s;"
    echo "            send_timeout 3600s;"
    echo "            proxy_buffering off;"
    echo "        }"
    echo "    }"
    echo "}"
  } > "$nginx_conf"

  local nginx_pid=""

  cleanup() {
    set +e
    if [[ -n "$nginx_pid" ]]; then
      kill "$nginx_pid" >/dev/null 2>&1 || true
    fi
    for pid in "${backend_pids[@]}"; do
      kill "$pid" >/dev/null 2>&1 || true
    done
    wait >/dev/null 2>&1 || true
  }
  trap cleanup EXIT INT TERM

  echo "[entrypoint] Starting internal nginx load balancer on :${public_port}"
  nginx -c "$nginx_conf" -g "daemon off;" &
  nginx_pid="$!"

  while true; do
    if ! kill -0 "$nginx_pid" >/dev/null 2>&1; then
      echo "[entrypoint] nginx exited unexpectedly"
      exit 1
    fi
    for pid in "${backend_pids[@]}"; do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        echo "[entrypoint] backend process ${pid} exited unexpectedly"
        exit 1
      fi
    done
    sleep 2
  done
}

has_default_cmd=false
if [[ $# -eq 2 && "$1" == "$DEFAULT_CMD_1" && "$2" == "$DEFAULT_CMD_2" ]]; then
  has_default_cmd=true
fi

# If user passes a custom command, respect it and bypass auto multi-GPU logic.
if [[ $# -gt 0 && "$has_default_cmd" != "true" ]]; then
  exec "$@"
fi

devices_csv="$(normalize_cuda_devices)"
start_internal_nginx_mode "$devices_csv"
