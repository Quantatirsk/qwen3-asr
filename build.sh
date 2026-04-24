#!/bin/bash
#
# Qwen3-ASR Docker Build Tool
# Step-by-step interactive or CLI parameter mode.

set -euo pipefail

# =============================================================================
# Defaults
# =============================================================================

REGISTRY="quantatrisk"
IMAGE_NAME="qwen3-asr"
VERSION="latest"
BUILD_TYPE="all"
PLATFORM="linux/amd64"
PUSH="false"
EXPORT_TAR="false"
EXPORT_DIR="."
NO_CACHE="false"
LANG_MODE="zh"

# =============================================================================
# Localization
# =============================================================================

_resolve_lang() {
  printf "%s" "$LANG_MODE"
}

_msg() {
  local k="$1" l; l="$(_resolve_lang)"
  if [[ "$l" == "zh" ]]; then
    case "$k" in
      title)     echo "Qwen3-ASR Docker 构建工具" ;;
      subtitle)  echo "按提示选择或直接回车使用默认值" ;;
      q_lang)    echo "界面语言" ;;
      q_type)    echo "构建目标" ;;
      q_arch)    echo "目标架构" ;;
      q_version) echo "镜像版本" ;;
      q_push)    echo "推送到仓库" ;;
      q_export)  echo "导出 tar.gz" ;;
      q_output)  echo "输出目录" ;;
      q_reg)     echo "仓库命名空间" ;;
      q_cache)   echo "禁用构建缓存" ;;
      q_confirm) echo "确认并开始构建" ;;
      summary)   echo "构建摘要" ;;
      cancel)    echo "已取消" ;;
      start)     echo "开始构建..." ;;
      done)      echo "构建完成" ;;
      recent)    echo "最近镜像" ;;
      opt_cpu)   echo "CPU" ;;
      opt_gpu)   echo "GPU" ;;
      opt_all)   echo "全部 (CPU + GPU)" ;;
      opt_amd64) echo "amd64 (x86_64)" ;;
      opt_arm64) echo "arm64 (Apple Silicon / ARM)" ;;
      opt_multi) echo "多架构 (amd64 + arm64)" ;;
      err_inv)   echo "无效输入" ;;
      err_opt)   echo "未知选项" ;;
      err_gpu)   echo "GPU 构建仅支持 amd64" ;;
      err_bx)    echo "需要 Docker Buildx" ;;
      yes)       echo "是" ;;
      no)        echo "否" ;;
    esac
  else
    case "$k" in
      title)     echo "Qwen3-ASR Docker Build Tool" ;;
      subtitle)  echo "Select options or press Enter for defaults" ;;
      q_lang)    echo "Language" ;;
      q_type)    echo "Build target" ;;
      q_arch)    echo "Architecture" ;;
      q_version) echo "Image version" ;;
      q_push)    echo "Push to registry" ;;
      q_export)  echo "Export tar.gz" ;;
      q_output)  echo "Output directory" ;;
      q_reg)     echo "Registry namespace" ;;
      q_cache)   echo "Disable build cache" ;;
      q_confirm) echo "Confirm and start build" ;;
      summary)   echo "Build summary" ;;
      cancel)    echo "Cancelled" ;;
      start)     echo "Starting build..." ;;
      done)      echo "Build complete" ;;
      recent)    echo "Recent images" ;;
      opt_cpu)   echo "CPU" ;;
      opt_gpu)   echo "GPU" ;;
      opt_all)   echo "All (CPU + GPU)" ;;
      opt_amd64) echo "amd64 (x86_64)" ;;
      opt_arm64) echo "arm64 (Apple Silicon / ARM)" ;;
      opt_multi) echo "Multi-arch (amd64 + arm64)" ;;
      err_inv)   echo "Invalid input" ;;
      err_opt)   echo "Unknown option" ;;
      err_gpu)   echo "GPU build only supports amd64" ;;
      err_bx)    echo "Docker Buildx is required" ;;
      yes)       echo "yes" ;;
      no)        echo "no" ;;
    esac
  fi
}

_label_bool() { [[ "$1" == "true" ]] && _msg yes || _msg no; }

# =============================================================================
# Helpers
# =============================================================================

info() { echo "[INFO] $1" >&2; }
warn() { echo "[WARN] $1" >&2; }
die()  { echo "[ERROR] $1" >&2; exit 1; }

parse_arch() {
  case "$1" in
    amd64) echo "linux/amd64" ;;
    arm64) echo "linux/arm64" ;;
    multi) echo "linux/amd64,linux/arm64" ;;
    *) die "$(_msg err_inv): $1" ;;
  esac
}

arch_label() {
  case "$1" in
    linux/amd64) echo "amd64" ;;
    linux/arm64) echo "arm64" ;;
    linux/amd64,linux/arm64) echo "multi" ;;
    *) echo "$1" ;;
  esac
}

ensure_buildx() {
  docker buildx version >/dev/null 2>&1 || die "$(_msg err_bx)"
  if ! docker buildx inspect qwen3-asr-builder >/dev/null 2>&1; then
    info "Creating buildx builder: qwen3-asr-builder"
    docker buildx create --name qwen3-asr-builder --driver docker-container --use >/dev/null
  else
    docker buildx use qwen3-asr-builder >/dev/null
  fi
}

export_compressor() {
  command -v pigz >/dev/null 2>&1 && echo "pigz -f" || echo "gzip -f"
}

# =============================================================================
# Build Core
# =============================================================================

build_image() {
  local target="$1" dockerfile="$2" tag="$3" platform="$4"
  local args=(buildx build --platform "$platform" -f "$dockerfile" -t "$tag")

  [[ "$NO_CACHE" == "true" ]] && args+=(--no-cache)

  if [[ "$platform" == *","* ]]; then
    [[ "$EXPORT_TAR" == "true" ]] && warn "Multi-arch cannot export tar.gz"
    args+=(--push)
  elif [[ "$PUSH" == "true" ]]; then
    args+=(--push)
  elif [[ "$EXPORT_TAR" == "true" ]]; then
    mkdir -p "$EXPORT_DIR"
    local suffix="${target}-${VERSION}-$(basename "$platform")"
    local tar="${EXPORT_DIR}/${IMAGE_NAME}-${suffix}.tar"
    args+=(--output "type=docker,dest=${tar}")
  else
    args+=(--load)
  fi

  info "Building ${target}: ${tag} (${platform})"
  docker "${args[@]}" .

  if [[ "$EXPORT_TAR" == "true" && "$platform" != *","* && "$PUSH" != "true" ]]; then
    local suffix="${target}-${VERSION}-$(basename "$platform")"
    local tar="${EXPORT_DIR}/${IMAGE_NAME}-${suffix}.tar"
    if [[ -f "$tar" ]]; then
      info "Compressing ${tar}"
      $(export_compressor) "$tar"
      info "Exported ${tar}.gz"
    fi
  fi
}

build_cpu() {
  local tag="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
  [[ "$VERSION" == "latest" ]] && tag="${REGISTRY}/${IMAGE_NAME}:cpu-latest"
  build_image "cpu" "Dockerfile.cpu" "$tag" "$PLATFORM"
}

build_gpu() {
  [[ "$PLATFORM" == *"arm64"* ]] && die "$(_msg err_gpu)"
  build_image "gpu" "Dockerfile.gpu" "${REGISTRY}/${IMAGE_NAME}:gpu-${VERSION}" "linux/amd64"
}

# =============================================================================
# Interactive: step-by-step
# =============================================================================

_ask() {
  local __var_name="$1" prompt="$2" default="$3" l
  local val
  l="$(_resolve_lang)"
  if [[ "$l" == "zh" ]]; then
    read -r -p "${prompt} (默认: ${default}): " val
  else
    read -r -p "${prompt} (default: ${default}): " val
  fi
  printf -v "$__var_name" "%s" "${val:-$default}"
}

_ask_bool() {
  local __var_name="$1" prompt="$2" default="$3" hint l
  local val
  l="$(_resolve_lang)"
  if [[ "$default" == "true" ]]; then
    hint="[Y/n]"
  else
    hint="[y/N]"
  fi
  if [[ "$l" == "zh" ]]; then
    read -r -p "${prompt} ${hint}: " val
  else
    read -r -p "${prompt} ${hint}: " val
  fi
  case "${val:-}" in
    [Yy]|[Yy][Ee][Ss]|是) printf -v "$__var_name" "%s" "true" ;;
    [Nn]|[Nn][Oo]|否)     printf -v "$__var_name" "%s" "false" ;;
    "")                   printf -v "$__var_name" "%s" "$default" ;;
    *) warn "$(_msg err_inv): ${val}"; printf -v "$__var_name" "%s" "$default" ;;
  esac
}

_ask_choice() {
  local __var_name="$1" prompt="$2" default="$3" l
  shift 3
  local labels=() values=() i=1
  while [[ $# -ge 2 ]]; do
    labels+=("$1")
    values+=("$2")
    shift 2
  done

  l="$(_resolve_lang)"
  echo >&2
  echo "${prompt}:" >&2
  local idx=1
  for label in "${labels[@]}"; do
    if [[ "$idx" -eq "$default" ]]; then
      if [[ "$l" == "zh" ]]; then
        echo "  ${idx}) ${label} [默认]" >&2
      else
        echo "  ${idx}) ${label} [default]" >&2
      fi
    else
      echo "  ${idx}) ${label}" >&2
    fi
    ((idx++))
  done

  local val
  if [[ "$l" == "zh" ]]; then
    read -r -p "请选择 [1-${#labels[@]}] (默认: ${default}): " val
  else
    read -r -p "Select [1-${#labels[@]}] (default: ${default}): " val
  fi
  val="${val:-$default}"

  if [[ "$val" =~ ^[0-9]+$ ]] && [[ "$val" -ge 1 && "$val" -le "${#labels[@]}" ]]; then
    printf -v "$__var_name" "%s" "${values[$((val-1))]}"
  else
    if [[ "$l" == "zh" ]]; then
      warn "$(_msg err_inv): ${val}，使用默认值 ${default}"
    else
      warn "$(_msg err_inv): ${val}, using default ${default}"
    fi
    printf -v "$__var_name" "%s" "${values[$((default-1))]}"
  fi
}

interactive_mode() {
  echo
  echo "========================================"
  echo "$(_msg title)"
  echo "$(_msg subtitle)"
  echo "========================================"
  echo

  # 1. Language first so subsequent prompts use it
  _ask_choice LANG_MODE "$(_msg q_lang)" 1 \
    "中文" "zh" \
    "English" "en"
  echo

  # 2. Build type (numbered)
  _ask_choice BUILD_TYPE "$(_msg q_type)" 3 \
    "$(_msg opt_cpu)" "cpu" \
    "$(_msg opt_gpu)" "gpu" \
    "$(_msg opt_all)" "all"

  # 3. Architecture (numbered)
  local default_arch=1
  [[ "$(arch_label "$PLATFORM")" == "arm64" ]] && default_arch=2
  [[ "$(arch_label "$PLATFORM")" == "multi" ]] && default_arch=3
  local arch_val
  _ask_choice arch_val "$(_msg q_arch)" "$default_arch" \
    "$(_msg opt_amd64)" "amd64" \
    "$(_msg opt_arm64)" "arm64" \
    "$(_msg opt_multi)" "multi"
  PLATFORM="$(parse_arch "$arch_val")"

  # 4. Free-form inputs
  _ask VERSION "$(_msg q_version)" "$VERSION"
  _ask_bool PUSH "$(_msg q_push)" "$PUSH"
  _ask_bool EXPORT_TAR "$(_msg q_export)" "$EXPORT_TAR"
  [[ "$EXPORT_TAR" == "true" ]] && _ask EXPORT_DIR "$(_msg q_output)" "$EXPORT_DIR"
  _ask REGISTRY "$(_msg q_reg)" "$REGISTRY"
  _ask_bool NO_CACHE "$(_msg q_cache)" "$NO_CACHE"

  # Summary
  echo
  echo "----------------------------------------"
  echo "$(_msg summary)"
  echo "----------------------------------------"
  echo "$(_msg q_lang):    $LANG_MODE"
  echo "$(_msg q_type):    $BUILD_TYPE"
  echo "$(_msg q_arch):    $(arch_label "$PLATFORM")"
  echo "$(_msg q_version): $VERSION"
  echo "$(_msg q_push):    $(_label_bool "$PUSH")"
  echo "$(_msg q_export):  $(_label_bool "$EXPORT_TAR")"
  [[ "$EXPORT_TAR" == "true" ]] && echo "$(_msg q_output):  $EXPORT_DIR"
  echo "$(_msg q_reg):     $REGISTRY"
  echo "$(_msg q_cache):   $(_label_bool "$NO_CACHE")"
  echo

  local confirm
  read -r -p "$(_msg q_confirm) [Y/n]: " confirm
  if [[ "$confirm" =~ ^[Nn]$ ]]; then info "$(_msg cancel)"; exit 0; fi

  info "$(_msg start)"
}

# =============================================================================
# CLI Mode
# =============================================================================

show_help() {
  cat <<EOF
$(_msg title)

Usage: ./build.sh [options]

Options:
  -t, --type TYPE     $(_msg q_type): cpu, gpu, all (default: all)
  -a, --arch ARCH     $(_msg q_arch): amd64, arm64, multi (default: amd64)
  -v, --version VER   $(_msg q_version) (default: latest)
  -p, --push          $(_msg q_push)
  -e, --export        $(_msg q_export)
  -o, --output DIR    $(_msg q_output) (default: .)
  -r, --registry REG  $(_msg q_reg) (default: quantatrisk)
  -n, --no-cache      $(_msg q_cache)
  -l, --lang LANG     $(_msg q_lang): zh, en (default: zh)
  -h, --help          Show this help

Examples:
  ./build.sh
  ./build.sh -t gpu -a amd64
  ./build.sh -t cpu -a multi -p
  ./build.sh -t all -v 1.0.0 -p -l zh
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -t|--type)     BUILD_TYPE="$2"; shift 2 ;;
      -a|--arch)     PLATFORM="$(parse_arch "$2")"; shift 2 ;;
      -v|--version)  VERSION="$2"; shift 2 ;;
      -p|--push)     PUSH="true"; shift ;;
      -e|--export)   EXPORT_TAR="true"; shift ;;
      -o|--output)   EXPORT_DIR="$2"; shift 2 ;;
      -r|--registry) REGISTRY="$2"; shift 2 ;;
      -n|--no-cache) NO_CACHE="true"; shift ;;
      -l|--lang)     LANG_MODE="$2"; shift 2 ;;
      -h|--help)     show_help; exit 0 ;;
      *) die "$(_msg err_opt): $1" ;;
    esac
  done
}

validate() {
  case "$BUILD_TYPE" in cpu|gpu|all) ;; *) die "$(_msg err_inv): type=$BUILD_TYPE" ;; esac
}

# =============================================================================
# Main
# =============================================================================

main() {
  if [[ $# -eq 0 && -t 0 ]]; then
    interactive_mode
  else
    parse_args "$@"
    validate
  fi

  ensure_buildx

  case "$BUILD_TYPE" in
    cpu) build_cpu ;;
    gpu) build_gpu ;;
    all) build_cpu; build_gpu ;;
  esac

  if [[ "$PUSH" != "true" && "$EXPORT_TAR" != "true" ]]; then
    info "$(_msg recent):"
    docker images | grep "${REGISTRY}/${IMAGE_NAME}" | head -10 || true
  fi

  info "$(_msg done)"
}

main "$@"
