#!/bin/bash
#
# Thin Docker build wrapper for FunASR-API.
# Keeps image build semantics in one place without interactive shell UI.

set -euo pipefail

REGISTRY="quantatrisk"
IMAGE_NAME="funasr-api"
VERSION="latest"
BUILD_TYPE="all"
PLATFORM="linux/amd64"
PUSH="false"
EXPORT_TAR="false"
EXPORT_DIR="."
NO_CACHE="false"

info() { echo "[INFO] $1"; }
warn() { echo "[WARN] $1"; }
die() { echo "[ERROR] $1" >&2; exit 1; }

show_help() {
  cat <<'EOF'
FunASR-API Docker build wrapper

Usage:
  ./build.sh [options]

Options:
  -t, --type TYPE       Build target: cpu, gpu, all (default: all)
  -a, --arch ARCH       Target arch: amd64, arm64, multi (default: amd64)
  -v, --version VER     Image tag version (default: latest)
  -p, --push            Push image after build
  -e, --export          Export single-arch image to tar.gz
  -o, --output DIR      Export directory (default: .)
  -r, --registry REG    Registry namespace (default: quantatrisk)
  -n, --no-cache        Disable Docker build cache
  -h, --help            Show this help

Examples:
  ./build.sh
  ./build.sh -t gpu -a amd64
  ./build.sh -t cpu -a multi -p
  ./build.sh -t all -v 1.0.0 -p
EOF
}

parse_arch() {
  case "$1" in
    amd64) echo "linux/amd64" ;;
    arm64) echo "linux/arm64" ;;
    multi) echo "linux/amd64,linux/arm64" ;;
    *) die "Unknown arch: $1 (expected: amd64, arm64, multi)" ;;
  esac
}

ensure_buildx() {
  docker buildx version >/dev/null 2>&1 || die "Docker Buildx is required"

  if ! docker buildx inspect funasr-builder >/dev/null 2>&1; then
    info "Creating buildx builder: funasr-builder"
    docker buildx create --name funasr-builder --driver docker-container --use >/dev/null
  else
    docker buildx use funasr-builder >/dev/null
  fi
}

export_compressor() {
  if command -v pigz >/dev/null 2>&1; then
    echo "pigz -f"
  else
    echo "gzip -f"
  fi
}

build_image() {
  local target="$1"
  local dockerfile="$2"
  local image_tag="$3"
  local platform="$4"

  local args=(buildx build --platform "$platform" -f "$dockerfile" -t "$image_tag")

  if [[ "$NO_CACHE" == "true" ]]; then
    args+=(--no-cache)
  fi

  if [[ "$platform" == *","* ]]; then
    if [[ "$EXPORT_TAR" == "true" ]]; then
      warn "Multi-arch build cannot export tar.gz, skipping export"
    fi
    args+=(--push)
  elif [[ "$PUSH" == "true" ]]; then
    args+=(--push)
  elif [[ "$EXPORT_TAR" == "true" ]]; then
    mkdir -p "$EXPORT_DIR"
    local suffix="${target}-${VERSION}-$(basename "$platform")"
    local tar_path="${EXPORT_DIR}/${IMAGE_NAME}-${suffix}.tar"
    args+=(--output "type=docker,dest=${tar_path}")
  else
    args+=(--load)
  fi

  info "Building ${target}: ${image_tag} (${platform})"
  docker "${args[@]}" .

  if [[ "$EXPORT_TAR" == "true" && "$platform" != *","* && "$PUSH" != "true" ]]; then
    local suffix="${target}-${VERSION}-$(basename "$platform")"
    local tar_path="${EXPORT_DIR}/${IMAGE_NAME}-${suffix}.tar"
    if [[ -f "$tar_path" ]]; then
      info "Compressing ${tar_path}"
      $(export_compressor) "$tar_path"
      info "Exported ${tar_path}.gz"
    fi
  fi
}

build_cpu() {
  local tag="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
  if [[ "$VERSION" == "latest" ]]; then
    tag="${REGISTRY}/${IMAGE_NAME}:cpu-latest"
  fi
  build_image "cpu" "Dockerfile.cpu" "$tag" "$PLATFORM"
}

build_gpu() {
  if [[ "$PLATFORM" == *"arm64"* ]]; then
    die "GPU build only supports amd64"
  fi

  local platform="linux/amd64"
  local tag="${REGISTRY}/${IMAGE_NAME}:gpu-${VERSION}"
  build_image "gpu" "Dockerfile.gpu" "$tag" "$platform"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    -a|--arch)
      PLATFORM="$(parse_arch "$2")"
      shift 2
      ;;
    -v|--version)
      VERSION="$2"
      shift 2
      ;;
    -p|--push)
      PUSH="true"
      shift
      ;;
    -e|--export)
      EXPORT_TAR="true"
      shift
      ;;
    -o|--output)
      EXPORT_DIR="$2"
      shift 2
      ;;
    -r|--registry)
      REGISTRY="$2"
      shift 2
      ;;
    -n|--no-cache)
      NO_CACHE="true"
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

case "$BUILD_TYPE" in
  cpu|gpu|all) ;;
  *) die "Unknown build type: $BUILD_TYPE (expected: cpu, gpu, all)" ;;
esac

ensure_buildx

info "Build type: ${BUILD_TYPE}"
info "Platform: ${PLATFORM}"
info "Version: ${VERSION}"
info "Push: ${PUSH}"
info "Export: ${EXPORT_TAR}"
info "No cache: ${NO_CACHE}"

case "$BUILD_TYPE" in
  cpu)
    build_cpu
    ;;
  gpu)
    build_gpu
    ;;
  all)
    build_cpu
    build_gpu
    ;;
esac

if [[ "$PUSH" != "true" && "$EXPORT_TAR" != "true" ]]; then
  info "Recent images:"
  docker images | grep "${REGISTRY}/${IMAGE_NAME}" | head -10 || true
fi
