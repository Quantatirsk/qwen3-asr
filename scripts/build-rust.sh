#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/vendor/qwenasr"
cargo build --release --features ffi --package qwen-asr

TARGET_DIR="${CARGO_TARGET_DIR:-target}"
case "$TARGET_DIR" in
  /*) ;;
  *) TARGET_DIR="$PWD/$TARGET_DIR" ;;
esac
case "$(uname -s)" in
  Darwin) LIBRARY="libqwen_asr.dylib" ;;
  Linux) LIBRARY="libqwen_asr.so" ;;
  *) printf 'Unsupported platform\n' >&2; exit 1 ;;
esac
printf 'Rust library: %s\n' "$TARGET_DIR/release/$LIBRARY"
