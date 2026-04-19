# qwen-asr-cli

CLI for [qwen-asr](https://crates.io/crates/qwen-asr): CPU-only Qwen3-ASR speech-to-text in pure Rust.

## Install

```bash
cargo install qwen-asr-cli

# Recommended: enable native CPU SIMD tuning
RUSTFLAGS="-C target-cpu=native" cargo install qwen-asr-cli
```

vDSP/Accelerate is auto-enabled on macOS via default features.

## Download Model

```bash
qwen-asr download qwen3-asr-0.6b
```

## Usage

```bash
# Transcribe a file
qwen-asr -d qwen3-asr-0.6b -i audio.wav

# Streaming mode
qwen-asr -d qwen3-asr-0.6b -i audio.wav --stream

# Live capture (macOS)
qwen-asr -d qwen3-asr-0.6b --live --stream --device "BlackHole 2ch"

# VAD live mode (macOS)
qwen-asr -d qwen3-asr-0.6b --live --vad --device "BlackHole 2ch"

# Forced alignment
qwen-asr -d qwen3-aligner-0.6b -i audio.wav --align "Hello world"

# All options
qwen-asr -h
```

See the [project README](https://github.com/huanglizhuo/QwenASR) for full documentation.

## License

MIT
