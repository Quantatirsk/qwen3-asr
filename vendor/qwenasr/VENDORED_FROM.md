# Vendored Source

This directory vendors the upstream `QwenASR` Rust workspace into this repository.

- Upstream repository: `https://github.com/huanglizhuo/QwenASR`
- Vendored commit: `4e85a19b05f034e106a345d279c68f50df718ab8`
- License: `MIT`

Local modifications included directly in the vendored source:

- Expose `qwen_asr_stream_set_past_text` in the C API for correct streaming behavior.
- Expose `qwen_asr_force_align_file` in the C API for service-side word timestamp alignment.

The vendored source is now the build source of truth for the CPU Rust backend.
