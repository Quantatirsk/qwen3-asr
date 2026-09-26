# Vendored Source

This directory vendors the upstream `QwenASR` Rust library into this repository.
The unused CLI workspace member is omitted.

- Upstream repository: `https://github.com/huanglizhuo/QwenASR`
- Vendored commit: `4e85a19b05f034e106a345d279c68f50df718ab8`
- License: `MIT`

Local modifications included directly in the vendored source:

- Expose `qwen_asr_stream_set_past_text` in the C API for correct streaming behavior.
- Expose `qwen_asr_force_align_file` in the C API for service-side word timestamp alignment.

- Restore the main-branch library from `dd53fc059f705ae3059ff9849abf2e3b648fbff1`.
- Share immutable model weights across contexts with an `Arc` cache.
- Add `qwen_asr_generate_pcm` for exact token-ID prompts and R2T2 generation; Python owns tokenization and the streaming policy.
- Add `bf16-decoder` as an explicit experimental alternative to ARM64 INT8 decode. BF16 weights use FP32 CPU arithmetic.

The vendored source is the build source of truth for the R2T2 CPU backend.
