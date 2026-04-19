# qwen_asr

CPU-only Qwen3-ASR speech recognition in pure Rust. No Python, no ONNX runtime,
no framework dependencies — just `libc` and BLAS. BF16 weights stay memory-mapped
for minimal RAM usage; SIMD kernels (NEON / AVX2+FMA) accelerate inference.

## Prerequisites

- Rust 1.70+
- BLAS: Accelerate (macOS, linked automatically) or OpenBLAS (Linux)

## Building

Platform-specific optimizations are detected automatically at compile time:

| Platform | BLAS | SIMD |
|----------|------|------|
| macOS (Apple Silicon) | Accelerate + vDSP | NEON (always available) |
| macOS (Intel) | Accelerate + vDSP | AVX2+FMA |
| Linux (x86_64) | OpenBLAS | AVX2+FMA |
| Linux (aarch64) | OpenBLAS | NEON |
| Other | OpenBLAS | Generic scalar fallback |

For best performance, build with native CPU tuning so the compiler can emit
AVX2+FMA instructions on x86_64:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

On AArch64 (Apple Silicon, ARM Linux) NEON is baseline — no extra flags needed,
though `-C target-cpu=native` is still recommended for other micro-architecture
tuning.

**Important:** Always use `--release` mode. Debug builds are 10-50x slower due
to missing optimizations and are not usable for real-time inference.

## Model Download

```bash
# Install huggingface-cli if needed
pip install huggingface_hub

# Download the 0.6B model (~1.3 GB)
huggingface_hub download Qwen/Qwen3-ASR-0.6B --local-dir qwen3-asr-0.6b

# Download the 0.6B forced-aligner model (~1.3 GB)
huggingface_hub download Qwen/Qwen3-ASR-0.6B-Aligner --local-dir qwen3-aligner-0.6b
```

## Usage

```rust,no_run
use qwen_asr::context::QwenCtx;
use qwen_asr::transcribe;

fn main() {
    // Load model (returns None on failure)
    let mut ctx = QwenCtx::load("qwen3-asr-0.6b").expect("failed to load model");

    // Transcribe a WAV file
    let text = transcribe::transcribe(&mut ctx, "audio.wav").unwrap();
    println!("{}", text);
}
```

### Segmented Mode

For long audio files, split into overlapping segments to reduce memory usage
and improve accuracy:

```rust,no_run
use qwen_asr::context::QwenCtx;
use qwen_asr::transcribe;

let mut ctx = QwenCtx::load("qwen3-asr-0.6b").unwrap();
ctx.segment_sec = 30.0; // split every ~30 seconds

let text = transcribe::transcribe(&mut ctx, "long-meeting.wav").unwrap();
```

### Raw PCM Input

```rust,no_run
use qwen_asr::context::QwenCtx;
use qwen_asr::transcribe;

let mut ctx = QwenCtx::load("qwen3-asr-0.6b").unwrap();

// f32 samples at 16 kHz, mono, range [-1, 1]
let samples: Vec<f32> = load_audio_somehow();
let text = transcribe::transcribe_audio(&mut ctx, &samples).unwrap();
```

### Streaming API

For real-time incremental transcription, use `StreamState` and `stream_push_audio`:

```rust,no_run
use qwen_asr::context::QwenCtx;
use qwen_asr::transcribe::{StreamState, stream_push_audio};

let mut ctx = QwenCtx::load("qwen3-asr-0.6b").unwrap();
let mut state = StreamState::new();

// As audio arrives (e.g., from a microphone), accumulate samples
let mut all_samples: Vec<f32> = Vec::new();
loop {
    let new_audio = get_audio_chunk(); // your audio source
    all_samples.extend_from_slice(&new_audio);

    // Push all accumulated audio; stream_push_audio tracks its own cursor
    if let Some(delta) = stream_push_audio(&mut ctx, &all_samples, &mut state, false) {
        if !delta.is_empty() {
            print!("{}", delta); // incremental output
        }
    }
}

// Finalize to flush remaining tokens
stream_push_audio(&mut ctx, &all_samples, &mut state, true);
```

### Forced Alignment

Produce word-level timestamps for a known transcript. Requires the
ForcedAligner model variant (`Qwen3-ASR-0.6B-Aligner`).

```rust,no_run
use qwen_asr::context::QwenCtx;
use qwen_asr::align;

let mut ctx = QwenCtx::load("qwen3-aligner-0.6b").unwrap();
let samples: Vec<f32> = load_audio_somehow();

let results = align::forced_align(&mut ctx, &samples, "Hello world", "English")
    .expect("alignment failed");

for r in &results {
    println!("{}: {:.0} ms – {:.0} ms", r.text, r.start_ms, r.end_ms);
}
```

CLI:

```bash
qwen-asr -d qwen3-aligner-0.6b -i audio.wav --align "Hello world" --align-language English
```

Each `AlignResult` contains the word text, `start_ms`, and `end_ms` timestamps.
For CJK languages the text is split at character level; for others it is split on
whitespace.

## Feature Flags

| Feature   | Default | Description |
|-----------|---------|-------------|
| `blas`    | yes     | Link Accelerate (macOS) or OpenBLAS (Linux) for matrix ops |
| `vdsp`    | yes     | Use vDSP/vForce from Accelerate for dot products and exp (macOS only) |
| `ios`     | no      | Build C-FFI API for iOS integration |
| `android` | no      | Build C-FFI + JNI API for Android integration |

## Performance

Benchmarks on Apple M2 Pro (10-core), 0.6B model:

| Mode | Audio | Wall Time | Realtime Factor |
|------|-------|-----------|-----------------|
| Offline | 11 s | 1.8 s | 6.2x |
| Offline | 28 s | 4.0 s | 7.0x |
| Segmented (-S 30) | 45 s | 4.6 s | 9.8x |
| Streaming | 28 s | 10.4 s | 2.7x |
| Streaming (live) | 51 s | 14.1 s | 3.6x |

## License

MIT
