//! CPU-only Qwen3-ASR speech recognition in pure Rust.
//!
//! BLAS and SIMD optimizations are selected automatically at compile time based
//! on the target platform — Accelerate + NEON on macOS/aarch64, OpenBLAS + AVX2
//! on Linux/x86_64, etc. For best performance on x86_64, build with:
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo build --release
//! ```
//!
//! **Important:** Always build in release mode (`--release`). Debug builds are
//! 10–50x slower and unusable for real-time inference.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use qwen_asr::context::QwenCtx;
//! use qwen_asr::transcribe;
//!
//! let mut ctx = QwenCtx::load("qwen3-asr-0.6b").expect("model not found");
//! let text = transcribe::transcribe(&mut ctx, "audio.wav").unwrap();
//! println!("{text}");
//! ```
//!
//! # Forced Alignment
//!
//! With the aligner model variant you can obtain word-level timestamps for a
//! known transcript:
//!
//! ```rust,no_run
//! use qwen_asr::context::QwenCtx;
//! use qwen_asr::align;
//!
//! let mut ctx = QwenCtx::load("qwen3-aligner-0.6b").expect("aligner model not found");
//! let samples: Vec<f32> = vec![]; // 16 kHz mono f32 PCM
//! let results = align::forced_align(&mut ctx, &samples, "Hello world", "English").unwrap();
//! for r in &results {
//!     println!("{}: {:.0} – {:.0} ms", r.text, r.start_ms, r.end_ms);
//! }
//! ```
//!
//! # Module Guide
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`context`] | Engine state — start here with [`context::QwenCtx::load`] |
//! | [`transcribe`] | Offline, segmented, and streaming transcription |
//! | [`audio`] | WAV loading, resampling, mel spectrogram |
//! | [`align`] | Forced alignment (word/character timestamps) |
//! | [`config`] | Model configuration and variant detection |
//! | [`tokenizer`] | GPT-2 byte-level BPE tokenizer |
//!
//! The remaining modules (`encoder`, `decoder`, `kernels`, `safetensors`) are
//! implementation details and not intended for direct use.

pub mod config;
pub mod safetensors;
pub mod audio;
pub mod tokenizer;
pub mod kernels;
pub mod encoder;
pub mod decoder;
pub mod context;
pub mod transcribe;
pub mod align;
#[cfg(any(feature = "ios", feature = "android", feature = "macos-ffi"))]
pub mod c_api;
#[cfg(feature = "android")]
pub mod jni_api;

/// Returns a list of compile-time optimization flags enabled for this build.
pub fn optimization_flags() -> Vec<&'static str> {
    let mut flags = Vec::new();

    if cfg!(feature = "vdsp") {
        flags.push("vDSP/Accelerate");
    }
    if cfg!(feature = "blas") && !cfg!(feature = "vdsp") {
        flags.push("BLAS");
    }

    // Architecture-specific SIMD
    if cfg!(target_arch = "aarch64") {
        flags.push("NEON");
        if cfg!(target_feature = "dotprod") {
            flags.push("DotProd");
        }
    } else if cfg!(target_arch = "x86_64") {
        if cfg!(target_feature = "avx2") {
            flags.push("AVX2");
        } else if cfg!(target_feature = "avx") {
            flags.push("AVX");
        } else if cfg!(target_feature = "sse4.1") {
            flags.push("SSE4.1");
        }
        if cfg!(target_feature = "fma") {
            flags.push("FMA");
        }
    }

    if flags.is_empty() {
        flags.push("generic");
    }

    flags
}
