//! C-FFI API for iOS and native integration.
//!
//! Compile with: `cargo build --release --target aarch64-apple-ios --features ios`

use std::ffi::{CStr, CString};
use std::fmt::Write;
use std::os::raw::c_char;

use crate::align;
use crate::audio;
use crate::context::QwenCtx;
use crate::kernels;
use crate::transcribe;

/// Opaque handle to the ASR engine.
pub struct QwenAsrEngine {
    ctx: QwenCtx,
}

fn push_json_string(dst: &mut String, value: &str) {
    dst.push('"');
    for ch in value.chars() {
        match ch {
            '"' => dst.push_str("\\\""),
            '\\' => dst.push_str("\\\\"),
            '\n' => dst.push_str("\\n"),
            '\r' => dst.push_str("\\r"),
            '\t' => dst.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(dst, "\\u{:04x}", c as u32);
            }
            c => dst.push(c),
        }
    }
    dst.push('"');
}

/// Load model from a directory path. Returns null on failure.
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_load_model(
    model_dir: *const c_char,
    n_threads: i32,
    verbosity: i32,
) -> *mut QwenAsrEngine {
    if model_dir.is_null() {
        return std::ptr::null_mut();
    }
    let dir = match CStr::from_ptr(model_dir).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    kernels::set_verbose(verbosity);

    let threads = if n_threads <= 0 {
        kernels::get_num_cpus()
    } else {
        n_threads as usize
    };
    kernels::set_threads(threads);

    match QwenCtx::load(dir) {
        Some(ctx) => Box::into_raw(Box::new(QwenAsrEngine { ctx })),
        None => std::ptr::null_mut(),
    }
}

/// Transcribe a WAV file. Returns a heap-allocated C string (caller must free with qwen_asr_free_string).
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_transcribe_file(
    engine: *mut QwenAsrEngine,
    wav_path: *const c_char,
) -> *mut c_char {
    if engine.is_null() || wav_path.is_null() {
        return std::ptr::null_mut();
    }
    let eng = &mut *engine;
    let path = match CStr::from_ptr(wav_path).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match transcribe::transcribe(&mut eng.ctx, path) {
        Some(text) => match CString::new(text) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Transcribe raw PCM samples (f32, 16kHz, mono).
/// Returns a heap-allocated C string (caller must free with qwen_asr_free_string).
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_transcribe_pcm(
    engine: *mut QwenAsrEngine,
    samples: *const f32,
    n_samples: i32,
) -> *mut c_char {
    if engine.is_null() || samples.is_null() || n_samples <= 0 {
        return std::ptr::null_mut();
    }
    let eng = &mut *engine;
    let pcm = std::slice::from_raw_parts(samples, n_samples as usize);

    match transcribe::transcribe_audio(&mut eng.ctx, pcm) {
        Some(text) => match CString::new(text) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Transcribe raw WAV buffer (entire file contents including header).
/// Returns a heap-allocated C string (caller must free with qwen_asr_free_string).
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_transcribe_wav_buffer(
    engine: *mut QwenAsrEngine,
    wav_data: *const u8,
    wav_len: i32,
) -> *mut c_char {
    if engine.is_null() || wav_data.is_null() || wav_len <= 0 {
        return std::ptr::null_mut();
    }
    let eng = &mut *engine;
    let data = std::slice::from_raw_parts(wav_data, wav_len as usize);

    let samples = match audio::parse_wav_buffer(data) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };

    match transcribe::transcribe_audio(&mut eng.ctx, &samples) {
        Some(text) => match CString::new(text) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Force-align a WAV file against a known transcript.
/// Returns a heap-allocated JSON string:
/// [{"text":"Hello","start_ms":80.0,"end_ms":520.0}, ...]
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_force_align_file(
    engine: *mut QwenAsrEngine,
    wav_path: *const c_char,
    text: *const c_char,
    language: *const c_char,
) -> *mut c_char {
    if engine.is_null() || wav_path.is_null() || text.is_null() || language.is_null() {
        return std::ptr::null_mut();
    }
    let eng = &mut *engine;
    let path = match CStr::from_ptr(wav_path).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let transcript = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let align_language = match CStr::from_ptr(language).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let samples = match audio::load_wav(path) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let results = match align::forced_align(&mut eng.ctx, &samples, transcript, align_language) {
        Some(items) => items,
        None => return std::ptr::null_mut(),
    };

    let mut json = String::from("[");
    for (index, item) in results.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        json.push_str("{\"text\":");
        push_json_string(&mut json, &item.text);
        let _ = write!(
            &mut json,
            ",\"start_ms\":{:.3},\"end_ms\":{:.3}}}",
            item.start_ms,
            item.end_ms,
        );
    }
    json.push(']');

    match CString::new(json) {
        Ok(cs) => cs.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Set segmentation seconds (0 = no segmentation).
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_set_segment_sec(engine: *mut QwenAsrEngine, sec: f32) {
    if !engine.is_null() {
        (*engine).ctx.segment_sec = sec;
    }
}

/// Set language (e.g. "English", "Chinese"). Empty string = auto-detect.
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_set_language(
    engine: *mut QwenAsrEngine,
    language: *const c_char,
) -> i32 {
    if engine.is_null() || language.is_null() {
        return -1;
    }
    let lang = match CStr::from_ptr(language).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    match (*engine).ctx.set_force_language(lang) {
        Ok(()) => 0,
        Err(()) => -1,
    }
}

/// Free a string returned by qwen_asr_transcribe_*.
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Free the engine.
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_free(engine: *mut QwenAsrEngine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

// ========================================================================
// Streaming API
// ========================================================================

/// Opaque handle to streaming state.
pub struct QwenAsrStreamState {
    state: transcribe::StreamState,
    /// Accumulated audio buffer (stream_push_audio requires full buffer).
    audio_buf: Vec<f32>,
}

/// Create a new streaming state. Returns null on failure.
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_stream_new() -> *mut QwenAsrStreamState {
    Box::into_raw(Box::new(QwenAsrStreamState {
        state: transcribe::StreamState::new(),
        audio_buf: Vec::new(),
    }))
}

/// Free a streaming state.
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_stream_free(stream: *mut QwenAsrStreamState) {
    if !stream.is_null() {
        drop(Box::from_raw(stream));
    }
}

/// Reset streaming state for a new utterance (reuses allocations).
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_stream_reset(stream: *mut QwenAsrStreamState) {
    if !stream.is_null() {
        let s = &mut *stream;
        s.state.reset();
        s.audio_buf.clear();
    }
}

/// Push new audio samples and get incremental text delta.
///
/// `samples` / `n_samples`: new PCM chunk (f32, 16 kHz, mono).
/// `finalize`: set to 1 to signal end-of-stream and flush remaining tokens.
///
/// Returns a heap-allocated C string with newly emitted text (may be empty),
/// or null if nothing was emitted. Caller must free with `qwen_asr_free_string`.
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_stream_push(
    engine: *mut QwenAsrEngine,
    stream: *mut QwenAsrStreamState,
    samples: *const f32,
    n_samples: i32,
    finalize: i32,
) -> *mut c_char {
    if engine.is_null() || stream.is_null() {
        return std::ptr::null_mut();
    }

    let eng = &mut *engine;
    let s = &mut *stream;

    // Append new samples to accumulated buffer
    if !samples.is_null() && n_samples > 0 {
        let new_samples = std::slice::from_raw_parts(samples, n_samples as usize);
        s.audio_buf.extend_from_slice(new_samples);
    }

    // Call the Rust streaming API with the full accumulated buffer
    match transcribe::stream_push_audio(
        &mut eng.ctx,
        &s.audio_buf,
        &mut s.state,
        finalize != 0,
    ) {
        Some(delta) if !delta.is_empty() => match CString::new(delta) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        _ => std::ptr::null_mut(),
    }
}

/// Get the full accumulated transcription result so far.
/// Returns a heap-allocated C string. Caller must free with `qwen_asr_free_string`.
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_stream_get_result(
    stream: *mut QwenAsrStreamState,
) -> *mut c_char {
    if stream.is_null() {
        return std::ptr::null_mut();
    }
    let s = &*stream;
    let text = s.state.text();
    if text.is_empty() {
        return std::ptr::null_mut();
    }
    match CString::new(text) {
        Ok(cs) => cs.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Configure streaming chunk size in seconds (default 2.0).
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_stream_set_chunk_sec(engine: *mut QwenAsrEngine, sec: f32) {
    if !engine.is_null() && sec > 0.0 {
        (*engine).ctx.stream_chunk_sec = sec;
    }
}

/// Configure token rollback window (default 5).
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_stream_set_rollback(engine: *mut QwenAsrEngine, tokens: i32) {
    if !engine.is_null() && tokens >= 0 {
        (*engine).ctx.stream_rollback = tokens;
    }
}

/// Configure unfixed chunks count before emitting (default 2).
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_stream_set_unfixed_chunks(engine: *mut QwenAsrEngine, chunks: i32) {
    if !engine.is_null() && chunks >= 0 {
        (*engine).ctx.stream_unfixed_chunks = chunks;
    }
}

/// Configure max new tokens per chunk (default 32).
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_stream_set_max_new_tokens(engine: *mut QwenAsrEngine, tokens: i32) {
    if !engine.is_null() && tokens > 0 {
        (*engine).ctx.stream_max_new_tokens = tokens;
    }
}

/// Configure whether past text is reused as streaming context.
#[no_mangle]
pub unsafe extern "C" fn qwen_asr_stream_set_past_text(
    engine: *mut QwenAsrEngine,
    enabled: i32,
) {
    if !engine.is_null() {
        (*engine).ctx.past_text_conditioning = enabled != 0;
    }
}
