mod download;
#[cfg(target_os = "macos")]
mod live_capture;

use qwen_asr::{audio, config, context, kernels, transcribe, align};
use config::*;
use context::QwenCtx;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn stream_token(piece: &str) {
    use std::io::Write;
    print!("{}", piece);
    std::io::stdout().flush().ok();
}

fn usage(prog: &str) {
    eprintln!("qwen-asr — Qwen3-ASR speech-to-text (pure Rust)\n");
    eprintln!("Usage: {} -d <model_dir> (-i <input.wav> | --stdin | --live) [options]\n", prog);
    eprintln!("Required:");
    eprintln!("  -d <dir>      Model directory (with *.safetensors, vocab.json)");
    eprintln!("  -i <file>     Input WAV file (16-bit PCM, any sample rate)");
    eprintln!("  --stdin       Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono)");
    eprintln!("\nLive capture (macOS only):");
    eprintln!("  --live                      Capture from audio input device in real time");
    eprintln!("  --device <name>             Input device name (default: system default)");
    eprintln!("  --list-devices              List available audio input devices and exit");
    eprintln!("  --vad                       Live VAD mode: detect speech segments, transcribe each");
    eprintln!("\nOptions:");
    eprintln!("  -t <n>        Number of threads (default: all CPUs)");
    eprintln!("  -S <secs>     Segment target seconds (default: 0 = full-audio decode)");
    eprintln!("  -W <secs>     Segment-cutting silence search window ± seconds (default: 3.0)");
    eprintln!("  --stream      Streaming mode: process in chunks with prefix rollback");
    eprintln!("  --stream-max-new-tokens <n>  Max generated tokens per stream step (default: 32)");
    eprintln!("  --stream-chunk-sec <secs>   Chunk size for streaming (default: 2.0, min ~1.0)");
    eprintln!("  --enc-window-sec <secs>    Encoder attention window in seconds (1..8, default 8)");
    eprintln!("  --past-text <yes|no|auto>  Reuse previously decoded text as context");
    eprintln!("  --skip-silence              Drop long silent spans before inference");
    eprintln!("  --prompt <text>            System prompt for biasing");
    eprintln!("  --language <lang>          Force output language");
    eprintln!("\nAlignment mode (requires ForcedAligner model):");
    eprintln!("  --align <text>             Align transcript to audio (word-level timestamps)");
    eprintln!("  --align-language <lang>    Language for word splitting (default: English)");
    eprintln!("  --profile     Print per-operation timing breakdown");
    eprintln!("  --debug       Debug output (per-layer details)");
    eprintln!("  --silent      No status output (only transcription on stdout)");
    eprintln!("\nModel management:");
    eprintln!("  {} download [--list] [<model>] [--output <dir>]", prog);
    eprintln!("  -h            Show this help");
}

fn parse_past_text_mode(s: &str) -> Option<i32> {
    match s.to_lowercase().as_str() {
        "yes" => Some(1),
        "no" => Some(0),
        "auto" => Some(-1),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Handle `download` subcommand: qwen-asr download [args...]
    if args.len() >= 2 && args[1] == "download" {
        download::handle_download_command(&args[2..]);
        return;
    }

    // Handle --list-devices (no model needed)
    if args.iter().any(|a| a == "--list-devices") {
        #[cfg(target_os = "macos")]
        {
            live_capture::print_devices();
        }
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("--list-devices is only supported on macOS.");
            eprintln!("On Linux, use: arecord -l");
        }
        return;
    }

    let mut model_dir: Option<String> = None;
    let mut input_wav: Option<String> = None;
    let mut verbosity = 1i32;
    let mut use_stdin = false;
    let mut live_mode = false;
    let mut device_name: Option<String> = None;
    let mut n_threads = 0i32;
    let mut segment_sec: f32 = -1.0;
    let mut search_sec: f32 = -1.0;
    let mut stream_mode = false;
    let mut vad_mode = false;
    let mut stream_max_new_tokens: i32 = -1;
    let mut stream_chunk_sec: f32 = -1.0;
    let mut enc_window_sec: f32 = -1.0;
    let mut prompt_text: Option<String> = None;
    let mut force_language: Option<String> = None;
    let mut past_text_mode: i32 = -1; // -1 auto, 0 off, 1 on
    let mut skip_silence = false;
    let mut profile = false;
    let mut align_text: Option<String> = None;
    let mut align_language: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-d" => {
                i += 1;
                model_dir = args.get(i).cloned();
            }
            "-i" => {
                i += 1;
                input_wav = args.get(i).cloned();
            }
            "-t" => {
                i += 1;
                n_threads = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "-S" => {
                i += 1;
                segment_sec = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
            }
            "-W" => {
                i += 1;
                search_sec = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
            }
            "--stream" => {
                stream_mode = true;
            }
            "--vad" => {
                vad_mode = true;
            }
            "--stream-max-new-tokens" => {
                i += 1;
                stream_max_new_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1);
            }
            "--stream-chunk-sec" => {
                i += 1;
                stream_chunk_sec = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
            }
            "--enc-window-sec" => {
                i += 1;
                enc_window_sec = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
            }
            "--past-text" => {
                i += 1;
                if let Some(s) = args.get(i) {
                    match parse_past_text_mode(s) {
                        Some(m) => past_text_mode = m,
                        None => {
                            eprintln!("Error: --past-text must be one of yes|no|auto, got '{}'", s);
                            std::process::exit(1);
                        }
                    }
                }
            }
            "--skip-silence" => {
                skip_silence = true;
            }
            "--prompt" => {
                i += 1;
                prompt_text = args.get(i).cloned();
            }
            "--language" => {
                i += 1;
                force_language = args.get(i).cloned();
            }
            "--align" => {
                i += 1;
                align_text = args.get(i).cloned();
            }
            "--align-language" => {
                i += 1;
                align_language = args.get(i).cloned();
            }
            "--stdin" => {
                use_stdin = true;
            }
            "--live" => {
                live_mode = true;
            }
            "--device" => {
                i += 1;
                device_name = args.get(i).cloned();
            }
            "--list-devices" => {
                // Already handled above, but don't error
            }
            "--profile" => {
                profile = true;
            }
            "--debug" => {
                verbosity = 2;
            }
            "--silent" => {
                verbosity = 0;
            }
            "-h" | "--help" => {
                usage(&args[0]);
                return;
            }
            other => {
                eprintln!("Unknown option: {}", other);
                usage(&args[0]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let model_dir = match model_dir {
        Some(d) => d,
        None => {
            usage(&args[0]);
            std::process::exit(1);
        }
    };

    // Auto-prompt to download if model directory doesn't exist
    if !std::path::Path::new(&model_dir).exists() {
        if let Some(model) = download::find_model(&model_dir) {
            if download::prompt_download(&model_dir) {
                if let Err(e) = download::download_model(model, &model_dir) {
                    eprintln!("Download failed: {}", e);
                    std::process::exit(1);
                }
                eprintln!(); // blank line before model loading
            } else {
                eprintln!("Aborted.");
                std::process::exit(1);
            }
        } else {
            eprintln!("Error: Model directory '{}' not found.", model_dir);
            eprintln!();
            download::list_models();
            std::process::exit(1);
        }
    }

    if input_wav.is_none() && !use_stdin && !live_mode {
        usage(&args[0]);
        std::process::exit(1);
    }

    // Check mutual exclusivity of input modes
    let input_count = [input_wav.is_some(), use_stdin, live_mode].iter().filter(|&&x| x).count();
    if input_count > 1 {
        eprintln!("Error: -i, --stdin, and --live are mutually exclusive");
        std::process::exit(1);
    }

    kernels::set_verbose(verbosity);
    if profile {
        kernels::set_profile(true);
        kernels::profile_reset();
    }
    let emit_tokens = verbosity > 0;

    // Initialize thread pool
    if n_threads <= 0 {
        n_threads = kernels::get_num_cpus() as i32;
    }
    kernels::set_threads(n_threads as usize);

    // Print optimization info
    if verbosity >= 1 {
        let opts = qwen_asr::optimization_flags();
        eprintln!(
            "Optimizations: {} | {} threads | {}",
            opts.join(", "),
            n_threads,
            std::env::consts::ARCH,
        );
    }

    // Load model
    let mut ctx = match QwenCtx::load(&model_dir) {
        Some(c) => c,
        None => {
            eprintln!("Failed to load model from {}", model_dir);
            std::process::exit(1);
        }
    };

    // Apply settings
    if segment_sec >= 0.0 {
        ctx.segment_sec = segment_sec;
    }
    if search_sec >= 0.0 {
        ctx.search_sec = search_sec;
    }
    if enc_window_sec >= 0.0 {
        let window_frames = (enc_window_sec * 100.0 + 0.5) as usize;
        ctx.config.enc_n_window_infer = window_frames.clamp(100, 800);
    }
    if stream_max_new_tokens > 0 {
        ctx.stream_max_new_tokens = stream_max_new_tokens;
    }
    if stream_chunk_sec > 0.0 {
        ctx.stream_chunk_sec = stream_chunk_sec;
    }
    if past_text_mode >= 0 {
        ctx.past_text_conditioning = past_text_mode == 1;
    } else if stream_mode {
        ctx.past_text_conditioning = true;
    }
    if skip_silence {
        ctx.skip_silence = true;
    }
    if let Some(ref prompt) = prompt_text {
        if ctx.set_prompt(prompt).is_err() {
            eprintln!("Failed to set --prompt text");
            std::process::exit(1);
        }
    }
    if let Some(ref lang) = force_language {
        if ctx.set_force_language(lang).is_err() {
            eprintln!("Unsupported language for --language: {}", lang);
            eprintln!(
                "Supported languages: {}",
                SUPPORTED_LANGUAGES.join(",")
            );
            std::process::exit(1);
        }
    }

    // Alignment mode
    if let Some(ref atext) = align_text {
        let lang = align_language.as_deref().unwrap_or("English");
        let lang_normalized = match normalize_language(lang) {
            Some(l) => l,
            None => {
                eprintln!("Unsupported --align-language: {}", lang);
                eprintln!("Supported languages: {}", SUPPORTED_LANGUAGES.join(","));
                std::process::exit(1);
            }
        };

        let samples = if use_stdin {
            audio::read_pcm_stdin()
        } else {
            audio::load_wav(input_wav.as_ref().unwrap())
        };
        let samples = match samples {
            Some(s) => s,
            None => {
                eprintln!("Failed to load audio");
                std::process::exit(1);
            }
        };

        match align::forced_align(&mut ctx, &samples, atext, &lang_normalized) {
            Some(results) => {
                // Output JSON array
                println!("[");
                for (i, r) in results.iter().enumerate() {
                    let comma = if i + 1 < results.len() { "," } else { "" };
                    // Escape the text for JSON
                    let escaped = r.text.replace('\\', "\\\\").replace('"', "\\\"");
                    println!(
                        "  {{\"text\": \"{}\", \"start\": {:.0}, \"end\": {:.0}}}{}",
                        escaped, r.start_ms, r.end_ms, comma
                    );
                }
                println!("]");
            }
            None => {
                eprintln!("Alignment failed");
                std::process::exit(1);
            }
        }

        if verbosity >= 1 {
            eprintln!(
                "Alignment: {:.0} ms (encoding: {:.0}ms, decoding: {:.0}ms)",
                ctx.perf_total_ms, ctx.perf_encode_ms, ctx.perf_decode_ms
            );
        }

        if profile {
            kernels::profile_report();
        }
        return;
    }

    // Set token callback
    if emit_tokens {
        ctx.token_cb = Some(Box::new(stream_token));
    }

    // Live capture mode
    if live_mode {
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("Error: --live is only supported on macOS.");
            eprintln!("On Linux, pipe audio via: arecord -f S16_LE -r 16000 -c 1 | qwen-asr -d <model> --stdin");
            std::process::exit(1);
        }

        #[cfg(target_os = "macos")]
        {
            run_live_capture(&mut ctx, device_name.as_deref(), stream_mode, vad_mode, verbosity, profile);
            return;
        }
    }

    // Transcribe
    let text = if stream_mode {
        let samples = if use_stdin {
            audio::read_pcm_stdin()
        } else {
            audio::load_wav(input_wav.as_ref().unwrap())
        };
        match samples {
            Some(s) => transcribe::transcribe_stream(&mut ctx, &s),
            None => None,
        }
    } else if use_stdin {
        transcribe::transcribe_stdin(&mut ctx)
    } else {
        transcribe::transcribe(&mut ctx, input_wav.as_ref().unwrap())
    };

    match text {
        Some(t) => {
            if emit_tokens {
                println!();
            } else {
                println!("{}", t);
            }
        }
        None => {
            eprintln!("Transcription failed");
            std::process::exit(1);
        }
    }

    if verbosity >= 1 {
        let tokens_per_sec = if ctx.perf_total_ms > 0.0 {
            1000.0 * ctx.perf_text_tokens as f64 / ctx.perf_total_ms
        } else {
            0.0
        };
        eprintln!(
            "Inference: {:.0} ms, {} text tokens ({:.2} tok/s, encoding: {:.0}ms, decoding: {:.0}ms)",
            ctx.perf_total_ms, ctx.perf_text_tokens, tokens_per_sec,
            ctx.perf_encode_ms, ctx.perf_decode_ms
        );
        if ctx.perf_audio_ms > 0.0 && ctx.perf_total_ms > 0.0 {
            let audio_s = ctx.perf_audio_ms / 1000.0;
            let infer_s = ctx.perf_total_ms / 1000.0;
            eprintln!(
                "Audio: {:.1} s processed in {:.1} s ({:.2}x realtime)",
                audio_s, infer_s, audio_s / infer_s
            );
        }
    }

    if profile {
        kernels::profile_report();
    }
}

// ========================================================================
// Live Capture Loop (macOS only)
// ========================================================================

#[cfg(target_os = "macos")]
fn run_live_capture(
    ctx: &mut QwenCtx,
    device_name: Option<&str>,
    stream_mode: bool,
    vad_mode: bool,
    verbosity: i32,
    profile: bool,
) {
    use std::time::Duration;

    // Resolve device
    let device_id = if let Some(name) = device_name {
        match live_capture::find_device_by_name(name) {
            Some(dev) => {
                if verbosity >= 1 {
                    eprintln!("Using input device: {} ({} ch)", dev.name, dev.input_channels);
                }
                dev.id
            }
            None => {
                eprintln!("Error: No input device matching '{}'", name);
                if name.to_lowercase().contains("blackhole") {
                    eprintln!();
                    eprintln!("BlackHole does not appear to be installed.");
                    eprintln!("Install it with: brew install blackhole-2ch");
                    eprintln!("Then set it up as a Multi-Output Device in Audio MIDI Setup.");
                    eprintln!("See: https://github.com/ExistentialAudio/BlackHole");
                }
                eprintln!();
                live_capture::print_devices();
                std::process::exit(1);
            }
        }
    } else {
        match live_capture::default_input_device() {
            Some(id) => {
                if verbosity >= 1 {
                    let devices = live_capture::list_input_devices();
                    if let Some(dev) = devices.iter().find(|d| d.id == id) {
                        eprintln!("Using default input device: {}", dev.name);
                    }
                }
                id
            }
            None => {
                eprintln!("Error: No default input device found");
                std::process::exit(1);
            }
        }
    };

    // Start capture
    let (rx, _handle, device_rate) = match live_capture::start_capture(device_id) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: Failed to start audio capture: {}", e);
            std::process::exit(1);
        }
    };

    let mode_label = if stream_mode { "streaming" } else if vad_mode { "VAD" } else { "segmented" };
    if verbosity >= 1 {
        if vad_mode {
            eprintln!("Listening (VAD segmented)... press Ctrl+C to stop\n");
        } else {
            eprintln!(
                "Listening ({}, {:.1}s chunks)... press Ctrl+C to stop\n",
                mode_label,
                if stream_mode { ctx.stream_chunk_sec } else { ctx.segment_sec }
            );
        }
    }

    // Set up Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl+C handler");

    // Configure context
    ctx.past_text_conditioning = true;
    ctx.reset_perf();

    // Audio accumulation
    let target_rate = 16000;
    let mut raw_buf: Vec<f32> = Vec::new();
    let mut resampled_buf: Vec<f32> = Vec::new();
    let needs_resample = (device_rate - target_rate as f64).abs() > 1.0;
    let wall_start = std::time::Instant::now();

    if stream_mode {
        // ---- Streaming mode: incremental stream_push_audio ----
        //
        // We accumulate audio and call stream_push_audio() which only
        // processes NEW audio incrementally (persistent encoder cache,
        // LCP-reused decoder prefill, monotonic token commit).
        //
        // Buffer reset after ~120s to bound memory.
        let max_window_samples: usize = 120 * target_rate as usize;
        let mut stream_state = transcribe::StreamState::new();

        // Set token callback for direct printing
        ctx.token_cb = None; // stream_push_audio returns delta text, we print it

        // Text-emission timeout: flush rollback tokens after no new text for 5s
        let mut last_text_time: Option<std::time::Instant> = None;
        let text_flush_secs = 5.0_f32;
        let mut flushed = false;

        while running.load(Ordering::SeqCst) {
            // Receive audio
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(chunk) => raw_buf.extend_from_slice(&chunk),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            while let Ok(chunk) = rx.try_recv() {
                raw_buf.extend_from_slice(&chunk);
            }

            // Resample
            if needs_resample {
                if !raw_buf.is_empty() {
                    let resampled = qwen_asr::audio::resample(
                        &raw_buf, device_rate as i32, target_rate,
                    );
                    resampled_buf.extend_from_slice(&resampled);
                    raw_buf.clear();
                }
            } else {
                resampled_buf.append(&mut raw_buf);
            }

            // Reset window if buffer exceeds max
            if resampled_buf.len() > max_window_samples {
                // Flush rollback tokens before reset
                if let Some(delta) = transcribe::stream_push_audio(
                    ctx, &resampled_buf, &mut stream_state, true
                ) {
                    if !delta.is_empty() {
                        print!("{}", delta);
                    }
                }
                println!();
                resampled_buf.clear();
                stream_state.reset();
                last_text_time = None;
                flushed = false;
                continue;
            }

            // Determine if we should finalize: flush rollback tokens
            // when no new text has been emitted for 5 seconds
            let finalize = !flushed
                && last_text_time.is_some_and(|t| t.elapsed().as_secs_f32() >= text_flush_secs);

            // Process all available full chunks
            if resampled_buf.len() > stream_state.audio_cursor() {
                if let Some(delta) = transcribe::stream_push_audio(
                    ctx, &resampled_buf, &mut stream_state, finalize
                ) {
                    if !delta.is_empty() {
                        print!("{}", delta);
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                        last_text_time = Some(std::time::Instant::now());
                        flushed = false;
                    } else if finalize {
                        flushed = true; // Don't keep calling finalize
                    }
                }
            }
        }

        // Final flush
        if !raw_buf.is_empty() && needs_resample {
            let resampled = qwen_asr::audio::resample(
                &raw_buf, device_rate as i32, target_rate,
            );
            resampled_buf.extend_from_slice(&resampled);
        } else {
            resampled_buf.append(&mut raw_buf);
        }

        if resampled_buf.len() > stream_state.audio_cursor() {
            if let Some(delta) = transcribe::stream_push_audio(
                ctx, &resampled_buf, &mut stream_state, true // finalize: flush rollback
            ) {
                if !delta.is_empty() {
                    print!("{}", delta);
                }
            }
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        println!();
    } else if vad_mode {
        // ---- VAD mode: energy-based speech detection + segment transcription ----
        //
        // Detect speech using RMS energy. When speech ends (silence > 1.5s),
        // transcribe the accumulated speech segment using transcribe_audio().
        // This gives better accuracy than streaming (full segment context)
        // with automatic speech boundary detection.
        let speech_threshold: f32 = 0.001;
        let silence_hangover_secs = 1.5_f32;
        let min_segment_secs = 0.5_f32;
        let max_segment_secs = 30.0_f32;
        let min_segment_samples = (min_segment_secs * target_rate as f32) as usize;
        let max_segment_samples = (max_segment_secs * target_rate as f32) as usize;
        let check_samples = (target_rate as usize) * 30 / 1000; // 30ms window for RMS

        let mut speech_active = false;
        let mut silence_start: Option<std::time::Instant> = None;
        let mut speech_start_idx: usize = 0;

        // Keep a small pre-speech buffer to avoid clipping word beginnings
        let pre_speech_samples = (target_rate as usize) / 4; // 250ms lookback

        // Disable token callback — we print the full result after each segment
        ctx.token_cb = None;

        // Cross-segment context: accumulate text to use as prompt for next segment
        let mut accumulated_text = String::new();

        while running.load(Ordering::SeqCst) {
            // Receive audio
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(chunk) => raw_buf.extend_from_slice(&chunk),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            while let Ok(chunk) = rx.try_recv() {
                raw_buf.extend_from_slice(&chunk);
            }

            // Resample
            if needs_resample {
                if !raw_buf.is_empty() {
                    let resampled = qwen_asr::audio::resample(
                        &raw_buf, device_rate as i32, target_rate,
                    );
                    resampled_buf.extend_from_slice(&resampled);
                    raw_buf.clear();
                }
            } else {
                resampled_buf.append(&mut raw_buf);
            }

            // Compute RMS energy of latest 30ms
            let buf_len = resampled_buf.len();
            let rms = if buf_len >= check_samples {
                let tail = &resampled_buf[buf_len - check_samples..];
                let sum_sq: f32 = tail.iter().map(|&s| s * s).sum();
                (sum_sq / check_samples as f32).sqrt()
            } else {
                0.0
            };
            let is_speech = rms >= speech_threshold;

            // Periodic RMS debug output
            if verbosity >= 2 && buf_len % (target_rate as usize * 2) < check_samples {
                eprintln!("  [VAD] rms={:.6} threshold={:.4} speech={}",
                    rms, speech_threshold, if speech_active { "active" } else { "inactive" });
            }

            if !speech_active {
                if is_speech {
                    // Speech started — mark the start with lookback
                    speech_active = true;
                    silence_start = None;
                    speech_start_idx = buf_len.saturating_sub(pre_speech_samples);
                    if verbosity >= 2 {
                        eprintln!("  [VAD] speech start at {:.1}s",
                            buf_len as f32 / target_rate as f32);
                    }
                } else {
                    // No speech — bound buffer to avoid unlimited growth
                    // Keep only last 0.5s for lookback context
                    let keep = (target_rate as usize) / 2;
                    if resampled_buf.len() > keep * 4 {
                        let drain = resampled_buf.len() - keep;
                        resampled_buf.drain(..drain);
                    }
                }
            } else {
                // Speech is active
                let segment_len = buf_len - speech_start_idx;

                if is_speech {
                    // Still speaking — reset silence timer
                    silence_start = None;

                    // Force-flush if segment exceeds max duration
                    if segment_len >= max_segment_samples {
                        if verbosity >= 2 {
                            eprintln!("  [VAD] max segment reached ({:.1}s), flushing",
                                segment_len as f32 / target_rate as f32);
                        }
                        let segment = &resampled_buf[speech_start_idx..];
                        // Set previous text as context
                        if !accumulated_text.is_empty() {
                            ctx.prompt = Some(accumulated_text.clone());
                            ctx.prompt_tokens_ready = false;
                        }
                        ctx.reset_perf();
                        if let Some(text) = transcribe::transcribe_audio(ctx, segment) {
                            if !text.is_empty() {
                                println!("{}", text);
                                accumulated_text.push_str(&text);
                            }
                        }
                        resampled_buf.clear();
                        speech_active = false;
                        silence_start = None;
                    }
                } else {
                    // Silence during speech — track duration
                    if silence_start.is_none() {
                        silence_start = Some(std::time::Instant::now());
                    }
                    if let Some(start) = silence_start {
                        if start.elapsed().as_secs_f32() >= silence_hangover_secs {
                            // End of utterance — transcribe the segment
                            if segment_len >= min_segment_samples {
                                // Trim trailing silence (keep only 200ms of it)
                                let trail_keep = (target_rate as usize) / 5;
                                let seg_end = (buf_len - check_samples + trail_keep).min(buf_len);
                                let segment = &resampled_buf[speech_start_idx..seg_end];

                                if verbosity >= 2 {
                                    eprintln!("  [VAD] speech end, segment {:.1}s",
                                        segment.len() as f32 / target_rate as f32);
                                }

                                ctx.reset_perf();
                                // Set previous text as context
                                if !accumulated_text.is_empty() {
                                    ctx.prompt = Some(accumulated_text.clone());
                                    ctx.prompt_tokens_ready = false;
                                }
                                let t0 = std::time::Instant::now();
                                if let Some(text) = transcribe::transcribe_audio(ctx, segment) {
                                    if !text.is_empty() {
                                        println!("{}", text);
                                        accumulated_text.push_str(&text);
                                        if verbosity >= 1 {
                                            let audio_secs = segment.len() as f32 / target_rate as f32;
                                            let compute_secs = t0.elapsed().as_secs_f32();
                                            eprintln!(
                                                "  ({:.1}s audio in {:.1}s, {:.1}x realtime)",
                                                audio_secs, compute_secs,
                                                audio_secs / compute_secs.max(0.001)
                                            );
                                        }
                                    }
                                }
                            } else if verbosity >= 2 {
                                eprintln!("  [VAD] segment too short ({:.2}s), discarding",
                                    segment_len as f32 / target_rate as f32);
                            }

                            resampled_buf.clear();
                            speech_active = false;
                            silence_start = None;
                        }
                    }
                }
            }
        }

        // Flush remaining speech on Ctrl+C
        if speech_active && resampled_buf.len() > speech_start_idx + min_segment_samples {
            let segment = &resampled_buf[speech_start_idx..];
            ctx.reset_perf();
            if let Some(text) = transcribe::transcribe_audio(ctx, segment) {
                if !text.is_empty() {
                    println!("{}", text);
                }
            }
        }
    } else {
        // ---- Segmented mode: independent segments ----
        if ctx.segment_sec <= 0.0 {
            ctx.segment_sec = 5.0;
        }
        let segment_samples_16k = (ctx.segment_sec * target_rate as f32) as usize;

        while running.load(Ordering::SeqCst) {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(chunk) => raw_buf.extend_from_slice(&chunk),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            while let Ok(chunk) = rx.try_recv() {
                raw_buf.extend_from_slice(&chunk);
            }

            if needs_resample {
                if !raw_buf.is_empty() {
                    let resampled = qwen_asr::audio::resample(
                        &raw_buf, device_rate as i32, target_rate,
                    );
                    resampled_buf.extend_from_slice(&resampled);
                    raw_buf.clear();
                }
            } else {
                resampled_buf.append(&mut raw_buf);
            }

            if resampled_buf.len() >= segment_samples_16k {
                ctx.reset_perf();
                let _text = transcribe::transcribe_audio(ctx, &resampled_buf);
                resampled_buf.clear();
                if verbosity > 0 {
                    println!();
                }
            }
        }

        // Flush remaining
        if !raw_buf.is_empty() && needs_resample {
            let resampled = qwen_asr::audio::resample(
                &raw_buf, device_rate as i32, target_rate,
            );
            resampled_buf.extend_from_slice(&resampled);
        } else {
            resampled_buf.append(&mut raw_buf);
        }
        if !resampled_buf.is_empty() {
            ctx.reset_perf();
            let _text = transcribe::transcribe_audio(ctx, &resampled_buf);
            if verbosity > 0 {
                println!();
            }
        }
    }

    // ---- Benchmark summary ----
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let audio_s = resampled_buf.len() as f64 / target_rate as f64;

    if verbosity >= 1 {
        eprintln!("\nStopped.");
        let tokens_per_sec = if ctx.perf_total_ms > 0.0 {
            1000.0 * ctx.perf_text_tokens as f64 / ctx.perf_total_ms
        } else {
            0.0
        };
        eprintln!(
            "Inference: {:.0} ms, {} text tokens ({:.2} tok/s, encoding: {:.0}ms, decoding: {:.0}ms)",
            ctx.perf_total_ms, ctx.perf_text_tokens, tokens_per_sec,
            ctx.perf_encode_ms, ctx.perf_decode_ms
        );
        if audio_s > 0.0 && ctx.perf_total_ms > 0.0 {
            let infer_s = ctx.perf_total_ms / 1000.0;
            eprintln!(
                "Audio: {:.1} s processed in {:.1} s compute ({:.2}x realtime), {:.1} s wall clock",
                audio_s, infer_s, audio_s / infer_s, wall_ms / 1000.0
            );
        }
    }

    if profile {
        kernels::profile_report();
    }
}
