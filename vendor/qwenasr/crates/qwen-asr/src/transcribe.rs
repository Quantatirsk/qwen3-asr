//! Offline, segmented, and streaming transcription orchestration.

use crate::audio;
use crate::config::*;
use crate::context::QwenCtx;
use crate::decoder::{self, tok_embed_bf16_to_f32};
use crate::kernels;
use crate::tokenizer::QwenTokenizer;

use std::time::Instant;

// Prompt token sequences
const PREFIX_HEAD: &[i32] = &[151644, 8948, 198];
const PREFIX_TAIL: &[i32] = &[151645, 198, 151644, 872, 198, 151669];
const SUFFIX_BASE: &[i32] = &[151670, 151645, 198, 151644, 77091, 198];

// Streaming robustness constants (matching C reference)
const STREAM_DEGEN_MAX_PERIOD: usize = 6;
const STREAM_DEGEN_MIN_REPEATS: usize = 4;
const STREAM_STALE_CHUNKS: i32 = 4;
const STREAM_RESET_INTERVAL_CHUNKS: i32 = 45;
const STREAM_RESET_CARRY_TOKENS: usize = 24;
const STREAM_MAX_ENC_WINDOWS: usize = 4;

fn get_time_ms() -> f64 {
    // Use monotonic clock
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_secs_f64() * 1000.0
}

fn elapsed_ms(t0: f64) -> f64 {
    get_time_ms() - t0
}

/// Returns `(best_reps, best_period)`: how many times a block of `best_period`
/// tokens repeats at the tail of `tokens`. Used for streaming degeneracy detection.
fn stream_tail_repeat_blocks(tokens: &[i32], max_period: usize) -> (usize, usize) {
    let n = tokens.len();
    if n < 2 { return (1, 0); }
    let period_cap = (n / 2).min(if max_period > 0 { max_period } else { n / 2 });
    let mut best_reps = 1usize;
    let mut best_period = 0usize;
    for p in 1..=period_cap {
        let mut reps = 1usize;
        while (reps + 1) * p <= n {
            let a = &tokens[n - (reps + 1) * p .. n - reps * p];
            let b = &tokens[n - reps * p       .. n - (reps - 1) * p];
            if a != b { break; }
            reps += 1;
        }
        if reps > best_reps { best_reps = reps; best_period = p; }
    }
    (best_reps, best_period)
}

fn load_tokenizer(model_dir: &str) -> Option<QwenTokenizer> {
    let vocab_path = format!("{}/vocab.json", model_dir);
    QwenTokenizer::load(&vocab_path)
}

/// Transcribe a single segment. Returns (text, n_text_tokens).
fn transcribe_segment(
    ctx: &mut QwenCtx,
    samples: &[f32],
    tokenizer: &QwenTokenizer,
    past_tokens: Option<&[i32]>,
) -> Option<(String, i32)> {
    let cfg = &ctx.config.clone();
    let dim = cfg.dec_hidden;
    let seg_t0 = get_time_ms();
    let mut n_text_tokens = 0i32;

    // Mel spectrogram
    let t0 = get_time_ms();
    let (mel, mel_frames) = audio::mel_spectrogram(samples)?;
    let mel_ms = elapsed_ms(t0);

    if kernels::verbose() >= 2 {
        eprintln!("  Mel: {} frames ({:.0} ms)", mel_frames, mel_ms);
    }

    // Encoder
    let t0 = get_time_ms();
    let (enc_output, enc_seq_len) = ctx.encoder.forward(cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))?;
    let enc_ms = elapsed_ms(t0);

    if kernels::verbose() >= 2 {
        eprintln!("  Encoder: {} tokens ({:.0} ms)", enc_seq_len, enc_ms);
    }

    if !ctx.prepare_prompt_tokens(tokenizer) {
        return None;
    }

    // Build input embeddings
    let n_prompt_tokens = ctx.prompt_tokens.as_ref().map_or(0, |t| t.len());
    let n_force_prompt_tokens = ctx.force_prompt_tokens.as_ref().map_or(0, |t| t.len());
    let n_past = past_tokens.map_or(0, |t| t.len());
    let n_past_prompt_tokens = if n_past > 0 { n_past + 1 } else { 0 }; // +1 for <asr_text>

    let prefix_len = PREFIX_HEAD.len() + n_prompt_tokens + PREFIX_TAIL.len();
    let suffix_len = SUFFIX_BASE.len() + n_force_prompt_tokens;
    let total_seq = prefix_len + enc_seq_len + suffix_len + n_past_prompt_tokens;

    let mut input_embeds = vec![0.0f32; total_seq * dim];
    let tok_emb = ctx.decoder.tok_embeddings_bf16;

    // Embed prefix head
    let mut off = 0;
    for &tok in PREFIX_HEAD {
        unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim) };
        off += 1;
    }

    // Optional prompt
    if let Some(ref ptoks) = ctx.prompt_tokens {
        for &tok in ptoks {
            unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim) };
            off += 1;
        }
    }

    // Prefix tail
    for &tok in PREFIX_TAIL {
        unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim) };
        off += 1;
    }

    // Encoder output
    for i in 0..enc_seq_len {
        input_embeds[(prefix_len + i) * dim..(prefix_len + i + 1) * dim]
            .copy_from_slice(&enc_output[i * dim..(i + 1) * dim]);
    }

    // Suffix base
    let suffix_off = prefix_len + enc_seq_len;
    for (i, &tok) in SUFFIX_BASE.iter().enumerate() {
        unsafe { tok_embed_bf16_to_f32(
            &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim],
            tok_emb, tok, dim,
        ) };
    }

    // Force language tokens
    if let Some(ref ftoks) = ctx.force_prompt_tokens {
        for (i, &tok) in ftoks.iter().enumerate() {
            unsafe { tok_embed_bf16_to_f32(
                &mut input_embeds[(suffix_off + SUFFIX_BASE.len() + i) * dim
                    ..(suffix_off + SUFFIX_BASE.len() + i + 1) * dim],
                tok_emb, tok, dim,
            ) };
        }
    }

    // Past text conditioning tokens
    let past_off = suffix_off + suffix_len;
    if let Some(ptoks) = past_tokens {
        for (i, &tok) in ptoks.iter().enumerate() {
            unsafe { tok_embed_bf16_to_f32(
                &mut input_embeds[(past_off + i) * dim..(past_off + i + 1) * dim],
                tok_emb, tok, dim,
            ) };
        }
        unsafe { tok_embed_bf16_to_f32(
            &mut input_embeds[(past_off + ptoks.len()) * dim..(past_off + ptoks.len() + 1) * dim],
            tok_emb, TOKEN_ASR_TEXT, dim,
        ) };
    }

    // Decoder prefill
    let t0 = get_time_ms();
    ctx.kv_cache.len = 0;
    let prefill_len = total_seq - 1;
    decoder::decoder_prefill(
        &ctx.decoder, cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
        &mut ctx.dec_bufs, &input_embeds, prefill_len,
    );

    // First token from last prefill position
    let last_embed = &input_embeds[prefill_len * dim..(prefill_len + 1) * dim];
    let mut token = decoder::decoder_forward(
        &ctx.decoder, cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
        &mut ctx.dec_bufs, last_embed,
    );

    let prefill_ms = elapsed_ms(t0);
    if kernels::verbose() >= 2 {
        eprintln!("  Prefill: {} tokens ({:.0} ms)", total_seq, prefill_ms);
    }

    // Autoregressive decode
    let t0 = get_time_ms();
    let max_tokens = 2048;
    let mut n_generated = 0;
    let mut past_asr_text = n_force_prompt_tokens > 0 || n_past > 0;

    let mut text_bytes: Vec<u8> = Vec::new();
    let mut tmp_embed = vec![0.0f32; dim];

    while n_generated < max_tokens {
        n_generated += 1;

        if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
            break;
        }

        if token == TOKEN_ASR_TEXT {
            past_asr_text = true;
        } else if past_asr_text {
            let piece_bytes = tokenizer.decode_bytes(token);
            text_bytes.extend_from_slice(piece_bytes);
            n_text_tokens += 1;

            if let Some(ref cb) = ctx.token_cb {
                // For the callback, provide lossy UTF-8 for display purposes
                cb(&String::from_utf8_lossy(piece_bytes));
            }
        }

        unsafe { tok_embed_bf16_to_f32(&mut tmp_embed, tok_emb, token, dim) };
        token = decoder::decoder_forward(
            &ctx.decoder, cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
            &mut ctx.dec_bufs, &tmp_embed,
        );
    }

    let decode_ms = elapsed_ms(t0);
    if kernels::verbose() >= 2 {
        eprintln!(
            "  Decode: {} tokens ({:.0} ms, {:.1} ms/token)",
            n_generated, decode_ms,
            if n_generated > 0 { decode_ms / n_generated as f64 } else { 0.0 }
        );
    }

    // Trim whitespace — convert accumulated bytes to UTF-8 first
    let text = String::from_utf8_lossy(&text_bytes);
    let trimmed = text.trim().to_string();

    ctx.perf_total_ms += elapsed_ms(seg_t0);
    ctx.perf_text_tokens += n_text_tokens;
    ctx.perf_encode_ms += mel_ms + enc_ms;
    ctx.perf_decode_ms += prefill_ms + decode_ms;

    Some((trimmed, n_text_tokens))
}

// ========================================================================
// Segment-based splitting
// ========================================================================

fn find_split_point(samples: &[f32], target_sample: usize, search_sec: f32) -> usize {
    let search_half = (search_sec * SAMPLE_RATE as f32) as usize;
    let lo = target_sample.saturating_sub(search_half);
    let hi = (target_sample + search_half).min(samples.len());

    let win_samples = 1600; // 100ms at 16kHz
    let mut best_energy = 1e30f32;
    let mut best_center = target_sample;

    let mut pos = lo;
    while pos + win_samples <= hi {
        let end = (pos + win_samples).min(samples.len());
        let mut energy = 0.0f32;
        for &s in samples.iter().take(end).skip(pos) {
            energy += s * s;
        }
        energy /= (end - pos) as f32;
        if energy < best_energy {
            best_energy = energy;
            best_center = pos + (end - pos) / 2;
        }
        pos += win_samples / 2;
    }

    best_center
}

fn should_insert_boundary_space(prev_ch: u8, next_ch: u8) -> bool {
    if prev_ch == 0 || next_ch == 0 { return false; }
    if (prev_ch as char).is_whitespace() { return false; }
    if (next_ch as char).is_whitespace() { return false; }
    if (next_ch as char).is_ascii_punctuation() { return false; }
    true
}

// ========================================================================
// Public API
// ========================================================================

/// Transcribe audio samples (f32, 16 kHz, mono, range [-1, 1]).
///
/// When `ctx.segment_sec > 0` and the audio exceeds that duration, it is
/// automatically split at low-energy boundaries.
/// Returns `None` if the tokenizer or encoder fails to initialize.
pub fn transcribe_audio(ctx: &mut QwenCtx, samples: &[f32]) -> Option<String> {
    ctx.reset_perf();
    ctx.perf_audio_ms = 1000.0 * samples.len() as f64 / SAMPLE_RATE as f64;

    let audio_samples = if ctx.skip_silence {
        let compacted = audio::compact_silence(samples);
        if kernels::verbose() >= 1 {
            let used_pct = 100.0 * compacted.len() as f32 / samples.len().max(1) as f32;
            eprintln!(
                "Silence skip: used {:.1}%, skipped {:.1}% ({} -> {} samples)",
                used_pct, 100.0 - used_pct, samples.len(), compacted.len()
            );
        }
        compacted
    } else {
        samples.to_vec()
    };

    if kernels::verbose() >= 2 {
        eprintln!(
            "Audio: {} samples ({:.1} seconds)",
            audio_samples.len(),
            audio_samples.len() as f32 / SAMPLE_RATE as f32
        );
    }

    let tokenizer = load_tokenizer(&ctx.model_dir)?;
    if !ctx.prepare_prompt_tokens(&tokenizer) {
        return None;
    }

    let target_samples = (ctx.segment_sec * SAMPLE_RATE as f32) as usize;
    let search = ctx.search_sec.min(ctx.segment_sec / 2.0);
    let margin_samples = (search * SAMPLE_RATE as f32) as usize;

    // No splitting if segment_sec is 0 or audio fits in one segment
    if ctx.segment_sec <= 0.0 || audio_samples.len() <= target_samples + margin_samples {
        let (text, _) = transcribe_segment(ctx, &audio_samples, &tokenizer, None)?;
        return Some(text);
    }

    // Build split points
    let mut splits = vec![0usize];
    let mut pos = 0;
    while pos + target_samples + margin_samples < audio_samples.len() {
        let split = find_split_point(&audio_samples, pos + target_samples, search);
        splits.push(split);
        pos = split;
        if splits.len() >= 127 { break; }
    }
    let n_splits = splits.len();

    if kernels::verbose() >= 2 {
        eprintln!("Splitting into {} segments", n_splits);
    }

    let mut result = String::new();
    let min_samples = SAMPLE_RATE as usize / 2;
    let use_past_text = ctx.past_text_conditioning;

    for s in 0..n_splits {
        let core_start = splits[s];
        let core_end = if s + 1 < n_splits { splits[s + 1] } else { audio_samples.len() };
        let seg_start = core_start;
        let seg_end = core_end;
        let seg_samples = seg_end - seg_start;

        if kernels::verbose() >= 2 {
            eprintln!(
                "Segment {}/{}: {:.1}-{:.1}s ({} samples)",
                s + 1, n_splits,
                seg_start as f32 / SAMPLE_RATE as f32,
                seg_end as f32 / SAMPLE_RATE as f32,
                seg_samples
            );
        }

        let seg_buf: Vec<f32>;
        let seg_ptr = if seg_samples < min_samples {
            seg_buf = {
                let mut buf = vec![0.0f32; min_samples];
                buf[..seg_samples].copy_from_slice(&audio_samples[seg_start..seg_end]);
                buf
            };
            &seg_buf[..]
        } else {
            &audio_samples[seg_start..seg_end]
        };

        let past_tokens: Option<Vec<i32>> = if use_past_text && !result.is_empty() {
            tokenizer.encode(&result)
        } else {
            None
        };

        let (seg_text, _seg_text_tokens) = match transcribe_segment(
            ctx, seg_ptr, &tokenizer, past_tokens.as_deref(),
        ) {
            Some(r) => r,
            None => continue,
        };

        if seg_text.is_empty() { continue; }

        let need_space = if !result.is_empty() {
            let prev = *result.as_bytes().last().unwrap_or(&0);
            let next = *seg_text.as_bytes().first().unwrap_or(&0);
            should_insert_boundary_space(prev, next)
        } else {
            false
        };

        if need_space {
            result.push(' ');
            if let Some(ref cb) = ctx.token_cb {
                cb(" ");
            }
        }

        if let Some(ref cb) = ctx.token_cb {
            if ctx.past_text_conditioning {
                cb(&seg_text);
            }
        }

        result.push_str(&seg_text);
    }

    Some(result)
}

/// Convenience wrapper: load a WAV file and transcribe it.
///
/// Equivalent to [`audio::load_wav`](crate::audio::load_wav) followed by
/// [`transcribe_audio`].
pub fn transcribe(ctx: &mut QwenCtx, wav_path: &str) -> Option<String> {
    let samples = audio::load_wav(wav_path)?;
    transcribe_audio(ctx, &samples)
}

/// Transcribe from stdin.
pub fn transcribe_stdin(ctx: &mut QwenCtx) -> Option<String> {
    let samples = audio::read_pcm_stdin()?;
    transcribe_audio(ctx, &samples)
}

/// Streaming transcription: processes audio in chunks, emitting tokens via
/// `ctx.token_cb` as they become stable.
///
/// Trades throughput for lower latency compared to offline mode. If no
/// `token_cb` is set, falls back to a single offline decode of the full audio.
pub fn transcribe_stream(ctx: &mut QwenCtx, samples: &[f32]) -> Option<String> {
    let cfg = ctx.config.clone();
    let dim = cfg.dec_hidden;
    let chunk_samples = (ctx.stream_chunk_sec * SAMPLE_RATE as f32) as usize;
    let rollback = ctx.stream_rollback;
    let unfixed_chunks = ctx.stream_unfixed_chunks;
    let max_new_tokens = if ctx.stream_max_new_tokens > 0 { ctx.stream_max_new_tokens } else { 32 };

    let audio_samples = if ctx.skip_silence {
        audio::compact_silence(samples)
    } else {
        samples.to_vec()
    };

    ctx.reset_perf();
    ctx.perf_audio_ms = 1000.0 * samples.len() as f64 / SAMPLE_RATE as f64;

    // If no token callback, fall back to offline decode
    if ctx.token_cb.is_none() {
        if kernels::verbose() >= 2 {
            eprintln!("Streaming: no token callback, using direct final refinement");
        }
        let tokenizer = load_tokenizer(&ctx.model_dir)?;
        ctx.prepare_prompt_tokens(&tokenizer);
        let (text, _) = transcribe_segment(ctx, &audio_samples, &tokenizer, None)?;
        return Some(text);
    }

    let tokenizer = load_tokenizer(&ctx.model_dir)?;
    if !ctx.prepare_prompt_tokens(&tokenizer) {
        return None;
    }

    let enc_window_frames = cfg.enc_n_window_infer.clamp(100, 800);
    let enc_window_samples = enc_window_frames * HOP_LENGTH;

    let tok_emb = ctx.decoder.tok_embeddings_bf16;

    let mut raw_tokens: Vec<i32> = Vec::new();
    let mut stable_text_tokens: Vec<i32> = Vec::new();
    let mut result_bytes: Vec<u8> = Vec::new();
    let mut tmp_embed = vec![0.0f32; dim];

    let mut chunk_idx = 0i32;
    let mut audio_cursor = 0usize;

    // Encoder window cache
    struct EncWindow {
        seq_len: usize,
        enc_output: Vec<f32>,
    }
    let mut enc_cache: Vec<EncWindow> = Vec::new();
    let mut enc_cached_seq_total = 0usize;

    // Previous prefill embeddings for LCP reuse
    let mut prev_prefill_embeds: Vec<f32> = Vec::new();
    let mut prev_prefill_len = 0usize;

    // Streaming robustness state
    let mut stale_count = 0i32;
    let mut prev_tail_snapshot: Vec<i32> = Vec::new();
    let mut enc_cache_base_windows = 0usize;

    while audio_cursor < audio_samples.len() {
        let chunk_t0 = get_time_ms();
        audio_cursor = (audio_cursor + chunk_samples).min(audio_samples.len());
        let is_final = audio_cursor >= audio_samples.len();

        // Encoder
        let t0 = get_time_ms();
        let full_end = (audio_cursor / enc_window_samples) * enc_window_samples;

        // Cache completed windows (base offset accounts for windows cleared on re-anchor)
        while (enc_cache_base_windows + enc_cache.len()) * enc_window_samples < full_end {
            let ws = (enc_cache_base_windows + enc_cache.len()) * enc_window_samples;
            let (mel, mel_frames) = audio::mel_spectrogram(&audio_samples[ws..ws + enc_window_samples])?;
            let (win_enc, win_seq) = ctx.encoder.forward(&cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))?;
            enc_cached_seq_total += win_seq;
            enc_cache.push(EncWindow { seq_len: win_seq, enc_output: win_enc });
        }

        // Encode partial tail
        let mut partial_seq = 0;
        let mut partial_enc: Vec<f32> = Vec::new();
        if full_end < audio_cursor {
            let _partial_samples = audio_cursor - full_end;
            if let Some((mel, mel_frames)) = audio::mel_spectrogram(&audio_samples[full_end..audio_cursor]) {
                if let Some((enc, seq)) = ctx.encoder.forward(&cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs)) {
                    partial_seq = seq;
                    partial_enc = enc;
                }
            }
        }

        let enc_seq_len = enc_cached_seq_total + partial_seq;
        if enc_seq_len == 0 {
            chunk_idx += 1;
            continue;
        }

        // Assemble encoder output
        let mut enc_output = vec![0.0f32; enc_seq_len * dim];
        let mut enc_off = 0;
        for w in &enc_cache {
            enc_output[enc_off * dim..(enc_off + w.seq_len) * dim]
                .copy_from_slice(&w.enc_output);
            enc_off += w.seq_len;
        }
        if partial_seq > 0 {
            enc_output[enc_off * dim..(enc_off + partial_seq) * dim]
                .copy_from_slice(&partial_enc);
        }

        let enc_ms = elapsed_ms(t0);
        ctx.perf_encode_ms += enc_ms;

        // Prefix rollback
        let n_prefix_tokens = if ctx.past_text_conditioning && chunk_idx >= unfixed_chunks && !raw_tokens.is_empty() {
            (raw_tokens.len() as i32 - rollback).max(0) as usize
        } else {
            0
        };

        // Build input embeddings
        let n_prompt_tokens = ctx.prompt_tokens.as_ref().map_or(0, |t| t.len());
        let n_force_prompt_tokens = ctx.force_prompt_tokens.as_ref().map_or(0, |t| t.len());
        let prefix_len = PREFIX_HEAD.len() + n_prompt_tokens + PREFIX_TAIL.len();
        let suffix_len = SUFFIX_BASE.len() + n_force_prompt_tokens;
        let total_seq = prefix_len + enc_seq_len + suffix_len + n_prefix_tokens;

        let mut input_embeds = vec![0.0f32; total_seq * dim];
        let mut off = 0;

        for &tok in PREFIX_HEAD {
            unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim) };
            off += 1;
        }
        if let Some(ref ptoks) = ctx.prompt_tokens {
            for &tok in ptoks {
                unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim) };
                off += 1;
            }
        }
        for &tok in PREFIX_TAIL {
            unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim) };
            off += 1;
        }

        for i in 0..enc_seq_len {
            input_embeds[(prefix_len + i) * dim..(prefix_len + i + 1) * dim]
                .copy_from_slice(&enc_output[i * dim..(i + 1) * dim]);
        }

        let suffix_off = prefix_len + enc_seq_len;
        for (i, &tok) in SUFFIX_BASE.iter().enumerate() {
            unsafe { tok_embed_bf16_to_f32(
                &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim],
                tok_emb, tok, dim,
            ) };
        }
        if let Some(ref ftoks) = ctx.force_prompt_tokens {
            for (i, &tok) in ftoks.iter().enumerate() {
                unsafe { tok_embed_bf16_to_f32(
                    &mut input_embeds[(suffix_off + SUFFIX_BASE.len() + i) * dim
                        ..(suffix_off + SUFFIX_BASE.len() + i + 1) * dim],
                    tok_emb, tok, dim,
                ) };
            }
        }

        let text_off = suffix_off + suffix_len;
        for i in 0..n_prefix_tokens {
            unsafe { tok_embed_bf16_to_f32(
                &mut input_embeds[(text_off + i) * dim..(text_off + i + 1) * dim],
                tok_emb, raw_tokens[i], dim,
            ) };
        }

        // Decoder prefill with LCP reuse
        let t0 = get_time_ms();
        let prefill_len = total_seq - 1;

        let mut reused_prefill = 0;
        if !prev_prefill_embeds.is_empty() && prev_prefill_len > 0 {
            let cmp_len = prefill_len.min(prev_prefill_len);
            let _row_bytes = dim * std::mem::size_of::<f32>();
            while reused_prefill < cmp_len {
                let a = &prev_prefill_embeds[reused_prefill * dim..(reused_prefill + 1) * dim];
                let b = &input_embeds[reused_prefill * dim..(reused_prefill + 1) * dim];
                if a != b { break; }
                reused_prefill += 1;
            }
        }

        ctx.kv_cache.len = reused_prefill;
        let delta_prefill = prefill_len - reused_prefill;
        if delta_prefill > 0 {
            decoder::decoder_prefill(
                &ctx.decoder, &cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
                &mut ctx.dec_bufs,
                &input_embeds[reused_prefill * dim..],
                delta_prefill,
            );
        }

        let last_embed = &input_embeds[prefill_len * dim..(prefill_len + 1) * dim];
        let mut token = decoder::decoder_forward(
            &ctx.decoder, &cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
            &mut ctx.dec_bufs, last_embed,
        );

        // Save for next chunk
        prev_prefill_embeds = input_embeds[..prefill_len * dim].to_vec();
        prev_prefill_len = prefill_len;

        let prefill_ms = elapsed_ms(t0);
        ctx.perf_decode_ms += prefill_ms;

        // Autoregressive decode
        let t0 = get_time_ms();
        let mut chunk_tokens: Vec<i32> = Vec::new();
        let mut n_generated = 0;

        while n_generated < max_new_tokens {
            n_generated += 1;
            if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END { break; }
            chunk_tokens.push(token);
            unsafe { tok_embed_bf16_to_f32(&mut tmp_embed, tok_emb, token, dim); }
            token = decoder::decoder_forward(
                &ctx.decoder, &cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
                &mut ctx.dec_bufs, &tmp_embed,
            );
        }

        let decode_ms = elapsed_ms(t0);
        ctx.perf_decode_ms += decode_ms;

        // Update raw token history
        raw_tokens.truncate(n_prefix_tokens);
        raw_tokens.extend_from_slice(&chunk_tokens);

        // Streaming degeneracy detection
        if raw_tokens == prev_tail_snapshot {
            stale_count += 1;
        } else {
            stale_count = 0;
            prev_tail_snapshot = raw_tokens.clone();
        }
        let (best_reps, _) = stream_tail_repeat_blocks(&raw_tokens, STREAM_DEGEN_MAX_PERIOD);
        let is_degen = stale_count >= STREAM_STALE_CHUNKS
            || best_reps >= STREAM_DEGEN_MIN_REPEATS;

        if is_degen {
            if kernels::verbose() >= 2 {
                eprintln!("[stream degen] reset at chunk {} (stale={}, reps={})",
                    chunk_idx, stale_count, best_reps);
            }
            let carry = stable_text_tokens.len().min(STREAM_RESET_CARRY_TOKENS);
            let carry_start = stable_text_tokens.len() - carry;
            raw_tokens.clear();
            if carry > 0 {
                raw_tokens.push(TOKEN_ASR_TEXT);
                raw_tokens.extend_from_slice(&stable_text_tokens[carry_start..]);
            }
            prev_prefill_embeds.clear();
            prev_prefill_len = 0;
            stale_count = 0;
            prev_tail_snapshot.clear();
        }

        // Periodic re-anchor: reset context every STREAM_RESET_INTERVAL_CHUNKS chunks
        if chunk_idx > 0 && chunk_idx % STREAM_RESET_INTERVAL_CHUNKS == 0 {
            if kernels::verbose() >= 2 {
                eprintln!("[stream reanchor] at chunk {}", chunk_idx);
            }
            let carry = stable_text_tokens.len().min(STREAM_RESET_CARRY_TOKENS);
            let carry_start = stable_text_tokens.len() - carry;
            raw_tokens.clear();
            if carry > 0 {
                raw_tokens.push(TOKEN_ASR_TEXT);
                raw_tokens.extend_from_slice(&stable_text_tokens[carry_start..]);
            }
            prev_prefill_embeds.clear();
            prev_prefill_len = 0;
            stale_count = 0;
            prev_tail_snapshot.clear();
            enc_cache_base_windows += enc_cache.len();
            enc_cache.clear();
            enc_cached_seq_total = 0;
        }

        // Parse text region
        let text_start = if n_force_prompt_tokens == 0 {
            raw_tokens.iter().position(|&t| t == TOKEN_ASR_TEXT)
                .map(|p| p + 1)
                .unwrap_or(0)
        } else {
            0
        };
        let n_text_tokens = raw_tokens.len().saturating_sub(text_start);

        // Fixed frontier
        let candidate_len = if is_final {
            n_text_tokens
        } else if chunk_idx >= unfixed_chunks {
            (n_text_tokens as i32 - rollback).max(0) as usize
        } else {
            0
        };

        // Monotonic commit
        let candidate_tokens = &raw_tokens[text_start..];
        let _lcp = stable_text_tokens.iter().zip(candidate_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();

        let emit_from = stable_text_tokens.len();
        let emit_to = candidate_len.max(emit_from);

        for i in emit_from..emit_to {
            if i < candidate_tokens.len() {
                if i >= stable_text_tokens.len() {
                    stable_text_tokens.push(candidate_tokens[i]);
                }
                let piece_bytes = tokenizer.decode_bytes(candidate_tokens[i]);
                if let Some(ref cb) = ctx.token_cb {
                    cb(&String::from_utf8_lossy(piece_bytes));
                }
                ctx.perf_text_tokens += 1;
                result_bytes.extend_from_slice(piece_bytes);
            }
        }

        ctx.perf_total_ms += elapsed_ms(chunk_t0);
        chunk_idx += 1;
    }

    Some(String::from_utf8_lossy(&result_bytes).trim().to_string())
}// ========================================================================
// Incremental Streaming API
// ========================================================================

/// Encoder window cached output.
struct EncWindow {
    seq_len: usize,
    enc_output: Vec<f32>,
}

/// Persistent state for incremental streaming transcription.
///
/// Create once, then call [`stream_push_audio`] each time new audio
/// arrives.  The state keeps encoder caches, token history, and decoder
/// prefill embeddings so that only *new* work is performed per call.
pub struct StreamState {
    // Encoder
    enc_cache: Vec<EncWindow>,
    enc_cached_seq_total: usize,
    enc_cache_base_windows: usize,

    // Decoder token history
    raw_tokens: Vec<i32>,
    stable_text_tokens: Vec<i32>,
    result_bytes: Vec<u8>,

    // Prefill LCP reuse
    prev_prefill_embeds: Vec<f32>,
    prev_prefill_len: usize,

    // Streaming robustness
    prev_tail_snapshot: Vec<i32>,
    stale_count: i32,

    // Lazy partial encoding: skip re-encoding every other chunk
    last_partial_cursor: usize,
    last_partial_enc: Vec<f32>,
    last_partial_seq: usize,

    // Audio cursor
    audio_cursor: usize,
    chunk_idx: i32,

    // Tokenizer (loaded once)
    tokenizer: Option<QwenTokenizer>,
    prompt_prepared: bool,
}

impl Default for StreamState {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamState {
    /// Create a new empty streaming state.
    pub fn new() -> Self {
        StreamState {
            enc_cache: Vec::new(),
            enc_cached_seq_total: 0,
            enc_cache_base_windows: 0,
            raw_tokens: Vec::new(),
            stable_text_tokens: Vec::new(),
            result_bytes: Vec::new(),
            prev_prefill_embeds: Vec::new(),
            prev_prefill_len: 0,
            prev_tail_snapshot: Vec::new(),
            stale_count: 0,
            last_partial_cursor: 0,
            last_partial_enc: Vec::new(),
            last_partial_seq: 0,
            audio_cursor: 0,
            chunk_idx: 0,
            tokenizer: None,
            prompt_prepared: false,
        }
    }

    /// Reset state for a new streaming window (e.g., after 30s limit).
    pub fn reset(&mut self) {
        self.enc_cache.clear();
        self.enc_cached_seq_total = 0;
        self.enc_cache_base_windows = 0;
        self.raw_tokens.clear();
        self.stable_text_tokens.clear();
        self.result_bytes.clear();
        self.prev_prefill_embeds.clear();
        self.prev_prefill_len = 0;
        self.prev_tail_snapshot.clear();
        self.stale_count = 0;
        self.last_partial_cursor = 0;
        self.last_partial_enc.clear();
        self.last_partial_seq = 0;
        self.audio_cursor = 0;
        self.chunk_idx = 0;
        // Keep tokenizer and prompt_prepared
    }

    /// Get the current stable transcription result.
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.result_bytes).into_owned()
    }

    /// Get how many samples have been processed so far.
    pub fn audio_cursor(&self) -> usize {
        self.audio_cursor
    }
}

/// Process all available new audio incrementally.
///
/// `samples` is the **full** audio buffer accumulated so far (16 kHz mono f32).
/// Processes all full chunks from `state.audio_cursor` to end of `samples`.
/// When `finalize` is true, also processes any remaining partial chunk and
/// emits all rollback-buffered tokens.
/// Returns the newly emitted text delta (if any).
pub fn stream_push_audio(
    ctx: &mut QwenCtx,
    samples: &[f32],
    state: &mut StreamState,
    finalize: bool,
) -> Option<String> {
    let cfg = ctx.config.clone();
    let dim = cfg.dec_hidden;
    let chunk_samples = (ctx.stream_chunk_sec * SAMPLE_RATE as f32) as usize;
    let rollback = ctx.stream_rollback;
    let unfixed_chunks = ctx.stream_unfixed_chunks;
    let max_new_tokens = if ctx.stream_max_new_tokens > 0 { ctx.stream_max_new_tokens } else { 32 };

    // Lazy-init tokenizer
    if state.tokenizer.is_none() {
        state.tokenizer = load_tokenizer(&ctx.model_dir);
    }
    let tokenizer = state.tokenizer.as_ref()?;

    if !state.prompt_prepared {
        if !ctx.prepare_prompt_tokens(tokenizer) {
            return None;
        }
        state.prompt_prepared = true;
    }

    // Check if we have enough audio for at least one chunk (or finalizing)
    let available = samples.len().saturating_sub(state.audio_cursor);
    if available < chunk_samples && !finalize {
        return Some(String::new());
    }
    if available == 0 {
        return Some(String::new());
    }

    let enc_window_frames = cfg.enc_n_window_infer.clamp(100, 800);
    let enc_window_samples = enc_window_frames * HOP_LENGTH;
    let tok_emb = ctx.decoder.tok_embeddings_bf16;
    let mut tmp_embed = vec![0.0f32; dim];
    let mut delta_bytes: Vec<u8> = Vec::new();

    // ---- Process full chunks, plus remainder if finalizing ----
    while state.audio_cursor < samples.len() {
        let remaining = samples.len() - state.audio_cursor;
        if remaining < chunk_samples && !finalize {
            break; // Wait for more audio
        }

        let chunk_t0 = get_time_ms();
        state.audio_cursor = (state.audio_cursor + chunk_samples).min(samples.len());
        let is_final = finalize && state.audio_cursor >= samples.len();

    // ---- Encoder: only encode new windows ----
    let t0 = get_time_ms();
    let full_end = (state.audio_cursor / enc_window_samples) * enc_window_samples;

    // Cache newly completed windows (base offset accounts for windows cleared on re-anchor)
    while (state.enc_cache_base_windows + state.enc_cache.len()) * enc_window_samples < full_end {
        let ws = (state.enc_cache_base_windows + state.enc_cache.len()) * enc_window_samples;
        let (mel, mel_frames) = audio::mel_spectrogram(&samples[ws..ws + enc_window_samples])?;
        let (win_enc, win_seq) = ctx.encoder.forward(&cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))?;
        state.enc_cached_seq_total += win_seq;
        state.enc_cache.push(EncWindow { seq_len: win_seq, enc_output: win_enc });
    }

    // Encode partial tail — with lazy re-encoding for LCP optimization.
    // Only re-encode when enough new audio has accumulated (every 2 chunks),
    // on the first chunk, or when finalizing. On skip chunks, the reused
    // encoder output gives near-perfect LCP matching, cutting prefill cost.
    let enc_update_threshold = chunk_samples * 2;
    let partial_age = state.audio_cursor.saturating_sub(state.last_partial_cursor);
    let need_encode = state.last_partial_cursor == 0
        || partial_age >= enc_update_threshold
        || is_final;

    let partial_seq;
    let partial_enc;
    if need_encode && full_end < state.audio_cursor {
        if let Some((mel, mel_frames)) = audio::mel_spectrogram(&samples[full_end..state.audio_cursor]) {
            if let Some((enc, seq)) = ctx.encoder.forward(&cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs)) {
                partial_seq = seq;
                partial_enc = enc;
                state.last_partial_cursor = state.audio_cursor;
                state.last_partial_enc = partial_enc.clone();
                state.last_partial_seq = partial_seq;
            } else {
                partial_seq = state.last_partial_seq;
                partial_enc = state.last_partial_enc.clone();
            }
        } else {
            partial_seq = state.last_partial_seq;
            partial_enc = state.last_partial_enc.clone();
        }
    } else if full_end < state.audio_cursor {
        // Reuse previous partial encoding (skip chunk)
        partial_seq = state.last_partial_seq;
        partial_enc = state.last_partial_enc.clone();
    } else {
        partial_seq = 0;
        partial_enc = Vec::new();
    }

    let enc_seq_len = state.enc_cached_seq_total + partial_seq;
    if enc_seq_len == 0 {
        state.chunk_idx += 1;
        return Some(String::new());
    }

    // Assemble encoder output (cached windows + partial)
    let mut enc_output = vec![0.0f32; enc_seq_len * dim];
    let mut enc_off = 0;
    for w in &state.enc_cache {
        enc_output[enc_off * dim..(enc_off + w.seq_len) * dim]
            .copy_from_slice(&w.enc_output);
        enc_off += w.seq_len;
    }
    if partial_seq > 0 {
        enc_output[enc_off * dim..(enc_off + partial_seq) * dim]
            .copy_from_slice(&partial_enc);
    }

    let enc_ms = elapsed_ms(t0);
    ctx.perf_encode_ms += enc_ms;

    // ---- Prefix rollback ----
    let n_prefix_tokens = if ctx.past_text_conditioning
        && state.chunk_idx >= unfixed_chunks
        && !state.raw_tokens.is_empty()
    {
        (state.raw_tokens.len() as i32 - rollback).max(0) as usize
    } else {
        0
    };

    // ---- Build input embeddings ----
    let n_prompt_tokens = ctx.prompt_tokens.as_ref().map_or(0, |t| t.len());
    let n_force_prompt_tokens = ctx.force_prompt_tokens.as_ref().map_or(0, |t| t.len());
    let prefix_len = PREFIX_HEAD.len() + n_prompt_tokens + PREFIX_TAIL.len();
    let suffix_len = SUFFIX_BASE.len() + n_force_prompt_tokens;
    let total_seq = prefix_len + enc_seq_len + suffix_len + n_prefix_tokens;

    let mut input_embeds = vec![0.0f32; total_seq * dim];
    let mut off = 0;

    for &tok in PREFIX_HEAD {
        unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim); }
        off += 1;
    }
    if let Some(ref ptoks) = ctx.prompt_tokens {
        for &tok in ptoks {
            unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim); }
            off += 1;
        }
    }
    for &tok in PREFIX_TAIL {
        unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim); }
        off += 1;
    }

    for i in 0..enc_seq_len {
        input_embeds[(prefix_len + i) * dim..(prefix_len + i + 1) * dim]
            .copy_from_slice(&enc_output[i * dim..(i + 1) * dim]);
    }

    let suffix_off = prefix_len + enc_seq_len;
    for (i, &tok) in SUFFIX_BASE.iter().enumerate() {
        unsafe { tok_embed_bf16_to_f32(
            &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim],
            tok_emb, tok, dim,
        ); }
    }
    if let Some(ref ftoks) = ctx.force_prompt_tokens {
        for (i, &tok) in ftoks.iter().enumerate() {
            unsafe { tok_embed_bf16_to_f32(
                &mut input_embeds[(suffix_off + SUFFIX_BASE.len() + i) * dim
                    ..(suffix_off + SUFFIX_BASE.len() + i + 1) * dim],
                tok_emb, tok, dim,
            ); }
        }
    }

    let text_off = suffix_off + suffix_len;
    for i in 0..n_prefix_tokens {
        unsafe { tok_embed_bf16_to_f32(
            &mut input_embeds[(text_off + i) * dim..(text_off + i + 1) * dim],
            tok_emb, state.raw_tokens[i], dim,
        ); }
    }

    // ---- Decoder prefill with LCP reuse ----
    let t0 = get_time_ms();
    let prefill_len = total_seq - 1;

    let mut reused_prefill = 0;
    if !state.prev_prefill_embeds.is_empty() && state.prev_prefill_len > 0 {
        let cmp_len = prefill_len.min(state.prev_prefill_len);
        while reused_prefill < cmp_len {
            let a = &state.prev_prefill_embeds[reused_prefill * dim..(reused_prefill + 1) * dim];
            let b = &input_embeds[reused_prefill * dim..(reused_prefill + 1) * dim];
            if a != b { break; }
            reused_prefill += 1;
        }
    }

    ctx.kv_cache.len = reused_prefill;
    let delta_prefill = prefill_len - reused_prefill;
    if delta_prefill > 0 {
        decoder::decoder_prefill(
            &ctx.decoder, &cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
            &mut ctx.dec_bufs,
            &input_embeds[reused_prefill * dim..],
            delta_prefill,
        );
    }

    let last_embed = &input_embeds[prefill_len * dim..(prefill_len + 1) * dim];
    let mut token = decoder::decoder_forward(
        &ctx.decoder, &cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
        &mut ctx.dec_bufs, last_embed,
    );

    // Save for next chunk
    state.prev_prefill_embeds = input_embeds[..prefill_len * dim].to_vec();
    state.prev_prefill_len = prefill_len;

    let prefill_ms = elapsed_ms(t0);
    ctx.perf_decode_ms += prefill_ms;

    if kernels::verbose() >= 2 {
        eprintln!(
            "  [stream chunk {}] encoder: {:.0}ms, prefill: {}/{} reused ({:.0}ms, delta={})",
            state.chunk_idx, enc_ms, reused_prefill, prefill_len, prefill_ms, delta_prefill
        );
    }

    // ---- Autoregressive decode ----
    let t0 = get_time_ms();
    let mut chunk_tokens: Vec<i32> = Vec::new();
    let mut n_generated = 0;

    while n_generated < max_new_tokens {
        n_generated += 1;
        if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END { break; }
        chunk_tokens.push(token);
        unsafe { tok_embed_bf16_to_f32(&mut tmp_embed, tok_emb, token, dim); }
        token = decoder::decoder_forward(
            &ctx.decoder, &cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
            &mut ctx.dec_bufs, &tmp_embed,
        );
    }

    let decode_ms = elapsed_ms(t0);
    ctx.perf_decode_ms += decode_ms;

    // ---- Detect speech end (decoder produced EOT on silence) ----
    // When chunk_tokens is empty, the decoder saw silence/end-of-speech.
    // Commit ALL remaining rollback-buffered tokens BEFORE truncation,
    // since truncate will remove them.
    let speech_ended = chunk_tokens.is_empty()
        && !state.raw_tokens.is_empty()
        && state.chunk_idx >= unfixed_chunks;

    if speech_ended {
        // Emit remaining rollback tokens from current raw_tokens (before truncation)
        let text_start = if n_force_prompt_tokens == 0 {
            state.raw_tokens.iter().position(|&t| t == TOKEN_ASR_TEXT)
                .map(|p| p + 1)
                .unwrap_or(0)
        } else {
            0
        };
        let candidate_tokens = &state.raw_tokens[text_start..];
        let n_text = candidate_tokens.len();
        let emit_from = state.stable_text_tokens.len();
        for i in emit_from..n_text {
            if i < candidate_tokens.len() {
                if i >= state.stable_text_tokens.len() {
                    state.stable_text_tokens.push(candidate_tokens[i]);
                }
                let piece_bytes = tokenizer.decode_bytes(candidate_tokens[i]);
                if let Some(ref cb) = ctx.token_cb {
                    cb(&String::from_utf8_lossy(piece_bytes));
                }
                ctx.perf_text_tokens += 1;
                state.result_bytes.extend_from_slice(piece_bytes);
                delta_bytes.extend_from_slice(piece_bytes);
            }
        }
    }

    // ---- Update raw token history ----
    state.raw_tokens.truncate(n_prefix_tokens);
    state.raw_tokens.extend_from_slice(&chunk_tokens);

    // ---- Streaming degeneracy detection ----
    if !speech_ended {
        if state.raw_tokens == state.prev_tail_snapshot {
            state.stale_count += 1;
        } else {
            state.stale_count = 0;
            state.prev_tail_snapshot = state.raw_tokens.clone();
        }
        let (best_reps, _) = stream_tail_repeat_blocks(&state.raw_tokens, STREAM_DEGEN_MAX_PERIOD);
        let is_degen = state.stale_count >= STREAM_STALE_CHUNKS
            || best_reps >= STREAM_DEGEN_MIN_REPEATS;

        if is_degen {
            if kernels::verbose() >= 2 {
                eprintln!("[stream degen] reset at chunk {} (stale={}, reps={})",
                    state.chunk_idx, state.stale_count, best_reps);
            }
            let carry = state.stable_text_tokens.len().min(STREAM_RESET_CARRY_TOKENS);
            let carry_start = state.stable_text_tokens.len() - carry;
            state.raw_tokens.clear();
            if carry > 0 {
                state.raw_tokens.push(TOKEN_ASR_TEXT);
                state.raw_tokens.extend_from_slice(&state.stable_text_tokens[carry_start..]);
            }
            state.prev_prefill_embeds.clear();
            state.prev_prefill_len = 0;
            state.stale_count = 0;
            state.prev_tail_snapshot.clear();
            if state.enc_cache.len() >= STREAM_MAX_ENC_WINDOWS {
                state.enc_cache_base_windows += state.enc_cache.len();
                state.enc_cache.clear();
                state.enc_cached_seq_total = 0;
            }
        }

        // Periodic re-anchor: reset context every STREAM_RESET_INTERVAL_CHUNKS chunks
        if state.chunk_idx > 0 && state.chunk_idx % STREAM_RESET_INTERVAL_CHUNKS == 0 {
            if kernels::verbose() >= 2 {
                eprintln!("[stream reanchor] at chunk {}", state.chunk_idx);
            }
            let carry = state.stable_text_tokens.len().min(STREAM_RESET_CARRY_TOKENS);
            let carry_start = state.stable_text_tokens.len() - carry;
            state.raw_tokens.clear();
            if carry > 0 {
                state.raw_tokens.push(TOKEN_ASR_TEXT);
                state.raw_tokens.extend_from_slice(&state.stable_text_tokens[carry_start..]);
            }
            state.prev_prefill_embeds.clear();
            state.prev_prefill_len = 0;
            state.stale_count = 0;
            state.prev_tail_snapshot.clear();
            if state.enc_cache.len() >= STREAM_MAX_ENC_WINDOWS {
                state.enc_cache_base_windows += state.enc_cache.len();
                state.enc_cache.clear();
                state.enc_cached_seq_total = 0;
            }
        }
    }

    // ---- Parse text region and emit stable tokens (non-speech-ended case) ----
    if !speech_ended {
        let text_start = if n_force_prompt_tokens == 0 {
            state.raw_tokens.iter().position(|&t| t == TOKEN_ASR_TEXT)
                .map(|p| p + 1)
                .unwrap_or(0)
        } else {
            0
        };
        let n_text_tokens = state.raw_tokens.len().saturating_sub(text_start);

        let candidate_len = if is_final {
            n_text_tokens
        } else if state.chunk_idx >= unfixed_chunks {
            (n_text_tokens as i32 - rollback).max(0) as usize
        } else {
            0
        };

        let candidate_tokens = &state.raw_tokens[text_start..];
        let emit_from = state.stable_text_tokens.len();
        let emit_to = candidate_len.max(emit_from);

        for i in emit_from..emit_to {
            if i < candidate_tokens.len() {
                if i >= state.stable_text_tokens.len() {
                    state.stable_text_tokens.push(candidate_tokens[i]);
                }
                let piece_bytes = tokenizer.decode_bytes(candidate_tokens[i]);
                if let Some(ref cb) = ctx.token_cb {
                    cb(&String::from_utf8_lossy(piece_bytes));
                }
                ctx.perf_text_tokens += 1;
                state.result_bytes.extend_from_slice(piece_bytes);
                delta_bytes.extend_from_slice(piece_bytes);
            }
        }
    }

        ctx.perf_total_ms += elapsed_ms(chunk_t0);
        state.chunk_idx += 1;

        // Stop processing after speech ends — no point encoding more silence
        if speech_ended {
            break;
        }
    } // end while loop

    Some(String::from_utf8_lossy(&delta_bytes).into_owned())
}
