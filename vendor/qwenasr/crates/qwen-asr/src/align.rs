//! Forced alignment: word- and character-level timestamps.

use crate::audio;
use crate::config::*;
use crate::context::QwenCtx;
use crate::decoder::{self, tok_embed_bf16_to_f32};
use crate::kernels;
use crate::tokenizer::QwenTokenizer;

use std::time::Instant;

// Reuse the same prompt structure as transcribe
const PREFIX_HEAD: &[i32] = &[151644, 8948, 198];
const PREFIX_TAIL: &[i32] = &[151645, 198, 151644, 872, 198, 151669];
const SUFFIX_BASE: &[i32] = &[151670, 151645, 198, 151644, 77091, 198];

/// A single word (or character for CJK) with its aligned time span.
///
/// * `text`     – the word or character this entry covers.
/// * `start_ms` – start time in milliseconds from the beginning of the audio.
/// * `end_ms`   – end time in milliseconds.
#[derive(Debug, Clone)]
pub struct AlignResult {
    pub text: String,
    pub start_ms: f32,
    pub end_ms: f32,
}

fn get_time_ms() -> f64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_secs_f64() * 1000.0
}

fn elapsed_ms(t0: f64) -> f64 {
    get_time_ms() - t0
}

/// Split text into words based on language.
/// English/space-delimited: split on whitespace.
/// CJK (Chinese, Japanese, Korean, Cantonese): character-level split.
fn split_words(text: &str, language: &str) -> Vec<String> {
    let is_cjk = matches!(
        language,
        "Chinese" | "Japanese" | "Korean" | "Cantonese"
    );

    if is_cjk {
        // Character-level split (each Unicode character is a "word")
        text.chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| c.to_string())
            .collect()
    } else {
        // Space-delimited split
        text.split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }
}

/// Build the input token sequence for forced alignment.
/// Interleaves <timestamp><timestamp> between words and appends at end.
/// Returns (word_list, token_ids) where token_ids includes audio markers + text + timestamps.
fn encode_timestamp(
    text: &str,
    language: &str,
    tokenizer: &QwenTokenizer,
) -> Option<(Vec<String>, Vec<i32>)> {
    let words = split_words(text, language);
    if words.is_empty() {
        return None;
    }

    // Build the text with interleaved timestamp tokens:
    // word1 <ts><ts> word2 <ts><ts> ... wordN <ts><ts>
    // But we tokenize each word separately and insert timestamp token IDs between them.
    let mut token_ids: Vec<i32> = Vec::new();

    for (i, word) in words.iter().enumerate() {
        let word_tokens = tokenizer.encode(word)?;
        token_ids.extend_from_slice(&word_tokens);
        // Append <timestamp><timestamp> after each word (including the last)
        token_ids.push(TOKEN_TIMESTAMP);
        token_ids.push(TOKEN_TIMESTAMP);

        if i == 0 && word_tokens.is_empty() {
            return None; // First word must produce tokens
        }
    }

    Some((words, token_ids))
}

/// Find the longest increasing subsequence indices.
fn longest_increasing_subsequence(vals: &[f32]) -> Vec<usize> {
    let n = vals.len();
    if n == 0 {
        return Vec::new();
    }

    // dp[i] = length of LIS ending at i
    let mut dp = vec![1usize; n];
    let mut prev = vec![usize::MAX; n];

    for i in 1..n {
        for j in 0..i {
            if vals[j] <= vals[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                prev[i] = j;
            }
        }
    }

    // Find the end of the longest sequence
    let mut best_len = 0;
    let mut best_end = 0;
    for (i, &dp_val) in dp.iter().enumerate().take(n) {
        if dp_val > best_len {
            best_len = dp_val;
            best_end = i;
        }
    }

    // Trace back
    let mut lis_indices = Vec::with_capacity(best_len);
    let mut idx = best_end;
    loop {
        lis_indices.push(idx);
        if prev[idx] == usize::MAX {
            break;
        }
        idx = prev[idx];
    }
    lis_indices.reverse();
    lis_indices
}

/// Fix anomalous timestamps using LIS + interpolation.
fn fix_timestamps(timestamps: &mut [f32]) {
    if timestamps.len() <= 1 {
        return;
    }

    let lis_indices = longest_increasing_subsequence(timestamps);
    if lis_indices.len() == timestamps.len() {
        return; // Already monotonically increasing
    }

    // Mark which indices are in the LIS (normal)
    let n = timestamps.len();
    let mut is_normal = vec![false; n];
    for &idx in &lis_indices {
        is_normal[idx] = true;
    }

    // Fix anomalous regions
    let mut i = 0;
    while i < n {
        if is_normal[i] {
            i += 1;
            continue;
        }

        // Find the extent of this anomalous block
        let block_start = i;
        while i < n && !is_normal[i] {
            i += 1;
        }
        let block_end = i; // exclusive
        let block_len = block_end - block_start;

        // Get boundary values
        let left_val = if block_start > 0 { timestamps[block_start - 1] } else { 0.0 };
        let right_val = if block_end < n { timestamps[block_end] } else { left_val + block_len as f32 * 80.0 };

        if block_len <= 2 {
            // Small block: fill with nearest normal value
            let fill = if block_start > 0 { left_val } else { right_val };
            for ts in timestamps.iter_mut().take(block_end).skip(block_start) {
                *ts = fill;
            }
        } else {
            // Larger block: linearly interpolate
            for j in 0..block_len {
                let t = (j + 1) as f32 / (block_len + 1) as f32;
                timestamps[block_start + j] = left_val + t * (right_val - left_val);
            }
        }
    }
}

/// Perform forced alignment on audio samples with a known transcript.
///
/// Requires a ForcedAligner model (one where `config.classify_num > 0`).
/// Returns `None` if the model is not an aligner, the text is empty, or
/// encoding fails.
///
/// `language` controls how `text` is split into units: for CJK languages
/// (`"Chinese"`, `"Japanese"`, `"Korean"`, `"Cantonese"`) each character
/// becomes a separate entry; for all other languages the text is split on
/// whitespace.
///
/// The returned vector has one [`AlignResult`] per word/character with
/// monotonically non-decreasing timestamps (anomalies are corrected via
/// LIS + interpolation).
pub fn forced_align(
    ctx: &mut QwenCtx,
    samples: &[f32],
    text: &str,
    language: &str,
) -> Option<Vec<AlignResult>> {
    let shared = ctx.shared.clone();
    let cfg = &shared.config.clone();
    let dim = cfg.dec_hidden;
    let seg_time = cfg.timestamp_segment_time;

    if !cfg.is_aligner() {
        eprintln!("align: model is not a forced aligner (classify_num=0)");
        return None;
    }

    let vocab_path = format!("{}/vocab.json", shared.model_dir);
    let tokenizer = QwenTokenizer::load(&vocab_path)?;

    ctx.reset_perf();
    ctx.perf_audio_ms = 1000.0 * samples.len() as f64 / SAMPLE_RATE as f64;

    let seg_t0 = get_time_ms();

    // Step 1: Tokenize text with timestamp interleaving
    let (words, text_tokens) = encode_timestamp(text, language, &tokenizer)?;

    if kernels::verbose() >= 2 {
        eprintln!("  Align: {} words, {} text tokens", words.len(), text_tokens.len());
    }

    // Step 2: Mel spectrogram + encoder
    let t0 = get_time_ms();
    let (mel, mel_frames) = audio::mel_spectrogram(samples)?;
    let mel_ms = elapsed_ms(t0);

    let t0 = get_time_ms();
    let (enc_output, enc_seq_len) = shared.encoder.forward(cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))?;
    let enc_ms = elapsed_ms(t0);

    if kernels::verbose() >= 2 {
        eprintln!("  Mel: {} frames ({:.0} ms), Encoder: {} tokens ({:.0} ms)",
                  mel_frames, mel_ms, enc_seq_len, enc_ms);
    }

    // Step 3: Build input embeddings
    // Structure: PREFIX_HEAD + PREFIX_TAIL + [encoder output] + SUFFIX_BASE + text_tokens
    // (No prompt tokens or language forcing for alignment)
    let prefix_len = PREFIX_HEAD.len() + PREFIX_TAIL.len();
    let suffix_len = SUFFIX_BASE.len();
    let total_seq = prefix_len + enc_seq_len + suffix_len + text_tokens.len();

    let mut input_embeds = vec![0.0f32; total_seq * dim];
    let tok_emb = shared.decoder.tok_embeddings_bf16;

    let mut off = 0;
    for &tok in PREFIX_HEAD {
        unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim); }
        off += 1;
    }
    for &tok in PREFIX_TAIL {
        unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim); }
        off += 1;
    }

    // Encoder output
    for i in 0..enc_seq_len {
        input_embeds[(prefix_len + i) * dim..(prefix_len + i + 1) * dim]
            .copy_from_slice(&enc_output[i * dim..(i + 1) * dim]);
    }

    // Suffix
    let suffix_off = prefix_len + enc_seq_len;
    for (i, &tok) in SUFFIX_BASE.iter().enumerate() {
        unsafe { tok_embed_bf16_to_f32(
            &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim],
            tok_emb, tok, dim,
        ); }
    }

    // Text tokens (with interleaved <timestamp> tokens)
    let text_off = suffix_off + suffix_len;
    for (i, &tok) in text_tokens.iter().enumerate() {
        unsafe { tok_embed_bf16_to_f32(
            &mut input_embeds[(text_off + i) * dim..(text_off + i + 1) * dim],
            tok_emb, tok, dim,
        ); }
    }

    // Step 4: Single prefill pass → logits for all positions
    let t0 = get_time_ms();
    ctx.kv_cache.len = 0;

    let logits = decoder::decoder_prefill_logits(
        &shared.decoder, cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
        &mut ctx.dec_bufs, &input_embeds, total_seq,
    );
    let prefill_ms = elapsed_ms(t0);

    if kernels::verbose() >= 2 {
        eprintln!("  Prefill: {} tokens ({:.0} ms)", total_seq, prefill_ms);
    }

    // Step 5: Extract timestamps from <timestamp> positions
    let out_dim = cfg.lm_head_dim();
    let mut raw_timestamps: Vec<f32> = Vec::new();

    for (i, &tok) in text_tokens.iter().enumerate() {
        if tok == TOKEN_TIMESTAMP {
            let pos = text_off + i;
            let logit_row = &logits[pos * out_dim..(pos + 1) * out_dim];
            // Argmax
            let mut best_idx = 0;
            let mut best_val = logit_row[0];
            for (j, &val) in logit_row.iter().enumerate().take(out_dim).skip(1) {
                if val > best_val {
                    best_val = val;
                    best_idx = j;
                }
            }
            raw_timestamps.push(best_idx as f32 * seg_time);
        }
    }

    // Step 6: Fix timestamps (LIS + interpolation)
    fix_timestamps(&mut raw_timestamps);

    // Step 7: Pair consecutive timestamps into (start, end) per word
    // Each word has 2 timestamps: start, end
    let mut results = Vec::with_capacity(words.len());
    for (i, word) in words.iter().enumerate() {
        let start_ms = raw_timestamps.get(i * 2).copied().unwrap_or(0.0);
        let end_ms = raw_timestamps.get(i * 2 + 1).copied().unwrap_or(start_ms);
        results.push(AlignResult {
            text: word.clone(),
            start_ms,
            end_ms,
        });
    }

    ctx.perf_total_ms += elapsed_ms(seg_t0);
    ctx.perf_encode_ms += mel_ms + enc_ms;
    ctx.perf_decode_ms += prefill_ms;

    Some(results)
}
