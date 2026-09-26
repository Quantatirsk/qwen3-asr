//! Greedy PCM generation with exact caller-supplied tokenizer IDs.

use crate::audio;
use crate::config::{TOKEN_ENDOFTEXT, TOKEN_IM_END};
use crate::context::QwenCtx;
use crate::decoder::{decoder_forward, decoder_prefill, tok_embed_bf16_to_f32};

pub struct Generation {
    pub token_ids: Vec<i32>,
    pub reached_limit: bool,
}

fn valid_input(
    samples: &[f32],
    before: &[i32],
    after: &[i32],
    budget: usize,
    vocab: usize,
) -> bool {
    !samples.is_empty()
        && samples.len() <= 61 * 16000
        && samples.iter().all(|sample| sample.is_finite())
        && !before.is_empty()
        && !after.is_empty()
        && before.len() + after.len() <= 8192
        && (1..=4096).contains(&budget)
        && before
            .iter()
            .chain(after)
            .all(|id| *id >= 0 && (*id as usize) < vocab)
}

/// Replace the caller's single audio placeholder with encoder embeddings.
/// Callers exclusively own the context for the duration of this operation.
pub fn generate_pcm(
    ctx: &mut QwenCtx,
    samples: &[f32],
    before: &[i32],
    after: &[i32],
    budget: usize,
) -> Option<Generation> {
    let shared = ctx.shared.clone();
    let cfg = &shared.config;
    if cfg.is_aligner() || !valid_input(samples, before, after, budget, cfg.vocab_size) {
        return None;
    }
    let (mel, frames) = audio::mel_spectrogram(samples)?;
    let (encoded, audio_tokens) =
        shared
            .encoder
            .forward(cfg, &mel, frames, Some(&mut ctx.enc_bufs))?;
    let total = before.len() + audio_tokens + after.len();
    if total + budget > 16384 {
        return None;
    }
    let dim = cfg.dec_hidden;
    let embedding = shared.decoder.tok_embeddings_bf16;
    let mut input = vec![0.0f32; total * dim];
    for (index, &id) in before.iter().enumerate() {
        unsafe {
            tok_embed_bf16_to_f32(
                &mut input[index * dim..(index + 1) * dim],
                embedding,
                id,
                dim,
            )
        };
    }
    let audio_end = before.len() + audio_tokens;
    input[before.len() * dim..audio_end * dim].copy_from_slice(&encoded);
    for (index, &id) in after.iter().enumerate() {
        let offset = audio_end + index;
        unsafe {
            tok_embed_bf16_to_f32(
                &mut input[offset * dim..(offset + 1) * dim],
                embedding,
                id,
                dim,
            )
        };
    }
    ctx.kv_cache.len = 0;
    decoder_prefill(
        &shared.decoder,
        cfg,
        &mut ctx.kv_cache,
        &mut ctx.rope_cache,
        &mut ctx.dec_bufs,
        &input,
        total - 1,
    );
    let mut token = decoder_forward(
        &shared.decoder,
        cfg,
        &mut ctx.kv_cache,
        &mut ctx.rope_cache,
        &mut ctx.dec_bufs,
        &input[(total - 1) * dim..],
    );
    let mut result = Generation {
        token_ids: Vec::new(),
        reached_limit: true,
    };
    let mut next = vec![0.0f32; dim];
    for step in 0..budget {
        if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
            result.reached_limit = false;
            break;
        }
        result.token_ids.push(token);
        if step + 1 < budget {
            unsafe { tok_embed_bf16_to_f32(&mut next, embedding, token, dim) };
            token = decoder_forward(
                &shared.decoder,
                cfg,
                &mut ctx.kv_cache,
                &mut ctx.rope_cache,
                &mut ctx.dec_bufs,
                &next,
            );
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::valid_input;

    #[test]
    fn rejects_invalid_generation_inputs() {
        assert!(valid_input(&[0.0], &[1], &[2], 16, 100));
        assert!(!valid_input(&[f32::NAN], &[1], &[2], 16, 100));
        assert!(!valid_input(&[0.0], &[-1], &[2], 16, 100));
        assert!(!valid_input(&[0.0], &[1], &[100], 16, 100));
        assert!(!valid_input(&[], &[1], &[2], 16, 100));
        assert!(!valid_input(&[0.0], &[], &[2], 16, 100));
        assert!(!valid_input(&[0.0], &[1], &[2], 4097, 100));
    }
}
