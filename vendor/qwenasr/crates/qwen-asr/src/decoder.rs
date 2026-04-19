//! Qwen3 LLM decoder with GQA, KV cache, and generation.

use crate::config::*;
use crate::kernels;
use crate::safetensors::MultiSafetensors;

pub struct DecLayer {
    pub wq_weight_bf16: *const u16,
    pub wk_weight_bf16: *const u16,
    pub wv_weight_bf16: *const u16,
    pub wo_weight_bf16: *const u16,
    pub q_norm_weight: Vec<f32>,
    pub k_norm_weight: Vec<f32>,
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub gate_weight_bf16: *const u16,
    pub up_weight_bf16: *const u16,
    pub down_weight_bf16: *const u16,
    pub gate_up_fused_bf16: Vec<u16>, // owned, interleaved
    /// INT8 quantized attention weights + per-row scales
    pub wq_int8: Vec<i8>,
    pub wq_int8_scales: Vec<f32>,
    pub wk_int8: Vec<i8>,
    pub wk_int8_scales: Vec<f32>,
    pub wv_int8: Vec<i8>,
    pub wv_int8_scales: Vec<f32>,
    pub wo_int8: Vec<i8>,
    pub wo_int8_scales: Vec<f32>,
    /// INT8 quantized FFN weights + per-row scales
    pub gate_up_int8: Vec<i8>,
    pub gate_up_int8_scales: Vec<f32>,
    pub down_int8: Vec<i8>,
    pub down_int8_scales: Vec<f32>,
}

unsafe impl Send for DecLayer {}
unsafe impl Sync for DecLayer {}

pub struct Decoder {
    pub tok_embeddings_bf16: *const u16,
    pub layers: Vec<DecLayer>,
    pub norm: Vec<f32>,
    /// Separate lm_head for forced aligner (None = tied weights with tok_embeddings)
    pub lm_head_bf16: Option<*const u16>,
    /// INT8 quantized lm_head weights for fast argmax
    pub lm_head_int8: Option<Vec<i8>>,
    pub lm_head_int8_scales: Option<Vec<f32>>,
}

unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

fn load_f32(ms: &MultiSafetensors, name: &str) -> Option<Vec<f32>> {
    let result = ms.get_f32(name);
    if result.is_none() {
        eprintln!("decoder: weight not found: {}", name);
    }
    result
}

fn load_bf16_direct(ms: &MultiSafetensors, name: &str) -> Option<*const u16> {
    let result = ms.get_bf16_direct(name);
    if result.is_none() {
        eprintln!("decoder: weight not found: {}", name);
    }
    result
}

impl Decoder {
    pub fn load(ms: &MultiSafetensors, cfg: &QwenConfig) -> Option<Self> {
        let tok_embeddings_bf16 = load_bf16_direct(ms, "thinker.model.embed_tokens.weight")?;

        let mut layers = Vec::new();
        for i in 0..cfg.dec_layers {
            let lp = format!("thinker.model.layers.{}", i);

            let wq = load_bf16_direct(ms, &format!("{}.self_attn.q_proj.weight", lp))?;
            let wk = load_bf16_direct(ms, &format!("{}.self_attn.k_proj.weight", lp))?;
            let wv = load_bf16_direct(ms, &format!("{}.self_attn.v_proj.weight", lp))?;
            let wo = load_bf16_direct(ms, &format!("{}.self_attn.o_proj.weight", lp))?;

            let q_norm = load_f32(ms, &format!("{}.self_attn.q_norm.weight", lp))?;
            let k_norm = load_f32(ms, &format!("{}.self_attn.k_norm.weight", lp))?;
            let input_norm = load_f32(ms, &format!("{}.input_layernorm.weight", lp))?;
            let post_attn_norm = load_f32(ms, &format!("{}.post_attention_layernorm.weight", lp))?;

            let gate_bf16 = load_bf16_direct(ms, &format!("{}.mlp.gate_proj.weight", lp))?;
            let up_bf16 = load_bf16_direct(ms, &format!("{}.mlp.up_proj.weight", lp))?;
            let down_bf16 = load_bf16_direct(ms, &format!("{}.mlp.down_proj.weight", lp))?;

            // Fuse gate+up: interleave rows
            let inter = cfg.dec_intermediate;
            let hidden = cfg.dec_hidden;
            let mut gate_up_fused = vec![0u16; 2 * inter * hidden];
            unsafe {
                let gate_slice = std::slice::from_raw_parts(gate_bf16, inter * hidden);
                let up_slice = std::slice::from_raw_parts(up_bf16, inter * hidden);
                for r in 0..inter {
                    gate_up_fused[2 * r * hidden..(2 * r + 1) * hidden]
                        .copy_from_slice(&gate_slice[r * hidden..(r + 1) * hidden]);
                    gate_up_fused[(2 * r + 1) * hidden..(2 * r + 2) * hidden]
                        .copy_from_slice(&up_slice[r * hidden..(r + 1) * hidden]);
                }
            }

            // INT8 quantize all decoder layer weights
            let q_dim = cfg.dec_heads * cfg.dec_head_dim;
            let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
            let (wq_int8, wq_int8_scales) = kernels::quantize_bf16_weights_to_int8(wq, q_dim, hidden);
            let (wk_int8, wk_int8_scales) = kernels::quantize_bf16_weights_to_int8(wk, kv_dim, hidden);
            let (wv_int8, wv_int8_scales) = kernels::quantize_bf16_weights_to_int8(wv, kv_dim, hidden);
            let (wo_int8, wo_int8_scales) = kernels::quantize_bf16_weights_to_int8(wo, hidden, q_dim);
            let (gate_up_int8, gate_up_int8_scales) = kernels::quantize_bf16_weights_to_int8(
                gate_up_fused.as_ptr(), 2 * inter, hidden,
            );
            let (down_int8, down_int8_scales) = kernels::quantize_bf16_weights_to_int8(
                down_bf16, hidden, inter,
            );

            layers.push(DecLayer {
                wq_weight_bf16: wq,
                wk_weight_bf16: wk,
                wv_weight_bf16: wv,
                wo_weight_bf16: wo,
                q_norm_weight: q_norm,
                k_norm_weight: k_norm,
                input_norm,
                post_attn_norm,
                gate_weight_bf16: gate_bf16,
                up_weight_bf16: up_bf16,
                down_weight_bf16: down_bf16,
                gate_up_fused_bf16: gate_up_fused,
                wq_int8, wq_int8_scales,
                wk_int8, wk_int8_scales,
                wv_int8, wv_int8_scales,
                wo_int8, wo_int8_scales,
                gate_up_int8, gate_up_int8_scales,
                down_int8, down_int8_scales,
            });
        }

        let norm = load_f32(ms, "thinker.model.norm.weight")?;

        // Load separate lm_head if present (forced aligner has untied lm_head)
        let lm_head_bf16 = if cfg.classify_num > 0 {
            let ptr = load_bf16_direct(ms, "thinker.lm_head.weight")?;
            Some(ptr)
        } else {
            // For normal ASR, lm_head is tied with tok_embeddings (no separate weight)
            ms.get_bf16_direct("thinker.lm_head.weight")
        };

        // Quantize lm_head to INT8 for fast argmax
        let lm_weight = lm_head_bf16.unwrap_or(tok_embeddings_bf16);
        let lm_out_dim = cfg.lm_head_dim();
        let lm_in_dim = cfg.dec_hidden;
        let (lm_int8, lm_scales) = kernels::quantize_bf16_weights_to_int8(lm_weight, lm_out_dim, lm_in_dim);

        Some(Decoder {
            tok_embeddings_bf16,
            layers,
            norm,
            lm_head_bf16,
            lm_head_int8: Some(lm_int8),
            lm_head_int8_scales: Some(lm_scales),
        })
    }
}

// ========================================================================
// KV Cache
// ========================================================================

pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
    pub max_seq: usize,
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl KvCache {
    /// Layout: `[layer][head][pos][head_dim]` — head-contiguous for cache-friendly attention.
    pub fn new(n_layers: usize, max_seq: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let total = n_layers * n_kv_heads * max_seq * head_dim;
        KvCache {
            k: vec![0.0f32; total],
            v: vec![0.0f32; total],
            len: 0,
            max_seq,
            n_layers,
            n_kv_heads,
            head_dim,
        }
    }

    pub fn grow(&mut self, required: usize) {
        if required <= self.max_seq {
            return;
        }

        let mut new_max = self.max_seq;
        while new_max < required {
            new_max *= 2;
        }

        let old_head_stride = self.max_seq * self.head_dim;
        let new_head_stride = new_max * self.head_dim;
        let total = self.n_layers * self.n_kv_heads * new_head_stride;

        let mut new_k = vec![0.0f32; total];
        let mut new_v = vec![0.0f32; total];

        let copy_len = self.len * self.head_dim;
        for l in 0..self.n_layers {
            for h in 0..self.n_kv_heads {
                let old_off = (l * self.n_kv_heads + h) * old_head_stride;
                let new_off = (l * self.n_kv_heads + h) * new_head_stride;
                new_k[new_off..new_off + copy_len]
                    .copy_from_slice(&self.k[old_off..old_off + copy_len]);
                new_v[new_off..new_off + copy_len]
                    .copy_from_slice(&self.v[old_off..old_off + copy_len]);
            }
        }

        self.k = new_k;
        self.v = new_v;
        self.max_seq = new_max;
    }

    /// Write K for all heads at a given position (from interleaved kv_dim buffer).
    pub fn k_write_pos(&mut self, layer: usize, pos: usize, src: &[f32]) {
        let head_stride = self.max_seq * self.head_dim;
        for h in 0..self.n_kv_heads {
            let dst_off = (layer * self.n_kv_heads + h) * head_stride + pos * self.head_dim;
            let src_off = h * self.head_dim;
            self.k[dst_off..dst_off + self.head_dim]
                .copy_from_slice(&src[src_off..src_off + self.head_dim]);
        }
    }

    /// Write V for all heads at a given position (from interleaved kv_dim buffer).
    pub fn v_write_pos(&mut self, layer: usize, pos: usize, src: &[f32]) {
        let head_stride = self.max_seq * self.head_dim;
        for h in 0..self.n_kv_heads {
            let dst_off = (layer * self.n_kv_heads + h) * head_stride + pos * self.head_dim;
            let src_off = h * self.head_dim;
            self.v[dst_off..dst_off + self.head_dim]
                .copy_from_slice(&src[src_off..src_off + self.head_dim]);
        }
    }

    /// Base pointer for K data of a specific layer (head-contiguous layout).
    /// Layout within layer: `[head][pos][head_dim]`, stride between heads = max_seq * head_dim.
    pub fn k_layer_base(&self, layer: usize) -> *const f32 {
        let off = layer * self.n_kv_heads * self.max_seq * self.head_dim;
        unsafe { self.k.as_ptr().add(off) }
    }

    pub fn v_layer_base(&self, layer: usize) -> *const f32 {
        let off = layer * self.n_kv_heads * self.max_seq * self.head_dim;
        unsafe { self.v.as_ptr().add(off) }
    }

    /// Stride between heads (in floats): max_seq * head_dim.
    pub fn head_stride(&self) -> usize {
        self.max_seq * self.head_dim
    }
}

// ========================================================================
// RoPE Cache
// ========================================================================

pub struct RopeCache {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub inv_freq: Vec<f32>,
    pub cap: usize,
    pub head_dim: usize,
}

impl Default for RopeCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RopeCache {
    pub fn new() -> Self {
        RopeCache {
            cos: Vec::new(),
            sin: Vec::new(),
            inv_freq: Vec::new(),
            cap: 0,
            head_dim: 0,
        }
    }

    pub fn ensure(&mut self, required_pos: usize, head_dim: usize, theta: f32) {
        if self.head_dim != head_dim || self.inv_freq.is_empty() {
            let half = head_dim / 2;
            self.inv_freq = (0..half)
                .map(|d| 1.0 / theta.powf((2 * d) as f32 / head_dim as f32))
                .collect();
            self.head_dim = head_dim;
        }

        if required_pos <= self.cap {
            return;
        }

        let mut new_cap = if self.cap > 0 { self.cap } else { 1024 };
        while new_cap < required_pos {
            new_cap *= 2;
        }

        self.cos.resize(new_cap * head_dim, 0.0);
        self.sin.resize(new_cap * head_dim, 0.0);

        let half = head_dim / 2;
        for pos in self.cap..new_cap {
            let p = pos as f32;
            for d in 0..half {
                let angle = p * self.inv_freq[d];
                let c = angle.cos();
                let s = angle.sin();
                self.cos[pos * head_dim + d] = c;
                self.cos[pos * head_dim + half + d] = c;
                self.sin[pos * head_dim + d] = s;
                self.sin[pos * head_dim + half + d] = s;
            }
        }

        self.cap = new_cap;
    }

    pub fn cos_at(&self, pos: usize) -> &[f32] {
        &self.cos[pos * self.head_dim..(pos + 1) * self.head_dim]
    }

    pub fn sin_at(&self, pos: usize) -> &[f32] {
        &self.sin[pos * self.head_dim..(pos + 1) * self.head_dim]
    }

    pub fn cos_range(&self, start: usize, len: usize) -> &[f32] {
        &self.cos[start * self.head_dim..(start + len) * self.head_dim]
    }

    pub fn sin_range(&self, start: usize, len: usize) -> &[f32] {
        &self.sin[start * self.head_dim..(start + len) * self.head_dim]
    }
}

// ========================================================================
// Decoder Forward
// ========================================================================

pub struct DecoderBuffers {
    // Single-token decode buffers
    pub x: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub proj_out: Vec<f32>,
    pub gate_buf: Vec<f32>,
    pub ffn_out: Vec<f32>,

    // Prefill buffers
    pub pref_x: Vec<f32>,
    pub pref_x_norm: Vec<f32>,
    pub pref_q: Vec<f32>,
    pub pref_k: Vec<f32>,
    pub pref_v: Vec<f32>,
    pub pref_attn_out: Vec<f32>,
    pub pref_proj_out: Vec<f32>,
    pub pref_ffn_out: Vec<f32>,
    pub pref_gate_up: Vec<f32>,
    pub pref_gate: Vec<f32>,
    pub pref_seq_cap: usize,

    // Reusable scratch for BF16→F32 conversion in prefill path
    pub bf16_scratch: Vec<f32>,
}

impl DecoderBuffers {
    pub fn new(cfg: &QwenConfig) -> Self {
        let dim = cfg.dec_hidden;
        let q_dim = cfg.dec_heads * cfg.dec_head_dim;
        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let intermediate = cfg.dec_intermediate;

        // Largest weight matrix is gate_up_fused: 2 * intermediate * hidden
        let max_weight = (2 * intermediate * dim).max(q_dim * dim).max(kv_dim * dim);

        DecoderBuffers {
            x: vec![0.0f32; dim],
            x_norm: vec![0.0f32; dim],
            q: vec![0.0f32; q_dim],
            k: vec![0.0f32; kv_dim],
            v: vec![0.0f32; kv_dim],
            attn_out: vec![0.0f32; q_dim],
            proj_out: vec![0.0f32; dim],
            gate_buf: vec![0.0f32; 2 * intermediate],
            ffn_out: vec![0.0f32; intermediate],
            pref_x: Vec::new(),
            pref_x_norm: Vec::new(),
            pref_q: Vec::new(),
            pref_k: Vec::new(),
            pref_v: Vec::new(),
            pref_attn_out: Vec::new(),
            pref_proj_out: Vec::new(),
            pref_ffn_out: Vec::new(),
            pref_gate_up: Vec::new(),
            pref_gate: Vec::new(),
            pref_seq_cap: 0,
            bf16_scratch: vec![0.0f32; max_weight],
        }
    }

    pub fn ensure_prefill(&mut self, seq_len: usize, cfg: &QwenConfig) {
        if seq_len <= self.pref_seq_cap {
            return;
        }

        let dim = cfg.dec_hidden;
        let q_dim = cfg.dec_heads * cfg.dec_head_dim;
        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let intermediate = cfg.dec_intermediate;

        let mut new_cap = if self.pref_seq_cap > 0 { self.pref_seq_cap } else { 64 };
        while new_cap < seq_len {
            new_cap *= 2;
        }

        self.pref_x.resize(new_cap * dim, 0.0);
        self.pref_x_norm.resize(new_cap * dim, 0.0);
        self.pref_q.resize(new_cap * q_dim, 0.0);
        self.pref_k.resize(new_cap * kv_dim, 0.0);
        self.pref_v.resize(new_cap * kv_dim, 0.0);
        self.pref_attn_out.resize(new_cap * q_dim, 0.0);
        self.pref_proj_out.resize(new_cap * dim, 0.0);
        self.pref_ffn_out.resize(new_cap * dim, 0.0);
        self.pref_gate_up.resize(new_cap * 2 * intermediate, 0.0);
        self.pref_gate.resize(new_cap * intermediate, 0.0);
        self.pref_seq_cap = new_cap;
    }
}

/// Decoder prefill: process multiple tokens.
pub fn decoder_prefill(
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embeds: &[f32],
    seq_len: usize,
) {
    let dim = cfg.dec_hidden;
    let n_heads = cfg.dec_heads;
    let n_kv_heads = cfg.dec_kv_heads;
    let head_dim = cfg.dec_head_dim;
    let intermediate = cfg.dec_intermediate;
    let eps = cfg.dec_rms_norm_eps;
    let theta = cfg.dec_rope_theta;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    // Ensure KV cache
    let needed = kv_cache.len + seq_len;
    if needed > kv_cache.max_seq {
        kv_cache.grow(needed + 1024);
    }

    bufs.ensure_prefill(seq_len, cfg);

    let x = &mut bufs.pref_x[..seq_len * dim];
    x.copy_from_slice(&input_embeds[..seq_len * dim]);

    let start_pos = kv_cache.len;
    rope.ensure(start_pos + seq_len, head_dim, theta);
    let rope_cos = rope.cos_range(start_pos, seq_len);
    let rope_sin = rope.sin_range(start_pos, seq_len);

    let scale = 1.0 / (head_dim as f32).sqrt();

    for (layer_idx, layer) in decoder.layers.iter().enumerate() {
        let x_norm = &mut bufs.pref_x_norm[..seq_len * dim];
        kernels::rms_norm(x_norm, &bufs.pref_x[..seq_len * dim], &layer.input_norm, seq_len, dim, eps);

        let q = &mut bufs.pref_q[..seq_len * q_dim];
        let k = &mut bufs.pref_k[..seq_len * kv_dim];
        let v = &mut bufs.pref_v[..seq_len * kv_dim];

        unsafe { kernels::linear_nobias_bf16_scratch(q, x_norm, layer.wq_weight_bf16, seq_len, dim, q_dim, &mut bufs.bf16_scratch); }
        unsafe { kernels::linear_nobias_bf16_scratch(k, x_norm, layer.wk_weight_bf16, seq_len, dim, kv_dim, &mut bufs.bf16_scratch); }
        unsafe { kernels::linear_nobias_bf16_scratch(v, x_norm, layer.wv_weight_bf16, seq_len, dim, kv_dim, &mut bufs.bf16_scratch); }

        kernels::rms_norm_per_head(q, &layer.q_norm_weight, seq_len, n_heads, head_dim, eps);
        kernels::rms_norm_per_head(k, &layer.k_norm_weight, seq_len, n_kv_heads, head_dim, eps);

        kernels::apply_rope_neox(q, rope_cos, rope_sin, seq_len, n_heads, head_dim);
        kernels::apply_rope_neox(k, rope_cos, rope_sin, seq_len, n_kv_heads, head_dim);

        // Store K, V in cache (scatter to head-contiguous layout)
        for s in 0..seq_len {
            kv_cache.k_write_pos(layer_idx, start_pos + s, &bufs.pref_k[s * kv_dim..(s + 1) * kv_dim]);
            kv_cache.v_write_pos(layer_idx, start_pos + s, &bufs.pref_v[s * kv_dim..(s + 1) * kv_dim]);
        }

        let total_seq = start_pos + seq_len;
        let k_base = kv_cache.k_layer_base(layer_idx);
        let v_base = kv_cache.v_layer_base(layer_idx);
        let head_stride = kv_cache.head_stride();

        let attn_out = &mut bufs.pref_attn_out[..seq_len * q_dim];
        kernels::causal_attention(attn_out, q, k_base, v_base,
                                 head_stride,
                                 seq_len, total_seq, n_heads, n_kv_heads,
                                 head_dim, scale, start_pos);

        let proj_out = &mut bufs.pref_proj_out[..seq_len * dim];
        unsafe { kernels::linear_nobias_bf16_scratch(proj_out, attn_out, layer.wo_weight_bf16, seq_len, q_dim, dim, &mut bufs.bf16_scratch); }
        kernels::add_inplace(&mut bufs.pref_x[..seq_len * dim], proj_out, seq_len * dim);

        // Post-attention RMSNorm + SwiGLU MLP
        let x_norm2 = &mut bufs.pref_x_norm[..seq_len * dim];
        kernels::rms_norm(x_norm2, &bufs.pref_x[..seq_len * dim], &layer.post_attn_norm, seq_len, dim, eps);

        let gate_up = &mut bufs.pref_gate_up[..seq_len * 2 * intermediate];
        unsafe { kernels::linear_nobias_bf16_scratch(gate_up, x_norm2, layer.gate_up_fused_bf16.as_ptr(), seq_len, dim, 2 * intermediate, &mut bufs.bf16_scratch); }

        let gate = &mut bufs.pref_gate[..seq_len * intermediate];
        kernels::swiglu_multiply(gate, gate_up, seq_len, intermediate);

        let ffn_out = &mut bufs.pref_ffn_out[..seq_len * dim];
        unsafe { kernels::linear_nobias_bf16_scratch(ffn_out, gate, layer.down_weight_bf16, seq_len, intermediate, dim, &mut bufs.bf16_scratch); }
        kernels::add_inplace(&mut bufs.pref_x[..seq_len * dim], ffn_out, seq_len * dim);
    }

    kv_cache.len = start_pos + seq_len;
}

/// Decoder single-token forward: returns greedy token ID.
pub fn decoder_forward(
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embed: &[f32],
) -> i32 {
    let dim = cfg.dec_hidden;
    let n_heads = cfg.dec_heads;
    let n_kv_heads = cfg.dec_kv_heads;
    let head_dim = cfg.dec_head_dim;
    let intermediate = cfg.dec_intermediate;
    let eps = cfg.dec_rms_norm_eps;
    let theta = cfg.dec_rope_theta;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    bufs.x[..dim].copy_from_slice(&input_embed[..dim]);

    let pos = kv_cache.len;

    if pos >= kv_cache.max_seq {
        kv_cache.grow(pos + 1024);
    }

    rope.ensure(pos + 1, head_dim, theta);
    let rope_cos = rope.cos_at(pos);
    let rope_sin = rope.sin_at(pos);

    let scale = 1.0 / (head_dim as f32).sqrt();

    for (layer_idx, layer) in decoder.layers.iter().enumerate() {
        kernels::rms_norm(&mut bufs.x_norm[..dim], &bufs.x[..dim], &layer.input_norm, 1, dim, eps);

        // INT8 fused QKV projection
        kernels::linear_nobias_int8_qkv(
            &mut bufs.q[..q_dim], &mut bufs.k[..kv_dim], &mut bufs.v[..kv_dim],
            &bufs.x_norm[..dim],
            &layer.wq_int8, &layer.wq_int8_scales,
            &layer.wk_int8, &layer.wk_int8_scales,
            &layer.wv_int8, &layer.wv_int8_scales,
            dim, q_dim, kv_dim,
        );

        kernels::rms_norm_per_head(&mut bufs.q[..q_dim], &layer.q_norm_weight, 1, n_heads, head_dim, eps);
        kernels::rms_norm_per_head(&mut bufs.k[..kv_dim], &layer.k_norm_weight, 1, n_kv_heads, head_dim, eps);

        kernels::apply_rope_neox(&mut bufs.q[..q_dim], rope_cos, rope_sin, 1, n_heads, head_dim);
        kernels::apply_rope_neox(&mut bufs.k[..kv_dim], rope_cos, rope_sin, 1, n_kv_heads, head_dim);

        kv_cache.k_write_pos(layer_idx, pos, &bufs.k[..kv_dim]);
        kv_cache.v_write_pos(layer_idx, pos, &bufs.v[..kv_dim]);

        let total_seq = pos + 1;
        let k_base = kv_cache.k_layer_base(layer_idx);
        let v_base = kv_cache.v_layer_base(layer_idx);
        let head_stride = kv_cache.head_stride();

        kernels::causal_attention(&mut bufs.attn_out[..q_dim], &bufs.q[..q_dim],
                                 k_base, v_base,
                                 head_stride,
                                 1, total_seq, n_heads, n_kv_heads,
                                 head_dim, scale, pos);

        // INT8 O-projection with fused residual add: x += attn_out @ wo
        kernels::linear_nobias_int8_addto(&mut bufs.x[..dim], &bufs.attn_out[..q_dim],
                                          &layer.wo_int8, &layer.wo_int8_scales, q_dim, dim);

        kernels::rms_norm(&mut bufs.x_norm[..dim], &bufs.x[..dim], &layer.post_attn_norm, 1, dim, eps);

        // INT8 gate_up + SwiGLU
        kernels::linear_nobias_int8_swiglu(
            &mut bufs.ffn_out[..intermediate], &bufs.x_norm[..dim],
            &layer.gate_up_int8, &layer.gate_up_int8_scales, dim, intermediate,
        );
        // INT8 down-projection with fused residual add: x += ffn_out @ down
        kernels::linear_nobias_int8_addto(&mut bufs.x[..dim], &bufs.ffn_out[..intermediate],
                                          &layer.down_int8, &layer.down_int8_scales, intermediate, dim);
    }

    kv_cache.len = pos + 1;

    // Final norm + streaming argmax (use x_norm as temp to avoid heap allocation)
    kernels::rms_norm(&mut bufs.x_norm[..dim], &bufs.x[..dim], &decoder.norm, 1, dim, eps);
    bufs.x[..dim].copy_from_slice(&bufs.x_norm[..dim]);
    let lm_out_dim = cfg.lm_head_dim();

    // Use INT8 quantized argmax if available (2x less bandwidth)
    if let (Some(ref int8_data), Some(ref scales)) = (&decoder.lm_head_int8, &decoder.lm_head_int8_scales) {
        return kernels::argmax_matvec_int8(&bufs.x[..dim], int8_data, scales, dim, lm_out_dim) as i32;
    }

    let lm_weight = decoder.lm_head_bf16.unwrap_or(decoder.tok_embeddings_bf16);
    kernels::argmax_matvec_bf16(&bufs.x[..dim], lm_weight, dim, lm_out_dim) as i32
}

/// Decoder prefill that returns per-position logits (for forced aligner).
/// Returns `[seq_len × out_dim]` logits where out_dim = classify_num.
pub fn decoder_prefill_logits(
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embeds: &[f32],
    seq_len: usize,
) -> Vec<f32> {
    let dim = cfg.dec_hidden;
    let eps = cfg.dec_rms_norm_eps;
    let out_dim = cfg.lm_head_dim();

    // Run the standard prefill to get hidden states
    decoder_prefill(decoder, cfg, kv_cache, rope, bufs, input_embeds, seq_len);

    // After prefill, pref_x contains the final hidden states for all positions.
    // Apply final RMS norm and lm_head projection.
    let x = &bufs.pref_x[..seq_len * dim];
    let mut x_norm = vec![0.0f32; seq_len * dim];
    kernels::rms_norm(&mut x_norm, x, &decoder.norm, seq_len, dim, eps);

    let lm_weight = decoder.lm_head_bf16.unwrap_or(decoder.tok_embeddings_bf16);

    // Project each position through lm_head: [seq_len × dim] × [out_dim × dim]^T → [seq_len × out_dim]
    let mut logits = vec![0.0f32; seq_len * out_dim];
    unsafe { kernels::linear_nobias_bf16_scratch(
        &mut logits, &x_norm, lm_weight,
        seq_len, dim, out_dim, &mut bufs.bf16_scratch,
    ); }

    logits
}

/// Convert a token embedding from bf16 to f32.
///
/// # Safety
/// tok_emb_bf16 must point to valid memory for at least (token_id + 1) * dim bf16 values.
pub unsafe fn tok_embed_bf16_to_f32(dst: &mut [f32], tok_emb_bf16: *const u16, token_id: i32, dim: usize) {
    let src = unsafe { std::slice::from_raw_parts(tok_emb_bf16.add(token_id as usize * dim), dim) };
    kernels::bf16_to_f32_buf(dst, src);
}
