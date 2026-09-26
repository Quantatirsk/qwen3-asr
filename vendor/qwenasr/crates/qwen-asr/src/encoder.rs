//! Audio encoder: Conv2D stem + windowed transformer + projection cascade.

use crate::config::*;
use crate::kernels;
use crate::safetensors::MultiSafetensors;

pub struct EncLayer {
    pub wq_weight: Vec<f32>,
    pub wq_bias: Vec<f32>,
    pub wk_weight: Vec<f32>,
    pub wk_bias: Vec<f32>,
    pub wv_weight: Vec<f32>,
    pub wv_bias: Vec<f32>,
    pub wo_weight: Vec<f32>,
    pub wo_bias: Vec<f32>,
    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Vec<f32>,
    pub fc1_weight: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2_weight: Vec<f32>,
    pub fc2_bias: Vec<f32>,
    pub ffn_norm_weight: Vec<f32>,
    pub ffn_norm_bias: Vec<f32>,
}

pub struct EncoderBuffers {
    pub x_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub proj_out: Vec<f32>,
    pub ffn_mid: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub cap_tokens: usize,
}

impl Default for EncoderBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl EncoderBuffers {
    pub fn new() -> Self {
        EncoderBuffers {
            x_norm: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            attn_out: Vec::new(),
            proj_out: Vec::new(),
            ffn_mid: Vec::new(),
            ffn_out: Vec::new(),
            cap_tokens: 0,
        }
    }

    pub fn ensure(&mut self, total_tokens: usize, d_model: usize, ffn_dim: usize) {
        if total_tokens <= self.cap_tokens {
            return;
        }
        let mut new_cap = if self.cap_tokens > 0 { self.cap_tokens } else { 256 };
        while new_cap < total_tokens {
            new_cap *= 2;
        }
        self.x_norm.resize(new_cap * d_model, 0.0);
        self.q.resize(new_cap * d_model, 0.0);
        self.k.resize(new_cap * d_model, 0.0);
        self.v.resize(new_cap * d_model, 0.0);
        self.attn_out.resize(new_cap * d_model, 0.0);
        self.proj_out.resize(new_cap * d_model, 0.0);
        self.ffn_mid.resize(new_cap * ffn_dim, 0.0);
        self.ffn_out.resize(new_cap * d_model, 0.0);
        self.cap_tokens = new_cap;
    }
}

pub struct Encoder {
    pub conv1_weight: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    pub conv2_weight: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    pub conv3_weight: Vec<f32>,
    pub conv3_bias: Vec<f32>,
    pub conv_out_weight: Vec<f32>,
    pub layers: Vec<EncLayer>,
    pub ln_post_weight: Vec<f32>,
    pub ln_post_bias: Vec<f32>,
    pub proj1_weight: Vec<f32>,
    pub proj1_bias: Vec<f32>,
    pub proj2_weight: Vec<f32>,
    pub proj2_bias: Vec<f32>,
}

const ENC_PREFIX: &str = "thinker.audio_tower.";

fn load_f32(ms: &MultiSafetensors, name: &str) -> Option<Vec<f32>> {
    let result = ms.get_f32(name);
    if result.is_none() {
        eprintln!("encoder: weight not found: {}", name);
    }
    result
}

fn load_bf16_as_f32(ms: &MultiSafetensors, name: &str) -> Option<Vec<f32>> {
    let (si, t) = ms.find(name).or_else(|| {
        eprintln!("encoder: weight not found: {}", name);
        None
    })?;

    let bf16_ptr = ms.shards[si].get_bf16_direct(t)?;
    let n = t.numel();
    let mut f32_data = vec![0.0f32; n];
    let src = unsafe { std::slice::from_raw_parts(bf16_ptr, n) };
    for i in 0..n {
        f32_data[i] = f32::from_bits((src[i] as u32) << 16);
    }
    Some(f32_data)
}

impl Encoder {
    pub fn load(ms: &MultiSafetensors, cfg: &QwenConfig) -> Option<Self> {
        let p = ENC_PREFIX;

        let conv1_weight = load_f32(ms, &format!("{}conv2d1.weight", p))?;
        let conv1_bias = load_f32(ms, &format!("{}conv2d1.bias", p))?;
        let conv2_weight = load_f32(ms, &format!("{}conv2d2.weight", p))?;
        let conv2_bias = load_f32(ms, &format!("{}conv2d2.bias", p))?;
        let conv3_weight = load_f32(ms, &format!("{}conv2d3.weight", p))?;
        let conv3_bias = load_f32(ms, &format!("{}conv2d3.bias", p))?;
        let conv_out_weight = load_bf16_as_f32(ms, &format!("{}conv_out.weight", p))?;

        let mut layers = Vec::new();
        for i in 0..cfg.enc_layers {
            let lp = format!("{}layers.{}", p, i);

            let layer = EncLayer {
                wq_weight: load_bf16_as_f32(ms, &format!("{}.self_attn.q_proj.weight", lp))?,
                wq_bias: load_f32(ms, &format!("{}.self_attn.q_proj.bias", lp))?,
                wk_weight: load_bf16_as_f32(ms, &format!("{}.self_attn.k_proj.weight", lp))?,
                wk_bias: load_f32(ms, &format!("{}.self_attn.k_proj.bias", lp))?,
                wv_weight: load_bf16_as_f32(ms, &format!("{}.self_attn.v_proj.weight", lp))?,
                wv_bias: load_f32(ms, &format!("{}.self_attn.v_proj.bias", lp))?,
                wo_weight: load_bf16_as_f32(ms, &format!("{}.self_attn.out_proj.weight", lp))?,
                wo_bias: load_f32(ms, &format!("{}.self_attn.out_proj.bias", lp))?,
                attn_norm_weight: load_f32(ms, &format!("{}.self_attn_layer_norm.weight", lp))?,
                attn_norm_bias: load_f32(ms, &format!("{}.self_attn_layer_norm.bias", lp))?,
                fc1_weight: load_bf16_as_f32(ms, &format!("{}.fc1.weight", lp))?,
                fc1_bias: load_f32(ms, &format!("{}.fc1.bias", lp))?,
                fc2_weight: load_bf16_as_f32(ms, &format!("{}.fc2.weight", lp))?,
                fc2_bias: load_f32(ms, &format!("{}.fc2.bias", lp))?,
                ffn_norm_weight: load_f32(ms, &format!("{}.final_layer_norm.weight", lp))?,
                ffn_norm_bias: load_f32(ms, &format!("{}.final_layer_norm.bias", lp))?,
            };
            layers.push(layer);
        }

        let ln_post_weight = load_f32(ms, &format!("{}ln_post.weight", p))?;
        let ln_post_bias = load_f32(ms, &format!("{}ln_post.bias", p))?;
        let proj1_weight = load_bf16_as_f32(ms, &format!("{}proj1.weight", p))?;
        let proj1_bias = load_f32(ms, &format!("{}proj1.bias", p))?;
        let proj2_weight = load_bf16_as_f32(ms, &format!("{}proj2.weight", p))?;
        let proj2_bias = load_f32(ms, &format!("{}proj2.bias", p))?;

        Some(Encoder {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            conv3_weight,
            conv3_bias,
            conv_out_weight,
            layers,
            ln_post_weight,
            ln_post_bias,
            proj1_weight,
            proj1_bias,
            proj2_weight,
            proj2_bias,
        })
    }

    /// Run encoder forward pass on mel spectrogram.
    /// mel: [128, mel_frames], returns [total_tokens, output_dim].
    pub fn forward(&self, cfg: &QwenConfig, mel: &[f32], mel_frames: usize, enc_bufs: Option<&mut EncoderBuffers>) -> Option<(Vec<f32>, usize)> {
        let d_model = cfg.enc_d_model;
        let n_heads = cfg.enc_heads;
        let head_dim = cfg.enc_head_dim;
        let ffn_dim = cfg.enc_ffn_dim;
        let output_dim = cfg.enc_output_dim;
        let chunk_size = cfg.enc_chunk_size;
        let n_window_infer = cfg.enc_n_window_infer;

        // Determine tokens per full chunk
        let tokens_per_chunk = {
            let w = chunk_size;
            let w1 = (w + 2 - 3) / 2 + 1;
            let w2 = (w1 + 2 - 3) / 2 + 1;
            (w2 + 2 - 3) / 2 + 1
        };

        let n_chunks = mel_frames.div_ceil(chunk_size);

        // Pre-calculate total tokens
        let mut total_tokens = 0;
        let mut chunk_sizes = Vec::new();
        for c in 0..n_chunks {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(mel_frames);
            let chunk_w = end - start;
            let w1 = (chunk_w + 2 - 3) / 2 + 1;
            let w2 = (w1 + 2 - 3) / 2 + 1;
            let w3 = (w2 + 2 - 3) / 2 + 1;
            total_tokens += w3;
            chunk_sizes.push((start, end, w3));
        }

        // Main sequence buffer: [total_tokens, d_model]
        let mut x = vec![0.0f32; total_tokens * d_model];
        let mut token_offset = 0;

        // Process each chunk through Conv2D + reshape + project + sinusoidal PE
        for &(start, end, w3) in &chunk_sizes {
            let chunk_w = end - start;

            // Extract chunk mel: [128, chunk_w]
            let mut chunk_mel = vec![0.0f32; 128 * chunk_w];
            for m in 0..128 {
                chunk_mel[m * chunk_w..(m + 1) * chunk_w]
                    .copy_from_slice(&mel[m * mel_frames + start..m * mel_frames + end]);
            }

            // Conv2D layer 1: [1, 128, chunk_w] -> [480, h1, w1]
            let h1 = (128 + 2 - 3) / 2 + 1; // 64
            let w1 = (chunk_w + 2 - 3) / 2 + 1;
            let mut c1 = vec![0.0f32; CONV_HIDDEN * h1 * w1];
            kernels::conv2d(
                &mut c1, &chunk_mel, &self.conv1_weight, Some(&self.conv1_bias),
                1, CONV_HIDDEN, 128, chunk_w, 3, 3, 2, 1,
            );
            kernels::gelu(&mut c1, CONV_HIDDEN * h1 * w1);

            // Conv2D layer 2: [480, h1, w1] -> [480, h2, w2]
            let h2 = (h1 + 2 - 3) / 2 + 1; // 32
            let w2 = (w1 + 2 - 3) / 2 + 1;
            let mut c2 = vec![0.0f32; CONV_HIDDEN * h2 * w2];
            kernels::conv2d(
                &mut c2, &c1, &self.conv2_weight, Some(&self.conv2_bias),
                CONV_HIDDEN, CONV_HIDDEN, h1, w1, 3, 3, 2, 1,
            );
            kernels::gelu(&mut c2, CONV_HIDDEN * h2 * w2);

            // Conv2D layer 3: [480, h2, w2] -> [480, h3, w3]
            let h3 = (h2 + 2 - 3) / 2 + 1; // 16
            let _w3_calc = (w2 + 2 - 3) / 2 + 1;
            debug_assert_eq!(_w3_calc, w3);
            let mut c3 = vec![0.0f32; CONV_HIDDEN * h3 * w3];
            kernels::conv2d(
                &mut c3, &c2, &self.conv3_weight, Some(&self.conv3_bias),
                CONV_HIDDEN, CONV_HIDDEN, h2, w2, 3, 3, 2, 1,
            );
            kernels::gelu(&mut c3, CONV_HIDDEN * h3 * w3);

            // Reshape [480, h3, w3] -> [w3, 480*h3]
            // Loop order: ch → f → t for sequential reads from c3
            let conv_proj_dim = CONV_HIDDEN * h3;
            let mut reshaped = vec![0.0f32; w3 * conv_proj_dim];
            for ch in 0..CONV_HIDDEN {
                for f in 0..h3 {
                    let src_off = ch * h3 * w3 + f * w3;
                    let dst_col = ch * h3 + f;
                    for t in 0..w3 {
                        reshaped[t * conv_proj_dim + dst_col] = c3[src_off + t];
                    }
                }
            }

            // Project: [w3, 7680] -> [w3, d_model]
            let projected = &mut x[token_offset * d_model..(token_offset + w3) * d_model];
            kernels::linear_nobias(projected, &reshaped, &self.conv_out_weight, w3, conv_proj_dim, d_model);

            // Add per-chunk sinusoidal PE
            let mut pe = vec![0.0f32; w3 * d_model];
            kernels::sinusoidal_pe(&mut pe, w3, d_model);
            kernels::add_inplace(projected, &pe, w3 * d_model);

            token_offset += w3;
        }

        // Build attention window boundaries
        let window_token_size = tokens_per_chunk * (n_window_infer / chunk_size);
        let n_windows = total_tokens.div_ceil(window_token_size);
        let mut window_starts = vec![0i32; n_windows + 1];
        for (w, ws) in window_starts.iter_mut().enumerate().take(n_windows) {
            *ws = (w * window_token_size) as i32;
        }
        window_starts[n_windows] = total_tokens as i32;

        // Transformer layer scratch buffers (reusable or fresh)
        let mut _owned_bufs;
        let bufs: &mut EncoderBuffers = match enc_bufs {
            Some(b) => { b.ensure(total_tokens, d_model, ffn_dim); b }
            None => { _owned_bufs = EncoderBuffers::new(); _owned_bufs.ensure(total_tokens, d_model, ffn_dim); &mut _owned_bufs }
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let td = total_tokens * d_model;
        let tf = total_tokens * ffn_dim;

        for layer in &self.layers {
            // Self-attention
            kernels::layer_norm(&mut bufs.x_norm[..td], &x, &layer.attn_norm_weight, &layer.attn_norm_bias,
                              total_tokens, d_model, 1e-5);

            kernels::linear(&mut bufs.q[..td], &bufs.x_norm[..td], &layer.wq_weight, Some(&layer.wq_bias),
                          total_tokens, d_model, d_model);
            kernels::linear(&mut bufs.k[..td], &bufs.x_norm[..td], &layer.wk_weight, Some(&layer.wk_bias),
                          total_tokens, d_model, d_model);
            kernels::linear(&mut bufs.v[..td], &bufs.x_norm[..td], &layer.wv_weight, Some(&layer.wv_bias),
                          total_tokens, d_model, d_model);

            kernels::bidirectional_attention(&mut bufs.attn_out[..td], &bufs.q[..td], &bufs.k[..td], &bufs.v[..td],
                                           total_tokens, n_heads, head_dim, scale,
                                           &window_starts, n_windows);

            // Fused: x += wo_bias + attn_out @ wo_weight.T
            kernels::linear_accumulate(&mut x, &bufs.attn_out[..td], &layer.wo_weight, Some(&layer.wo_bias),
                          total_tokens, d_model, d_model);

            // FFN
            kernels::layer_norm(&mut bufs.x_norm[..td], &x, &layer.ffn_norm_weight, &layer.ffn_norm_bias,
                              total_tokens, d_model, 1e-5);

            kernels::linear(&mut bufs.ffn_mid[..tf], &bufs.x_norm[..td], &layer.fc1_weight, Some(&layer.fc1_bias),
                          total_tokens, d_model, ffn_dim);
            kernels::gelu(&mut bufs.ffn_mid[..tf], tf);
            // Fused: x += fc2_bias + ffn_mid @ fc2_weight.T
            kernels::linear_accumulate(&mut x, &bufs.ffn_mid[..tf], &layer.fc2_weight, Some(&layer.fc2_bias),
                          total_tokens, ffn_dim, d_model);
        }

        // Final LayerNorm: use x_norm as temp, then swap into x
        kernels::layer_norm(&mut bufs.x_norm[..td], &x, &self.ln_post_weight, &self.ln_post_bias,
                          total_tokens, d_model, 1e-5);
        x[..td].copy_from_slice(&bufs.x_norm[..td]);

        // Projection: proj1 (GELU) -> proj2 (reuse scratch buffers)
        kernels::linear(&mut bufs.q[..td], &x, &self.proj1_weight, Some(&self.proj1_bias),
                       total_tokens, d_model, d_model);
        kernels::gelu(&mut bufs.q[..td], td);

        let mut enc_output = vec![0.0f32; total_tokens * output_dim];
        kernels::linear(&mut enc_output, &bufs.q[..td], &self.proj2_weight, Some(&self.proj2_bias),
                       total_tokens, d_model, output_dim);

        Some((enc_output, total_tokens))
    }
}
