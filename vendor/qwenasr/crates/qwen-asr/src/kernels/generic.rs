//! Generic (portable) implementations of hot kernels.

#[inline]
pub fn bf16_to_f32(bf16: u16) -> f32 {
    f32::from_bits((bf16 as u32) << 16)
}

/// # Safety
/// w_bf16 must point to at least out_dim * in_dim valid bf16 values.
pub unsafe fn bf16_matvec_fused(y: &mut [f32], x: &[f32], w_bf16: *const u16, bias: Option<&[f32]>, in_dim: usize, out_dim: usize) {
    for o in 0..out_dim {
        let w_row = unsafe { std::slice::from_raw_parts(w_bf16.add(o * in_dim), in_dim) };
        let mut sum = bias.map_or(0.0f32, |b| b[o]);
        for k in 0..in_dim {
            sum += bf16_to_f32(w_row[k]) * x[k];
        }
        y[o] = sum;
    }
}

/// # Safety
/// w_bf16 must point to at least end * in_dim valid bf16 values.
pub unsafe fn argmax_bf16_range(x: &[f32], w_bf16: *const u16, in_dim: usize, start: usize, end: usize) -> (usize, f32) {
    let mut best = start;
    let mut best_val = -1e30f32;

    for o in start..end {
        let w_row = unsafe { std::slice::from_raw_parts(w_bf16.add(o * in_dim), in_dim) };
        let mut sum = 0.0f32;
        for k in 0..in_dim {
            sum += bf16_to_f32(w_row[k]) * x[k];
        }
        if sum > best_val {
            best_val = sum;
            best = o;
        }
    }
    (best, best_val)
}

pub fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

pub fn vec_scale_inplace(dst: &mut [f32], scale: f32, n: usize) {
    for val in dst.iter_mut().take(n) {
        *val *= scale;
    }
}

pub fn vec_axpy_inplace(dst: &mut [f32], src: &[f32], alpha: f32, n: usize) {
    for i in 0..n {
        dst[i] += alpha * src[i];
    }
}

pub fn vec_scale_add(dst: &mut [f32], src: &[f32], correction: f32, n: usize) {
    for i in 0..n {
        dst[i] = dst[i] * correction + src[i];
    }
}
