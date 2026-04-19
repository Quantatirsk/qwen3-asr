/// ARM NEON implementations of hot kernels.
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// # Safety
/// Uses NEON intrinsics; caller must ensure slices have equal lengths.
#[cfg(target_arch = "aarch64")]
pub unsafe fn bf16_to_f32_buf(dst: &mut [f32], src: &[u16]) {
    let n = src.len();
    let mut i = 0usize;

    while i + 8 <= n {
        let raw = vld1q_u16(src.as_ptr().add(i));
        let lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(raw), 16));
        let hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(raw), 16));
        vst1q_f32(dst.as_mut_ptr().add(i), lo);
        vst1q_f32(dst.as_mut_ptr().add(i + 4), hi);
        i += 8;
    }

    while i < n {
        dst[i] = f32::from_bits((src[i] as u32) << 16);
        i += 1;
    }
}

/// # Safety
/// w_bf16 must point to at least out_dim * in_dim valid bf16 values.
#[cfg(target_arch = "aarch64")]
pub unsafe fn bf16_matvec_fused(y: &mut [f32], x: &[f32], w_bf16: *const u16, bias: Option<&[f32]>, in_dim: usize, out_dim: usize) {
    let mut o = 0usize;

    // Process 2 output rows at a time
    while o + 1 < out_dim {
        let w0 = w_bf16.add(o * in_dim);
        let w1 = w_bf16.add((o + 1) * in_dim);
        let mut s0 = bias.map_or(0.0f32, |b| b[o]);
        let mut s1 = bias.map_or(0.0f32, |b| b[o + 1]);

        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        let mut b0 = vdupq_n_f32(0.0);
        let mut b1 = vdupq_n_f32(0.0);
        let mut b2 = vdupq_n_f32(0.0);
        let mut b3 = vdupq_n_f32(0.0);
        let mut k = 0usize;

        while k + 32 <= in_dim {
            let x0 = vld1q_f32(x.as_ptr().add(k));
            let x1 = vld1q_f32(x.as_ptr().add(k + 4));
            let x2 = vld1q_f32(x.as_ptr().add(k + 8));
            let x3 = vld1q_f32(x.as_ptr().add(k + 12));
            let x4 = vld1q_f32(x.as_ptr().add(k + 16));
            let x5 = vld1q_f32(x.as_ptr().add(k + 20));
            let x6 = vld1q_f32(x.as_ptr().add(k + 24));
            let x7 = vld1q_f32(x.as_ptr().add(k + 28));

            let r0a = vld1q_u16(w0.add(k));
            let r0b = vld1q_u16(w0.add(k + 8));
            let r0c = vld1q_u16(w0.add(k + 16));
            let r0d = vld1q_u16(w0.add(k + 24));
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)), x1);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)), x2);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)), x3);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0c), 16)), x4);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0c), 16)), x5);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0d), 16)), x6);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0d), 16)), x7);

            let r1a = vld1q_u16(w1.add(k));
            let r1b = vld1q_u16(w1.add(k + 8));
            let r1c = vld1q_u16(w1.add(k + 16));
            let r1d = vld1q_u16(w1.add(k + 24));
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1a), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1a), 16)), x1);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1b), 16)), x2);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1b), 16)), x3);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1c), 16)), x4);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1c), 16)), x5);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1d), 16)), x6);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1d), 16)), x7);

            k += 32;
        }
        while k + 8 <= in_dim {
            let xv0 = vld1q_f32(x.as_ptr().add(k));
            let xv1 = vld1q_f32(x.as_ptr().add(k + 4));
            let r0 = vld1q_u16(w0.add(k));
            let r1 = vld1q_u16(w1.add(k));
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0), 16)), xv0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0), 16)), xv1);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1), 16)), xv0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1), 16)), xv1);
            k += 8;
        }
        s0 += vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        s1 += vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));

        while k < in_dim {
            let wv0 = f32::from_bits(((*w0.add(k)) as u32) << 16);
            let wv1 = f32::from_bits(((*w1.add(k)) as u32) << 16);
            s0 += wv0 * x[k];
            s1 += wv1 * x[k];
            k += 1;
        }
        y[o] = s0;
        y[o + 1] = s1;
        o += 2;
    }

    // Handle remaining odd row
    while o < out_dim {
        let w_row = w_bf16.add(o * in_dim);
        let mut sum = bias.map_or(0.0f32, |b| b[o]);
        let mut k = 0usize;

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        while k + 8 <= in_dim {
            let bf = vld1q_u16(w_row.add(k));
            acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)),
                             vld1q_f32(x.as_ptr().add(k)));
            acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)),
                             vld1q_f32(x.as_ptr().add(k + 4)));
            k += 8;
        }
        sum += vaddvq_f32(vaddq_f32(acc0, acc1));

        while k < in_dim {
            let w_val = f32::from_bits(((*w_row.add(k)) as u32) << 16);
            sum += w_val * x[k];
            k += 1;
        }
        y[o] = sum;
        o += 1;
    }
}

/// # Safety
/// w_bf16 must point to at least end * in_dim valid bf16 values.
#[cfg(target_arch = "aarch64")]
pub unsafe fn argmax_bf16_range(x: &[f32], w_bf16: *const u16, in_dim: usize, start: usize, end: usize) -> (usize, f32) {
    let mut best = start;
    let mut best_val = -1e30f32;
    let mut o = start;

    // Process 2 rows at a time
    while o + 1 < end {
        let w0 = w_bf16.add(o * in_dim);
        let w1 = w_bf16.add((o + 1) * in_dim);
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        let mut b0 = vdupq_n_f32(0.0);
        let mut b1 = vdupq_n_f32(0.0);
        let mut b2 = vdupq_n_f32(0.0);
        let mut b3 = vdupq_n_f32(0.0);
        let mut k = 0usize;

        while k + 32 <= in_dim {
            let x0 = vld1q_f32(x.as_ptr().add(k));
            let x1 = vld1q_f32(x.as_ptr().add(k + 4));
            let x2 = vld1q_f32(x.as_ptr().add(k + 8));
            let x3 = vld1q_f32(x.as_ptr().add(k + 12));
            let x4 = vld1q_f32(x.as_ptr().add(k + 16));
            let x5 = vld1q_f32(x.as_ptr().add(k + 20));
            let x6 = vld1q_f32(x.as_ptr().add(k + 24));
            let x7 = vld1q_f32(x.as_ptr().add(k + 28));

            let r0a = vld1q_u16(w0.add(k));
            let r0b = vld1q_u16(w0.add(k + 8));
            let r0c = vld1q_u16(w0.add(k + 16));
            let r0d = vld1q_u16(w0.add(k + 24));
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)), x1);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)), x2);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)), x3);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0c), 16)), x4);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0c), 16)), x5);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0d), 16)), x6);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0d), 16)), x7);

            let r1a = vld1q_u16(w1.add(k));
            let r1b = vld1q_u16(w1.add(k + 8));
            let r1c = vld1q_u16(w1.add(k + 16));
            let r1d = vld1q_u16(w1.add(k + 24));
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1a), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1a), 16)), x1);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1b), 16)), x2);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1b), 16)), x3);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1c), 16)), x4);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1c), 16)), x5);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1d), 16)), x6);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1d), 16)), x7);

            k += 32;
        }

        let s0_v = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        let s1_v = vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));

        let mut s0 = s0_v;
        let mut s1 = s1_v;
        while k < in_dim {
            let wv0 = f32::from_bits(((*w0.add(k)) as u32) << 16);
            let wv1 = f32::from_bits(((*w1.add(k)) as u32) << 16);
            s0 += wv0 * x[k];
            s1 += wv1 * x[k];
            k += 1;
        }

        if s0 > best_val { best_val = s0; best = o; }
        if s1 > best_val { best_val = s1; best = o + 1; }
        o += 2;
    }

    while o < end {
        let w_row = w_bf16.add(o * in_dim);
        let mut sum = 0.0f32;
        let mut k = 0usize;

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        while k + 8 <= in_dim {
            let bf = vld1q_u16(w_row.add(k));
            acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)),
                             vld1q_f32(x.as_ptr().add(k)));
            acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)),
                             vld1q_f32(x.as_ptr().add(k + 4)));
            k += 8;
        }
        sum += vaddvq_f32(vaddq_f32(acc0, acc1));

        while k < in_dim {
            let w_val = f32::from_bits(((*w_row.add(k)) as u32) << 16);
            sum += w_val * x[k];
            k += 1;
        }
        if sum > best_val { best_val = sum; best = o; }
        o += 1;
    }

    (best, best_val)
}

/// # Safety
/// Uses NEON intrinsics; slices must have at least n elements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    while i + 16 <= n {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a.as_ptr().add(i + 4)), vld1q_f32(b.as_ptr().add(i + 4)));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a.as_ptr().add(i + 8)), vld1q_f32(b.as_ptr().add(i + 8)));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a.as_ptr().add(i + 12)), vld1q_f32(b.as_ptr().add(i + 12)));
        i += 16;
    }

    while i + 4 <= n {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
        i += 4;
    }

    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc2), vaddq_f32(acc1, acc3)));
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// # Safety
/// Uses NEON intrinsics; dst must have at least n elements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn vec_scale_inplace(dst: &mut [f32], scale: f32, n: usize) {
    let mut i = 0usize;
    let s = vdupq_n_f32(scale);
    while i + 8 <= n {
        let d0 = vld1q_f32(dst.as_ptr().add(i));
        let d1 = vld1q_f32(dst.as_ptr().add(i + 4));
        vst1q_f32(dst.as_mut_ptr().add(i), vmulq_f32(d0, s));
        vst1q_f32(dst.as_mut_ptr().add(i + 4), vmulq_f32(d1, s));
        i += 8;
    }
    while i < n {
        dst[i] *= scale;
        i += 1;
    }
}

/// # Safety
/// Uses NEON intrinsics; dst and src must have at least n elements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn vec_axpy_inplace(dst: &mut [f32], src: &[f32], alpha: f32, n: usize) {
    let mut i = 0usize;
    let a = vdupq_n_f32(alpha);
    while i + 8 <= n {
        let d0 = vld1q_f32(dst.as_ptr().add(i));
        let s0 = vld1q_f32(src.as_ptr().add(i));
        let d1 = vld1q_f32(dst.as_ptr().add(i + 4));
        let s1 = vld1q_f32(src.as_ptr().add(i + 4));
        vst1q_f32(dst.as_mut_ptr().add(i), vfmaq_f32(d0, s0, a));
        vst1q_f32(dst.as_mut_ptr().add(i + 4), vfmaq_f32(d1, s1, a));
        i += 8;
    }
    while i < n {
        dst[i] += alpha * src[i];
        i += 1;
    }
}

/// # Safety
/// Uses NEON intrinsics; dst and src must have at least n elements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn vec_scale_add(dst: &mut [f32], src: &[f32], correction: f32, n: usize) {
    let mut i = 0usize;
    let c = vdupq_n_f32(correction);
    while i + 8 <= n {
        let d0 = vld1q_f32(dst.as_ptr().add(i));
        let s0 = vld1q_f32(src.as_ptr().add(i));
        let d1 = vld1q_f32(dst.as_ptr().add(i + 4));
        let s1 = vld1q_f32(src.as_ptr().add(i + 4));
        vst1q_f32(dst.as_mut_ptr().add(i), vfmaq_f32(s0, d0, c));
        vst1q_f32(dst.as_mut_ptr().add(i + 4), vfmaq_f32(s1, d1, c));
        i += 8;
    }
    while i < n {
        dst[i] = dst[i] * correction + src[i];
        i += 1;
    }
}

/// NEON-accelerated RMS norm for a single row.
///
/// # Safety
/// Uses NEON intrinsics; all slices must have at least hidden elements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn rms_norm_row(out: &mut [f32], x: &[f32], weight: &[f32], hidden: usize, eps: f32) {
    // Sum of squares
    let mut i = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    while i + 8 <= hidden {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        acc0 = vfmaq_f32(acc0, x0, x0);
        acc1 = vfmaq_f32(acc1, x1, x1);
        i += 8;
    }
    let mut sum_sq = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < hidden {
        sum_sq += x[i] * x[i];
        i += 1;
    }

    let rms_inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
    let rms_v = vdupq_n_f32(rms_inv);

    // Scale: out = x * rms_inv * weight
    i = 0;
    while i + 8 <= hidden {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        let w0 = vld1q_f32(weight.as_ptr().add(i));
        let w1 = vld1q_f32(weight.as_ptr().add(i + 4));
        vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(vmulq_f32(x0, rms_v), w0));
        vst1q_f32(out.as_mut_ptr().add(i + 4), vmulq_f32(vmulq_f32(x1, rms_v), w1));
        i += 8;
    }
    while i < hidden {
        out[i] = x[i] * rms_inv * weight[i];
        i += 1;
    }
}

/// NEON-accelerated in-place RMS norm for a single row: x[i] = x[i] * rms_inv * weight[i].
///
/// # Safety
/// Uses NEON intrinsics; x and weight must have at least `hidden` elements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn rms_norm_inplace(x: &mut [f32], weight: &[f32], hidden: usize, eps: f32) {
    let ptr = x.as_mut_ptr();
    // Sum of squares
    let mut i = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    while i + 8 <= hidden {
        let x0 = vld1q_f32(ptr.add(i));
        let x1 = vld1q_f32(ptr.add(i + 4));
        acc0 = vfmaq_f32(acc0, x0, x0);
        acc1 = vfmaq_f32(acc1, x1, x1);
        i += 8;
    }
    let mut sum_sq = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < hidden {
        sum_sq += *ptr.add(i) * *ptr.add(i);
        i += 1;
    }

    let rms_inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
    let rms_v = vdupq_n_f32(rms_inv);

    // Scale in-place: x[i] = x[i] * rms_inv * weight[i]
    i = 0;
    while i + 8 <= hidden {
        let x0 = vld1q_f32(ptr.add(i));
        let x1 = vld1q_f32(ptr.add(i + 4));
        let w0 = vld1q_f32(weight.as_ptr().add(i));
        let w1 = vld1q_f32(weight.as_ptr().add(i + 4));
        vst1q_f32(ptr.add(i), vmulq_f32(vmulq_f32(x0, rms_v), w0));
        vst1q_f32(ptr.add(i + 4), vmulq_f32(vmulq_f32(x1, rms_v), w1));
        i += 8;
    }
    while i < hidden {
        *ptr.add(i) = *ptr.add(i) * rms_inv * weight[i];
        i += 1;
    }
}

/// NEON-accelerated layer norm for a single row.
///
/// # Safety
/// Uses NEON intrinsics; all slices must have at least hidden elements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn layer_norm_row(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32], hidden: usize, eps: f32) {
    // Pass 1: compute mean
    let mut i = 0usize;
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    while i + 8 <= hidden {
        sum0 = vaddq_f32(sum0, vld1q_f32(x.as_ptr().add(i)));
        sum1 = vaddq_f32(sum1, vld1q_f32(x.as_ptr().add(i + 4)));
        i += 8;
    }
    let mut mean = vaddvq_f32(vaddq_f32(sum0, sum1));
    while i < hidden {
        mean += x[i];
        i += 1;
    }
    mean /= hidden as f32;

    // Pass 2: compute variance
    let mean_v = vdupq_n_f32(mean);
    i = 0;
    let mut var0 = vdupq_n_f32(0.0);
    let mut var1 = vdupq_n_f32(0.0);
    while i + 8 <= hidden {
        let d0 = vsubq_f32(vld1q_f32(x.as_ptr().add(i)), mean_v);
        let d1 = vsubq_f32(vld1q_f32(x.as_ptr().add(i + 4)), mean_v);
        var0 = vfmaq_f32(var0, d0, d0);
        var1 = vfmaq_f32(var1, d1, d1);
        i += 8;
    }
    let mut var = vaddvq_f32(vaddq_f32(var0, var1));
    while i < hidden {
        let d = x[i] - mean;
        var += d * d;
        i += 1;
    }

    let inv_std = 1.0 / (var / hidden as f32 + eps).sqrt();
    let inv_v = vdupq_n_f32(inv_std);

    // Pass 3: normalize
    i = 0;
    while i + 8 <= hidden {
        let x0 = vsubq_f32(vld1q_f32(x.as_ptr().add(i)), mean_v);
        let x1 = vsubq_f32(vld1q_f32(x.as_ptr().add(i + 4)), mean_v);
        let w0 = vld1q_f32(weight.as_ptr().add(i));
        let w1 = vld1q_f32(weight.as_ptr().add(i + 4));
        let b0 = vld1q_f32(bias.as_ptr().add(i));
        let b1 = vld1q_f32(bias.as_ptr().add(i + 4));
        vst1q_f32(out.as_mut_ptr().add(i), vfmaq_f32(b0, vmulq_f32(x0, inv_v), w0));
        vst1q_f32(out.as_mut_ptr().add(i + 4), vfmaq_f32(b1, vmulq_f32(x1, inv_v), w1));
        i += 8;
    }
    while i < hidden {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
        i += 1;
    }
}

/// Fast exp approximation using NEON (7th-order polynomial, ~1e-4 relative error for |x| < 88).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn fast_exp_neon(x: float32x4_t) -> float32x4_t {
    // exp(x) ≈ 2^(x * log2e) using integer trick + polynomial refinement
    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);

    let val = vmulq_f32(x, log2e);
    // Clamp to prevent overflow
    let val = vminq_f32(val, vdupq_n_f32(126.0));
    let val = vmaxq_f32(val, vdupq_n_f32(-126.0));

    // Integer part
    let ipart = vcvtq_s32_f32(val);
    let fpart = vsubq_f32(val, vcvtq_f32_s32(ipart));

    // 2^ipart using bit manipulation
    let exp_i = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ipart, vdupq_n_s32(127)), 23));

    // 2^fpart using polynomial: 1 + fpart*ln2*(1 + fpart*ln2/2*(1 + fpart*ln2/3*(1 + ...)))
    let f = vmulq_f32(fpart, ln2);
    let c2 = vdupq_n_f32(0.5);
    let c3 = vdupq_n_f32(1.0 / 6.0);
    let c4 = vdupq_n_f32(1.0 / 24.0);
    let c5 = vdupq_n_f32(1.0 / 120.0);

    let mut p = vfmaq_f32(c4, c5, f);
    p = vfmaq_f32(c3, p, f);
    p = vfmaq_f32(c2, p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0), p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0), p, f);

    vmulq_f32(exp_i, p)
}

/// NEON-accelerated exp() in-place using fast polynomial approximation.
///
/// # Safety
/// Uses NEON intrinsics.
#[cfg(target_arch = "aarch64")]
pub unsafe fn exp_inplace(x: &mut [f32]) {
    let n = x.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let v = vld1q_f32(x.as_ptr().add(i));
        vst1q_f32(x.as_mut_ptr().add(i), fast_exp_neon(v));
        i += 4;
    }
    while i < n {
        x[i] = x[i].exp();
        i += 1;
    }
}

/// NEON-accelerated SwiGLU: out[j] = silu(gate[2j]) * gate[2j+1] for interleaved gate/up.
///
/// # Safety
/// Uses NEON intrinsics; gate_up must have at least 2*n elements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn swiglu_interleaved(out: &mut [f32], gate_up: &[f32], n: usize) {
    let one = vdupq_n_f32(1.0);
    let mut j = 0usize;

    // Process 8 elements per iteration (2x float32x4)
    while j + 8 <= n {
        let pair0 = vld1q_f32(gate_up.as_ptr().add(2 * j));
        let pair1 = vld1q_f32(gate_up.as_ptr().add(2 * j + 4));
        let gates0 = vuzp1q_f32(pair0, pair1);
        let ups0 = vuzp2q_f32(pair0, pair1);

        let pair2 = vld1q_f32(gate_up.as_ptr().add(2 * j + 8));
        let pair3 = vld1q_f32(gate_up.as_ptr().add(2 * j + 12));
        let gates1 = vuzp1q_f32(pair2, pair3);
        let ups1 = vuzp2q_f32(pair2, pair3);

        let exp0 = fast_exp_neon(vnegq_f32(gates0));
        let exp1 = fast_exp_neon(vnegq_f32(gates1));
        let silu0 = vdivq_f32(gates0, vaddq_f32(one, exp0));
        let silu1 = vdivq_f32(gates1, vaddq_f32(one, exp1));

        vst1q_f32(out.as_mut_ptr().add(j), vmulq_f32(silu0, ups0));
        vst1q_f32(out.as_mut_ptr().add(j + 4), vmulq_f32(silu1, ups1));
        j += 8;
    }

    while j + 4 <= n {
        let pair0 = vld1q_f32(gate_up.as_ptr().add(2 * j));
        let pair1 = vld1q_f32(gate_up.as_ptr().add(2 * j + 4));
        let gates = vuzp1q_f32(pair0, pair1);
        let ups = vuzp2q_f32(pair0, pair1);
        let exp_ng = fast_exp_neon(vnegq_f32(gates));
        let silu_g = vdivq_f32(gates, vaddq_f32(one, exp_ng));
        vst1q_f32(out.as_mut_ptr().add(j), vmulq_f32(silu_g, ups));
        j += 4;
    }

    while j < n {
        let g = gate_up[2 * j];
        let u = gate_up[2 * j + 1];
        let g_silu = g / (1.0 + (-g).exp());
        out[j] = g_silu * u;
        j += 1;
    }
}

/// NEON-accelerated GELU (tanh approximation).
///
/// # Safety
/// Uses NEON intrinsics; x must have at least n elements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn gelu_inplace(x: &mut [f32], n: usize) {
    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let coeff = vdupq_n_f32(0.797_884_6); // sqrt(2/pi)
    let c3 = vdupq_n_f32(0.044715);
    let mut i = 0usize;

    while i + 4 <= n {
        let v = vld1q_f32(x.as_ptr().add(i));
        let v3 = vmulq_f32(vmulq_f32(v, v), v);
        let inner = vmulq_f32(coeff, vfmaq_f32(v, c3, v3)); // coeff * (v + c3 * v^3)
        // tanh(x) ≈ (1 - 2/(exp(2x)+1)) = approximate via fast_exp
        let exp2x = fast_exp_neon(vmulq_f32(vdupq_n_f32(2.0), inner));
        let tanh_v = vsubq_f32(one, vdivq_f32(vdupq_n_f32(2.0), vaddq_f32(exp2x, one)));
        let result = vmulq_f32(half, vmulq_f32(v, vaddq_f32(one, tanh_v)));
        vst1q_f32(x.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < n {
        let val = x[i];
        let x3 = val * val * val;
        let inner = 0.797_884_6_f32 * (val + 0.044715 * x3);
        x[i] = 0.5 * val * (1.0 + inner.tanh());
        i += 1;
    }
}

/// Quantize BF16 weight matrix to INT8 per-row with absmax scaling.
/// Returns (int8_data, scales) where scales[row] is the per-row scale factor.
///
/// # Safety
/// w_bf16 must point to at least out_dim * in_dim valid bf16 values.
/// in_dim must be a multiple of 16 for alignment.
#[cfg(target_arch = "aarch64")]
pub unsafe fn quantize_bf16_to_int8(w_bf16: *const u16, out_dim: usize, in_dim: usize) -> (Vec<i8>, Vec<f32>) {
    let mut int8_data = vec![0i8; out_dim * in_dim];
    let mut scales = vec![0.0f32; out_dim];

    for row in 0..out_dim {
        let w_row = w_bf16.add(row * in_dim);

        // Find absmax of the row
        let mut k = 0;
        let mut vmax = vdupq_n_f32(0.0);
        let abs_mask = vdupq_n_u32(0x7FFF_FFFF);
        while k + 8 <= in_dim {
            let r0 = vld1q_u16(w_row.add(k));
            let f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0), 16));
            let f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0), 16));
            let a0 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(f0), abs_mask));
            let a1 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(f1), abs_mask));
            vmax = vmaxq_f32(vmax, vmaxq_f32(a0, a1));
            k += 8;
        }
        let mut max_abs = vmaxvq_f32(vmax);
        while k < in_dim {
            let v = f32::from_bits((*w_row.add(k) as u32) << 16).abs();
            if v > max_abs { max_abs = v; }
            k += 1;
        }

        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv_scale = 127.0 / max_abs.max(1e-10);
        scales[row] = scale;

        // Quantize row
        let dst = &mut int8_data[row * in_dim..(row + 1) * in_dim];
        k = 0;
        let inv_s = vdupq_n_f32(inv_scale);
        while k + 8 <= in_dim {
            let r0 = vld1q_u16(w_row.add(k));
            let f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0), 16));
            let f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0), 16));
            let q0 = vcvtq_s32_f32(vmulq_f32(f0, inv_s));
            let q1 = vcvtq_s32_f32(vmulq_f32(f1, inv_s));
            let q16 = vqmovn_s32(q0);
            let q16b = vqmovn_s32(q1);
            let q8 = vqmovn_s16(vcombine_s16(q16, q16b));
            vst1_s8(dst.as_mut_ptr().add(k) as *mut i8, q8);
            k += 8;
        }
        while k < in_dim {
            let v = f32::from_bits((*w_row.add(k) as u32) << 16);
            let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            dst[k] = q;
            k += 1;
        }
    }

    (int8_data, scales)
}

/// SDOT via inline assembly (stable Rust, avoids unstable vdotq_s32)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot_s32(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    core::arch::asm!(
        "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags),
    );
    acc
}

/// INT8 matvec: y = W_int8 @ x_int8 * (x_scale * w_scales[row])
/// Produces f32 output. Optionally adds bias (for fused residual add).
///
/// # Safety
/// Uses NEON SDOT via inline asm.
#[cfg(target_arch = "aarch64")]
pub unsafe fn matvec_int8(
    y: &mut [f32], x_int8: *const i8, x_scale: f32,
    w_int8: *const i8, w_scales: &[f32],
    bias: Option<&[f32]>,
    in_dim: usize, out_dim: usize,
) {
    let mut o = 0;
    while o + 1 < out_dim {
        let w0 = w_int8.add(o * in_dim);
        let w1 = w_int8.add((o + 1) * in_dim);
        let mut acc0a = vdupq_n_s32(0);
        let mut acc0b = vdupq_n_s32(0);
        let mut acc1a = vdupq_n_s32(0);
        let mut acc1b = vdupq_n_s32(0);
        let mut k = 0;

        while k + 32 <= in_dim {
            let x0 = vld1q_s8(x_int8.add(k));
            let x1 = vld1q_s8(x_int8.add(k + 16));
            acc0a = sdot_s32(acc0a, x0, vld1q_s8(w0.add(k)));
            acc0b = sdot_s32(acc0b, x1, vld1q_s8(w0.add(k + 16)));
            acc1a = sdot_s32(acc1a, x0, vld1q_s8(w1.add(k)));
            acc1b = sdot_s32(acc1b, x1, vld1q_s8(w1.add(k + 16)));
            k += 32;
        }
        while k + 16 <= in_dim {
            let xv = vld1q_s8(x_int8.add(k));
            acc0a = sdot_s32(acc0a, xv, vld1q_s8(w0.add(k)));
            acc1a = sdot_s32(acc1a, xv, vld1q_s8(w1.add(k)));
            k += 16;
        }

        let sum0 = vaddvq_s32(vaddq_s32(acc0a, acc0b));
        let sum1 = vaddvq_s32(vaddq_s32(acc1a, acc1b));

        let mut v0 = sum0 as f32 * x_scale * w_scales[o];
        let mut v1 = sum1 as f32 * x_scale * w_scales[o + 1];

        // Scalar tail
        while k < in_dim {
            let xv = *x_int8.add(k) as i32;
            v0 += xv as f32 * (*w0.add(k) as i32) as f32 * x_scale * w_scales[o];
            v1 += xv as f32 * (*w1.add(k) as i32) as f32 * x_scale * w_scales[o + 1];
            k += 1;
        }

        if let Some(b) = bias {
            v0 += b[o];
            v1 += b[o + 1];
        }
        y[o] = v0;
        y[o + 1] = v1;
        o += 2;
    }
    while o < out_dim {
        let w_row = w_int8.add(o * in_dim);
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut k = 0;
        while k + 32 <= in_dim {
            acc0 = sdot_s32(acc0, vld1q_s8(x_int8.add(k)), vld1q_s8(w_row.add(k)));
            acc1 = sdot_s32(acc1, vld1q_s8(x_int8.add(k + 16)), vld1q_s8(w_row.add(k + 16)));
            k += 32;
        }
        while k + 16 <= in_dim {
            acc0 = sdot_s32(acc0, vld1q_s8(x_int8.add(k)), vld1q_s8(w_row.add(k)));
            k += 16;
        }
        let mut val = vaddvq_s32(vaddq_s32(acc0, acc1)) as f32 * x_scale * w_scales[o];
        while k < in_dim {
            val += (*x_int8.add(k) as f32) * (*w_row.add(k) as f32) * x_scale * w_scales[o];
            k += 1;
        }
        if let Some(b) = bias { val += b[o]; }
        y[o] = val;
        o += 1;
    }
}

/// INT8 argmax: find argmax of x @ W.T where W is int8-quantized.
/// x_int8: quantized input [in_dim], x_scale: input quantization scale
/// w_int8: quantized weights [out_dim, in_dim], w_scales: per-row scales [out_dim]
///
/// # Safety
/// Uses NEON SDOT via inline asm. in_dim should be a multiple of 16 for best perf.
#[cfg(target_arch = "aarch64")]
pub unsafe fn argmax_int8_range(
    x_int8: *const i8, x_scale: f32,
    w_int8: *const i8, w_scales: &[f32],
    in_dim: usize, start: usize, end: usize,
) -> (usize, f32) {
    let mut best = start;
    let mut best_val = -1e30f32;

    for o in start..end {
        let w_row = w_int8.add(o * in_dim);
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);
        let mut k = 0;

        while k + 64 <= in_dim {
            acc0 = sdot_s32(acc0, vld1q_s8(x_int8.add(k)), vld1q_s8(w_row.add(k)));
            acc1 = sdot_s32(acc1, vld1q_s8(x_int8.add(k + 16)), vld1q_s8(w_row.add(k + 16)));
            acc2 = sdot_s32(acc2, vld1q_s8(x_int8.add(k + 32)), vld1q_s8(w_row.add(k + 32)));
            acc3 = sdot_s32(acc3, vld1q_s8(x_int8.add(k + 48)), vld1q_s8(w_row.add(k + 48)));
            k += 64;
        }

        while k + 16 <= in_dim {
            acc0 = sdot_s32(acc0, vld1q_s8(x_int8.add(k)), vld1q_s8(w_row.add(k)));
            k += 16;
        }

        let sum_i32 = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc2), vaddq_s32(acc1, acc3)));
        let val = sum_i32 as f32 * x_scale * w_scales[o];

        // Scalar tail
        let mut tail_sum = 0i32;
        while k < in_dim {
            tail_sum += (*x_int8.add(k) as i32) * (*w_row.add(k) as i32);
            k += 1;
        }
        let val = val + tail_sum as f32 * x_scale * w_scales[o];

        if val > best_val {
            best_val = val;
            best = o;
        }
    }

    (best, best_val)
}
