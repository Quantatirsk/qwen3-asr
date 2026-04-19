// x86 AVX2+FMA implementations of hot kernels.
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn bf16_to_f32_buf(dst: &mut [f32], src: &[u16]) {
    let n = src.len();
    let mut i = 0usize;

    while i + 8 <= n {
        let raw = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        let wide = _mm256_cvtepu16_epi32(raw);
        let shifted = _mm256_slli_epi32(wide, 16);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_castsi256_ps(shifted));
        i += 8;
    }

    while i < n {
        dst[i] = f32::from_bits((src[i] as u32) << 16);
        i += 1;
    }
}

/// Convert 8 BF16 values (in a __m128i) to 8 f32 values (in a __m256).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bf16x8_to_f32(raw: __m128i) -> __m256 {
    // Zero-extend u16 -> u32, shift left 16 to put BF16 bits in f32 position
    let wide = _mm256_cvtepu16_epi32(raw);
    let shifted = _mm256_slli_epi32(wide, 16);
    _mm256_castsi256_ps(shifted)
}

/// Horizontal sum of __m256 -> f32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_ps(v: __m256) -> f32 {
    // Add high 128 to low 128
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    // Horizontal add twice to reduce 4 -> 2 -> 1
    let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
    let sum64 = _mm_add_ps(sum128, shuf); // [0+1, _, 2+3, _]
    let hi64 = _mm_movehl_ps(sum64, sum64); // [2+3, _, _, _]
    let sum32 = _mm_add_ss(sum64, hi64);
    _mm_cvtss_f32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bf16_matvec_fused(
    y: &mut [f32], x: &[f32], w_bf16: *const u16,
    bias: Option<&[f32]>, in_dim: usize, out_dim: usize,
) {
    let mut o = 0usize;

    // Process 2 output rows at a time
    while o + 1 < out_dim {
        let w0 = w_bf16.add(o * in_dim);
        let w1 = w_bf16.add((o + 1) * in_dim);
        let mut s0 = bias.map_or(0.0f32, |b| b[o]);
        let mut s1 = bias.map_or(0.0f32, |b| b[o + 1]);

        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut k = 0usize;

        // Main loop: 16 elements per iteration
        while k + 16 <= in_dim {
            let xlo = _mm256_loadu_ps(x.as_ptr().add(k));
            let xhi = _mm256_loadu_ps(x.as_ptr().add(k + 8));

            // Row 0
            let raw0lo = _mm_loadu_si128(w0.add(k) as *const __m128i);
            let raw0hi = _mm_loadu_si128(w0.add(k + 8) as *const __m128i);
            let w0lo = bf16x8_to_f32(raw0lo);
            let w0hi = bf16x8_to_f32(raw0hi);
            a0 = _mm256_fmadd_ps(w0lo, xlo, a0);
            a1 = _mm256_fmadd_ps(w0hi, xhi, a1);

            // Row 1
            let raw1lo = _mm_loadu_si128(w1.add(k) as *const __m128i);
            let raw1hi = _mm_loadu_si128(w1.add(k + 8) as *const __m128i);
            let w1lo = bf16x8_to_f32(raw1lo);
            let w1hi = bf16x8_to_f32(raw1hi);
            b0 = _mm256_fmadd_ps(w1lo, xlo, b0);
            b1 = _mm256_fmadd_ps(w1hi, xhi, b1);

            k += 16;
        }

        // 8-element cleanup
        while k + 8 <= in_dim {
            let xv = _mm256_loadu_ps(x.as_ptr().add(k));
            let r0 = bf16x8_to_f32(_mm_loadu_si128(w0.add(k) as *const __m128i));
            let r1 = bf16x8_to_f32(_mm_loadu_si128(w1.add(k) as *const __m128i));
            a0 = _mm256_fmadd_ps(r0, xv, a0);
            b0 = _mm256_fmadd_ps(r1, xv, b0);
            k += 8;
        }

        s0 += hsum_ps(_mm256_add_ps(a0, a1));
        s1 += hsum_ps(_mm256_add_ps(b0, b1));

        // Scalar tail
        while k < in_dim {
            let wv0 = f32::from_bits((*w0.add(k) as u32) << 16);
            let wv1 = f32::from_bits((*w1.add(k) as u32) << 16);
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

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        while k + 16 <= in_dim {
            let xlo = _mm256_loadu_ps(x.as_ptr().add(k));
            let xhi = _mm256_loadu_ps(x.as_ptr().add(k + 8));
            let wlo = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k) as *const __m128i));
            let whi = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k + 8) as *const __m128i));
            acc0 = _mm256_fmadd_ps(wlo, xlo, acc0);
            acc1 = _mm256_fmadd_ps(whi, xhi, acc1);
            k += 16;
        }

        while k + 8 <= in_dim {
            let xv = _mm256_loadu_ps(x.as_ptr().add(k));
            let wv = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k) as *const __m128i));
            acc0 = _mm256_fmadd_ps(wv, xv, acc0);
            k += 8;
        }

        sum += hsum_ps(_mm256_add_ps(acc0, acc1));

        while k < in_dim {
            let w_val = f32::from_bits((*w_row.add(k) as u32) << 16);
            sum += w_val * x[k];
            k += 1;
        }
        y[o] = sum;
        o += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn argmax_bf16_range(
    x: &[f32], w_bf16: *const u16, in_dim: usize, start: usize, end: usize,
) -> (usize, f32) {
    let mut best = start;
    let mut best_val = -1e30f32;
    let mut o = start;

    // Process 2 rows at a time
    while o + 1 < end {
        let w0 = w_bf16.add(o * in_dim);
        let w1 = w_bf16.add((o + 1) * in_dim);
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut k = 0usize;

        while k + 16 <= in_dim {
            let xlo = _mm256_loadu_ps(x.as_ptr().add(k));
            let xhi = _mm256_loadu_ps(x.as_ptr().add(k + 8));

            let r0lo = bf16x8_to_f32(_mm_loadu_si128(w0.add(k) as *const __m128i));
            let r0hi = bf16x8_to_f32(_mm_loadu_si128(w0.add(k + 8) as *const __m128i));
            a0 = _mm256_fmadd_ps(r0lo, xlo, a0);
            a1 = _mm256_fmadd_ps(r0hi, xhi, a1);

            let r1lo = bf16x8_to_f32(_mm_loadu_si128(w1.add(k) as *const __m128i));
            let r1hi = bf16x8_to_f32(_mm_loadu_si128(w1.add(k + 8) as *const __m128i));
            b0 = _mm256_fmadd_ps(r1lo, xlo, b0);
            b1 = _mm256_fmadd_ps(r1hi, xhi, b1);

            k += 16;
        }

        let mut s0 = hsum_ps(_mm256_add_ps(a0, a1));
        let mut s1 = hsum_ps(_mm256_add_ps(b0, b1));

        while k < in_dim {
            let wv0 = f32::from_bits((*w0.add(k) as u32) << 16);
            let wv1 = f32::from_bits((*w1.add(k) as u32) << 16);
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

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        while k + 16 <= in_dim {
            let xlo = _mm256_loadu_ps(x.as_ptr().add(k));
            let xhi = _mm256_loadu_ps(x.as_ptr().add(k + 8));
            let wlo = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k) as *const __m128i));
            let whi = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k + 8) as *const __m128i));
            acc0 = _mm256_fmadd_ps(wlo, xlo, acc0);
            acc1 = _mm256_fmadd_ps(whi, xhi, acc1);
            k += 16;
        }
        sum += hsum_ps(_mm256_add_ps(acc0, acc1));

        while k < in_dim {
            let w_val = f32::from_bits((*w_row.add(k) as u32) << 16);
            sum += w_val * x[k];
            k += 1;
        }
        if sum > best_val { best_val = sum; best = o; }
        o += 1;
    }

    (best, best_val)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while i + 32 <= n {
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            acc0,
        );
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 8)),
            _mm256_loadu_ps(b.as_ptr().add(i + 8)),
            acc1,
        );
        acc2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 16)),
            _mm256_loadu_ps(b.as_ptr().add(i + 16)),
            acc2,
        );
        acc3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 24)),
            _mm256_loadu_ps(b.as_ptr().add(i + 24)),
            acc3,
        );
        i += 32;
    }

    while i + 8 <= n {
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            acc0,
        );
        i += 8;
    }

    let mut sum = hsum_ps(_mm256_add_ps(
        _mm256_add_ps(acc0, acc1),
        _mm256_add_ps(acc2, acc3),
    ));

    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vec_scale_inplace(dst: &mut [f32], scale: f32, n: usize) {
    let mut i = 0usize;
    let s = _mm256_set1_ps(scale);

    while i + 32 <= n {
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i)), s));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 8)), s));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 16), _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 16)), s));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 24), _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 24)), s));
        i += 32;
    }

    while i + 8 <= n {
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i)), s));
        i += 8;
    }

    while i < n {
        dst[i] *= scale;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn vec_axpy_inplace(dst: &mut [f32], src: &[f32], alpha: f32, n: usize) {
    let mut i = 0usize;
    let a = _mm256_set1_ps(alpha);

    while i + 32 <= n {
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(_mm256_loadu_ps(src.as_ptr().add(i)), a, _mm256_loadu_ps(dst.as_ptr().add(i))));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), _mm256_fmadd_ps(_mm256_loadu_ps(src.as_ptr().add(i + 8)), a, _mm256_loadu_ps(dst.as_ptr().add(i + 8))));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 16), _mm256_fmadd_ps(_mm256_loadu_ps(src.as_ptr().add(i + 16)), a, _mm256_loadu_ps(dst.as_ptr().add(i + 16))));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 24), _mm256_fmadd_ps(_mm256_loadu_ps(src.as_ptr().add(i + 24)), a, _mm256_loadu_ps(dst.as_ptr().add(i + 24))));
        i += 32;
    }

    while i + 8 <= n {
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(_mm256_loadu_ps(src.as_ptr().add(i)), a, _mm256_loadu_ps(dst.as_ptr().add(i))));
        i += 8;
    }

    while i < n {
        dst[i] += alpha * src[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn vec_scale_add(dst: &mut [f32], src: &[f32], correction: f32, n: usize) {
    let mut i = 0usize;
    let c = _mm256_set1_ps(correction);

    while i + 32 <= n {
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(_mm256_loadu_ps(dst.as_ptr().add(i)), c, _mm256_loadu_ps(src.as_ptr().add(i))));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), _mm256_fmadd_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 8)), c, _mm256_loadu_ps(src.as_ptr().add(i + 8))));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 16), _mm256_fmadd_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 16)), c, _mm256_loadu_ps(src.as_ptr().add(i + 16))));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 24), _mm256_fmadd_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 24)), c, _mm256_loadu_ps(src.as_ptr().add(i + 24))));
        i += 32;
    }

    while i + 8 <= n {
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(_mm256_loadu_ps(dst.as_ptr().add(i)), c, _mm256_loadu_ps(src.as_ptr().add(i))));
        i += 8;
    }

    while i < n {
        dst[i] = dst[i] * correction + src[i];
        i += 1;
    }
}

/// AVX2-accelerated RMS norm for a single row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rms_norm_row(out: &mut [f32], x: &[f32], weight: &[f32], hidden: usize, eps: f32) {
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while i + 16 <= hidden {
        let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let x1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(x0, x0, acc0);
        acc1 = _mm256_fmadd_ps(x1, x1, acc1);
        i += 16;
    }
    while i + 8 <= hidden {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(xv, xv, acc0);
        i += 8;
    }

    let mut sum_sq = hsum_ps(_mm256_add_ps(acc0, acc1));
    while i < hidden {
        sum_sq += x[i] * x[i];
        i += 1;
    }

    let rms_inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
    let rms_v = _mm256_set1_ps(rms_inv);

    i = 0;
    while i + 8 <= hidden {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let wv = _mm256_loadu_ps(weight.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_mul_ps(xv, rms_v), wv));
        i += 8;
    }
    while i < hidden {
        out[i] = x[i] * rms_inv * weight[i];
        i += 1;
    }
}

/// AVX2-accelerated layer norm for a single row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn layer_norm_row(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32], hidden: usize, eps: f32) {
    // Pass 1: compute mean
    let mut i = 0usize;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    while i + 16 <= hidden {
        sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(x.as_ptr().add(i)));
        sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(x.as_ptr().add(i + 8)));
        i += 16;
    }
    while i + 8 <= hidden {
        sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(x.as_ptr().add(i)));
        i += 8;
    }
    let mut mean = hsum_ps(_mm256_add_ps(sum0, sum1));
    while i < hidden {
        mean += x[i];
        i += 1;
    }
    mean /= hidden as f32;

    // Pass 2: compute variance
    let mean_v = _mm256_set1_ps(mean);
    i = 0;
    let mut var0 = _mm256_setzero_ps();
    let mut var1 = _mm256_setzero_ps();
    while i + 16 <= hidden {
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), mean_v);
        let d1 = _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), mean_v);
        var0 = _mm256_fmadd_ps(d0, d0, var0);
        var1 = _mm256_fmadd_ps(d1, d1, var1);
        i += 16;
    }
    while i + 8 <= hidden {
        let d = _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), mean_v);
        var0 = _mm256_fmadd_ps(d, d, var0);
        i += 8;
    }
    let mut var = hsum_ps(_mm256_add_ps(var0, var1));
    while i < hidden {
        let d = x[i] - mean;
        var += d * d;
        i += 1;
    }

    let inv_std = 1.0 / (var / hidden as f32 + eps).sqrt();
    let inv_v = _mm256_set1_ps(inv_std);

    // Pass 3: normalize
    i = 0;
    while i + 8 <= hidden {
        let xv = _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), mean_v);
        let wv = _mm256_loadu_ps(weight.as_ptr().add(i));
        let bv = _mm256_loadu_ps(bias.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_fmadd_ps(_mm256_mul_ps(xv, inv_v), wv, bv));
        i += 8;
    }
    while i < hidden {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
        i += 1;
    }
}

/// Fast exp approximation using AVX2+FMA (~1e-4 relative error for |x| < 88).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn fast_exp_avx(x: __m256) -> __m256 {
    let log2e = _mm256_set1_ps(1.442695041);
    let ln2 = _mm256_set1_ps(0.6931471806);

    let val = _mm256_mul_ps(x, log2e);
    let val = _mm256_min_ps(val, _mm256_set1_ps(126.0));
    let val = _mm256_max_ps(val, _mm256_set1_ps(-126.0));

    let ipart = _mm256_cvtps_epi32(val);
    let fpart = _mm256_sub_ps(val, _mm256_cvtepi32_ps(ipart));

    let exp_i = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(ipart, _mm256_set1_epi32(127)), 23));

    let f = _mm256_mul_ps(fpart, ln2);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(1.0 / 6.0);
    let c4 = _mm256_set1_ps(1.0 / 24.0);
    let c5 = _mm256_set1_ps(1.0 / 120.0);

    let mut p = _mm256_fmadd_ps(c5, f, c4);
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0));

    _mm256_mul_ps(exp_i, p)
}

/// AVX2-accelerated exp() in-place using fast polynomial approximation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn exp_inplace(x: &mut [f32]) {
    let n = x.len();
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), fast_exp_avx(v));
        i += 8;
    }
    while i < n {
        x[i] = x[i].exp();
        i += 1;
    }
}

/// AVX2-accelerated GELU (tanh approximation).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gelu_inplace(x: &mut [f32], n: usize) {
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let two = _mm256_set1_ps(2.0);
    let coeff = _mm256_set1_ps(0.7978845608028654);
    let c3 = _mm256_set1_ps(0.044715);
    let mut i = 0usize;

    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        let v2 = _mm256_mul_ps(v, v);
        let v3 = _mm256_mul_ps(v2, v);
        let inner = _mm256_mul_ps(coeff, _mm256_fmadd_ps(c3, v3, v));
        let exp2x = fast_exp_avx(_mm256_mul_ps(two, inner));
        let tanh_v = _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(exp2x, one)));
        let result = _mm256_mul_ps(half, _mm256_mul_ps(v, _mm256_add_ps(one, tanh_v)));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < n {
        let val = x[i];
        let x3 = val * val * val;
        let inner = 0.7978845608028654f32 * (val + 0.044715 * x3);
        x[i] = 0.5 * val * (1.0 + inner.tanh());
        i += 1;
    }
}

/// AVX2-accelerated SwiGLU with interleaved gate/up.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn swiglu_interleaved(out: &mut [f32], gate_up: &[f32], n: usize) {
    let one = _mm256_set1_ps(1.0);
    let mut j = 0usize;

    while j + 8 <= n {
        // Load 16 floats: [g0,u0,g1,u1,g2,u2,g3,u3] x2
        let lo = _mm256_loadu_ps(gate_up.as_ptr().add(2 * j));
        let hi = _mm256_loadu_ps(gate_up.as_ptr().add(2 * j + 8));

        // Deinterleave using shuffle + permute
        let shuf_lo = _mm256_shuffle_ps(lo, hi, 0b10_00_10_00); // g0,g1,g4,g5,g2,g3,g6,g7
        let shuf_hi = _mm256_shuffle_ps(lo, hi, 0b11_01_11_01); // u0,u1,u4,u5,u2,u3,u6,u7
        let gates = _mm256_permutevar8x32_ps(shuf_lo, _mm256_setr_epi32(0,1,4,5,2,3,6,7));
        let ups = _mm256_permutevar8x32_ps(shuf_hi, _mm256_setr_epi32(0,1,4,5,2,3,6,7));

        let neg_g = _mm256_sub_ps(_mm256_setzero_ps(), gates);
        let exp_ng = fast_exp_avx(neg_g);
        let denom = _mm256_add_ps(one, exp_ng);
        let silu_g = _mm256_div_ps(gates, denom);

        _mm256_storeu_ps(out.as_mut_ptr().add(j), _mm256_mul_ps(silu_g, ups));
        j += 8;
    }

    while j < n {
        let g = gate_up[2 * j];
        let u = gate_up[2 * j + 1];
        let g_silu = g / (1.0 + (-g).exp());
        out[j] = g_silu * u;
        j += 1;
    }
}
