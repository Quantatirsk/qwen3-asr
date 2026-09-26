//! WAV loading, resampling, and mel spectrogram computation.

use crate::config::*;
use crate::kernels;

const N_FFT: usize = 400;
const N_FREQ: usize = N_FFT / 2 + 1; // 201

fn read_u16_le(data: &[u8]) -> u16 {
    u16::from_le_bytes([data[0], data[1]])
}

fn read_u32_le(data: &[u8]) -> u32 {
    u32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

/// Parse a WAV file buffer into f32 samples at 16 kHz mono.
///
/// Accepts 16-bit PCM WAV at any sample rate (resampled automatically) and
/// any channel count (downmixed to mono). Returns `None` if the buffer is not
/// a valid WAV or uses an unsupported encoding (e.g. float, compressed).
pub fn parse_wav_buffer(data: &[u8]) -> Option<Vec<f32>> {
    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        eprintln!("parse_wav_buffer: not a valid WAV file");
        return None;
    }

    let mut channels = 0i32;
    let mut sample_rate = 0i32;
    let mut bits_per_sample = 0i32;
    let mut audio_format = 0i32;
    let mut pcm_data: Option<&[u8]> = None;

    let mut p = 12;
    while p + 8 <= data.len() {
        let chunk_id = &data[p..p + 4];
        let chunk_size = read_u32_le(&data[p + 4..]) as usize;
        if p + 8 + chunk_size > data.len() {
            break;
        }

        if chunk_id == b"fmt " && chunk_size >= 16 {
            audio_format = read_u16_le(&data[p + 8..]) as i32;
            channels = read_u16_le(&data[p + 10..]) as i32;
            sample_rate = read_u32_le(&data[p + 12..]) as i32;
            bits_per_sample = read_u16_le(&data[p + 22..]) as i32;
        } else if chunk_id == b"data" {
            let end = (p + 8 + chunk_size).min(data.len());
            pcm_data = Some(&data[p + 8..end]);
        }

        p += 8 + chunk_size;
        if chunk_size & 1 != 0 {
            p += 1;
        }
    }

    if audio_format != 1 || bits_per_sample != 16 || channels < 1 {
        eprintln!(
            "parse_wav_buffer: unsupported format (need 16-bit PCM, got fmt={} bits={})",
            audio_format, bits_per_sample
        );
        return None;
    }

    let pcm = pcm_data?;
    let n_frames = pcm.len() / (channels as usize * 2);
    let mut samples = vec![0.0f32; n_frames];

    for i in 0..n_frames {
        if channels == 1 {
            let val = i16::from_le_bytes([pcm[i * 2], pcm[i * 2 + 1]]);
            samples[i] = val as f32 / 32768.0;
        } else {
            let mut sum = 0.0f32;
            for c in 0..channels as usize {
                let off = (i * channels as usize + c) * 2;
                let val = i16::from_le_bytes([pcm[off], pcm[off + 1]]);
                sum += val as f32;
            }
            samples[i] = (sum / channels as f32) / 32768.0;
        }
    }

    // Resample to 16kHz if needed
    if sample_rate != SAMPLE_RATE {
        samples = resample(&samples, sample_rate, SAMPLE_RATE);
    }

    Some(samples)
}

/// Read a WAV file from disk and return f32 samples at 16 kHz mono.
///
/// Equivalent to `std::fs::read` + [`parse_wav_buffer`].
pub fn load_wav(path: &str) -> Option<Vec<f32>> {
    let data = std::fs::read(path).ok()?;
    parse_wav_buffer(&data)
}

/// Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono).
pub fn read_pcm_stdin() -> Option<Vec<f32>> {
    use std::io::Read;
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).ok()?;

    if buf.len() < 4 {
        eprintln!("read_pcm_stdin: no data on stdin");
        return None;
    }

    if &buf[0..4] == b"RIFF" {
        if kernels::verbose() >= 2 {
            eprintln!("Detected WAV format on stdin");
        }
        return parse_wav_buffer(&buf);
    }

    // Raw s16le 16kHz mono
    if kernels::verbose() >= 2 {
        eprintln!("Treating stdin as raw s16le 16kHz mono");
    }
    let n_frames = buf.len() / 2;
    let mut samples = vec![0.0f32; n_frames];
    for i in 0..n_frames {
        let val = i16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
        samples[i] = val as f32 / 32768.0;
    }
    Some(samples)
}

/// Kaiser-windowed sinc resampler.
pub fn resample(samples: &[f32], from_rate: i32, to_rate: i32) -> Vec<f32> {
    let n_frames = samples.len();
    let new_n = (n_frames as i64 * to_rate as i64 / from_rate as i64) as usize;
    let mut resampled = vec![0.0f32; new_n];

    let sinc_half = 16;
    let kaiser_beta = 6.0f64;
    let ratio = to_rate as f64 / from_rate as f64;
    let cutoff = if ratio < 1.0 { ratio } else { 1.0 };

    // Bessel I0 approximation
    fn bessel_i0(x: f64) -> f64 {
        let mut sum = 1.0;
        let mut term = 1.0;
        let xx = x * x;
        for k in 1..=20 {
            term *= xx / (4.0 * k as f64 * k as f64);
            sum += term;
        }
        sum
    }

    let inv_i0_beta = 1.0 / bessel_i0(kaiser_beta);

    for (i, resampled_val) in resampled.iter_mut().enumerate().take(new_n) {
        let src_pos = i as f64 / ratio;
        let center = src_pos as i32;
        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;

        let j_lo = center - sinc_half + 1;
        let j_hi = center + sinc_half;

        for j in j_lo..=j_hi {
            let d = j as f64 - src_pos;
            let x = d * cutoff;

            // Sinc
            let s = if x.abs() < 1e-9 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };

            // Kaiser window
            let npos = d / sinc_half as f64;
            let w = if npos <= -1.0 || npos >= 1.0 {
                0.0
            } else {
                bessel_i0(kaiser_beta * (1.0 - npos * npos).sqrt()) * inv_i0_beta
            };

            let coeff = s * w * cutoff;
            if j >= 0 && (j as usize) < n_frames {
                acc += samples[j as usize] as f64 * coeff;
            }
            wsum += coeff;
        }

        *resampled_val = if wsum > 1e-9 { (acc / wsum) as f32 } else { 0.0 };
    }

    resampled
}

// ========================================================================
// Mel Filter Bank (Slaney-style)
// ========================================================================

fn hertz_to_mel(freq: f32) -> f32 {
    let min_log_hertz = 1000.0f32;
    let min_log_mel = 15.0f32;
    let logstep = 27.0 / (6.4f32).ln();
    let mels = 3.0 * freq / 200.0;
    if freq >= min_log_hertz {
        min_log_mel + (freq / min_log_hertz).ln() * logstep
    } else {
        mels
    }
}

fn mel_to_hertz(mels: f32) -> f32 {
    let min_log_hertz = 1000.0f32;
    let min_log_mel = 15.0f32;
    let logstep = (6.4f32).ln() / 27.0;
    if mels >= min_log_mel {
        min_log_hertz * (logstep * (mels - min_log_mel)).exp()
    } else {
        200.0 * mels / 3.0
    }
}

fn build_mel_filters() -> Vec<f32> {
    let mut filters = vec![0.0f32; MEL_BINS * N_FREQ];

    let mut fft_freqs = vec![0.0f32; N_FREQ];
    for (i, freq) in fft_freqs.iter_mut().enumerate().take(N_FREQ) {
        *freq = i as f32 * (SAMPLE_RATE as f32 / 2.0) / (N_FREQ - 1) as f32;
    }

    let mel_min = hertz_to_mel(0.0);
    let mel_max = hertz_to_mel(SAMPLE_RATE as f32 / 2.0);

    let n_filters = MEL_BINS;
    let mut filter_freqs = vec![0.0f32; n_filters + 2];
    let mut filter_diff = vec![0.0f32; n_filters + 1];

    for (i, filter_freq) in filter_freqs.iter_mut().enumerate().take(n_filters + 2) {
        let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_filters + 1) as f32;
        *filter_freq = mel_to_hertz(mel);
    }
    for i in 0..n_filters + 1 {
        filter_diff[i] = filter_freqs[i + 1] - filter_freqs[i];
        if filter_diff[i] == 0.0 {
            filter_diff[i] = 1e-6;
        }
    }

    for m in 0..n_filters {
        let enorm = 2.0 / (filter_freqs[m + 2] - filter_freqs[m]);
        for f in 0..N_FREQ {
            let down = (fft_freqs[f] - filter_freqs[m]) / filter_diff[m];
            let up = (filter_freqs[m + 2] - fft_freqs[f]) / filter_diff[m + 1];
            let val = down.min(up).max(0.0);
            filters[m * N_FREQ + f] = val * enorm;
        }
    }

    filters
}

// ========================================================================
// Mel Spectrogram
// ========================================================================

/// Compute a 128-bin log-mel spectrogram from 16 kHz audio samples.
///
/// Returns `(mel_flat, n_frames)` where `mel_flat` has shape `[128, n_frames]`
/// in row-major order. Returns `None` if the audio is too short to produce
/// even one frame.
pub fn mel_spectrogram(samples: &[f32]) -> Option<(Vec<f32>, usize)> {
    let n_samples = samples.len();
    let n_fft = N_FFT;
    let n_freqs = N_FREQ;
    let pad_len = n_fft / 2;

    // Reflect-pad the signal
    let padded_len = n_samples + 2 * pad_len;
    let mut padded = vec![0.0f32; padded_len];

    for (i, padded_val) in padded.iter_mut().enumerate().take(pad_len) {
        let src = pad_len - i;
        *padded_val = if src < n_samples { samples[src] } else { 0.0 };
    }
    padded[pad_len..pad_len + n_samples].copy_from_slice(samples);
    for i in 0..pad_len {
        let src = n_samples as i32 - 2 - i as i32;
        padded[pad_len + n_samples + i] = if src >= 0 { samples[src as usize] } else { 0.0 };
    }

    let n_frames_total = (padded_len - n_fft) / HOP_LENGTH + 1;
    let n_frames = n_frames_total - 1; // drop last frame
    if n_frames == 0 {
        eprintln!("mel_spectrogram: audio too short ({} samples)", n_samples);
        return None;
    }

    static MEL_FILTERS: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    let mel_filters = MEL_FILTERS.get_or_init(build_mel_filters);

    // Periodic Hann window (cached)
    static HANN_WINDOW: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    let window = HANN_WINDOW.get_or_init(|| {
        let mut w = vec![0.0f32; WINDOW_SIZE];
        for (i, w_val) in w.iter_mut().enumerate().take(WINDOW_SIZE) {
            *w_val = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / WINDOW_SIZE as f32).cos());
        }
        w
    });

    // Precompute DFT tables (cached)
    static DFT_TABLES: std::sync::OnceLock<(Vec<f32>, Vec<f32>)> = std::sync::OnceLock::new();
    let (dft_cos, dft_sin) = DFT_TABLES.get_or_init(|| {
        let mut cos_tbl = vec![0.0f32; N_FREQ * N_FFT];
        let mut sin_tbl = vec![0.0f32; N_FREQ * N_FFT];
        for k in 0..N_FREQ {
            for n in 0..N_FFT {
                let angle = 2.0 * std::f32::consts::PI * k as f32 * n as f32 / N_FFT as f32;
                cos_tbl[k * N_FFT + n] = angle.cos();
                sin_tbl[k * N_FFT + n] = angle.sin();
            }
        }
        (cos_tbl, sin_tbl)
    });

    // Batched computation via BLAS sgemm:
    // 1. Pre-compute all windowed frames: windowed[N_FFT × n_frames] column-major
    let mut windowed_all = vec![0.0f32; N_FFT * n_frames];
    for t in 0..n_frames {
        let start = t * HOP_LENGTH;
        for n in 0..N_FFT {
            windowed_all[n * n_frames + t] = padded[start + n] * window[n];
        }
    }

    // 2. DFT via BLAS: re = dft_cos @ windowed_all, im = dft_sin @ windowed_all
    //    [N_FREQ × N_FFT] @ [N_FFT × n_frames] = [N_FREQ × n_frames]
    let mut re = vec![0.0f32; n_freqs * n_frames];
    let mut im = vec![0.0f32; n_freqs * n_frames];
    kernels::matmul_nn(&mut re, dft_cos, &windowed_all, n_freqs, N_FFT, n_frames);
    kernels::matmul_nn(&mut im, dft_sin, &windowed_all, n_freqs, N_FFT, n_frames);
    drop(windowed_all);

    // 3. Power spectrum: power[k * n_frames + t] = re² + im²
    let mut power = vec![0.0f32; n_freqs * n_frames];
    for i in 0..n_freqs * n_frames {
        power[i] = re[i] * re[i] + im[i] * im[i];
    }
    drop(re);
    drop(im);

    // 4. Mel filter bank via BLAS: mel_raw = mel_filters @ power
    //    [MEL_BINS × N_FREQ] @ [N_FREQ × n_frames] = [MEL_BINS × n_frames]
    let mut mel = vec![0.0f32; MEL_BINS * n_frames];
    kernels::matmul_nn(&mut mel, mel_filters, &power, MEL_BINS, n_freqs, n_frames);
    drop(power);

    // 5. Log, clamp, normalize
    let mut global_max = -1e30f32;
    for val in mel.iter_mut() {
        *val = (*val).max(1e-10).log10();
        if *val > global_max { global_max = *val; }
    }
    let min_val = global_max - 8.0;
    for val in mel.iter_mut() {
        *val = ((*val).max(min_val) + 4.0) / 4.0;
    }

    Some((mel, n_frames))
}

/// Drop long silent spans. Adaptive RMS gating with spike rejection.
pub fn compact_silence(samples: &[f32]) -> Vec<f32> {
    let n_samples = samples.len();
    if n_samples == 0 {
        return Vec::new();
    }

    let win = 160; // 10ms at 16kHz
    let base_thresh = 0.002f32;
    let max_thresh = 0.025f32;
    let smooth_alpha = 0.2f32;
    let min_voice_windows = 5;
    let pad_voice_windows = 3;
    let pass_windows = 60;

    let n_win = n_samples.div_ceil(win);
    let mut rms_vals = vec![0.0f32; n_win];

    for (w, rms_val) in rms_vals.iter_mut().enumerate().take(n_win) {
        let start = w * win;
        let end = (start + win).min(n_samples);
        let len = end - start;
        let mut energy = 0.0f32;
        for sample in samples.iter().take(end).skip(start) {
            energy += sample * sample;
        }
        *rms_val = (energy / len.max(1) as f32).sqrt();
    }

    // Smooth RMS
    let mut smooth_vals = vec![0.0f32; n_win];
    let mut smooth = rms_vals[0];
    for w in 0..n_win {
        smooth = (1.0 - smooth_alpha) * smooth + smooth_alpha * rms_vals[w];
        smooth_vals[w] = smooth;
    }

    // Adaptive threshold from 25th percentile
    let mut sorted = smooth_vals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p25 = ((n_win - 1) as f32 * 0.25) as usize;
    let noise_floor = sorted[p25];
    let thresh = (noise_floor * 1.8).clamp(base_thresh, max_thresh);

    let mut is_voice = vec![false; n_win];
    for w in 0..n_win {
        is_voice[w] = smooth_vals[w] > thresh;
    }

    // Remove short voice bursts
    let mut i = 0;
    while i < n_win {
        if !is_voice[i] {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < n_win && is_voice[j] {
            j += 1;
        }
        if j - i < min_voice_windows {
            for is_voice_val in is_voice.iter_mut().take(j).skip(i) {
                *is_voice_val = false;
            }
        }
        i = j;
    }

    // Pad voice edges
    let mut padded_voice = vec![false; n_win];
    for (w, &voice) in is_voice.iter().enumerate().take(n_win) {
        if !voice {
            continue;
        }
        let a = w.saturating_sub(pad_voice_windows);
        let b = (w + pad_voice_windows).min(n_win - 1);
        for padded_val in padded_voice.iter_mut().take(b + 1).skip(a) {
            *padded_val = true;
        }
    }

    let mut out = Vec::with_capacity(n_samples);
    let mut silence_count = 0;

    for (w, &pv) in padded_voice.iter().enumerate().take(n_win) {
        let start = w * win;
        let end = (start + win).min(n_samples);

        if pv {
            out.extend_from_slice(&samples[start..end]);
            silence_count = 0;
        } else {
            silence_count += 1;
            if silence_count <= pass_windows {
                out.extend_from_slice(&samples[start..end]);
            }
        }
    }

    if out.is_empty() {
        let keep = n_samples.min(SAMPLE_RATE as usize / 2);
        out.extend_from_slice(&samples[..keep]);
    }

    out
}
