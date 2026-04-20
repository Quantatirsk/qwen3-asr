//! BLAS/vDSP bindings, thread pool, and SIMD kernel dispatch.

pub mod generic;
#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "x86_64")]
pub mod avx;

use std::thread;

// BLAS extern bindings
#[cfg(all(feature = "blas", target_vendor = "apple"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

// vDSP/vForce bindings (macOS Accelerate)
#[cfg(all(feature = "vdsp", target_vendor = "apple"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn vDSP_dotpr(
        a: *const f32, a_stride: i32,
        b: *const f32, b_stride: i32,
        result: *mut f32,
        n: u64,
    );
    fn vDSP_vsmul(
        a: *const f32, a_stride: i32,
        scalar: *const f32,
        c: *mut f32, c_stride: i32,
        n: u64,
    );
    fn vDSP_vsma(
        a: *const f32, a_stride: i32,
        scalar: *const f32,
        b: *const f32, b_stride: i32,
        c: *mut f32, c_stride: i32,
        n: u64,
    );
    fn vvexpf(dst: *mut f32, src: *const f32, n: *const i32);
}

#[cfg(all(feature = "blas", not(target_vendor = "apple")))]
extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

#[cfg(feature = "blas")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(feature = "blas")]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(feature = "blas")]
const CBLAS_TRANS: i32 = 112;

// Verbose flag
static VERBOSE: AtomicI32 = AtomicI32::new(0);

// ========================================================================
// Profiling support
// ========================================================================

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::time::Instant;

static PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_profile(enabled: bool) {
    PROFILE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn is_profiling() -> bool {
    PROFILE_ENABLED.load(Ordering::Relaxed)
}

macro_rules! define_profile_counters {
    ($($name:ident),+) => {
        pub struct ProfileCounters {
            $(pub $name: (AtomicU64, AtomicU64),)+ // (total_ns, call_count)
        }

        impl ProfileCounters {
            pub const fn new() -> Self {
                ProfileCounters {
                    $($name: (AtomicU64::new(0), AtomicU64::new(0)),)+
                }
            }
        }

        impl Default for ProfileCounters {
            fn default() -> Self {
                Self::new()
            }
        }

        impl ProfileCounters {
            pub fn reset(&self) {
                $(
                    self.$name.0.store(0, Ordering::Relaxed);
                    self.$name.1.store(0, Ordering::Relaxed);
                )+
            }

            pub fn report(&self) {
                $(
                    let ns = self.$name.0.load(Ordering::Relaxed);
                    let calls = self.$name.1.load(Ordering::Relaxed);
                    if calls > 0 {
                        let ms = ns as f64 / 1_000_000.0;
                        let avg = ms / calls as f64;
                        eprintln!("[profile] {}: {:.1}ms ({} calls, {:.2}ms avg)",
                                  stringify!($name), ms, calls, avg);
                    }
                )+
            }
        }
    }
}

define_profile_counters!(
    rms_norm, layer_norm, gelu, swiglu,
    bf16_matvec, bf16_to_f32_conv, attention_bidir, attention_causal,
    sgemm, conv2d_op, rope, add_inplace_op
);

pub static PROF: ProfileCounters = ProfileCounters::new();

pub struct ProfileGuard {
    start: Instant,
    counter: &'static (AtomicU64, AtomicU64),
}

impl ProfileGuard {
    #[inline]
    pub fn new(counter: &'static (AtomicU64, AtomicU64)) -> Option<Self> {
        if PROFILE_ENABLED.load(Ordering::Relaxed) {
            Some(ProfileGuard { start: Instant::now(), counter })
        } else {
            None
        }
    }
}

impl Drop for ProfileGuard {
    #[inline]
    fn drop(&mut self) {
        let ns = self.start.elapsed().as_nanos() as u64;
        self.counter.0.fetch_add(ns, Ordering::Relaxed);
        self.counter.1.fetch_add(1, Ordering::Relaxed);
    }
}

// Convenience: unused ProfileTimer alias removed

pub fn profile_reset() { PROF.reset(); }
pub fn profile_report() { PROF.report(); }

pub fn set_verbose(v: i32) {
    VERBOSE.store(v, Ordering::Relaxed);
}

pub fn verbose() -> i32 {
    VERBOSE.load(Ordering::Relaxed)
}

// ========================================================================
// Thread Pool (persistent, mutex+condvar, matches C approach)
// ========================================================================

use std::sync::{Mutex, Condvar, Arc, OnceLock};

const MAX_THREADS: usize = 16;

struct ThreadPool {
    // Mutex+condvar only used as slow-path fallback when spin-wait misses
    state: Mutex<bool>, // shutdown flag only
    work_cv: Condvar,
    // All dispatch data is lock-free via atomics
    gen_atomic: AtomicU64,
    done_atomic: AtomicUsize,
    fn_ptr_atomic: AtomicUsize,
    fn_call_atomic: AtomicUsize,
    n_threads_atomic: AtomicUsize,
}

static THREAD_POOL: OnceLock<Arc<ThreadPool>> = OnceLock::new();

fn get_pool() -> &'static Arc<ThreadPool> {
    THREAD_POOL.get_or_init(|| {
        Arc::new(ThreadPool {
            state: Mutex::new(false),
            work_cv: Condvar::new(),
            gen_atomic: AtomicU64::new(0),
            done_atomic: AtomicUsize::new(0),
            fn_ptr_atomic: AtomicUsize::new(0),
            fn_call_atomic: AtomicUsize::new(0),
            n_threads_atomic: AtomicUsize::new(1),
        })
    })
}

fn pool_worker(pool: Arc<ThreadPool>, tid: usize) {
    let mut last_gen: u64 = 0;
    loop {
        // Fast path: spin briefly on atomic generation counter
        let mut found = false;
        for _ in 0..512 {
            let gen = pool.gen_atomic.load(Ordering::Acquire);
            if gen != last_gen {
                last_gen = gen;
                found = true;
                break;
            }
            core::hint::spin_loop();
        }

        if !found {
            // Slow path: condvar wait (mutex only protects shutdown flag)
            let mut shutdown = match pool.state.lock() {
                Ok(s) => s,
                Err(p) => p.into_inner(),
            };
            while !*shutdown && pool.gen_atomic.load(Ordering::Relaxed) == last_gen {
                shutdown = match pool.work_cv.wait(shutdown) {
                    Ok(s) => s,
                    Err(p) => p.into_inner(),
                };
            }
            if *shutdown {
                return;
            }
            last_gen = pool.gen_atomic.load(Ordering::Acquire);
        }

        // Read dispatch data from atomics (ordered by gen_atomic Acquire)
        let fn_ptr = pool.fn_ptr_atomic.load(Ordering::Relaxed) as *const ();
        let fn_call: fn(*const (), usize, usize) = unsafe {
            core::mem::transmute(pool.fn_call_atomic.load(Ordering::Relaxed))
        };
        let n_threads = pool.n_threads_atomic.load(Ordering::Relaxed);

        fn_call(fn_ptr, tid, n_threads);
        pool.done_atomic.fetch_add(1, Ordering::Release);
    }
}

static SPAWNED_THREADS: AtomicUsize = AtomicUsize::new(0);

fn ensure_workers(pool: &Arc<ThreadPool>, n_threads: usize) {
    let spawned = SPAWNED_THREADS.load(Ordering::Relaxed);
    if spawned >= n_threads - 1 {
        return;
    }
    let start = spawned + 1;
    for tid in start..n_threads {
        let p = pool.clone();
        thread::Builder::new()
            .name(format!("qwen-worker-{}", tid))
            .spawn(move || pool_worker(p, tid))
            .expect("failed to spawn worker thread");
    }
    SPAWNED_THREADS.store(n_threads - 1, Ordering::Relaxed);
}

static THREAD_POOL_THREADS: AtomicUsize = AtomicUsize::new(1);

pub fn set_threads(n: usize) {
    let n = n.clamp(1, MAX_THREADS);
    THREAD_POOL_THREADS.store(n, Ordering::Relaxed);
    if n > 1 {
        let pool = get_pool();
        ensure_workers(pool, n);
    }
    if verbose() >= 2 {
        eprintln!("Thread pool: {} threads", n);
    }
}

pub fn get_num_threads() -> usize {
    THREAD_POOL_THREADS.load(Ordering::Relaxed)
}

pub fn get_num_cpus() -> usize {
    // On Apple Silicon, prefer performance cores only (E-cores bottleneck parallel_for).
    #[cfg(target_os = "macos")]
    {
        let perf_cores = get_perf_core_count();
        if perf_cores > 0 {
            return perf_cores;
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(target_os = "macos")]
fn get_perf_core_count() -> usize {
    extern "C" {
        fn sysctlbyname(name: *const i8, oldp: *mut libc::c_void, oldlenp: *mut usize,
                        newp: *const libc::c_void, newlen: usize) -> i32;
    }
    let name = c"hw.perflevel0.physicalcpu";
    let mut val: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    let ret = unsafe {
        sysctlbyname(name.as_ptr(), &mut val as *mut i32 as *mut libc::c_void,
                     &mut len, std::ptr::null(), 0)
    };
    if ret == 0 && val > 0 { val as usize } else { 0 }
}

/// Run a closure in parallel using the persistent thread pool.
/// The closure takes (thread_id, n_threads).
fn parallel_for<F: Fn(usize, usize) + Send + Sync>(f: F) {
    let n_threads = get_num_threads();
    if n_threads <= 1 {
        f(0, 1);
        return;
    }

    let pool = get_pool();

    // Trampoline: cast *const () back to &F and call it
    fn trampoline<F: Fn(usize, usize) + Send + Sync>(ptr: *const (), tid: usize, nt: usize) {
        let f = unsafe { &*(ptr as *const F) };
        f(tid, nt);
    }

    // Publish dispatch data via atomics (Relaxed OK: gen_atomic Release provides ordering)
    pool.done_atomic.store(0, Ordering::Relaxed);
    pool.fn_ptr_atomic.store(&f as *const F as *const () as usize, Ordering::Relaxed);
    pool.fn_call_atomic.store(trampoline::<F> as usize, Ordering::Relaxed);
    pool.n_threads_atomic.store(n_threads, Ordering::Relaxed);
    // Release: ensures all stores above are visible to workers that Acquire gen_atomic
    pool.gen_atomic.fetch_add(1, Ordering::Release);

    // Wake workers that fell through to condvar wait
    // Lock scope is minimal: just notify, no data to write
    {
        let _guard = match pool.state.lock() {
            Ok(s) => s,
            Err(p) => p.into_inner(),
        };
        pool.work_cv.notify_all();
    }

    // Main thread does tid=0
    f(0, n_threads);

    // Wait for workers: spin on atomic done counter
    let expected = n_threads - 1;
    loop {
        if pool.done_atomic.load(Ordering::Acquire) >= expected {
            break;
        }
        core::hint::spin_loop();
    }
}

// ========================================================================
// Dispatch helpers - pick NEON/AVX/generic at compile time
// ========================================================================

#[inline]
pub fn bf16_to_f32(bf16: u16) -> f32 {
    f32::from_bits((bf16 as u32) << 16)
}

pub fn bf16_to_f32_buf(dst: &mut [f32], src: &[u16]) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::bf16_to_f32_buf(dst, src); } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::bf16_to_f32_buf(dst, src); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for i in 0..src.len() {
        dst[i] = bf16_to_f32(src[i]);
    }
}

fn bf16_matvec_fused(y: &mut [f32], x: &[f32], w_bf16: *const u16, bias: Option<&[f32]>, in_dim: usize, out_dim: usize) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim); } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    generic::bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim);
}

fn argmax_bf16_range(x: &[f32], w_bf16: *const u16, in_dim: usize, start: usize, end: usize) -> (usize, f32) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::argmax_bf16_range(x, w_bf16, in_dim, start, end) } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::argmax_bf16_range(x, w_bf16, in_dim, start, end) } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    generic::argmax_bf16_range(x, w_bf16, in_dim, start, end)
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
    {
        let mut result = 0.0f32;
        unsafe { vDSP_dotpr(a.as_ptr(), 1, b.as_ptr(), 1, &mut result, n as u64); }
        result
    }

    #[cfg(all(target_arch = "aarch64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { neon::dot_f32(a, b, n) } }

    #[cfg(all(target_arch = "x86_64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { avx::dot_f32(a, b, n) } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", all(feature = "vdsp", target_vendor = "apple"))))]
    generic::dot_f32(a, b, n)
}

#[inline]
pub fn vec_scale_inplace(dst: &mut [f32], scale: f32, n: usize) {
    #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
    {
        unsafe { vDSP_vsmul(dst.as_ptr(), 1, &scale, dst.as_mut_ptr(), 1, n as u64); }
    }

    #[cfg(all(target_arch = "aarch64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { neon::vec_scale_inplace(dst, scale, n); } }

    #[cfg(all(target_arch = "x86_64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { avx::vec_scale_inplace(dst, scale, n); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", all(feature = "vdsp", target_vendor = "apple"))))]
    generic::vec_scale_inplace(dst, scale, n);
}

#[inline]
pub fn vec_axpy_inplace(dst: &mut [f32], src: &[f32], alpha: f32, n: usize) {
    #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
    {
        unsafe { vDSP_vsma(src.as_ptr(), 1, &alpha, dst.as_ptr(), 1, dst.as_mut_ptr(), 1, n as u64); }
    }

    #[cfg(all(target_arch = "aarch64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { neon::vec_axpy_inplace(dst, src, alpha, n); } }

    #[cfg(all(target_arch = "x86_64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { avx::vec_axpy_inplace(dst, src, alpha, n); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", all(feature = "vdsp", target_vendor = "apple"))))]
    generic::vec_axpy_inplace(dst, src, alpha, n);
}

#[inline]
pub fn vec_scale_add(dst: &mut [f32], src: &[f32], correction: f32, n: usize) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::vec_scale_add(dst, src, correction, n); } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::vec_scale_add(dst, src, correction, n); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    generic::vec_scale_add(dst, src, correction, n);
}

// ========================================================================
// Basic Operations
// ========================================================================

pub fn add_inplace(a: &mut [f32], b: &[f32], n: usize) {
    let _pg = ProfileGuard::new(&PROF.add_inplace_op);
    for i in 0..n { a[i] += b[i]; }
}

// ========================================================================
// Matrix Operations
// ========================================================================

/// C = A @ B (no transpose): A[M,K], B[K,N], C[M,N]
pub fn matmul_nn(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    #[cfg(feature = "blas")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0, c.as_mut_ptr(), n as i32,
        );
    }

    #[cfg(not(feature = "blas"))]
    {
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f32;
                for ki in 0..k {
                    sum += a[mi * k + ki] * b[ki * n + ni];
                }
                c[mi * n + ni] = sum;
            }
        }
    }
}

/// C = A @ B^T: A[M,K], B[N,K], C[M,N]
pub fn matmul_t(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    #[cfg(feature = "blas")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), k as i32,
            0.0, c.as_mut_ptr(), n as i32,
        );
    }

    #[cfg(not(feature = "blas"))]
    {
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f32;
                for ki in 0..k {
                    sum += a[mi * k + ki] * b[ni * k + ki];
                }
                c[mi * n + ni] = sum;
            }
        }
    }
}

/// y = x @ W^T + b: x[seq,in], W[out,in], b[out], y[seq,out]
pub fn linear(y: &mut [f32], x: &[f32], w: &[f32], b: Option<&[f32]>, seq_len: usize, in_dim: usize, out_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.sgemm);
    #[cfg(feature = "blas")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
            seq_len as i32, out_dim as i32, in_dim as i32,
            1.0, x.as_ptr(), in_dim as i32,
            w.as_ptr(), in_dim as i32,
            0.0, y.as_mut_ptr(), out_dim as i32,
        );
        if let Some(b) = b {
            for s in 0..seq_len {
                for o in 0..out_dim {
                    y[s * out_dim + o] += b[o];
                }
            }
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        for s in 0..seq_len {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let w_row = &w[o * in_dim..(o + 1) * in_dim];
                let mut sum = b.map_or(0.0, |b| b[o]);
                for i in 0..in_dim {
                    sum += x_row[i] * w_row[i];
                }
                y[s * out_dim + o] = sum;
            }
        }
    }
}

pub fn linear_nobias(y: &mut [f32], x: &[f32], w: &[f32], seq_len: usize, in_dim: usize, out_dim: usize) {
    linear(y, x, w, None, seq_len, in_dim, out_dim);
}

/// y += bias + x @ w.T  (accumulate into existing y, fusing residual add)
pub fn linear_accumulate(y: &mut [f32], x: &[f32], w: &[f32], b: Option<&[f32]>, seq_len: usize, in_dim: usize, out_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.sgemm);
    #[cfg(feature = "blas")]
    unsafe {
        // Add bias to y first (y already has residual)
        if let Some(b) = b {
            for s in 0..seq_len {
                let row = &mut y[s * out_dim..(s + 1) * out_dim];
                for o in 0..out_dim {
                    row[o] += b[o];
                }
            }
        }
        // y = 1.0 * x @ w.T + 1.0 * y  (accumulate matmul into y)
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
            seq_len as i32, out_dim as i32, in_dim as i32,
            1.0, x.as_ptr(), in_dim as i32,
            w.as_ptr(), in_dim as i32,
            1.0, y.as_mut_ptr(), out_dim as i32,
        );
    }

    #[cfg(not(feature = "blas"))]
    {
        for s in 0..seq_len {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let w_row = &w[o * in_dim..(o + 1) * in_dim];
                let mut sum = b.map_or(0.0, |bb| bb[o]);
                for i in 0..in_dim {
                    sum += x_row[i] * w_row[i];
                }
                y[s * out_dim + o] += sum;
            }
        }
    }
}

fn bf16_to_f32_view(src: *const u16, n: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; n];
    let src_slice = unsafe { std::slice::from_raw_parts(src, n) };
    bf16_to_f32_buf(&mut buf, src_slice);
    buf
}

/// Threaded bf16 matvec
fn bf16_matvec_threaded(y: &mut [f32], x: &[f32], w_bf16: *const u16, bias: Option<&[f32]>, in_dim: usize, out_dim: usize) {
    let n_threads = get_num_threads();
    if n_threads <= 1 {
        bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim);
        return;
    }

    let y_ptr = y.as_mut_ptr();
    let x_ptr = x.as_ptr();
    let w_ptr = w_bf16;
    let bias_ptr = bias.map(|b| b.as_ptr());

    // SAFETY: Each thread writes to non-overlapping segments of y
    let y_send = y_ptr as usize;
    let x_send = x_ptr as usize;
    let w_send = w_ptr as usize;
    let bias_send = bias_ptr.map(|p| p as usize);

    parallel_for(|tid, nt| {
        let chunk = out_dim.div_ceil(nt);
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }

        let y_local = unsafe { std::slice::from_raw_parts_mut((y_send as *mut f32).add(start), end - start) };
        let x_local = unsafe { std::slice::from_raw_parts(x_send as *const f32, in_dim) };
        let w_local = unsafe { (w_send as *const u16).add(start * in_dim) };
        let bias_local = bias_send.map(|p| unsafe { std::slice::from_raw_parts((p as *const f32).add(start), end - start) });

        bf16_matvec_fused(y_local, x_local, w_local, bias_local, in_dim, end - start);
    });
}

/// Like linear_nobias_bf16 for seq_len=1, but ADDS to the destination: y[i] += W[i] @ x.
/// Achieves fused residual add by passing y as its own "bias".
pub fn linear_nobias_bf16_addto(y: &mut [f32], x: &[f32], w_bf16: *const u16, in_dim: usize, out_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    // SAFETY: bf16_matvec_fused reads bias[i] before writing y[i], so aliasing y as bias is safe.
    let bias = unsafe { std::slice::from_raw_parts(y.as_ptr(), out_dim) };
    bf16_matvec_threaded(y, x, w_bf16, Some(bias), in_dim, out_dim);
}

pub fn linear_nobias_bf16(y: &mut [f32], x: &[f32], w_bf16: *const u16, seq_len: usize, in_dim: usize, out_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    if seq_len == 1 {
        bf16_matvec_threaded(y, x, w_bf16, None, in_dim, out_dim);
        return;
    }
    let w_f32 = bf16_to_f32_view(w_bf16, out_dim * in_dim);
    linear_nobias(y, x, &w_f32, seq_len, in_dim, out_dim);
}

/// Like linear_nobias_bf16 but reuses a caller-provided scratch buffer for bf16→f32 conversion.
/// # Safety
/// Caller must ensure w_bf16 points to at least out_dim * in_dim valid bf16 values.
pub unsafe fn linear_nobias_bf16_scratch(y: &mut [f32], x: &[f32], w_bf16: *const u16, seq_len: usize, in_dim: usize, out_dim: usize, scratch: &mut [f32]) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    if seq_len == 1 {
        bf16_matvec_threaded(y, x, w_bf16, None, in_dim, out_dim);
        return;
    }
    let n = out_dim * in_dim;
    let src = unsafe { std::slice::from_raw_parts(w_bf16, n) };
    bf16_to_f32_buf(&mut scratch[..n], src);
    linear_nobias(y, x, &scratch[..n], seq_len, in_dim, out_dim);
}

pub fn linear_bf16(y: &mut [f32], x: &[f32], w_bf16: *const u16, b: Option<&[f32]>, seq_len: usize, in_dim: usize, out_dim: usize) {
    if seq_len == 1 {
        bf16_matvec_threaded(y, x, w_bf16, b, in_dim, out_dim);
        return;
    }
    let w_f32 = bf16_to_f32_view(w_bf16, out_dim * in_dim);
    linear(y, x, &w_f32, b, seq_len, in_dim, out_dim);
}

/// Fused Q/K/V matvec for single-token decode
#[allow(clippy::too_many_arguments)]
pub fn linear_nobias_bf16_qkv(
    q: &mut [f32], k: &mut [f32], v: &mut [f32], x: &[f32],
    wq: *const u16, wk: *const u16, wv: *const u16,
    in_dim: usize, q_dim: usize, kv_dim: usize,
) {
    let n_threads = get_num_threads();
    if n_threads <= 1 {
        bf16_matvec_fused(q, x, wq, None, in_dim, q_dim);
        bf16_matvec_fused(k, x, wk, None, in_dim, kv_dim);
        bf16_matvec_fused(v, x, wv, None, in_dim, kv_dim);
        return;
    }

    let total_dim = q_dim + 2 * kv_dim;
    let q_ptr = q.as_mut_ptr() as usize;
    let k_ptr = k.as_mut_ptr() as usize;
    let v_ptr = v.as_mut_ptr() as usize;
    let x_ptr = x.as_ptr() as usize;
    let wq_ptr = wq as usize;
    let wk_ptr = wk as usize;
    let wv_ptr = wv as usize;

    parallel_for(|tid, nt| {
        let chunk = total_dim.div_ceil(nt);
        let start = tid * chunk;
        let end = (start + chunk).min(total_dim);
        if start >= end { return; }

        let x_local = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, in_dim) };
        let q_end = q_dim;
        let k_end = q_end + kv_dim;

        // Q range
        if start < q_end {
            let s = start;
            let e = end.min(q_end);
            if s < e {
                let y_local = unsafe { std::slice::from_raw_parts_mut((q_ptr as *mut f32).add(s), e - s) };
                let w_local = unsafe { (wq_ptr as *const u16).add(s * in_dim) };
                bf16_matvec_fused(y_local, x_local, w_local, None, in_dim, e - s);
            }
        }

        // K range
        if end > q_end && start < k_end {
            let s = start.saturating_sub(q_end);
            let e_abs = end.min(k_end);
            let e = e_abs - q_end;
            if s < e {
                let y_local = unsafe { std::slice::from_raw_parts_mut((k_ptr as *mut f32).add(s), e - s) };
                let w_local = unsafe { (wk_ptr as *const u16).add(s * in_dim) };
                bf16_matvec_fused(y_local, x_local, w_local, None, in_dim, e - s);
            }
        }

        // V range
        if end > k_end {
            let s = start.saturating_sub(k_end);
            let e_abs = end.min(total_dim);
            let e = e_abs - k_end;
            if s < e {
                let y_local = unsafe { std::slice::from_raw_parts_mut((v_ptr as *mut f32).add(s), e - s) };
                let w_local = unsafe { (wv_ptr as *const u16).add(s * in_dim) };
                bf16_matvec_fused(y_local, x_local, w_local, None, in_dim, e - s);
            }
        }
    });
}

/// Fused gate_up matvec + SwiGLU for single-token decode.
/// Computes: ffn_out[j] = silu(gate[j]) * up[j] where gate/up come from interleaved gate_up_fused matvec.
/// Keeps gate_up output in L1 cache for the SwiGLU operation.
pub fn linear_nobias_bf16_swiglu(
    ffn_out: &mut [f32],
    x: &[f32],
    gate_up_bf16: *const u16,
    in_dim: usize,
    intermediate: usize,
) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    let n_threads = get_num_threads();

    if n_threads <= 1 {
        // Single-threaded: compute gate_up, then SwiGLU inline
        let mut gate_buf = vec![0.0f32; 2 * intermediate];
        bf16_matvec_fused(&mut gate_buf, x, gate_up_bf16, None, in_dim, 2 * intermediate);
        for j in 0..intermediate {
            let g = gate_buf[2 * j];
            let u = gate_buf[2 * j + 1];
            ffn_out[j] = g / (1.0 + (-g).exp()) * u;
        }
        return;
    }

    let x_ptr = x.as_ptr() as usize;
    let w_ptr = gate_up_bf16 as usize;
    let ffn_ptr = ffn_out.as_mut_ptr() as usize;

    parallel_for(|tid, nt| {
        let chunk = intermediate.div_ceil(nt);
        let start = tid * chunk;
        let end = (start + chunk).min(intermediate);
        if start >= end { return; }
        let n_rows = end - start;

        let x_local = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, in_dim) };
        let w_local = unsafe { (w_ptr as *const u16).add(2 * start * in_dim) };

        // Compute gate_up for this chunk (thread-local stack buffer)
        let mut gate_up_local = vec![0.0f32; 2 * n_rows];
        bf16_matvec_fused(&mut gate_up_local, x_local, w_local, None, in_dim, 2 * n_rows);

        // Apply SwiGLU inline while data is hot in L1
        let ffn_local = unsafe { std::slice::from_raw_parts_mut((ffn_ptr as *mut f32).add(start), n_rows) };
        for j in 0..n_rows {
            let g = gate_up_local[2 * j];
            let u = gate_up_local[2 * j + 1];
            ffn_local[j] = g / (1.0 + (-g).exp()) * u;
        }
    });
}

/// INT8 threaded matvec: y = W_int8 @ x + bias  (x is f32, quantized on the fly)
fn int8_matvec_threaded(y: &mut [f32], x: &[f32], w_int8: &[i8], w_scales: &[f32], bias: Option<&[f32]>, in_dim: usize, out_dim: usize) {
    let (x_int8, x_scale) = quantize_f32_to_int8(x);
    let n_threads = get_num_threads();

    #[cfg(target_arch = "aarch64")]
    {
        if n_threads <= 1 {
            unsafe {
                neon::matvec_int8(y, x_int8.as_ptr(), x_scale, w_int8.as_ptr(), w_scales, bias, in_dim, out_dim);
            }
            return;
        }

        let x_int8_ptr = x_int8.as_ptr() as usize;
        let w_int8_ptr = w_int8.as_ptr() as usize;
        let w_scales_ptr = w_scales.as_ptr() as usize;
        let y_ptr = y.as_mut_ptr() as usize;
        let bias_ptr = bias.map(|b| b.as_ptr() as usize);

        parallel_for(|tid, nt| {
            let chunk = out_dim.div_ceil(nt);
            let start = tid * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end { return; }

            let y_local = unsafe { std::slice::from_raw_parts_mut((y_ptr as *mut f32).add(start), end - start) };
            let w_local = unsafe { (w_int8_ptr as *const i8).add(start * in_dim) };
            let w_scales_local = unsafe { std::slice::from_raw_parts((w_scales_ptr as *const f32).add(start), end - start) };
            let bias_local = bias_ptr.map(|p| unsafe { std::slice::from_raw_parts((p as *const f32).add(start), end - start) });

            unsafe {
                neon::matvec_int8(y_local, x_int8_ptr as *const i8, x_scale, w_local, w_scales_local, bias_local, in_dim, end - start);
            }
        });
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        if n_threads <= 1 || out_dim <= 1 {
            for row in 0..out_dim {
                let w_row = &w_int8[row * in_dim..(row + 1) * in_dim];
                let mut acc = 0i32;
                for col in 0..in_dim {
                    acc += (x_int8[col] as i32) * (w_row[col] as i32);
                }
                let mut value = (acc as f32) * x_scale * w_scales[row];
                if let Some(bias_values) = bias {
                    value += bias_values[row];
                }
                y[row] = value;
            }
            return;
        }

        let x_int8_ptr = x_int8.as_ptr() as usize;
        let w_int8_ptr = w_int8.as_ptr() as usize;
        let w_scales_ptr = w_scales.as_ptr() as usize;
        let y_ptr = y.as_mut_ptr() as usize;
        let bias_ptr = bias.map(|values| values.as_ptr() as usize);

        parallel_for(|tid, nt| {
            let chunk = out_dim.div_ceil(nt);
            let start = tid * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end {
                return;
            }

            for row in start..end {
                let w_row = unsafe {
                    std::slice::from_raw_parts(
                        (w_int8_ptr as *const i8).add(row * in_dim),
                        in_dim,
                    )
                };
                let x_local = unsafe {
                    std::slice::from_raw_parts(x_int8_ptr as *const i8, in_dim)
                };
                let mut acc = 0i32;
                for col in 0..in_dim {
                    acc += (x_local[col] as i32) * (w_row[col] as i32);
                }
                let mut value = (acc as f32)
                    * x_scale
                    * unsafe { *((w_scales_ptr as *const f32).add(row)) };
                if let Some(ptr) = bias_ptr {
                    value += unsafe { *((ptr as *const f32).add(row)) };
                }
                unsafe {
                    *((y_ptr as *mut f32).add(row)) = value;
                }
            }
        });
    }
}

/// INT8 fused QKV matvec for single-token decode
#[allow(clippy::too_many_arguments)]
pub fn linear_nobias_int8_qkv(
    q: &mut [f32], k: &mut [f32], v: &mut [f32], x: &[f32],
    wq_int8: &[i8], wq_scales: &[f32],
    wk_int8: &[i8], wk_scales: &[f32],
    wv_int8: &[i8], wv_scales: &[f32],
    in_dim: usize, q_dim: usize, kv_dim: usize,
) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    let (x_int8, x_scale) = quantize_f32_to_int8(x);
    let n_threads = get_num_threads();

    #[cfg(target_arch = "aarch64")]
    {
        if n_threads <= 1 {
            unsafe {
                neon::matvec_int8(q, x_int8.as_ptr(), x_scale, wq_int8.as_ptr(), wq_scales, None, in_dim, q_dim);
                neon::matvec_int8(k, x_int8.as_ptr(), x_scale, wk_int8.as_ptr(), wk_scales, None, in_dim, kv_dim);
                neon::matvec_int8(v, x_int8.as_ptr(), x_scale, wv_int8.as_ptr(), wv_scales, None, in_dim, kv_dim);
            }
            return;
        }

        let total_dim = q_dim + 2 * kv_dim;
        let q_ptr = q.as_mut_ptr() as usize;
        let k_ptr = k.as_mut_ptr() as usize;
        let v_ptr = v.as_mut_ptr() as usize;
        let x_int8_ptr = x_int8.as_ptr() as usize;
        let wq_ptr = wq_int8.as_ptr() as usize;
        let wk_ptr = wk_int8.as_ptr() as usize;
        let wv_ptr = wv_int8.as_ptr() as usize;
        let wq_scales_ptr = wq_scales.as_ptr() as usize;
        let wk_scales_ptr = wk_scales.as_ptr() as usize;
        let wv_scales_ptr = wv_scales.as_ptr() as usize;

        parallel_for(|tid, nt| {
            let chunk = total_dim.div_ceil(nt);
            let start = tid * chunk;
            let end = (start + chunk).min(total_dim);
            if start >= end { return; }

            let q_end = q_dim;
            let k_end = q_end + kv_dim;

            // Q range
            if start < q_end {
                let s = start;
                let e = end.min(q_end);
                if s < e {
                    let y_local = unsafe { std::slice::from_raw_parts_mut((q_ptr as *mut f32).add(s), e - s) };
                    let w_local = unsafe { (wq_ptr as *const i8).add(s * in_dim) };
                    let scales_local = unsafe { std::slice::from_raw_parts((wq_scales_ptr as *const f32).add(s), e - s) };
                    unsafe { neon::matvec_int8(y_local, x_int8_ptr as *const i8, x_scale, w_local, scales_local, None, in_dim, e - s); }
                }
            }
            // K range
            if start < k_end && end > q_end {
                let s = start.max(q_end) - q_end;
                let e = end.min(k_end) - q_end;
                if s < e {
                    let y_local = unsafe { std::slice::from_raw_parts_mut((k_ptr as *mut f32).add(s), e - s) };
                    let w_local = unsafe { (wk_ptr as *const i8).add(s * in_dim) };
                    let scales_local = unsafe { std::slice::from_raw_parts((wk_scales_ptr as *const f32).add(s), e - s) };
                    unsafe { neon::matvec_int8(y_local, x_int8_ptr as *const i8, x_scale, w_local, scales_local, None, in_dim, e - s); }
                }
            }
            // V range
            if end > k_end {
                let s = start.max(k_end) - k_end;
                let e = end.min(total_dim) - k_end;
                if s < e {
                    let y_local = unsafe { std::slice::from_raw_parts_mut((v_ptr as *mut f32).add(s), e - s) };
                    let w_local = unsafe { (wv_ptr as *const i8).add(s * in_dim) };
                    let scales_local = unsafe { std::slice::from_raw_parts((wv_scales_ptr as *const f32).add(s), e - s) };
                    unsafe { neon::matvec_int8(y_local, x_int8_ptr as *const i8, x_scale, w_local, scales_local, None, in_dim, e - s); }
                }
            }
        });
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let total_dim = q_dim + 2 * kv_dim;
        if n_threads <= 1 || total_dim <= 1 {
            for row in 0..q_dim {
                let weights = &wq_int8[row * in_dim..(row + 1) * in_dim];
                let mut acc = 0i32;
                for col in 0..in_dim {
                    acc += (x_int8[col] as i32) * (weights[col] as i32);
                }
                q[row] = (acc as f32) * x_scale * wq_scales[row];
            }
            for row in 0..kv_dim {
                let weights = &wk_int8[row * in_dim..(row + 1) * in_dim];
                let mut acc = 0i32;
                for col in 0..in_dim {
                    acc += (x_int8[col] as i32) * (weights[col] as i32);
                }
                k[row] = (acc as f32) * x_scale * wk_scales[row];
            }
            for row in 0..kv_dim {
                let weights = &wv_int8[row * in_dim..(row + 1) * in_dim];
                let mut acc = 0i32;
                for col in 0..in_dim {
                    acc += (x_int8[col] as i32) * (weights[col] as i32);
                }
                v[row] = (acc as f32) * x_scale * wv_scales[row];
            }
            return;
        }

        let total_dim = q_dim + 2 * kv_dim;
        let q_ptr = q.as_mut_ptr() as usize;
        let k_ptr = k.as_mut_ptr() as usize;
        let v_ptr = v.as_mut_ptr() as usize;
        let x_int8_ptr = x_int8.as_ptr() as usize;
        let wq_ptr = wq_int8.as_ptr() as usize;
        let wk_ptr = wk_int8.as_ptr() as usize;
        let wv_ptr = wv_int8.as_ptr() as usize;
        let wq_scales_ptr = wq_scales.as_ptr() as usize;
        let wk_scales_ptr = wk_scales.as_ptr() as usize;
        let wv_scales_ptr = wv_scales.as_ptr() as usize;

        parallel_for(|tid, nt| {
            let chunk = total_dim.div_ceil(nt);
            let start = tid * chunk;
            let end = (start + chunk).min(total_dim);
            if start >= end {
                return;
            }

            for row in start..end {
                let (out_ptr, local_row, weights_ptr, scales_ptr) = if row < q_dim {
                    (q_ptr as *mut f32, row, wq_ptr as *const i8, wq_scales_ptr as *const f32)
                } else if row < q_dim + kv_dim {
                    (
                        k_ptr as *mut f32,
                        row - q_dim,
                        wk_ptr as *const i8,
                        wk_scales_ptr as *const f32,
                    )
                } else {
                    (
                        v_ptr as *mut f32,
                        row - q_dim - kv_dim,
                        wv_ptr as *const i8,
                        wv_scales_ptr as *const f32,
                    )
                };

                let weights = unsafe {
                    std::slice::from_raw_parts(weights_ptr.add(local_row * in_dim), in_dim)
                };
                let x_local = unsafe {
                    std::slice::from_raw_parts(x_int8_ptr as *const i8, in_dim)
                };
                let mut acc = 0i32;
                for col in 0..in_dim {
                    acc += (x_local[col] as i32) * (weights[col] as i32);
                }
                let value = (acc as f32) * x_scale * unsafe { *scales_ptr.add(local_row) };
                unsafe {
                    *out_ptr.add(local_row) = value;
                }
            }
        });
    }
}

/// INT8 fused gate_up + SwiGLU
pub fn linear_nobias_int8_swiglu(
    ffn_out: &mut [f32], x: &[f32],
    w_int8: &[i8], w_scales: &[f32],
    in_dim: usize, intermediate: usize,
) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    let (x_int8, x_scale) = quantize_f32_to_int8(x);
    let n_threads = get_num_threads();

    #[cfg(target_arch = "aarch64")]
    {
        let x_int8_ptr = x_int8.as_ptr() as usize;
        let w_int8_ptr = w_int8.as_ptr() as usize;
        let w_scales_ptr = w_scales.as_ptr() as usize;
        let ffn_ptr = ffn_out.as_mut_ptr() as usize;

        if n_threads <= 1 {
            let mut gate_buf = vec![0.0f32; 2 * intermediate];
            unsafe {
                neon::matvec_int8(&mut gate_buf, x_int8.as_ptr(), x_scale, w_int8.as_ptr(), w_scales, None, in_dim, 2 * intermediate);
            }
            for j in 0..intermediate {
                let g = gate_buf[2 * j];
                let u = gate_buf[2 * j + 1];
                ffn_out[j] = g / (1.0 + (-g).exp()) * u;
            }
            return;
        }

        parallel_for(|tid, nt| {
            let chunk = intermediate.div_ceil(nt);
            let start = tid * chunk;
            let end = (start + chunk).min(intermediate);
            if start >= end { return; }
            let n_rows = end - start;

            let w_local = unsafe { (w_int8_ptr as *const i8).add(2 * start * in_dim) };
            let w_scales_local = unsafe { std::slice::from_raw_parts((w_scales_ptr as *const f32).add(2 * start), 2 * n_rows) };

            let mut gate_up_local = vec![0.0f32; 2 * n_rows];
            unsafe {
                neon::matvec_int8(&mut gate_up_local, x_int8_ptr as *const i8, x_scale, w_local, w_scales_local, None, in_dim, 2 * n_rows);
            }

            let ffn_local = unsafe { std::slice::from_raw_parts_mut((ffn_ptr as *mut f32).add(start), n_rows) };
            for j in 0..n_rows {
                let g = gate_up_local[2 * j];
                let u = gate_up_local[2 * j + 1];
                ffn_local[j] = g / (1.0 + (-g).exp()) * u;
            }
        });
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        if n_threads <= 1 || intermediate <= 1 {
            for row in 0..intermediate {
                let gate_weights = &w_int8[(2 * row) * in_dim..(2 * row + 1) * in_dim];
                let up_weights = &w_int8[(2 * row + 1) * in_dim..(2 * row + 2) * in_dim];
                let mut gate_acc = 0i32;
                let mut up_acc = 0i32;
                for col in 0..in_dim {
                    let x_value = x_int8[col] as i32;
                    gate_acc += x_value * (gate_weights[col] as i32);
                    up_acc += x_value * (up_weights[col] as i32);
                }
                let gate = (gate_acc as f32) * x_scale * w_scales[2 * row];
                let up = (up_acc as f32) * x_scale * w_scales[2 * row + 1];
                ffn_out[row] = gate / (1.0 + (-gate).exp()) * up;
            }
            return;
        }

        let x_int8_ptr = x_int8.as_ptr() as usize;
        let w_int8_ptr = w_int8.as_ptr() as usize;
        let w_scales_ptr = w_scales.as_ptr() as usize;
        let out_ptr = ffn_out.as_mut_ptr() as usize;

        parallel_for(|tid, nt| {
            let chunk = intermediate.div_ceil(nt);
            let start = tid * chunk;
            let end = (start + chunk).min(intermediate);
            if start >= end {
                return;
            }

            for row in start..end {
                let gate_weights = unsafe {
                    std::slice::from_raw_parts(
                        (w_int8_ptr as *const i8).add((2 * row) * in_dim),
                        in_dim,
                    )
                };
                let up_weights = unsafe {
                    std::slice::from_raw_parts(
                        (w_int8_ptr as *const i8).add((2 * row + 1) * in_dim),
                        in_dim,
                    )
                };
                let x_local = unsafe {
                    std::slice::from_raw_parts(x_int8_ptr as *const i8, in_dim)
                };
                let mut gate_acc = 0i32;
                let mut up_acc = 0i32;
                for col in 0..in_dim {
                    let x_value = x_local[col] as i32;
                    gate_acc += x_value * (gate_weights[col] as i32);
                    up_acc += x_value * (up_weights[col] as i32);
                }
                let gate = (gate_acc as f32)
                    * x_scale
                    * unsafe { *((w_scales_ptr as *const f32).add(2 * row)) };
                let up = (up_acc as f32)
                    * x_scale
                    * unsafe { *((w_scales_ptr as *const f32).add(2 * row + 1)) };
                unsafe {
                    *((out_ptr as *mut f32).add(row)) = gate / (1.0 + (-gate).exp()) * up;
                }
            }
        });
    }
}

/// INT8 matvec with fused residual add: y += W_int8 @ x  (y acts as bias)
pub fn linear_nobias_int8_addto(y: &mut [f32], x: &[f32], w_int8: &[i8], w_scales: &[f32], in_dim: usize, out_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    let bias = unsafe { std::slice::from_raw_parts(y.as_ptr(), out_dim) };
    int8_matvec_threaded(y, x, w_int8, w_scales, Some(bias), in_dim, out_dim);
}

pub fn matmul_t_bf16(c: &mut [f32], a: &[f32], b_bf16: *const u16, m: usize, k: usize, n: usize) {
    if m == 1 {
        bf16_matvec_threaded(c, a, b_bf16, None, k, n);
    } else {
        let b_f32 = bf16_to_f32_view(b_bf16, n * k);
        matmul_t(c, a, &b_f32, m, k, n);
    }
}

// ========================================================================
// 2D Convolution (im2col + BLAS sgemm)
// ========================================================================

#[allow(clippy::too_many_arguments)]
fn im2col(input: &[f32], cols: &mut [f32], c_in: usize, h_in: usize, w_in: usize,
          kh: usize, kw: usize, stride: usize, padding: usize, h_out: usize, w_out: usize) {
    let col_len = h_out * w_out;
    for ic in 0..c_in {
        for ki in 0..kh {
            for kj in 0..kw {
                let col_row = (ic * kh + ki) * kw + kj;
                for oh in 0..h_out {
                    let ih = oh * stride + ki;
                    let ih = ih as isize - padding as isize;
                    for ow in 0..w_out {
                        let iw = ow * stride + kj;
                        let iw = iw as isize - padding as isize;
                        let val = if ih >= 0 && (ih as usize) < h_in && iw >= 0 && (iw as usize) < w_in {
                            input[ic * h_in * w_in + ih as usize * w_in + iw as usize]
                        } else {
                            0.0
                        };
                        cols[col_row * col_len + oh * w_out + ow] = val;
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d(out: &mut [f32], input: &[f32], weight: &[f32], bias: Option<&[f32]>,
              c_in: usize, c_out: usize, h_in: usize, w_in: usize,
              kh: usize, kw: usize, stride: usize, padding: usize) {
    let _pg = ProfileGuard::new(&PROF.conv2d_op);
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    let patch_size = c_in * kh * kw;
    let spatial_out = h_out * w_out;

    let mut cols = vec![0.0f32; patch_size * spatial_out];

    // Thread im2col across col_rows (each row is independent)
    let n_threads = get_num_threads();
    if n_threads > 1 && patch_size >= 16 {
        let input_ptr = input.as_ptr() as usize;
        let cols_ptr = cols.as_mut_ptr() as usize;
        parallel_for(|tid, nt| {
            let chunk = patch_size.div_ceil(nt);
            let start = tid * chunk;
            let end = (start + chunk).min(patch_size);
            if start >= end { return; }
            for col_row in start..end {
                let ic = col_row / (kh * kw);
                let rem = col_row % (kh * kw);
                let ki = rem / kw;
                let kj = rem % kw;
                for oh in 0..h_out {
                    let ih = (oh * stride + ki) as isize - padding as isize;
                    for ow in 0..w_out {
                        let iw = (ow * stride + kj) as isize - padding as isize;
                        let val = if ih >= 0 && (ih as usize) < h_in && iw >= 0 && (iw as usize) < w_in {
                            unsafe { *(input_ptr as *const f32).add(ic * h_in * w_in + ih as usize * w_in + iw as usize) }
                        } else {
                            0.0
                        };
                        unsafe { *(cols_ptr as *mut f32).add(col_row * spatial_out + oh * w_out + ow) = val; }
                    }
                }
            }
        });
    } else {
        im2col(input, &mut cols, c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);
    }

    // GEMM: weight[c_out, patch_size] @ cols[patch_size, spatial_out] = out[c_out, spatial_out]
    #[cfg(feature = "blas")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            c_out as i32, spatial_out as i32, patch_size as i32,
            1.0, weight.as_ptr(), patch_size as i32,
            cols.as_ptr(), spatial_out as i32,
            0.0, out.as_mut_ptr(), spatial_out as i32,
        );
    }

    #[cfg(not(feature = "blas"))]
    {
        for oc in 0..c_out {
            for s in 0..spatial_out {
                let mut sum = 0.0f32;
                for p in 0..patch_size {
                    sum += weight[oc * patch_size + p] * cols[p * spatial_out + s];
                }
                out[oc * spatial_out + s] = sum;
            }
        }
    }

    if let Some(bias) = bias {
        for oc in 0..c_out {
            let b = bias[oc];
            for s in 0..spatial_out {
                out[oc * spatial_out + s] += b;
            }
        }
    }
}

// ========================================================================
// Normalization
// ========================================================================

pub fn layer_norm(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32],
                  seq_len: usize, hidden: usize, eps: f32) {
    let _pg = ProfileGuard::new(&PROF.layer_norm);
    for s in 0..seq_len {
        let x_row = &x[s * hidden..(s + 1) * hidden];
        let out_row = &mut out[s * hidden..(s + 1) * hidden];

        #[cfg(target_arch = "aarch64")]
        { unsafe { neon::layer_norm_row(out_row, x_row, weight, bias, hidden, eps); } continue; }

        #[cfg(target_arch = "x86_64")]
        { unsafe { avx::layer_norm_row(out_row, x_row, weight, bias, hidden, eps); } continue; }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mean: f32 = x_row.iter().sum::<f32>() / hidden as f32;

            let var: f32 = x_row.iter().map(|&v| {
                let d = v - mean;
                d * d
            }).sum::<f32>() / hidden as f32;

            let inv_std = 1.0 / (var + eps).sqrt();

            for i in 0..hidden {
                out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
            }
        }
    }
}

pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], seq_len: usize, hidden: usize, eps: f32) {
    let _pg = ProfileGuard::new(&PROF.rms_norm);
    for s in 0..seq_len {
        let x_row = &x[s * hidden..(s + 1) * hidden];
        let out_row = &mut out[s * hidden..(s + 1) * hidden];

        #[cfg(target_arch = "aarch64")]
        { unsafe { neon::rms_norm_row(out_row, x_row, weight, hidden, eps); } continue; }

        #[cfg(target_arch = "x86_64")]
        { unsafe { avx::rms_norm_row(out_row, x_row, weight, hidden, eps); } continue; }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let sum_sq: f32 = x_row.iter().map(|&v| v * v).sum();
            let rms_inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
            for i in 0..hidden {
                out_row[i] = x_row[i] * rms_inv * weight[i];
            }
        }
    }
}

pub fn rms_norm_per_head(x: &mut [f32], weight: &[f32], seq_len: usize, n_heads: usize, head_dim: usize, eps: f32) {
    let hidden = n_heads * head_dim;
    for s in 0..seq_len {
        for h in 0..n_heads {
            let off = s * hidden + h * head_dim;

            #[cfg(target_arch = "aarch64")]
            {
                let vec = &mut x[off..off + head_dim];
                unsafe { neon::rms_norm_inplace(vec, weight, head_dim, eps); }
                continue;
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                let vec = &mut x[off..off + head_dim];
                let sum_sq: f32 = vec.iter().map(|&v| v * v).sum();
                let rms_inv = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();
                for d in 0..head_dim {
                    vec[d] = vec[d] * rms_inv * weight[d];
                }
            }
        }
    }
}

// ========================================================================
// Activation Functions
// ========================================================================

pub fn silu(x: &mut [f32], n: usize) {
    for val in x.iter_mut().take(n) {
        *val = *val / (1.0 + (-*val).exp());
    }
}

pub fn gelu(x: &mut [f32], n: usize) {
    let _pg = ProfileGuard::new(&PROF.gelu);
    let n_threads = get_num_threads();
    // Thread GELU for large buffers (encoder FFN: ~320K floats)
    if n_threads > 1 && n > 4096 {
        let x_ptr = x.as_mut_ptr() as usize;
        parallel_for(|tid, nt| {
            let chunk = n.div_ceil(nt);
            let start = tid * chunk;
            let end = (start + chunk).min(n);
            if start >= end { return; }
            let x_local = unsafe { std::slice::from_raw_parts_mut((x_ptr as *mut f32).add(start), end - start) };
            #[cfg(target_arch = "aarch64")]
            unsafe { neon::gelu_inplace(x_local, end - start); }
            #[cfg(target_arch = "x86_64")]
            unsafe { avx::gelu_inplace(x_local, end - start); }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            for i in 0..(end - start) {
                let val = x_local[i];
                let x3 = val * val * val;
                let inner = 0.7978845608028654f32 * (val + 0.044715 * x3);
                x_local[i] = 0.5 * val * (1.0 + inner.tanh());
            }
        });
        return;
    }
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::gelu_inplace(x, n); } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::gelu_inplace(x, n); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for i in 0..n {
        let val = x[i];
        let x3 = val * val * val;
        let inner = 0.7978845608028654f32 * (val + 0.044715 * x3);
        x[i] = 0.5 * val * (1.0 + inner.tanh());
    }
}

pub fn swiglu_multiply(out: &mut [f32], gate_up: &[f32], seq_len: usize, intermediate: usize) {
    let _pg = ProfileGuard::new(&PROF.swiglu);
    let total = seq_len * intermediate;
    let n_threads = get_num_threads();

    // Thread SwiGLU for large prefill buffers
    if n_threads > 1 && total > 4096 {
        let out_ptr = out.as_mut_ptr() as usize;
        let gu_ptr = gate_up.as_ptr() as usize;
        parallel_for(|tid, nt| {
            let chunk = seq_len.div_ceil(nt);
            let start = tid * chunk;
            let end = (start + chunk).min(seq_len);
            if start >= end { return; }
            for s in start..end {
                let gu = unsafe { std::slice::from_raw_parts((gu_ptr as *const f32).add(s * 2 * intermediate), 2 * intermediate) };
                let o = unsafe { std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(s * intermediate), intermediate) };
                #[cfg(target_arch = "aarch64")]
                { unsafe { neon::swiglu_interleaved(o, gu, intermediate); } continue; }
                #[cfg(target_arch = "x86_64")]
                { unsafe { avx::swiglu_interleaved(o, gu, intermediate); } continue; }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                for j in 0..intermediate {
                    let g = gu[2 * j];
                    let u = gu[2 * j + 1];
                    o[j] = g / (1.0 + (-g).exp()) * u;
                }
            }
        });
        return;
    }

    for s in 0..seq_len {
        let gu = &gate_up[s * 2 * intermediate..s * 2 * intermediate + 2 * intermediate];
        let o = &mut out[s * intermediate..(s + 1) * intermediate];

        #[cfg(target_arch = "aarch64")]
        { unsafe { neon::swiglu_interleaved(o, gu, intermediate); } continue; }

        #[cfg(target_arch = "x86_64")]
        { unsafe { avx::swiglu_interleaved(o, gu, intermediate); } continue; }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        for j in 0..intermediate {
            let g = gu[2 * j];
            let u = gu[2 * j + 1];
            let g_silu = g / (1.0 + (-g).exp());
            o[j] = g_silu * u;
        }
    }
}

pub fn softmax(x: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for val in row.iter_mut().take(cols) {
            *val -= max_val;
        }

        #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
        {
            let n = cols as i32;
            unsafe { vvexpf(row.as_mut_ptr(), row.as_ptr(), &n); }
        }
        #[cfg(not(all(feature = "vdsp", target_vendor = "apple")))]
        {
            #[cfg(target_arch = "aarch64")]
            { unsafe { neon::exp_inplace(row); } }

            #[cfg(target_arch = "x86_64")]
            { unsafe { avx::exp_inplace(row); } }

            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            for c in 0..cols {
                row[c] = row[c].exp();
            }
        }

        let mut sum = 0.0f32;
        for val in row.iter().take(cols) {
            sum += val;
        }
        let inv_sum = 1.0 / sum;
        for val in row.iter_mut().take(cols) {
            *val *= inv_sum;
        }
    }
}

// ========================================================================
// Attention Operations
// ========================================================================

#[allow(clippy::too_many_arguments)]
fn bidirectional_attention_heads(out: &mut [f32], q: &[f32], k: &[f32], v: &[f32],
                                  n_heads: usize, head_dim: usize, scale: f32,
                                  window_starts: &[i32], n_windows: usize,
                                  head_start: usize, head_end: usize) {
    let hidden = n_heads * head_dim;

    for h in head_start..head_end {
        for w in 0..n_windows {
            let ws = window_starts[w] as usize;
            let we = window_starts[w + 1] as usize;

            for i in ws..we {
                let q_off = i * hidden + h * head_dim;
                let q_row = &q[q_off..q_off + head_dim];
                let o_row = &mut out[i * hidden + h * head_dim..i * hidden + h * head_dim + head_dim];

                let mut max_score = -1e30f32;
                let mut sum_exp = 0.0f32;
                for val in o_row.iter_mut().take(head_dim) { *val = 0.0; }

                for j in ws..we {
                    let k_off = j * hidden + h * head_dim;
                    let v_off = j * hidden + h * head_dim;
                    let k_row = &k[k_off..k_off + head_dim];
                    let v_row = &v[v_off..v_off + head_dim];

                    let score = dot_f32(q_row, k_row, head_dim) * scale;

                    if score > max_score {
                        let correction = (max_score - score).exp();
                        sum_exp = sum_exp * correction + 1.0;
                        vec_scale_add(o_row, v_row, correction, head_dim);
                        max_score = score;
                    } else {
                        let wt = (score - max_score).exp();
                        sum_exp += wt;
                        vec_axpy_inplace(o_row, v_row, wt, head_dim);
                    }
                }

                if sum_exp > 0.0 {
                    let inv_sum = 1.0 / sum_exp;
                    vec_scale_inplace(o_row, inv_sum, head_dim);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bidirectional_attention(out: &mut [f32], q: &[f32], k: &[f32], v: &[f32],
                               seq: usize, n_heads: usize, head_dim: usize, scale: f32,
                               window_starts: &[i32], n_windows: usize) {
    let _pg = ProfileGuard::new(&PROF.attention_bidir);
    let n_threads = get_num_threads();
    let hidden = n_heads * head_dim;

    if n_threads > 1 && n_heads >= 2 {
        let out_ptr = out.as_mut_ptr() as usize;
        let q_ptr = q.as_ptr() as usize;
        let k_ptr = k.as_ptr() as usize;
        let v_ptr = v.as_ptr() as usize;
        let ws_ptr = window_starts.as_ptr() as usize;

        parallel_for(|tid, nt| {
            let chunk = n_heads.div_ceil(nt);
            let h0 = tid * chunk;
            let h1 = (h0 + chunk).min(n_heads);
            if h0 >= h1 { return; }

            let out_local = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, seq * hidden) };
            let q_local = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, seq * hidden) };
            let k_local = unsafe { std::slice::from_raw_parts(k_ptr as *const f32, seq * hidden) };
            let v_local = unsafe { std::slice::from_raw_parts(v_ptr as *const f32, seq * hidden) };
            let ws_local = unsafe { std::slice::from_raw_parts(ws_ptr as *const i32, n_windows + 1) };

            bidirectional_attention_heads(out_local, q_local, k_local, v_local,
                                         n_heads, head_dim, scale,
                                         ws_local, n_windows, h0, h1);
        });
        return;
    }

    bidirectional_attention_heads(out, q, k, v, n_heads, head_dim, scale,
                                 window_starts, n_windows, 0, n_heads);
}

/// Two-pass causal attention using BLAS sgemm with head-contiguous KV cache.
/// K/V layout: `[head][pos][head_dim]` — each head's data is contiguous across positions.
///
/// Single-token (seq_q=1): online softmax with NEON dot products — avoids BLAS overhead,
/// scores allocation, and fuses all 3 passes into a single scan over KV positions.
///
/// Multi-token (seq_q>1): 3-pass BLAS sgemm approach.
#[cfg(feature = "blas")]
#[allow(clippy::too_many_arguments)]
fn causal_attention_heads(out: &mut [f32], q: &[f32],
                           k_base: *const f32, v_base: *const f32,
                           head_stride: usize,
                           seq_q: usize, seq_k: usize, n_heads: usize, n_kv_heads: usize,
                           head_dim: usize, scale: f32, q_offset: usize,
                           head_start: usize, head_end: usize) {
    let heads_per_kv = n_heads / n_kv_heads;
    let q_hidden = n_heads * head_dim;

    // Single-token path: online softmax without allocation or BLAS
    if seq_q == 1 {
        for h in head_start..head_end {
            let kv_h = h / heads_per_kv;
            let k_head = unsafe { k_base.add(kv_h * head_stride) };
            let v_head = unsafe { v_base.add(kv_h * head_stride) };
            let q_off = h * head_dim;
            let o_row = &mut out[q_off..q_off + head_dim];
            let k_end = (q_offset + 1).min(seq_k);

            if k_end == 0 {
                for val in o_row.iter_mut().take(head_dim) { *val = 0.0; }
                continue;
            }

            let q_row = &q[q_off..q_off + head_dim];

            // Online softmax: single pass over KV positions
            let mut max_score = -1e30f32;
            let mut sum_exp = 0.0f32;
            for val in o_row.iter_mut().take(head_dim) { *val = 0.0; }

            for j in 0..k_end {
                let k_row = unsafe { std::slice::from_raw_parts(k_head.add(j * head_dim), head_dim) };
                let v_row = unsafe { std::slice::from_raw_parts(v_head.add(j * head_dim), head_dim) };

                let score = dot_f32(q_row, k_row, head_dim) * scale;

                if score > max_score {
                    let correction = (max_score - score).exp();
                    sum_exp = sum_exp * correction + 1.0;
                    vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    let wt = (score - max_score).exp();
                    sum_exp += wt;
                    vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }

            if sum_exp > 0.0 {
                let inv_sum = 1.0 / sum_exp;
                vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
        return;
    }

    // Multi-token path: 3-pass BLAS sgemm
    let mut scores = vec![0.0f32; seq_k];

    for h in head_start..head_end {
        let kv_h = h / heads_per_kv;
        let k_head = unsafe { k_base.add(kv_h * head_stride) };
        let v_head = unsafe { v_base.add(kv_h * head_stride) };

        for i in 0..seq_q {
            let q_off = i * q_hidden + h * head_dim;
            let o_off = i * q_hidden + h * head_dim;
            let o_row = &mut out[o_off..o_off + head_dim];
            let global_pos = q_offset + i;
            let k_end = (global_pos + 1).min(seq_k);

            if k_end == 0 {
                for val in o_row.iter_mut().take(head_dim) { *val = 0.0; }
                continue;
            }

            // Pass 1: scores = K_h @ q_h
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                    k_end as i32, 1, head_dim as i32,
                    scale,
                    k_head, head_dim as i32,
                    q.as_ptr().add(q_off), 1,
                    0.0,
                    scores.as_mut_ptr(), 1,
                );
            }

            // Pass 2: Softmax
            let mut max_s = scores[0];
            for j in 1..k_end { if scores[j] > max_s { max_s = scores[j]; } }
            for j in 0..k_end { scores[j] -= max_s; }

            #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
            {
                let n = k_end as i32;
                unsafe { vvexpf(scores.as_mut_ptr(), scores.as_ptr(), &n); }
            }
            #[cfg(not(all(feature = "vdsp", target_vendor = "apple")))]
            {
                for j in 0..k_end { scores[j] = scores[j].exp(); }
            }

            let mut sum_exp = 0.0f32;
            for j in 0..k_end { sum_exp += scores[j]; }
            if sum_exp > 0.0 {
                let inv = 1.0 / sum_exp;
                for j in 0..k_end { scores[j] *= inv; }
            }

            // Pass 3: out = V_h^T @ softmax_scores
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR, CBLAS_TRANS, CBLAS_NO_TRANS,
                    head_dim as i32, 1, k_end as i32,
                    1.0,
                    v_head, head_dim as i32,
                    scores.as_ptr(), 1,
                    0.0,
                    o_row.as_mut_ptr(), 1,
                );
            }
        }
    }
}

/// Fallback: online softmax causal attention (no BLAS), head-contiguous KV layout.
#[cfg(not(feature = "blas"))]
#[allow(clippy::too_many_arguments)]
fn causal_attention_heads(out: &mut [f32], q: &[f32],
                           k_base: *const f32, v_base: *const f32,
                           head_stride: usize,
                           seq_q: usize, seq_k: usize, n_heads: usize, n_kv_heads: usize,
                           head_dim: usize, scale: f32, q_offset: usize,
                           head_start: usize, head_end: usize) {
    let heads_per_kv = n_heads / n_kv_heads;
    let q_hidden = n_heads * head_dim;

    for h in head_start..head_end {
        let kv_h = h / heads_per_kv;

        for i in 0..seq_q {
            let q_off = i * q_hidden + h * head_dim;
            let q_row = &q[q_off..q_off + head_dim];
            let o_row = &mut out[i * q_hidden + h * head_dim..i * q_hidden + h * head_dim + head_dim];
            let global_pos = q_offset + i;
            let k_end = (global_pos + 1).min(seq_k);

            let mut max_score = -1e30f32;
            let mut sum_exp = 0.0f32;
            for val in o_row.iter_mut().take(head_dim) { *val = 0.0; }

            for j in 0..k_end {
                let k_row = unsafe { std::slice::from_raw_parts(k_base.add(kv_h * head_stride + j * head_dim), head_dim) };
                let v_row = unsafe { std::slice::from_raw_parts(v_base.add(kv_h * head_stride + j * head_dim), head_dim) };

                let score = dot_f32(q_row, k_row, head_dim) * scale;

                if score > max_score {
                    let correction = (max_score - score).exp();
                    sum_exp = sum_exp * correction + 1.0;
                    vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    let wt = (score - max_score).exp();
                    sum_exp += wt;
                    vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }

            if sum_exp > 0.0 {
                let inv_sum = 1.0 / sum_exp;
                vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn causal_attention(out: &mut [f32], q: &[f32],
                         k_base: *const f32, v_base: *const f32,
                         head_stride: usize,
                         seq_q: usize, seq_k: usize, n_heads: usize, n_kv_heads: usize,
                         head_dim: usize, scale: f32, q_offset: usize) {
    let _pg = ProfileGuard::new(&PROF.attention_causal);
    let n_threads = get_num_threads();
    if n_threads > 1 && n_heads >= 2 {
        let out_ptr = out.as_mut_ptr() as usize;
        let q_ptr = q.as_ptr() as usize;
        let k_ptr = k_base as usize;
        let v_ptr = v_base as usize;
        let q_hidden = n_heads * head_dim;

        parallel_for(|tid, nt| {
            let chunk = n_heads.div_ceil(nt);
            let h0 = tid * chunk;
            let h1 = (h0 + chunk).min(n_heads);
            if h0 >= h1 { return; }

            let out_local = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, seq_q * q_hidden) };
            let q_local = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, seq_q * q_hidden) };

            causal_attention_heads(out_local, q_local,
                                   k_ptr as *const f32, v_ptr as *const f32,
                                   head_stride,
                                   seq_q, seq_k, n_heads, n_kv_heads,
                                   head_dim, scale, q_offset, h0, h1);
        });
        return;
    }

    causal_attention_heads(out, q, k_base, v_base, head_stride,
                            seq_q, seq_k, n_heads, n_kv_heads,
                            head_dim, scale, q_offset, 0, n_heads);
}

// ========================================================================
// Position Embeddings
// ========================================================================

pub fn sinusoidal_pe(pe: &mut [f32], n_pos: usize, d_model: usize) {
    let half = d_model / 2;
    let log_timescale = (10000.0f32).ln() / (half - 1) as f32;

    for p in 0..n_pos {
        let row = &mut pe[p * d_model..(p + 1) * d_model];
        for d in 0..half {
            let inv_timescale = (-(d as f32) * log_timescale).exp();
            let angle = p as f32 * inv_timescale;
            row[d] = angle.sin();
            row[half + d] = angle.cos();
        }
    }
}

pub fn compute_rope_neox(cos_out: &mut [f32], sin_out: &mut [f32], positions: &[i32],
                          seq: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;

    for s in 0..seq {
        let pos = positions[s] as f32;
        for d in 0..half {
            let freq = 1.0 / theta.powf((2 * d) as f32 / head_dim as f32);
            let angle = pos * freq;
            let c = angle.cos();
            let sn = angle.sin();
            cos_out[s * head_dim + d] = c;
            cos_out[s * head_dim + half + d] = c;
            sin_out[s * head_dim + d] = sn;
            sin_out[s * head_dim + half + d] = sn;
        }
    }
}

pub fn apply_rope_neox(x: &mut [f32], cos_vals: &[f32], sin_vals: &[f32],
                        seq: usize, n_heads: usize, head_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.rope);
    let half = head_dim / 2;
    let hidden = n_heads * head_dim;

    for s in 0..seq {
        let c = &cos_vals[s * head_dim..];
        let sn = &sin_vals[s * head_dim..];

        for h in 0..n_heads {
            let base = s * hidden + h * head_dim;
            let vec = &mut x[base..base + head_dim];

            #[cfg(target_arch = "aarch64")]
            {
                let mut d = 0usize;
                while d + 4 <= half {
                    unsafe {
                        use core::arch::aarch64::*;
                        let x1 = vld1q_f32(vec.as_ptr().add(d));
                        let x2 = vld1q_f32(vec.as_ptr().add(half + d));
                        let cv = vld1q_f32(c.as_ptr().add(d));
                        let sv = vld1q_f32(sn.as_ptr().add(d));
                        // vec[d] = x1*cos - x2*sin
                        let new1 = vfmsq_f32(vmulq_f32(x1, cv), x2, sv);
                        // vec[half+d] = x2*cos + x1*sin (cos[half+d]==cos[d])
                        let new2 = vfmaq_f32(vmulq_f32(x2, cv), x1, sv);
                        vst1q_f32(vec.as_mut_ptr().add(d), new1);
                        vst1q_f32(vec.as_mut_ptr().add(half + d), new2);
                    }
                    d += 4;
                }
                while d < half {
                    let x1 = vec[d];
                    let x2 = vec[half + d];
                    vec[d]        = x1 * c[d] - x2 * sn[d];
                    vec[half + d] = x2 * c[d] + x1 * sn[d];
                    d += 1;
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                for d in 0..half {
                    let x1 = vec[d];
                    let x2 = vec[half + d];
                    vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                    vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
                }
            }
        }
    }
}

/// Streaming argmax: finds argmax(W_bf16 @ x) without materializing full logits.
/// Quantize x (f32) to int8 with absmax scaling. Returns (x_int8, scale).
pub fn quantize_f32_to_int8(x: &[f32]) -> (Vec<i8>, f32) {
    let mut max_abs = 0.0f32;
    for &v in x { max_abs = max_abs.max(v.abs()); }
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv_scale = 127.0 / max_abs.max(1e-10);
    let int8: Vec<i8> = x.iter().map(|&v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8).collect();
    (int8, scale)
}

/// Quantize BF16 weights to INT8 per-row. Returns (int8_data, per_row_scales).
pub fn quantize_bf16_weights_to_int8(w_bf16: *const u16, out_dim: usize, in_dim: usize) -> (Vec<i8>, Vec<f32>) {
    #[cfg(target_arch = "aarch64")]
    unsafe { return neon::quantize_bf16_to_int8(w_bf16, out_dim, in_dim); }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut int8_data = vec![0i8; out_dim * in_dim];
        let mut scales = vec![0.0f32; out_dim];
        let src = unsafe { std::slice::from_raw_parts(w_bf16, out_dim * in_dim) };
        for row in 0..out_dim {
            let mut max_abs = 0.0f32;
            for k in 0..in_dim {
                let v = f32::from_bits((src[row * in_dim + k] as u32) << 16).abs();
                if v > max_abs { max_abs = v; }
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 127.0 / max_abs.max(1e-10);
            scales[row] = scale;
            for k in 0..in_dim {
                let v = f32::from_bits((src[row * in_dim + k] as u32) << 16);
                int8_data[row * in_dim + k] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
        (int8_data, scales)
    }
}

/// INT8 threaded argmax: find argmax(x @ W.T) using INT8 quantized weights.
pub fn argmax_matvec_int8(x: &[f32], w_int8: &[i8], w_scales: &[f32], in_dim: usize, out_dim: usize) -> usize {
    let (x_int8, x_scale) = quantize_f32_to_int8(x);
    let n_threads = get_num_threads();

    #[cfg(target_arch = "aarch64")]
    {
        if n_threads <= 1 {
            let (best, _) = unsafe {
                neon::argmax_int8_range(x_int8.as_ptr(), x_scale, w_int8.as_ptr(), w_scales, in_dim, 0, out_dim)
            };
            return best;
        }

        let mut best_indices = vec![0usize; n_threads];
        let mut best_vals = vec![-1e30f32; n_threads];

        let x_int8_ptr = x_int8.as_ptr() as usize;
        let w_int8_ptr = w_int8.as_ptr() as usize;
        let w_scales_ptr = w_scales.as_ptr() as usize;
        let bi_ptr = best_indices.as_mut_ptr() as usize;
        let bv_ptr = best_vals.as_mut_ptr() as usize;

        parallel_for(|tid, nt| {
            let chunk = out_dim.div_ceil(nt);
            let start = tid * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end {
                unsafe {
                    *(bv_ptr as *mut f32).add(tid) = -1e30;
                    *(bi_ptr as *mut usize).add(tid) = 0;
                }
                return;
            }

            let w_scales_local = unsafe { std::slice::from_raw_parts(w_scales_ptr as *const f32, out_dim) };
            let (best, best_val) = unsafe {
                neon::argmax_int8_range(x_int8_ptr as *const i8, x_scale, w_int8_ptr as *const i8, w_scales_local, in_dim, start, end)
            };
            unsafe {
                *(bi_ptr as *mut usize).add(tid) = best;
                *(bv_ptr as *mut f32).add(tid) = best_val;
            }
        });

        let mut best = best_indices[0];
        let mut best_val = best_vals[0];
        for i in 1..n_threads {
            if best_vals[i] > best_val {
                best_val = best_vals[i];
                best = best_indices[i];
            }
        }
        return best;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        if n_threads <= 1 || out_dim <= 1 {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for row in 0..out_dim {
                let weights = &w_int8[row * in_dim..(row + 1) * in_dim];
                let mut acc = 0i32;
                for col in 0..in_dim {
                    acc += (x_int8[col] as i32) * (weights[col] as i32);
                }
                let value = (acc as f32) * x_scale * w_scales[row];
                if value > best_val {
                    best_val = value;
                    best_idx = row;
                }
            }
            return best_idx;
        }

        let mut best_indices = vec![0usize; n_threads];
        let mut best_vals = vec![f32::NEG_INFINITY; n_threads];

        let x_int8_ptr = x_int8.as_ptr() as usize;
        let w_int8_ptr = w_int8.as_ptr() as usize;
        let w_scales_ptr = w_scales.as_ptr() as usize;
        let bi_ptr = best_indices.as_mut_ptr() as usize;
        let bv_ptr = best_vals.as_mut_ptr() as usize;

        parallel_for(|tid, nt| {
            let chunk = out_dim.div_ceil(nt);
            let start = tid * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end {
                unsafe {
                    *(bv_ptr as *mut f32).add(tid) = f32::NEG_INFINITY;
                    *(bi_ptr as *mut usize).add(tid) = 0;
                }
                return;
            }

            let x_local = unsafe {
                std::slice::from_raw_parts(x_int8_ptr as *const i8, in_dim)
            };
            let mut best_idx = start;
            let mut best_val = f32::NEG_INFINITY;
            for row in start..end {
                let weights = unsafe {
                    std::slice::from_raw_parts(
                        (w_int8_ptr as *const i8).add(row * in_dim),
                        in_dim,
                    )
                };
                let mut acc = 0i32;
                for col in 0..in_dim {
                    acc += (x_local[col] as i32) * (weights[col] as i32);
                }
                let value = (acc as f32)
                    * x_scale
                    * unsafe { *((w_scales_ptr as *const f32).add(row)) };
                if value > best_val {
                    best_val = value;
                    best_idx = row;
                }
            }

            unsafe {
                *(bi_ptr as *mut usize).add(tid) = best_idx;
                *(bv_ptr as *mut f32).add(tid) = best_val;
            }
        });

        let mut best = best_indices[0];
        let mut best_val = best_vals[0];
        for i in 1..n_threads {
            if best_vals[i] > best_val {
                best_val = best_vals[i];
                best = best_indices[i];
            }
        }
        return best;
    }
}

pub fn argmax_matvec_bf16(x: &[f32], w_bf16: *const u16, in_dim: usize, out_dim: usize) -> usize {
    let n_threads = get_num_threads();
    if n_threads <= 1 {
        let (best, _) = argmax_bf16_range(x, w_bf16, in_dim, 0, out_dim);
        return best;
    }

    let mut best_indices = vec![0usize; n_threads];
    let mut best_vals = vec![-1e30f32; n_threads];

    let x_ptr = x.as_ptr() as usize;
    let w_ptr = w_bf16 as usize;
    let bi_ptr = best_indices.as_mut_ptr() as usize;
    let bv_ptr = best_vals.as_mut_ptr() as usize;

    parallel_for(|tid, nt| {
        let chunk = out_dim.div_ceil(nt);
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end {
            unsafe {
                *(bv_ptr as *mut f32).add(tid) = -1e30;
                *(bi_ptr as *mut usize).add(tid) = 0;
            }
            return;
        }

        let x_local = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, in_dim) };
        let (best, best_val) = argmax_bf16_range(x_local, w_ptr as *const u16, in_dim, start, end);
        unsafe {
            *(bi_ptr as *mut usize).add(tid) = best;
            *(bv_ptr as *mut f32).add(tid) = best_val;
        }
    });

    let mut best = best_indices[0];
    let mut best_val = best_vals[0];
    for i in 1..n_threads {
        if best_vals[i] > best_val {
            best_val = best_vals[i];
            best = best_indices[i];
        }
    }
    best
}
