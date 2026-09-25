//! iai-callgrind instruction-count benchmarks for the AVX2 MLP kernels
//! (`simd_mlp::forward` → `forward_avx2`, `simd_mlp::backprop_step` →
//! `backprop_avx2`) and `mlp_train::add_l2_grad_layer1`. Lane: optmlp.
//!
//! Deterministic instruction/cycle-estimate counts — wall time on the
//! shared box is noisy. The public dispatchers are called rather than the
//! private `*_avx2` fns: the local `tier_cap` shim pins
//! `avx512_allowed()` to false, which is exactly the `ZENSIM_MAX_TIER=v3`
//! production path — dispatch then deterministically reaches the AVX2
//! kernels under valgrind (which cannot execute AVX-512). The two
//! `is_x86_feature_detected!` calls per dispatch are a small constant
//! cost, identical across baseline/optimized builds, so instruction-count
//! deltas are attributable to the kernels themselves.
//!
//! Shapes cover both real geometries:
//!   * 372×128 — V_X production (bench_mlp_kernels conventions)
//!   * 944×32  — the bitexact fold (`--max-features 944 --hidden 32`)
//!
//! `add_l2_grad_layer1` is private inside the 14k-line `mlp_train`
//! module, so it cannot be `#[path]`-included; the verbatim copy below
//! mirrors it 1:1 (kept in sync; the bitexact + unit-test gates exercise
//! the real function).

// Shim for the `crate::tier_cap::avx512_allowed()` call inside the
// #[path]-included simd_mlp.rs — the v3 cap semantics.
mod tier_cap {
    pub fn avx512_allowed() -> bool {
        false
    }
}

#[allow(dead_code)] // #[path]-includes the whole kernel file; this bench uses a subset
#[path = "../src/simd_mlp.rs"]
mod simd_mlp;

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};

const ALPHA: f64 = 0.01;

struct Xs64(u64);
impl Xs64 {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        (self.next() as f64 / u64::MAX as f64) * 2.0 - 1.0
    }
}

fn random_buf(rng: &mut Xs64, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.next_f64()).collect()
}

/// ~30% zeros, matching the V_22-IW v2 trainer's transformed features
/// (same convention as bench_mlp_kernels.rs).
fn random_sparse(rng: &mut Xs64, n: usize, zero_frac: f64) -> Vec<f64> {
    let thresh = 2.0 * zero_frac - 1.0;
    (0..n)
        .map(|_| {
            if rng.next_f64() < thresh {
                0.0
            } else {
                rng.next_f64()
            }
        })
        .collect()
}

struct FwdInputs {
    x: Vec<f64>,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    n_features: usize,
    n_hidden: usize,
}

fn build_fwd(n_features: usize, n_hidden: usize, seed: u64) -> FwdInputs {
    let mut rng = Xs64::new(seed);
    FwdInputs {
        x: random_sparse(&mut rng, n_features, 0.30),
        w1: random_buf(&mut rng, n_features * n_hidden),
        b1: random_buf(&mut rng, n_hidden),
        w2: random_buf(&mut rng, n_hidden),
        b2: vec![rng.next_f64()],
        n_features,
        n_hidden,
    }
}

struct BwdInputs {
    fwd: FwdInputs,
    h_pre: Vec<f64>,
    h: Vec<f64>,
    gw1: Vec<f64>,
    gb1: Vec<f64>,
    gw2: Vec<f64>,
    gb2: Vec<f64>,
}

fn build_bwd(n_features: usize, n_hidden: usize, seed: u64) -> BwdInputs {
    let fwd = build_fwd(n_features, n_hidden, seed);
    let mut rng = Xs64::new(seed ^ 0x5EED);
    let h_pre = random_buf(&mut rng, n_hidden);
    let h: Vec<f64> = h_pre
        .iter()
        .map(|&v| if v >= 0.0 { v } else { ALPHA * v })
        .collect();
    BwdInputs {
        gw1: random_buf(&mut rng, n_features * n_hidden),
        gb1: random_buf(&mut rng, n_hidden),
        gw2: random_buf(&mut rng, n_hidden),
        gb2: vec![rng.next_f64()],
        fwd,
        h_pre,
        h,
    }
}

// --- add_l2_grad_layer1: verbatim copy of mlp_train/mod.rs's private fn ---
// Ownership note: this copy is what the iai numbers measure; the shipped
// function is exercised bit-for-bit by the cargo-test + bitexact gates.
// Last synced: see benchmarks/optmlp_WORKLOG.md.
fn add_l2_grad_layer1(g: &mut [f64], w: &[f64], scale: f64, n_hidden: usize, mult: Option<&[f64]>) {
    let n = g.len().min(w.len());
    let (g, w) = (&mut g[..n], &w[..n]);
    let avx = l2_row_avx_available();
    let Some(mult) = mult else {
        if avx {
            unsafe { l2_row_avx(g, w, scale) };
            return;
        }
        for (g, &w) in g.iter_mut().zip(w.iter()) {
            *g += scale * w;
        }
        return;
    };
    if n_hidden == 0 {
        for (idx, (g, &w)) in g.iter_mut().zip(w.iter()).enumerate() {
            *g += scale * mult[idx / n_hidden] * w;
        }
        return;
    }
    for (feat, (grow, wrow)) in g.chunks_mut(n_hidden).zip(w.chunks(n_hidden)).enumerate() {
        let sm = scale * mult[feat];
        if avx {
            unsafe { l2_row_avx(grow, wrow, sm) };
        } else {
            for (g, &w) in grow.iter_mut().zip(wrow.iter()) {
                *g += sm * w;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn l2_row_avx(g: &mut [f64], w: &[f64], sm: f64) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };
    unsafe {
        let sm_v = _mm256_set1_pd(sm);
        let gp = g.as_mut_ptr();
        let wp = w.as_ptr();
        let n = g.len().min(w.len());
        for c in 0..n / 4 {
            let off = c * 4;
            let gv = _mm256_loadu_pd(gp.add(off));
            let wv = _mm256_loadu_pd(wp.add(off));
            _mm256_storeu_pd(gp.add(off), _mm256_add_pd(gv, _mm256_mul_pd(sm_v, wv)));
        }
        for o in n / 4 * 4..n {
            *gp.add(o) += sm * *wp.add(o);
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
unsafe fn l2_row_avx(g: &mut [f64], w: &[f64], sm: f64) {
    for (g, &w) in g.iter_mut().zip(w.iter()) {
        *g += sm * w;
    }
}

#[inline]
fn l2_row_avx_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// -----------------------------------------------------------------------------

#[library_benchmark]
#[bench::forward_372x128(args = (372usize, 128usize, 0xCAFEu64), setup = build_fwd)]
fn forward_v3(ix: FwdInputs) {
    let r = simd_mlp::forward(
        black_box(&ix.x),
        black_box(&ix.w1),
        black_box(&ix.b1),
        black_box(&ix.w2),
        black_box(&ix.b2),
        ix.n_features,
        ix.n_hidden,
        ALPHA,
    );
    black_box(r);
}

#[library_benchmark]
#[bench::forward_944x32(args = (944usize, 32usize, 0xF01Du64), setup = build_fwd)]
fn forward_v3_fold(ix: FwdInputs) {
    let r = simd_mlp::forward(
        black_box(&ix.x),
        black_box(&ix.w1),
        black_box(&ix.b1),
        black_box(&ix.w2),
        black_box(&ix.b2),
        ix.n_features,
        ix.n_hidden,
        ALPHA,
    );
    black_box(r);
}

#[library_benchmark]
#[bench::backprop_372x128(args = (372usize, 128usize, 0xCAFEu64), setup = build_bwd)]
fn backprop_v3(mut ix: BwdInputs) {
    simd_mlp::backprop_step(
        black_box(&ix.fwd.x),
        black_box(&ix.h_pre),
        black_box(&ix.h),
        black_box(0.1),
        black_box(&mut ix.gw1),
        black_box(&mut ix.gb1),
        black_box(&ix.fwd.w2),
        black_box(&mut ix.gw2),
        black_box(&mut ix.gb2),
        ix.fwd.n_features,
        ix.fwd.n_hidden,
        ALPHA,
    );
}

#[library_benchmark]
#[bench::backprop_944x32(args = (944usize, 32usize, 0xF01Du64), setup = build_bwd)]
fn backprop_v3_fold(mut ix: BwdInputs) {
    simd_mlp::backprop_step(
        black_box(&ix.fwd.x),
        black_box(&ix.h_pre),
        black_box(&ix.h),
        black_box(0.1),
        black_box(&mut ix.gw1),
        black_box(&mut ix.gb1),
        black_box(&ix.fwd.w2),
        black_box(&mut ix.gw2),
        black_box(&mut ix.gb2),
        ix.fwd.n_features,
        ix.fwd.n_hidden,
        ALPHA,
    );
}

struct L2Inputs {
    g: Vec<f64>,
    w: Vec<f64>,
    mult: Vec<f64>,
    n_hidden: usize,
}

fn build_l2(n_features: usize, n_hidden: usize) -> L2Inputs {
    let mut rng = Xs64::new(0xBADC0DE);
    L2Inputs {
        g: random_buf(&mut rng, n_features * n_hidden),
        w: random_buf(&mut rng, n_features * n_hidden),
        // mult>1 only on a minority of rows (coarse-scale convention).
        mult: (0..n_features)
            .map(|i| if i % 8 == 0 { 2.0 } else { 1.0 })
            .collect(),
        n_hidden,
    }
}

// L2 row loop at the 944×32 fold shape, `Some(mult)` arm (the arm the
// perf profile charged 5.9% to; the None arm is the same inner loop).
#[library_benchmark]
#[bench::l2_944x32(args = (944usize, 32usize), setup = build_l2)]
fn l2_grad_944x32(mut ix: L2Inputs) {
    add_l2_grad_layer1(
        black_box(&mut ix.g),
        black_box(&ix.w),
        black_box(1e-5),
        black_box(ix.n_hidden),
        black_box(Some(&ix.mult)),
    );
    black_box(ix.g);
}

// Same shape, `None` arm (uniform L2).
#[library_benchmark]
#[bench::l2_944x32_nomult(args = (944usize, 32usize), setup = build_l2)]
fn l2_grad_944x32_nomult(mut ix: L2Inputs) {
    add_l2_grad_layer1(
        black_box(&mut ix.g),
        black_box(&ix.w),
        black_box(1e-5),
        black_box(ix.n_hidden),
        black_box(None),
    );
    black_box(ix.g);
}

library_benchmark_group!(
    name = optmlp_group;
    benchmarks =
        forward_v3,
        forward_v3_fold,
        backprop_v3,
        backprop_v3_fold,
        l2_grad_944x32,
        l2_grad_944x32_nomult
);

main!(library_benchmark_groups = optmlp_group);
