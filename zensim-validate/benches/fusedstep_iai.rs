//! iai-callgrind instruction-count benchmarks for the K=1 fused pair
//! update (lane: fusedstep). Measures the per-pair layer-1 work:
//!
//!   UNFUSED: `backprop_step`×2 (incl. the gw1 row sweep) +
//!            `add_l2_grad_layer1` + `adam_update` on w1
//!   FUSED:   `backprop_grad_head`×2 + `adam_update_w1_fused`
//!
//! Deterministic instruction/cycle-estimate counts — wall time on the
//! shared box is noisy. Public dispatchers are called rather than the
//! private `*_avx2` fns. Note: the local `tier_cap` shim does NOT pin
//! `incant!`/`summon()` dispatch — `adam_update` and
//! `adam_update_w1_fused` probe host features directly; under
//! iai-callgrind/valgrind they reach v3 only because valgrind cannot
//! execute AVX-512, so `summon()` returns `None` for the v4 token. The
//! shim still matches the production `ZENSIM_MAX_TIER=v3` contract for
//! `simd_mlp` dispatch (which does consult `avx512_allowed()`). Dispatch
//! overhead is a small constant identical in both arms.
//!
//! Shapes cover both real geometries:
//!   * 372×128 — V_X production (bench_mlp_kernels conventions)
//!   * 944×32  — the bitexact fold (`--max-features 944 --hidden 32`)
//!
//! `add_l2_grad_layer1` is private inside `mlp_train`, so the verbatim
//! copy from `optmlp_iai.rs` is reused (kept 1:1 with the real fn; the
//! cargo-test + bitexact gates exercise the real one).

mod tier_cap {
    pub fn avx512_allowed() -> bool {
        false
    }
}

#[allow(dead_code)] // #[path]-includes the whole kernel file; this bench uses a subset
#[path = "../src/simd_mlp.rs"]
mod simd_mlp;

#[allow(dead_code)] // same for adam_simd.rs
#[path = "../src/adam_simd.rs"]
mod adam_simd;

use std::hint::black_box;

use adam_simd::{AdamUpdateArgs, AdamW1FusedArgs, adam_update, adam_update_w1_fused};
use iai_callgrind::{library_benchmark, library_benchmark_group, main};

const ALPHA: f64 = 0.01;
const L2: f64 = 1e-5;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
const LR: f64 = 0.005;

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

/// Per-pair training state: weights, Adam moments, and gradient buffers.
struct Pair {
    xa: Vec<f64>,
    xb: Vec<f64>,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    gw1: Vec<f64>,
    gb1: Vec<f64>,
    gw2: Vec<f64>,
    gb2: Vec<f64>,
    mw1: Vec<f64>,
    vw1: Vec<f64>,
    dh_a: Vec<f64>,
    dh_b: Vec<f64>,
    n_features: usize,
    n_hidden: usize,
    dl_dya: f64,
    dl_dyb: f64,
}

fn build_pair(n_features: usize, n_hidden: usize, seed: u64) -> Pair {
    let mut rng = Xs64::new(seed);
    let n = n_features * n_hidden;
    Pair {
        xa: random_sparse(&mut rng, n_features, 0.30),
        xb: random_sparse(&mut rng, n_features, 0.30),
        w1: random_buf(&mut rng, n),
        b1: random_buf(&mut rng, n_hidden),
        w2: random_buf(&mut rng, n_hidden),
        b2: vec![rng.next_f64()],
        // gw1 reads +0.0 on entry in the real K=1 path (zeroed by the
        // previous Adam step); zeros here keep the measured work honest.
        gw1: vec![0.0; n],
        gb1: vec![0.0; n_hidden],
        gw2: vec![0.0; n_hidden],
        gb2: vec![0.0],
        mw1: (0..n).map(|i| (i as f64 % 7.0) * 1e-4).collect(),
        vw1: (0..n).map(|i| 1e-6 + (i as f64 % 11.0) * 1e-4).collect(),
        dh_a: vec![0.0; n_hidden],
        dh_b: vec![0.0; n_hidden],
        n_features,
        n_hidden,
        dl_dya: 0.37,
        dl_dyb: -0.61,
    }
}

// --- add_l2_grad_layer1: verbatim copy (see optmlp_iai.rs header note) ---
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

fn adam_w1(p: &mut Pair) {
    adam_update(&mut AdamUpdateArgs {
        w: &mut p.w1,
        g: &mut p.gw1,
        m: &mut p.mw1,
        v: &mut p.vw1,
        beta1: BETA1,
        beta2: BETA2,
        eps: EPS,
        bc1: 1.0 - BETA1,
        bc2: 1.0 - BETA2,
        lr: LR,
    });
}

// ---------------- measured bodies ----------------

fn unfused_body(p: &mut Pair) {
    let (_, h_pre_a, h_a) = simd_mlp::forward(
        black_box(&p.xa),
        &p.w1,
        &p.b1,
        &p.w2,
        &p.b2,
        p.n_features,
        p.n_hidden,
        ALPHA,
    );
    let (_, h_pre_b, h_b) = simd_mlp::forward(
        black_box(&p.xb),
        &p.w1,
        &p.b1,
        &p.w2,
        &p.b2,
        p.n_features,
        p.n_hidden,
        ALPHA,
    );
    simd_mlp::backprop_step(
        black_box(&p.xa),
        &h_pre_a,
        &h_a,
        p.dl_dya,
        &mut p.gw1,
        &mut p.gb1,
        &p.w2,
        &mut p.gw2,
        &mut p.gb2,
        p.n_features,
        p.n_hidden,
        ALPHA,
    );
    simd_mlp::backprop_step(
        black_box(&p.xb),
        &h_pre_b,
        &h_b,
        p.dl_dyb,
        &mut p.gw1,
        &mut p.gb1,
        &p.w2,
        &mut p.gw2,
        &mut p.gb2,
        p.n_features,
        p.n_hidden,
        ALPHA,
    );
    add_l2_grad_layer1(&mut p.gw1, &p.w1, L2, p.n_hidden, None);
    adam_w1(p);
}

fn fused_body(p: &mut Pair) {
    let (_, h_pre_a, h_a) = simd_mlp::forward(
        black_box(&p.xa),
        &p.w1,
        &p.b1,
        &p.w2,
        &p.b2,
        p.n_features,
        p.n_hidden,
        ALPHA,
    );
    let (_, h_pre_b, h_b) = simd_mlp::forward(
        black_box(&p.xb),
        &p.w1,
        &p.b1,
        &p.w2,
        &p.b2,
        p.n_features,
        p.n_hidden,
        ALPHA,
    );
    simd_mlp::backprop_grad_head(
        &h_pre_a,
        &h_a,
        p.dl_dya,
        &mut p.dh_a,
        &mut p.gb1,
        &p.w2,
        &mut p.gw2,
        &mut p.gb2,
        p.n_hidden,
        ALPHA,
    );
    simd_mlp::backprop_grad_head(
        &h_pre_b,
        &h_b,
        p.dl_dyb,
        &mut p.dh_b,
        &mut p.gb1,
        &p.w2,
        &mut p.gw2,
        &mut p.gb2,
        p.n_hidden,
        ALPHA,
    );
    adam_update_w1_fused(&mut AdamW1FusedArgs {
        w: &mut p.w1,
        g: &mut p.gw1,
        m: &mut p.mw1,
        v: &mut p.vw1,
        xa: black_box(&p.xa),
        dha: &p.dh_a,
        xb: black_box(&p.xb),
        dhb: &p.dh_b,
        l2_scale: L2,
        l2_mult: None,
        n_hidden: p.n_hidden,
        beta1: BETA1,
        beta2: BETA2,
        eps: EPS,
        bc1: 1.0 - BETA1,
        bc2: 1.0 - BETA2,
        lr: LR,
    });
}

// ---------------- unfused arms ----------------

#[library_benchmark]
#[bench::p944x32(args = (944, 32, 0x94432), setup = build_pair)]
fn unfused_944x32(mut p: Pair) {
    unfused_body(&mut p)
}

#[library_benchmark]
#[bench::p372x128(args = (372, 128, 0x372128), setup = build_pair)]
fn unfused_372x128(mut p: Pair) {
    unfused_body(&mut p)
}

// ---------------- fused arms ----------------

#[library_benchmark]
#[bench::p944x32(args = (944, 32, 0x94432), setup = build_pair)]
fn fused_944x32(mut p: Pair) {
    fused_body(&mut p)
}

#[library_benchmark]
#[bench::p372x128(args = (372, 128, 0x372128), setup = build_pair)]
fn fused_372x128(mut p: Pair) {
    fused_body(&mut p)
}

// ---------------- bit-identity guard ----------------

// Runs the fused and unfused sequences on cloned state and asserts the
// w1/gw1/mw1/vw1 results are bit-identical — under callgrind this
// validates the fused kernels on the same input the counts measure.
#[library_benchmark]
#[bench::verify(args = (944, 32, 0x94432), setup = build_pair)]
fn fused_matches_unfused(p: Pair) {
    let mut uf = Pair {
        dh_a: vec![],
        dh_b: vec![],
        ..clone_pair(&p)
    };
    let mut fs = Pair {
        dh_a: vec![0.0; p.n_hidden],
        dh_b: vec![0.0; p.n_hidden],
        ..clone_pair(&p)
    };
    unfused_body(&mut uf);
    fused_body(&mut fs);
    for (name, a, b) in [
        ("w1", &uf.w1, &fs.w1),
        ("gw1", &uf.gw1, &fs.gw1),
        ("mw1", &uf.mw1, &fs.mw1),
        ("vw1", &uf.vw1, &fs.vw1),
        ("gw2", &uf.gw2, &fs.gw2),
        ("gb1", &uf.gb1, &fs.gb1),
        ("gb2", &uf.gb2, &fs.gb2),
    ] {
        assert!(
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "fusedstep iai verify: {name} differs"
        );
    }
}

fn clone_pair(p: &Pair) -> Pair {
    Pair {
        xa: p.xa.clone(),
        xb: p.xb.clone(),
        w1: p.w1.clone(),
        b1: p.b1.clone(),
        w2: p.w2.clone(),
        b2: p.b2.clone(),
        gw1: p.gw1.clone(),
        gb1: p.gb1.clone(),
        gw2: p.gw2.clone(),
        gb2: p.gb2.clone(),
        mw1: p.mw1.clone(),
        vw1: p.vw1.clone(),
        dh_a: p.dh_a.clone(),
        dh_b: p.dh_b.clone(),
        n_features: p.n_features,
        n_hidden: p.n_hidden,
        dl_dya: p.dl_dya,
        dl_dyb: p.dl_dyb,
    }
}

library_benchmark_group!(
    name = pair;
    benchmarks =
        unfused_944x32,
        unfused_372x128,
        fused_944x32,
        fused_372x128,
        fused_matches_unfused
);

main!(library_benchmark_groups = pair);
