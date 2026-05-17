//! zenbench microbench for the MLP forward + backprop_step kernels.
//!
//! Run:
//!   cargo bench --bench bench_mlp_kernels -p zensim-validate
//!
//! Tests three kernel shapes:
//!  * `forward_372x128`  — production training shape
//!  * `backprop_372x128` — production training shape
//!  * `pair_372x128`     — one RankNet pair update
//!                         (2× forward + 2× backprop, matching the
//!                         actual training inner loop)
//!
//! Each benchmark sets `Throughput::Elements(N)` to the FMA count so
//! the report shows MFLOPS-equivalent throughput, which is the only
//! cross-shape comparable number.
//!
//! Compares the dispatched SIMD path against the scalar fallback. Both
//! paths are kept in `simd_mlp` so the comparison is exact (no other
//! code differs).

#[path = "../src/simd_mlp.rs"]
#[allow(dead_code)]
mod simd_mlp;

use zenbench::prelude::*;

const N_FEATURES: usize = 372;
const N_HIDDEN: usize = 128;
const ALPHA: f64 = 0.01;
const FMA_FWD: u64 = (N_FEATURES * N_HIDDEN) as u64; // 47,616
const FMA_BWD: u64 = (N_FEATURES * N_HIDDEN) as u64; // ~47,616
const FMA_PAIR: u64 = 4 * FMA_FWD; // 2× forward + 2× backprop

/// Reproduce a deterministic input buffer with realistic sparsity.
/// 30% zeros matches the V_22-IW v2 trainer's auto-transform output
/// (winsorize + log-shift produce ~0.3 zero fraction in feature space).
fn build_inputs(seed: u64) -> Inputs {
    let mut rng = Xs64::new(seed);
    Inputs {
        x: random_sparse(&mut rng, N_FEATURES, 0.30),
        w1: random_buf(&mut rng, N_FEATURES * N_HIDDEN),
        b1: random_buf(&mut rng, N_HIDDEN),
        w2: random_buf(&mut rng, N_HIDDEN),
        b2: vec![rng.next_f64()],
    }
}

struct Inputs {
    x: Vec<f64>,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
}

fn bench_kernels(suite: &mut Suite) {
    let inputs = build_inputs(0xCAFE_F00D_DEAD_BEEF);

    suite.group("forward_372x128", |g| {
        g.throughput(Throughput::Elements(FMA_FWD));

        let ix = inputs.x.clone();
        let iw1 = inputs.w1.clone();
        let ib1 = inputs.b1.clone();
        let iw2 = inputs.w2.clone();
        let ib2 = inputs.b2.clone();
        g.bench("scalar", move |b| {
            b.iter(|| {
                let (y, h_pre, h) = forward_scalar_view(
                    black_box(&ix),
                    black_box(&iw1),
                    black_box(&ib1),
                    black_box(&iw2),
                    black_box(&ib2),
                    N_FEATURES,
                    N_HIDDEN,
                    ALPHA,
                );
                black_box((y, h_pre, h))
            })
        });

        let ix = inputs.x.clone();
        let iw1 = inputs.w1.clone();
        let ib1 = inputs.b1.clone();
        let iw2 = inputs.w2.clone();
        let ib2 = inputs.b2.clone();
        g.bench("simd_dispatch", move |b| {
            b.iter(|| {
                let (y, h_pre, h) = simd_mlp::forward(
                    black_box(&ix),
                    black_box(&iw1),
                    black_box(&ib1),
                    black_box(&iw2),
                    black_box(&ib2),
                    N_FEATURES,
                    N_HIDDEN,
                    ALPHA,
                );
                black_box((y, h_pre, h))
            })
        });
    });

    suite.group("backprop_372x128", |g| {
        g.throughput(Throughput::Elements(FMA_BWD));

        let (ix, iw2, ih_pre, ih, dl_dy) = backprop_static_inputs(&inputs);
        let gw1_init = vec![0.0f64; N_FEATURES * N_HIDDEN];
        let gb1_init = vec![0.0f64; N_HIDDEN];
        let gw2_init = vec![0.0f64; N_HIDDEN];
        let gb2_init = vec![0.0f64; 1];

        let ix_c = ix.clone();
        let iw2_c = iw2.clone();
        let ih_pre_c = ih_pre.clone();
        let ih_c = ih.clone();
        let gw1_c = gw1_init.clone();
        let gb1_c = gb1_init.clone();
        let gw2_c = gw2_init.clone();
        let gb2_c = gb2_init.clone();
        g.bench("scalar", move |b| {
            // We need fresh mutable gradient buffers per iteration to
            // measure the kernel itself (not Vec::clone). The cost of
            // the clone is consistent across the two paths and lives
            // in `before/after` per-round overhead — zenbench's
            // measurement excludes it.
            let ix = ix_c.clone();
            let iw2 = iw2_c.clone();
            let ih_pre = ih_pre_c.clone();
            let ih = ih_c.clone();
            let mut gw1 = gw1_c.clone();
            let mut gb1 = gb1_c.clone();
            let mut gw2 = gw2_c.clone();
            let mut gb2 = gb2_c.clone();
            b.iter(|| {
                backprop_scalar_view(
                    black_box(&ix),
                    black_box(&ih_pre),
                    black_box(&ih),
                    black_box(dl_dy),
                    black_box(&mut gw1),
                    black_box(&mut gb1),
                    black_box(&iw2),
                    black_box(&mut gw2),
                    black_box(&mut gb2),
                    N_FEATURES,
                    N_HIDDEN,
                    ALPHA,
                );
                black_box(&gw1[0]);
            });
        });

        let ix_c = ix.clone();
        let iw2_c = iw2.clone();
        let ih_pre_c = ih_pre.clone();
        let ih_c = ih.clone();
        let gw1_c = gw1_init.clone();
        let gb1_c = gb1_init.clone();
        let gw2_c = gw2_init.clone();
        let gb2_c = gb2_init.clone();
        g.bench("simd_dispatch", move |b| {
            let ix = ix_c.clone();
            let iw2 = iw2_c.clone();
            let ih_pre = ih_pre_c.clone();
            let ih = ih_c.clone();
            let mut gw1 = gw1_c.clone();
            let mut gb1 = gb1_c.clone();
            let mut gw2 = gw2_c.clone();
            let mut gb2 = gb2_c.clone();
            b.iter(|| {
                simd_mlp::backprop_step(
                    black_box(&ix),
                    black_box(&ih_pre),
                    black_box(&ih),
                    black_box(dl_dy),
                    black_box(&mut gw1),
                    black_box(&mut gb1),
                    black_box(&iw2),
                    black_box(&mut gw2),
                    black_box(&mut gb2),
                    N_FEATURES,
                    N_HIDDEN,
                    ALPHA,
                );
                black_box(&gw1[0]);
            });
        });
    });

    suite.group("pair_372x128", |g| {
        g.throughput(Throughput::Elements(FMA_PAIR));

        // 2× forward, 2× backprop (matches the RankNet inner-loop work
        // in `mlp_train::train_mlp_with_tv`).
        let inputs_a = inputs;
        let inputs_b = build_inputs(0xBEEF_F00D_1234_5678);
        let gw1_init = vec![0.0f64; N_FEATURES * N_HIDDEN];
        let gb1_init = vec![0.0f64; N_HIDDEN];
        let gw2_init = vec![0.0f64; N_HIDDEN];
        let gb2_init = vec![0.0f64; 1];

        let ia = inputs_a;
        let ib = inputs_b;
        let gw1_c = gw1_init.clone();
        let gb1_c = gb1_init.clone();
        let gw2_c = gw2_init.clone();
        let gb2_c = gb2_init.clone();
        g.bench("scalar", move |b| {
            let mut gw1 = gw1_c.clone();
            let mut gb1 = gb1_c.clone();
            let mut gw2 = gw2_c.clone();
            let mut gb2 = gb2_c.clone();
            b.iter(|| {
                let (ya, ha_pre, ha) = forward_scalar_view(
                    black_box(&ia.x),
                    black_box(&ia.w1),
                    black_box(&ia.b1),
                    black_box(&ia.w2),
                    black_box(&ia.b2),
                    N_FEATURES,
                    N_HIDDEN,
                    ALPHA,
                );
                let (yb, hb_pre, hb) = forward_scalar_view(
                    black_box(&ib.x),
                    black_box(&ib.w1),
                    black_box(&ib.b1),
                    black_box(&ib.w2),
                    black_box(&ib.b2),
                    N_FEATURES,
                    N_HIDDEN,
                    ALPHA,
                );
                let pair_diff = yb - ya;
                let target = 1.0f64;
                let z = -target * pair_diff;
                let sig_z = 1.0 / (1.0 + (-z).exp());
                let dl_da = target * sig_z;
                let dl_db = -target * sig_z;
                backprop_scalar_view(
                    &ia.x, &ha_pre, &ha, dl_da, &mut gw1, &mut gb1, &ia.w2, &mut gw2, &mut gb2,
                    N_FEATURES, N_HIDDEN, ALPHA,
                );
                backprop_scalar_view(
                    &ib.x, &hb_pre, &hb, dl_db, &mut gw1, &mut gb1, &ib.w2, &mut gw2, &mut gb2,
                    N_FEATURES, N_HIDDEN, ALPHA,
                );
                black_box(&gw1[0]);
            });
        });

        let ia = build_inputs(0xCAFE_F00D_DEAD_BEEF);
        let ib = build_inputs(0xBEEF_F00D_1234_5678);
        let gw1_c = gw1_init.clone();
        let gb1_c = gb1_init.clone();
        let gw2_c = gw2_init.clone();
        let gb2_c = gb2_init.clone();
        g.bench("simd_dispatch", move |b| {
            let mut gw1 = gw1_c.clone();
            let mut gb1 = gb1_c.clone();
            let mut gw2 = gw2_c.clone();
            let mut gb2 = gb2_c.clone();
            b.iter(|| {
                let (ya, ha_pre, ha) = simd_mlp::forward(
                    black_box(&ia.x),
                    black_box(&ia.w1),
                    black_box(&ia.b1),
                    black_box(&ia.w2),
                    black_box(&ia.b2),
                    N_FEATURES,
                    N_HIDDEN,
                    ALPHA,
                );
                let (yb, hb_pre, hb) = simd_mlp::forward(
                    black_box(&ib.x),
                    black_box(&ib.w1),
                    black_box(&ib.b1),
                    black_box(&ib.w2),
                    black_box(&ib.b2),
                    N_FEATURES,
                    N_HIDDEN,
                    ALPHA,
                );
                let pair_diff = yb - ya;
                let target = 1.0f64;
                let z = -target * pair_diff;
                let sig_z = 1.0 / (1.0 + (-z).exp());
                let dl_da = target * sig_z;
                let dl_db = -target * sig_z;
                simd_mlp::backprop_step(
                    &ia.x, &ha_pre, &ha, dl_da, &mut gw1, &mut gb1, &ia.w2, &mut gw2, &mut gb2,
                    N_FEATURES, N_HIDDEN, ALPHA,
                );
                simd_mlp::backprop_step(
                    &ib.x, &hb_pre, &hb, dl_db, &mut gw1, &mut gb1, &ib.w2, &mut gw2, &mut gb2,
                    N_FEATURES, N_HIDDEN, ALPHA,
                );
                black_box(&gw1[0]);
            });
        });
    });
}

zenbench::main!(bench_kernels);

// ============================================================================
// Scalar shim — calls the private scalar fallback through the public
// dispatch (forced by feature-gating? no — the dispatch is at runtime).
// Instead, we inline the scalar algorithm here so the comparison is
// fair (no dispatch overhead in the "scalar" arm).
// ============================================================================

fn forward_scalar_view(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    let mut h_pre = b1.to_vec();
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for (acc, &w) in h_pre.iter_mut().zip(row.iter()) {
            *acc += s * w;
        }
    }
    let h: Vec<f64> = h_pre
        .iter()
        .map(|&v| if v >= 0.0 { v } else { alpha * v })
        .collect();
    let mut y = b2[0];
    for o in 0..n_hidden {
        y += h[o] * w2[o];
    }
    (y, h_pre, h)
}

#[allow(clippy::too_many_arguments)]
fn backprop_scalar_view(
    x: &[f64],
    h_pre: &[f64],
    h: &[f64],
    dl_dy: f64,
    gw1: &mut [f64],
    gb1: &mut [f64],
    w2: &[f64],
    gw2: &mut [f64],
    gb2: &mut [f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) {
    for o in 0..n_hidden {
        gw2[o] += dl_dy * h[o];
    }
    gb2[0] += dl_dy;

    let mut dl_dh_pre = vec![0.0f64; n_hidden];
    for o in 0..n_hidden {
        let dh = dl_dy * w2[o];
        dl_dh_pre[o] = if h_pre[o] >= 0.0 { dh } else { alpha * dh };
    }

    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &mut gw1[i * n_hidden..(i + 1) * n_hidden];
        for (g, &dh) in row.iter_mut().zip(dl_dh_pre.iter()) {
            *g += s * dh;
        }
    }
    for (g, &dh) in gb1.iter_mut().zip(dl_dh_pre.iter()) {
        *g += dh;
    }
}

// ============================================================================
// Helpers — RNG + input synthesis.
// ============================================================================

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
        let bits = self.next();
        (bits as f64 / u64::MAX as f64) * 2.0 - 1.0
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

fn backprop_static_inputs(inputs: &Inputs) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let (_, h_pre, h) = forward_scalar_view(
        &inputs.x, &inputs.w1, &inputs.b1, &inputs.w2, &inputs.b2, N_FEATURES, N_HIDDEN, ALPHA,
    );
    (inputs.x.clone(), inputs.w2.clone(), h_pre, h, 0.1)
}
