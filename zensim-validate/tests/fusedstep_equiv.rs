//! Integration test: the fused K=1 pair update (backprop `dl_dh_pre` head +
//! `adam_update_w1_fused`) is bit-identical to the unfused sequence
//! (`backprop_step`×2 + layer-1 L2 + `adam_update`).
//!
//! `#[path]` pulls in `simd_mlp.rs` and `adam_simd.rs` directly, the same
//! convention as `bench_mlp_kernels.rs` / `adam_simd_equivalence.rs`; the
//! `tier_cap` shim matches the bench (allow every tier the host has).

mod tier_cap {
    pub fn avx512_allowed() -> bool {
        true
    }
}

#[path = "../src/simd_mlp.rs"]
#[allow(dead_code)]
mod simd_mlp;

#[path = "../src/adam_simd.rs"]
#[allow(dead_code)]
mod adam_simd;

use adam_simd::{AdamUpdateArgs, AdamW1FusedArgs, adam_update, adam_update_w1_fused};
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
const LR: f64 = 0.005;
const ALPHA: f64 = 0.01;

fn synth(n: usize, seed: &mut u64) -> Vec<f64> {
    let _ = n;
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 33) as u32) as f64 / u32::MAX as f64 * 2.0 - 1.0
        })
        .collect()
}

/// Bit-exact replica of `mlp_train::add_l2_grad_layer1` (which is private to
/// the trainer). Every arithmetic domain is mul+add on both the AVX and
/// scalar arms, so a scalar replica is bit-identical.
fn l2_oracle(g: &mut [f64], w: &[f64], scale: f64, n_hidden: usize, mult: Option<&[f64]>) {
    let n = g.len().min(w.len());
    let (g, w) = (&mut g[..n], &w[..n]);
    match mult {
        None => {
            for (g, &w) in g.iter_mut().zip(w.iter()) {
                *g += scale * w;
            }
        }
        Some(mult) => {
            for (feat, (grow, wrow)) in g.chunks_mut(n_hidden).zip(w.chunks(n_hidden)).enumerate() {
                let sm = scale * mult[feat];
                for (g, &w) in grow.iter_mut().zip(wrow.iter()) {
                    *g += sm * w;
                }
            }
        }
    }
}

fn adam(w: &mut [f64], g: &mut [f64], m: &mut [f64], v: &mut [f64]) {
    adam_update(&mut AdamUpdateArgs {
        w,
        g,
        m,
        v,
        beta1: BETA1,
        beta2: BETA2,
        eps: EPS,
        bc1: 1.0 - BETA1,
        bc2: 1.0 - BETA2,
        lr: LR,
    });
}

struct Net {
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
    mb1: Vec<f64>,
    vb1: Vec<f64>,
    mw2: Vec<f64>,
    vw2: Vec<f64>,
    mb2: Vec<f64>,
    vb2: Vec<f64>,
}

impl Net {
    fn new(nf: usize, nh: usize, seed: u64) -> Self {
        let mut s = seed;
        let nw1 = nf * nh;
        Net {
            w1: synth(nw1, &mut s),
            b1: synth(nh, &mut s),
            w2: synth(nh, &mut s),
            b2: synth(1, &mut s),
            // Non-zero gradient residue exercises the gw1 initial-value
            // read both paths must perform.
            gw1: synth(nw1, &mut s).iter().map(|x| x * 1e-4).collect(),
            gb1: synth(nh, &mut s),
            gw2: synth(nh, &mut s),
            gb2: synth(1, &mut s),
            mw1: synth(nw1, &mut s).iter().map(|x| x * 0.05).collect(),
            vw1: synth(nw1, &mut s)
                .iter()
                .map(|x| x.abs() * 0.1 + 1e-12)
                .collect(),
            mb1: synth(nh, &mut s),
            vb1: synth(nh, &mut s)
                .iter()
                .map(|x| x.abs() * 0.1 + 1e-12)
                .collect(),
            mw2: synth(nh, &mut s),
            vw2: synth(nh, &mut s)
                .iter()
                .map(|x| x.abs() * 0.1 + 1e-12)
                .collect(),
            mb2: synth(1, &mut s),
            vb2: synth(1, &mut s)
                .iter()
                .map(|x| x.abs() * 0.1 + 1e-12)
                .collect(),
        }
    }

    fn assert_same(&self, o: &Net, ctx: &str) {
        for (name, a, b) in [
            ("w1", &self.w1, &o.w1),
            ("gw1", &self.gw1, &o.gw1),
            ("mw1", &self.mw1, &o.mw1),
            ("vw1", &self.vw1, &o.vw1),
            ("b1", &self.b1, &o.b1),
            ("gb1", &self.gb1, &o.gb1),
            ("w2", &self.w2, &o.w2),
            ("gw2", &self.gw2, &o.gw2),
            ("b2", &self.b2, &o.b2),
            ("gb2", &self.gb2, &o.gb2),
        ] {
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{ctx}: {name}[{i}] differs: {x:e} vs {y:e}"
                );
            }
        }
    }
}

/// Unfused oracle: exactly the sequence `mlp_train` ran before the fusion —
/// two full `backprop_step`s, the layer-1 + layer-2 L2 passes, then four
/// independent `adam_update`s.
fn run_unfused(
    n: &mut Net,
    xa: &[f64],
    xb: &[f64],
    nf: usize,
    nh: usize,
    l2: f64,
    mult: Option<&[f64]>,
) {
    let (_, h_pre_a, h_a) = simd_mlp::forward(xa, &n.w1, &n.b1, &n.w2, &n.b2, nf, nh, ALPHA);
    let (_, h_pre_b, h_b) = simd_mlp::forward(xb, &n.w1, &n.b1, &n.w2, &n.b2, nf, nh, ALPHA);
    let dl_dya = 0.37_f64;
    let dl_dyb = -0.61_f64;
    simd_mlp::backprop_step(
        xa, &h_pre_a, &h_a, dl_dya, &mut n.gw1, &mut n.gb1, &n.w2, &mut n.gw2, &mut n.gb2, nf, nh,
        ALPHA,
    );
    simd_mlp::backprop_step(
        xb, &h_pre_b, &h_b, dl_dyb, &mut n.gw1, &mut n.gb1, &n.w2, &mut n.gw2, &mut n.gb2, nf, nh,
        ALPHA,
    );
    if l2 > 0.0 {
        l2_oracle(&mut n.gw1, &n.w1, l2, nh, mult);
        for (g, &w) in n.gw2.iter_mut().zip(n.w2.iter()) {
            *g += l2 * w;
        }
    }
    adam(&mut n.w1, &mut n.gw1, &mut n.mw1, &mut n.vw1);
    adam(&mut n.b1, &mut n.gb1, &mut n.mb1, &mut n.vb1);
    adam(&mut n.w2, &mut n.gw2, &mut n.mw2, &mut n.vw2);
    adam(&mut n.b2, &mut n.gb2, &mut n.mb2, &mut n.vb2);
}

/// Fused path: `backprop_grad_head`×2 produce `dl_dh_pre` + the
/// gw2/gb1/gb2 accumulations; `adam_update_w1_fused` folds both gw1 row
/// sweeps, the L2 term, and the w1 Adam step into one pass.
fn run_fused(
    n: &mut Net,
    xa: &[f64],
    xb: &[f64],
    nf: usize,
    nh: usize,
    l2: f64,
    mult: Option<&[f64]>,
) {
    let (_, h_pre_a, h_a) = simd_mlp::forward(xa, &n.w1, &n.b1, &n.w2, &n.b2, nf, nh, ALPHA);
    let (_, h_pre_b, h_b) = simd_mlp::forward(xb, &n.w1, &n.b1, &n.w2, &n.b2, nf, nh, ALPHA);
    let dl_dya = 0.37_f64;
    let dl_dyb = -0.61_f64;
    let mut dh_a = vec![0.0f64; nh];
    let mut dh_b = vec![0.0f64; nh];
    simd_mlp::backprop_grad_head(
        &h_pre_a, &h_a, dl_dya, &mut dh_a, &mut n.gb1, &n.w2, &mut n.gw2, &mut n.gb2, nh, ALPHA,
    );
    simd_mlp::backprop_grad_head(
        &h_pre_b, &h_b, dl_dyb, &mut dh_b, &mut n.gb1, &n.w2, &mut n.gw2, &mut n.gb2, nh, ALPHA,
    );
    if l2 > 0.0 {
        for (g, &w) in n.gw2.iter_mut().zip(n.w2.iter()) {
            *g += l2 * w;
        }
    }
    adam_update_w1_fused(&mut AdamW1FusedArgs {
        w: &mut n.w1,
        g: &mut n.gw1,
        m: &mut n.mw1,
        v: &mut n.vw1,
        xa,
        dha: &dh_a,
        xb,
        dhb: &dh_b,
        l2_scale: l2,
        l2_mult: mult,
        n_hidden: nh,
        beta1: BETA1,
        beta2: BETA2,
        eps: EPS,
        bc1: 1.0 - BETA1,
        bc2: 1.0 - BETA2,
        lr: LR,
    });
    adam(&mut n.b1, &mut n.gb1, &mut n.mb1, &mut n.vb1);
    adam(&mut n.w2, &mut n.gw2, &mut n.mw2, &mut n.vw2);
    adam(&mut n.b2, &mut n.gb2, &mut n.mb2, &mut n.vb2);
}

/// Deliberately-perturbed replica of the fused update: applies the L2 term
/// BEFORE side B's backprop contribution instead of after it. This is a
/// hand-written replica for the input-sensitivity check below — it never
/// calls the kernel under test. Any bit-difference vs the oracle proves
/// the test inputs are sensitive to accumulation order.
fn run_fused_wrong_order(
    n: &mut Net,
    xa: &[f64],
    xb: &[f64],
    nf: usize,
    nh: usize,
    l2: f64,
    mult: Option<&[f64]>,
) {
    let (_, h_pre_a, h_a) = simd_mlp::forward(xa, &n.w1, &n.b1, &n.w2, &n.b2, nf, nh, ALPHA);
    let (_, h_pre_b, h_b) = simd_mlp::forward(xb, &n.w1, &n.b1, &n.w2, &n.b2, nf, nh, ALPHA);
    let dl_dya = 0.37_f64;
    let dl_dyb = -0.61_f64;
    let mut dh_a = vec![0.0f64; nh];
    let mut dh_b = vec![0.0f64; nh];
    simd_mlp::backprop_grad_head(
        &h_pre_a, &h_a, dl_dya, &mut dh_a, &mut n.gb1, &n.w2, &mut n.gw2, &mut n.gb2, nh, ALPHA,
    );
    simd_mlp::backprop_grad_head(
        &h_pre_b, &h_b, dl_dyb, &mut dh_b, &mut n.gb1, &n.w2, &mut n.gw2, &mut n.gb2, nh, ALPHA,
    );
    if l2 > 0.0 {
        for (g, &w) in n.gw2.iter_mut().zip(n.w2.iter()) {
            *g += l2 * w;
        }
    }
    // Scalar perturbation of the fused element sequence: A → L2 → B.
    let l2_on = l2 > 0.0;
    for r in 0..nf {
        let sa = xa[r];
        let sb = xb[r];
        let sm = match mult {
            Some(m) => l2 * m[r],
            None => l2,
        };
        for j in 0..nh {
            let i = r * nh + j;
            let mut g = n.gw1[i];
            if sa != 0.0 {
                g = sa.mul_add(dh_a[j], g);
            }
            if l2_on {
                g += sm * n.w1[i];
            }
            if sb != 0.0 {
                g = sb.mul_add(dh_b[j], g);
            }
            n.gw1[i] = g;
        }
    }
    adam(&mut n.w1, &mut n.gw1, &mut n.mw1, &mut n.vw1);
    adam(&mut n.b1, &mut n.gb1, &mut n.mb1, &mut n.vb1);
    adam(&mut n.w2, &mut n.gw2, &mut n.mw2, &mut n.vw2);
    adam(&mut n.b2, &mut n.gb2, &mut n.mb2, &mut n.vb2);
}

#[test]
fn fused_pair_update_bit_identical() {
    // Run the whole equivalence sweep under every token permutation so
    // each dispatch tier's fused path — including the v3 kernel on hosts
    // where AVX-512 would otherwise route the dispatcher to v4 — is
    // compared against the real unfused code (both sides dispatch under
    // the same restriction). Without this, `fused_pair_*` never reaches
    // the v3 kernel on an AVX-512 host.
    let _lock = archmage::testing::lock_token_testing();
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        // nh % 4 == 0 in every case — the call-site gate's shape domain.
        // Includes the production shapes (944 features come from H32's real
        // feature count; 372×128 covers the bench shape).
        for &(nf, nh) in &[
            (4usize, 4usize),
            (8, 4),
            (16, 8),
            (13, 16),
            (944, 32),
            (372, 128),
            (944, 128),
        ] {
            for has_mult in [false, true] {
                for l2 in [0.0f64, 1e-5, 0.3] {
                    let mut seed = 0x1234_5678_9abc_def0u64 ^ (nf as u64) << 32 ^ nh as u64;
                    let xa: Vec<f64> = synth(nf, &mut seed)
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| if i % 7 == 3 { 0.0 } else { x.abs() })
                        .collect();
                    let xb: Vec<f64> = synth(nf, &mut seed)
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| if i % 5 == 1 { 0.0 } else { x.abs() })
                        .collect();
                    let mult: Option<Vec<f64>> = if has_mult {
                        Some(synth(nf, &mut seed).iter().map(|x| x.abs() + 0.5).collect())
                    } else {
                        None
                    };
                    let ctx = format!("{} nf={nf} nh={nh} mult={has_mult} l2={l2}", perm.label);
                    let mut o = Net::new(nf, nh, seed ^ 0xdead);
                    let mut f = Net::new(nf, nh, seed ^ 0xdead);
                    run_unfused(&mut o, &xa, &xb, nf, nh, l2, mult.as_deref());
                    run_fused(&mut f, &xa, &xb, nf, nh, l2, mult.as_deref());
                    f.assert_same(&o, &ctx);
                }
            }
        }
    });
    eprintln!("permutations run: {}", report.permutations_run);
}

#[test]
fn fused_pair_update_negative_control() {
    // Input-sensitivity check (NOT a kernel negative control): the
    // perturbed ordering (A → L2 → B, applied by a hand-written replica in
    // `run_fused_wrong_order`, not by the kernel) must produce observably
    // different bits; otherwise these test inputs could not detect an
    // order change at all. Run at production scale where ~30k lanes make
    // an accidental all-equal outcome astronomically unlikely, and require
    // only that at least one element differs. Kernel-level mutation
    // evidence (reordered v3 kernels are caught by the per-tier and
    // equivalence tests) is recorded in the lane report / Opus review
    // mutation table.
    for &(nf, nh) in &[(944usize, 32usize), (372, 128)] {
        let mut seed = 0xfeed_beef_cafe_f00du64 ^ (nf as u64) << 32 ^ nh as u64;
        let xa = synth(nf, &mut seed)
            .iter()
            .map(|x| x.abs() + 0.25)
            .collect::<Vec<_>>();
        let xb = synth(nf, &mut seed)
            .iter()
            .map(|x| x.abs() + 0.5)
            .collect::<Vec<_>>();
        let mut o = Net::new(nf, nh, seed ^ 0x1234);
        let mut p = Net::new(nf, nh, seed ^ 0x1234);
        run_unfused(&mut o, &xa, &xb, nf, nh, 1e-5, None);
        run_fused_wrong_order(&mut p, &xa, &xb, nf, nh, 1e-5, None);
        let mut diffs = 0usize;
        for (a, b) in [
            (&o.w1, &p.w1),
            (&o.gw1, &p.gw1),
            (&o.mw1, &p.mw1),
            (&o.vw1, &p.vw1),
        ] {
            diffs += a
                .iter()
                .zip(b.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
        }
        assert!(
            diffs > 0,
            "negative control failed to perturb: nf={nf} nh={nh} produced identical bits"
        );
    }
}
