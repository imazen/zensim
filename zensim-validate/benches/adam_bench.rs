//! Microbenchmark for AdamState::step's per-element kernel.
//!
//! Compares the scalar reference (`adam_update_scalar_ref`) against the
//! dispatched SIMD path (`adam_update`) on representative array sizes
//! taken from a real V_X training run:
//!
//! - `w1`: 372 × 128 = 47,616 (the dominant cost — feature → hidden)
//! - `b1`: 128
//! - `w2`: 128
//! - `b2`: 1
//!
//! Total: 47,873 params per Adam step × 4 (one call per array group)
//! × 9.5M pair updates over a 50,000-pair × 190-epoch V_22-IW run ≈
//! 9.5M Adam steps.

#[path = "../src/adam_simd.rs"]
mod adam_simd;

use adam_simd::{AdamUpdateArgs, adam_update, adam_update_scalar_ref};
#[cfg(target_arch = "x86_64")]
use adam_simd::{RsqrtPrecision, adam_update_rsqrt_v4, adam_update_rsqrt_v4_tiered};
use zenbench::black_box;

const N_W1: usize = 47_616;
const N_B1: usize = 128;
const N_W2: usize = 128;
const N_B2: usize = 1;
const N_TOTAL: usize = N_W1 + N_B1 + N_W2 + N_B2; // 47,873

// Adam hyperparameters — fixed at the trainer's defaults. `const` so
// the bench closures don't need to capture (and avoid 'static lifetime
// fights with the zenbench API).
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
const T_STEP: i32 = 100; // mid-training step; bias correction non-trivial.
const LR: f64 = 0.005;

#[inline]
fn bias_correction(beta: f64, t: i32) -> f64 {
    1.0 - beta.powi(t)
}

fn make_state(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut w = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut m = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    for i in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = ((state >> 33) as u32) as f64 / u32::MAX as f64;
        w.push(r * 2.0 - 1.0);
        g.push((r - 0.5) * 0.05);
        m.push(r * 0.01);
        // Mix of small (1e-12), medium (1e-4), and larger v values so
        // the sqrt + div hit a realistic dynamic range.
        v.push(match i % 4 {
            0 => 1e-12 + r * 1e-10,
            1 => 1e-8 + r * 1e-6,
            2 => 1e-4 + r * 1e-2,
            _ => r * 0.1,
        });
    }
    (w, g, m, v)
}

fn args_for<'a>(
    w: &'a mut [f64],
    g: &'a mut [f64],
    m: &'a mut [f64],
    v: &'a mut [f64],
) -> AdamUpdateArgs<'a> {
    AdamUpdateArgs {
        w,
        g,
        m,
        v,
        beta1: BETA1,
        beta2: BETA2,
        eps: EPS,
        bc1: bias_correction(BETA1, T_STEP),
        bc2: bias_correction(BETA2, T_STEP),
        lr: LR,
    }
}

type State = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn build_full_state() -> (State, State, State, State) {
    (
        make_state(N_W1),
        make_state(N_B1),
        make_state(N_W2),
        make_state(N_B2),
    )
}

fn run_scalar_full(s: &mut (State, State, State, State)) {
    {
        let mut a = args_for(&mut s.0.0, &mut s.0.1, &mut s.0.2, &mut s.0.3);
        adam_update_scalar_ref(&mut a);
    }
    {
        let mut a = args_for(&mut s.1.0, &mut s.1.1, &mut s.1.2, &mut s.1.3);
        adam_update_scalar_ref(&mut a);
    }
    {
        let mut a = args_for(&mut s.2.0, &mut s.2.1, &mut s.2.2, &mut s.2.3);
        adam_update_scalar_ref(&mut a);
    }
    {
        let mut a = args_for(&mut s.3.0, &mut s.3.1, &mut s.3.2, &mut s.3.3);
        adam_update_scalar_ref(&mut a);
    }
}

fn run_simd_full(s: &mut (State, State, State, State)) {
    {
        let mut a = args_for(&mut s.0.0, &mut s.0.1, &mut s.0.2, &mut s.0.3);
        adam_update(&mut a);
    }
    {
        let mut a = args_for(&mut s.1.0, &mut s.1.1, &mut s.1.2, &mut s.1.3);
        adam_update(&mut a);
    }
    {
        let mut a = args_for(&mut s.2.0, &mut s.2.1, &mut s.2.2, &mut s.2.3);
        adam_update(&mut a);
    }
    {
        let mut a = args_for(&mut s.3.0, &mut s.3.1, &mut s.3.2, &mut s.3.3);
        adam_update(&mut a);
    }
}

zenbench::main!(|suite| {
    // Full V_X step: 4 arrays (w1, b1, w2, b2). Mirrors the trainer's
    // AdamState::step() hot loop exactly.
    suite.compare("adam_step_full_47873_params", |group| {
        group.throughput(zenbench::Throughput::Elements(N_TOTAL as u64));
        group.throughput_unit("params");

        group.bench("scalar", |b| {
            b.with_input(build_full_state).run(|mut s| {
                run_scalar_full(&mut s);
                black_box(s)
            })
        });

        group.bench("simd", |b| {
            b.with_input(build_full_state).run(|mut s| {
                run_simd_full(&mut s);
                black_box(s)
            })
        });
    });

    // Per-array dominant cost — w1 = 47,616 elements drives ~99.5% of
    // the per-step work. Isolates the SIMD speedup on the longest
    // array, eliminating per-call dispatch overhead noise from the
    // 128/1-element arrays.
    suite.compare("adam_w1_only_47616_params", |group| {
        group.throughput(zenbench::Throughput::Elements(N_W1 as u64));
        group.throughput_unit("params");

        group.bench("scalar", |b| {
            b.with_input(|| make_state(N_W1)).run(|mut s| {
                let mut a = args_for(&mut s.0, &mut s.1, &mut s.2, &mut s.3);
                adam_update_scalar_ref(&mut a);
                black_box(s)
            })
        });

        group.bench("simd", |b| {
            b.with_input(|| make_state(N_W1)).run(|mut s| {
                let mut a = args_for(&mut s.0, &mut s.1, &mut s.2, &mut s.3);
                adam_update(&mut a);
                black_box(s)
            })
        });

        // rsqrt variants, one arm per precision tier (full = 2 NR steps,
        // nr1 = 1 NR step, estimate = raw 14-bit VRSQRT14PD/VRCP14PD).
        // The FULL tier historically DOES NOT BEAT vsqrtpd+vdivpd on
        // Zen 4 — the 2 NR iterations on rsqrt + 2 NR on rcp add more
        // cycles than the hardware sqrt/div latency saves. The lower
        // tiers exist per the 2026-08-05 tiered-precision ruling; their
        // costs are recorded in benchmarks/adam_rsqrt_tiers_2026-08-05.md.
        // If Zen 5 or Intel Sapphire Rapids changes the FP-unit balance,
        // rerun and reassess (methodology:
        // benchmarks/adam_simd_methodology_2026-05-17.md).
        #[cfg(target_arch = "x86_64")]
        group.bench("simd_rsqrt_full", |b| {
            b.with_input(|| make_state(N_W1)).run(|mut s| {
                let mut a = args_for(&mut s.0, &mut s.1, &mut s.2, &mut s.3);
                adam_update_rsqrt_v4(&mut a);
                black_box(s)
            })
        });

        #[cfg(target_arch = "x86_64")]
        group.bench("simd_rsqrt_nr1", |b| {
            b.with_input(|| make_state(N_W1)).run(|mut s| {
                let mut a = args_for(&mut s.0, &mut s.1, &mut s.2, &mut s.3);
                adam_update_rsqrt_v4_tiered(&mut a, RsqrtPrecision::Nr1);
                black_box(s)
            })
        });

        #[cfg(target_arch = "x86_64")]
        group.bench("simd_rsqrt_estimate", |b| {
            b.with_input(|| make_state(N_W1)).run(|mut s| {
                let mut a = args_for(&mut s.0, &mut s.1, &mut s.2, &mut s.3);
                adam_update_rsqrt_v4_tiered(&mut a, RsqrtPrecision::Estimate);
                black_box(s)
            })
        });
    });
});
