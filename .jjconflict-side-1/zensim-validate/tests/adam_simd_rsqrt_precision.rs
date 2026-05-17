//! Measure the precision delta of the rsqrt-based Adam path vs the
//! scalar reference. This is documentation, not a gate — the rsqrt path
//! is not currently shipped (it's slower than vsqrtpd+vdivpd on Zen 4),
//! but if the FP-unit balance changes in a future µarch the precision
//! delta needs to stay sane.

#[path = "../src/adam_simd.rs"]
mod adam_simd;

#[cfg(target_arch = "x86_64")]
use adam_simd::adam_update_rsqrt_v4;
use adam_simd::{AdamUpdateArgs, adam_update_scalar_ref};

fn synth_state(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut state = seed;
    let mut nxt = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32) as f64 / u32::MAX as f64
    };
    let mut w = vec![0.0; n];
    let mut g = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut v = vec![0.0; n];
    for i in 0..n {
        w[i] = nxt() * 2.0 - 1.0;
        g[i] = (nxt() * 2.0 - 1.0) * 0.1;
        m[i] = (nxt() * 2.0 - 1.0) * 0.05;
        v[i] = match i % 4 {
            0 => 1e-12 + nxt() * 1e-12,
            1 => 1e-8 + nxt() * 1e-7,
            2 => 1e-4 + nxt() * 1e-3,
            _ => nxt() * 0.1,
        };
    }
    (w, g, m, v)
}

fn make_args<'a>(
    w: &'a mut [f64],
    g: &'a mut [f64],
    m: &'a mut [f64],
    v: &'a mut [f64],
    t: u64,
) -> AdamUpdateArgs<'a> {
    let beta1 = 0.9f64;
    let beta2 = 0.999f64;
    let eps = 1e-8f64;
    let bc1 = 1.0 - beta1.powi(t as i32);
    let bc2 = 1.0 - beta2.powi(t as i32);
    AdamUpdateArgs {
        w,
        g,
        m,
        v,
        beta1,
        beta2,
        eps,
        bc1,
        bc2,
        lr: 0.005,
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn rsqrt_path_precision_vs_scalar() {
    let n = 47_616;
    let (w0, g0, m0, v0) = synth_state(n, 0xC0DE);
    let (mut wa, mut ga, mut ma, mut va) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
    let (mut wb, mut gb, mut mb, mut vb) = (w0, g0, m0, v0);

    adam_update_scalar_ref(&mut make_args(&mut wa, &mut ga, &mut ma, &mut va, 100));
    adam_update_rsqrt_v4(&mut make_args(&mut wb, &mut gb, &mut mb, &mut vb, 100));

    // Compute max + mean relative error in w.
    let mut max_rel: f64 = 0.0;
    let mut sum_rel: f64 = 0.0;
    let mut max_idx = 0usize;
    for i in 0..n {
        let rel = (wa[i] - wb[i]).abs() / (wa[i].abs().max(wb[i].abs()).max(1e-30));
        sum_rel += rel;
        if rel > max_rel {
            max_rel = rel;
            max_idx = i;
        }
    }
    let mean_rel = sum_rel / n as f64;
    eprintln!(
        "rsqrt-vs-scalar: max_rel={:.3e} mean_rel={:.3e} at i={} (v={:e})",
        max_rel, mean_rel, max_idx, va[max_idx]
    );
    // Loose gate — 1e-9 relative is well within RankNet's loss precision.
    assert!(
        max_rel < 1e-9,
        "rsqrt path drifted too far: max_rel={max_rel:e} at i={max_idx}"
    );
}
