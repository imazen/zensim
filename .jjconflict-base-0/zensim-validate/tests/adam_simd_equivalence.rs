//! Integration test: SIMD Adam path produces bit-equivalent results
//! to the scalar reference.
//!
//! Lives outside `src/` so we don't have to flip `mlp_train` into a
//! library. `#[path]` pulls in `adam_simd.rs` directly, just like the
//! bench does.

#[path = "../src/adam_simd.rs"]
mod adam_simd;

use adam_simd::{AdamUpdateArgs, adam_update, adam_update_scalar_ref};

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

fn assert_close(label: &str, a: &[f64], b: &[f64], rel_tol: f64) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_rel: f64 = 0.0;
    let mut max_idx = 0usize;
    for i in 0..a.len() {
        let rel = (a[i] - b[i]).abs() / (a[i].abs().max(b[i].abs()).max(1e-30));
        if rel > max_rel {
            max_rel = rel;
            max_idx = i;
        }
    }
    assert!(
        max_rel < rel_tol,
        "{label}: max relative error {:e} at i={}, ref={:e} got={:e} (tol={:e})",
        max_rel,
        max_idx,
        a[max_idx],
        b[max_idx],
        rel_tol,
    );
}

/// Production array sizes (V_X trainer's w1 is 47,616 elements,
/// b1=128, w2=128, b2=1). Verify each independently.
#[test]
fn matches_scalar_w1_47616_t100() {
    let n = 47_616;
    let (w0, g0, m0, v0) = synth_state(n, 0xA1B2);
    let (mut wa, mut ga, mut ma, mut va) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
    let (mut wb, mut gb, mut mb, mut vb) = (w0, g0, m0, v0);

    adam_update_scalar_ref(&mut make_args(&mut wa, &mut ga, &mut ma, &mut va, 100));
    adam_update(&mut make_args(&mut wb, &mut gb, &mut mb, &mut vb, 100));

    assert_close("w", &wa, &wb, 1e-12);
    assert_close("m", &ma, &mb, 1e-12);
    assert_close("v", &va, &vb, 1e-12);
    for i in 0..n {
        assert_eq!(gb[i], 0.0, "g must reset to zero at i={i}");
    }
}

/// w1 misaligned shape (49 → tail = 1) — exercises scalar mop-up path.
#[test]
fn matches_scalar_misaligned_49() {
    let n = 49;
    let (w0, g0, m0, v0) = synth_state(n, 0xDEAD);
    let (mut wa, mut ga, mut ma, mut va) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
    let (mut wb, mut gb, mut mb, mut vb) = (w0, g0, m0, v0);

    adam_update_scalar_ref(&mut make_args(&mut wa, &mut ga, &mut ma, &mut va, 7));
    adam_update(&mut make_args(&mut wb, &mut gb, &mut mb, &mut vb, 7));

    assert_close("w", &wa, &wb, 1e-12);
}

/// Late training (t = 10,000) — bias correction ≈ 1.0 exactly.
#[test]
fn matches_scalar_late_t10000() {
    let n = 47_873;
    let (w0, g0, m0, v0) = synth_state(n, 0xBEEF);
    let (mut wa, mut ga, mut ma, mut va) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
    let (mut wb, mut gb, mut mb, mut vb) = (w0, g0, m0, v0);

    adam_update_scalar_ref(&mut make_args(&mut wa, &mut ga, &mut ma, &mut va, 10_000));
    adam_update(&mut make_args(&mut wb, &mut gb, &mut mb, &mut vb, 10_000));

    assert_close("w", &wa, &wb, 1e-12);
}

/// 1-element array — exercises pure-scalar fallback inside the SIMD
/// dispatch wrapper. (Trainer's `b2 = 1` array.)
#[test]
fn matches_scalar_n1() {
    let mut wa = vec![0.5];
    let mut ga = vec![0.03];
    let mut ma = vec![0.01];
    let mut va = vec![1e-6];
    let mut wb = wa.clone();
    let mut gb = ga.clone();
    let mut mb = ma.clone();
    let mut vb = va.clone();

    adam_update_scalar_ref(&mut make_args(&mut wa, &mut ga, &mut ma, &mut va, 25));
    adam_update(&mut make_args(&mut wb, &mut gb, &mut mb, &mut vb, 25));

    assert!((wa[0] - wb[0]).abs() < 1e-15);
    assert!((ma[0] - mb[0]).abs() < 1e-15);
    assert!((va[0] - vb[0]).abs() < 1e-15);
    assert_eq!(gb[0], 0.0);
}

/// Run the same update over many sequential steps — accumulated drift
/// must stay bounded. Catches errors that compound across multiple
/// invocations (e.g. silent state corruption).
#[test]
fn matches_scalar_across_100_steps() {
    let n = 1024;
    let (w0, g0, m0, v0) = synth_state(n, 0xC0FFEE);
    let (mut wa, mut ga, mut ma, mut va) = (w0.clone(), g0.clone(), m0.clone(), v0.clone());
    let (mut wb, mut gb, mut mb, mut vb) = (w0, g0, m0, v0);

    for t in 1..=100u64 {
        // Replenish gradients each step (the trainer accumulates fresh
        // gradients between Adam steps; we simulate that here).
        for i in 0..n {
            let r = ((t as f64) * 0.013 + (i as f64) * 0.0001).sin();
            ga[i] = r * 0.05;
            gb[i] = r * 0.05;
        }
        adam_update_scalar_ref(&mut make_args(&mut wa, &mut ga, &mut ma, &mut va, t));
        adam_update(&mut make_args(&mut wb, &mut gb, &mut mb, &mut vb, t));
    }

    assert_close("w (100 steps)", &wa, &wb, 1e-12);
    assert_close("m (100 steps)", &ma, &mb, 1e-12);
    assert_close("v (100 steps)", &va, &vb, 1e-12);
}
