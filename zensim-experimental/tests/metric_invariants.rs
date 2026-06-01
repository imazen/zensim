//! Correct-by-construction invariant gate.
//!
//! A similarity metric must satisfy three axioms on its ENTIRE input
//! domain — not just on the in-distribution corpora SROCC is measured on:
//!
//!   1. **Boundedness**: `0 ≤ score ≤ 100` for every pair.
//!   2. **Self-identity is the unique maximum**: `score(x, x) = 100`
//!      and `score(x, y) ≤ 100`, with `< 100` for any real distortion.
//!   3. **Degradation monotonicity**: along a ladder of increasing
//!      distortion `D_t(x)`, `score(x, D_t(x))` is non-increasing in `t`.
//!
//! SROCC is rank-only and scale-invariant, so it is mathematically blind
//! to all three (see
//! `docs/METRIC_INVARIANTS_MECHANISM_AND_REDESIGN_2026-05-26.md`). This
//! gate exercises them directly across SYNTHETIC content (fractal,
//! checker, noise, smooth) — exactly the off-manifold region where an
//! unconstrained MLP misbehaves.
//!
//! [`zensim_experimental::linear_bounded()`] satisfies all three **by
//! construction** (non-negative weights × non-negative dissimilarity
//! features → distance `d ≥ 0`, mapped by the strictly-decreasing
//! bounded squash `100·exp(−(a/100)·d^b)`), so this gate passes for it.
//! The MLP profile `A` (V39) violates them — documented as a tracked
//! known-limit in `v39_known_limit_violations`, NOT relaxed.

mod common;

use common::generators::*;
use zensim::{RgbSlice, Zensim, ZensimProfile};

const W: usize = 128;
const H: usize = 128;
/// Float slack for the monotonicity comparison (box-blur diameter
/// rounding can perturb the distance by sub-ULP-scale amounts).
const MONO_EPS: f64 = 1e-6;

fn contents() -> Vec<(&'static str, Vec<[u8; 3]>)> {
    vec![
        ("mandelbrot", gen_mandelbrot(W, H)),
        ("checkerboard", gen_checkerboard(W, H, 8)),
        ("value_noise", gen_value_noise(W, H, 1)),
        ("color_blocks", gen_color_blocks(W, H)),
    ]
}

fn score(z: &Zensim, src: &[[u8; 3]], dst: &[[u8; 3]]) -> f64 {
    z.compute(&RgbSlice::new(src, W, H), &RgbSlice::new(dst, W, H))
        .unwrap()
        .score()
}

/// Axioms 1 + 2: boundedness and self-identity-maximal, across every
/// content class and a spread of distortion kinds.
#[test]
fn linear_bounded_is_bounded_and_self_identity_maximal() {
    let z = Zensim::new(zensim_experimental::linear_bounded()).with_parallel(false);
    for (name, src) in contents() {
        let s_id = score(&z, &src, &src);
        assert!(
            (s_id - 100.0).abs() < 1e-9,
            "{name}: self-identity must be exactly 100, got {s_id}"
        );
        let distortions: Vec<(&str, Vec<[u8; 3]>)> = vec![
            ("blur1", distort_blur(&src, W, H, 1)),
            ("blur5", distort_blur(&src, W, H, 5)),
            ("sharpen", distort_sharpen(&src, W, H)),
            ("color_shift", distort_color_shift(&src, W, H)),
            ("block", distort_block_artifacts(&src, W, H)),
        ];
        for (dname, dst) in &distortions {
            let s = score(&z, &src, dst);
            assert!(
                (0.0..=100.0).contains(&s),
                "{name}/{dname}: score {s} out of [0,100]"
            );
            assert!(
                s <= s_id + 1e-9,
                "{name}/{dname}: distorted score {s} exceeds self-identity {s_id}"
            );
        }
    }
}

/// Axiom 3: along an increasing blur ladder, the score is
/// non-increasing — a degraded image never scores higher than a
/// less-degraded one.
#[test]
fn linear_bounded_is_degradation_monotone() {
    let z = Zensim::new(zensim_experimental::linear_bounded()).with_parallel(false);
    for (name, src) in contents() {
        // t = 0 is the identity (score 100); t = 1..=6 is increasing blur.
        let mut prev = 100.0_f64;
        for r in 1..=6usize {
            let dst = distort_blur(&src, W, H, r);
            let s = score(&z, &src, &dst);
            assert!(
                (0.0..=100.0).contains(&s),
                "{name}: blur r={r} score {s} out of [0,100]"
            );
            assert!(
                s <= prev + MONO_EPS,
                "{name}: blur r={r} score {s} > previous {prev} (degradation must not raise the score)"
            );
            prev = s;
        }
    }
}

/// SROCC-equivalence guard: the bounded squash is a strictly-monotone
/// transform of the same distance as `PreviewV0_2`, so for any fixed
/// content the two profiles must rank a set of distortions identically.
#[test]
fn linear_bounded_preserves_v0_2_ranking() {
    let z_bounded = Zensim::new(zensim_experimental::linear_bounded()).with_parallel(false);
    let z_legacy = Zensim::new(ZensimProfile::PreviewV0_2).with_parallel(false);
    let src = gen_mandelbrot(W, H);
    let dists: Vec<Vec<[u8; 3]>> = (1..=6).map(|r| distort_blur(&src, W, H, r)).collect();
    let bounded: Vec<f64> = dists.iter().map(|d| score(&z_bounded, &src, d)).collect();
    let legacy: Vec<f64> = dists.iter().map(|d| score(&z_legacy, &src, d)).collect();
    // Same order: for every pair (i, j), the sign of the difference agrees.
    for i in 0..dists.len() {
        for j in (i + 1)..dists.len() {
            let sb = (bounded[i] - bounded[j]).signum();
            let sl = (legacy[i] - legacy[j]).signum();
            assert_eq!(
                sb, sl,
                "ranking divergence at ({i},{j}): bounded {bounded:?} vs legacy {legacy:?}"
            );
        }
    }
}

/// Axioms 1 + 2 for `ZensimProfile::A` — promoted into the gate 2026-05-27
/// when A was rotated to the masked-monotone-by-construction
/// `v47-strict-QAT-native` bake (this replaced the prior V39
/// `v39_known_limit_violations` characterization test, which asserted A
/// VIOLATED the axioms; V39 scored identity=0 and ranked heavier blur higher).
///
/// Differences from the `LinearBounded` gate: (1) A's self-identity is the
/// spline's top knot (~97.7), NOT exactly 100 — the axiom is that identity is
/// the unique *maximum*, not a fixed constant; (2) A has an INTENTIONAL
/// negative tail (a badly-corrupted pair scores well below 0 — the signal the
/// codec-regression / diffmap use case relies on), so there is NO [0, _] floor
/// — only the upper bound `≤ 100` and "no distortion exceeds identity".
#[test]
fn a_v47_is_bounded_above_and_self_identity_maximal() {
    let z = Zensim::new(ZensimProfile::A).with_parallel(false);
    for (name, src) in contents() {
        let s_id = score(&z, &src, &src);
        assert!(
            s_id <= 100.0 + 1e-9,
            "{name}: self-identity {s_id} exceeds the upper bound 100"
        );
        let distortions: Vec<(&str, Vec<[u8; 3]>)> = vec![
            ("blur1", distort_blur(&src, W, H, 1)),
            ("blur5", distort_blur(&src, W, H, 5)),
            ("sharpen", distort_sharpen(&src, W, H)),
            ("color_shift", distort_color_shift(&src, W, H)),
            ("block", distort_block_artifacts(&src, W, H)),
        ];
        for (dname, dst) in &distortions {
            let s = score(&z, &src, dst);
            assert!(
                s <= 100.0 + 1e-9,
                "{name}/{dname}: score {s} exceeds the upper bound 100"
            );
            assert!(
                s <= s_id + 1e-9,
                "{name}/{dname}: distorted score {s} exceeds self-identity {s_id}"
            );
        }
    }
}

/// Axiom 3 for `ZensimProfile::A`: along an increasing blur ladder the score
/// is non-increasing (degradation never raises the score). The score may go
/// negative (intentional corruption tail); only the ordering is gated.
#[test]
fn a_v47_is_degradation_monotone() {
    let z = Zensim::new(ZensimProfile::A).with_parallel(false);
    for (name, src) in contents() {
        let mut prev = score(&z, &src, &src); // identity (the maximum)
        for r in 1..=6usize {
            let dst = distort_blur(&src, W, H, r);
            let s = score(&z, &src, &dst);
            assert!(
                s <= 100.0 + 1e-9,
                "{name}: blur r={r} score {s} exceeds the upper bound 100"
            );
            assert!(
                s <= prev + MONO_EPS,
                "{name}: blur r={r} score {s} > previous {prev} (degradation must not raise the score)"
            );
            prev = s;
        }
    }
}
