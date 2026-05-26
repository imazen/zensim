//! Regression test: byte-identical images must return score=100 for
//! every profile, including the MLP-based PreviewV0_5* family.
//!
//! Bug history (2026-05-19): `compute_with_config_inner` short-circuits
//! to score=100 when `images_byte_identical` returns true, then
//! `apply_mlp_scoring` runs unconditionally on the (all-zero) feature
//! vector and OVERWRITES the 100 with the MLP's raw output via
//! `result.set_mlp_score`. For V0_5* bakes with `skip_score_mapping=true`,
//! the MLP's raw output on zero-feature input is ~0 (V0_5Balanced) or
//! ~2 (V0_5Compression / V0_5Ensemble) — catastrophic for any caller
//! that relies on the identity-image invariant (zensim-target's quality
//! dial defaulted to V0_3 as a workaround, per commits 5e3e6ce0 +
//! f0ea29fb).
//!
//! Fix: gate `apply_mlp_scoring` on whether the input pair was
//! byte-identical. Done at the `Zensim::compute*` callsite layer so
//! the metric.rs MLP path is unchanged for non-identical input.

use zensim::{RgbSlice, Zensim, ZensimProfile};

/// Build a small RGB image with non-trivial content. The exact pattern
/// is irrelevant — the regression target is the byte-identical
/// short-circuit, not numeric stability.
fn make_solid_pattern(w: usize, h: usize) -> Vec<[u8; 3]> {
    (0..w * h)
        .map(|i| {
            let x = ((i % w) * 255 / w.max(1)) as u8;
            let y = ((i / w) * 255 / h.max(1)) as u8;
            [x, y, x.wrapping_add(y)]
        })
        .collect()
}

fn assert_identity_returns_100(profile: ZensimProfile, label: &str) {
    const W: usize = 64;
    const H: usize = 64;
    let pixels = make_solid_pattern(W, H);
    // Use a separate clone so we're not relying on pointer identity.
    let copy = pixels.clone();
    let src = RgbSlice::new(&pixels, W, H);
    let dst = RgbSlice::new(&copy, W, H);

    let z = Zensim::new(profile).with_parallel(false);
    let r = z.compute(&src, &dst).expect("compute failed");
    let score = r.score();
    let raw = r.raw_distance();

    assert!(
        (score - 100.0).abs() < 1e-6,
        "{label}: identity-image short-circuit overwritten by MLP; \
         expected score≈100.0, got {score} (raw_distance={raw})"
    );
    assert!(
        raw.abs() < 1e-6,
        "{label}: identity-image raw_distance must be 0.0, got {raw}"
    );
}

#[test]
fn v05_balanced_identity_returns_100() {
    assert_identity_returns_100(ZensimProfile::PreviewV0_5Balanced, "PreviewV0_5Balanced");
}

#[test]
fn v05_alias_identity_returns_100() {
    // PreviewV0_5 is the back-compat alias for PreviewV0_5Balanced. Same
    // params; the test guards against accidental drift in the alias
    // mapping.
    assert_identity_returns_100(ZensimProfile::PreviewV0_5, "PreviewV0_5");
}

#[test]
fn v05_compression_identity_returns_100() {
    assert_identity_returns_100(
        ZensimProfile::PreviewV0_5Compression,
        "PreviewV0_5Compression",
    );
}

#[test]
fn v05_ensemble_identity_returns_100() {
    assert_identity_returns_100(ZensimProfile::PreviewV0_5Ensemble, "PreviewV0_5Ensemble");
}

#[test]
fn v03_identity_returns_100() {
    // V0_3 (the prior MLP-based ship at PreviewV0_3) also has the same
    // bug — the MLP runs after the byte-identical short-circuit and
    // overwrites the 100. Lock in the fix for it too.
    assert_identity_returns_100(ZensimProfile::A, "A");
}

#[test]
fn v04_identity_returns_100() {
    // V0_4 (D2 multi-bake) has the same bug since it routes through
    // apply_mlp_scoring too. The pre-fix v04_mlp.rs test only asserted
    // `score >= 90`; this test pins the invariant tighter.
    assert_identity_returns_100(ZensimProfile::PreviewV0_4, "PreviewV0_4");
}

#[test]
fn v02_identity_returns_100() {
    // V0_2 is the linear (non-MLP) profile. The short-circuit returns
    // 100 directly here; this test guards against any future change
    // that would route V0_2 through apply_mlp_scoring.
    assert_identity_returns_100(ZensimProfile::PreviewV0_2, "PreviewV0_2");
}
