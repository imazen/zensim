//! Smoke test for `PreviewV0_5CrossCodec` runtime wiring.
//!
//! EXP-CROSS-CODEC-METRIC (2026-05-19) shipped a per-sample-α
//! architecture bake trained with a cross-codec equivalence-pair
//! loss. The bake bytes were already shipped at
//! `zensim/weights/v_cross_codec_2026-05-19.bin`, but the profile
//! variant `PreviewV0_5CrossCodec` and its `ProfileParams` slot
//! landed afterward (this test guards that wiring).
//!
//! What this test covers:
//!   1. `include_bytes!` resolves — the bake file exists at build time.
//!   2. The variant's `name()` / `params()` dispatch works.
//!   3. The runtime `forward_one_bake` dispatches through the
//!      `zentrain.per_sample_alpha_head` metadata path (no panic,
//!      finite output).
//!   4. The bake bytes are distinct from the Tuner bake (i.e. the
//!      include_bytes! is pointed at the right file).
//!
//! What this test does NOT cover (lives elsewhere — `bake_verdict`
//! against canonical val parquets):
//!   * Cross-corpus SROCC (CID22 / KADID / TID / KonJND / AIC-3).
//!   * Cross-codec mean pairwise butter at target zensim score.
//!   * Full Mohammadi panel per band.
//!
//! NOTE on synthetic test patterns: per the v05_two_trail_profile_smoke
//! comment in `v04_mlp.rs`, per-sample-α bakes trained on natural-photo
//! corpora extrapolate to compressed / saturated outputs on synthetic
//! gradient patterns. Score-in-range and finiteness are reliable smoke
//! checks; precise score values or large within-test spread on
//! gradient patterns are NOT — those would require natural-image
//! fixtures or the held-out corpus tooling.

use zensim::{RgbSlice, Zensim, ZensimProfile};

/// Build a deterministic (ref, distorted) pair with adjustable
/// distortion strength. Higher `delta` → more visible difference.
fn make_pair_with_delta(w: usize, h: usize, delta: u8) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = w * h;
    let src: Vec<[u8; 3]> = (0..n)
        .map(|i| {
            let x = ((i % w) * 255 / w) as u8;
            let y = ((i / w) * 255 / h) as u8;
            [x, y, x.wrapping_add(y)]
        })
        .collect();
    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|&[r, g, b]| {
            [
                r.saturating_add(delta),
                g.saturating_add(delta.saturating_sub(2)),
                b.saturating_add(delta / 2),
            ]
        })
        .collect();
    (src, dst)
}

#[test]
fn cross_codec_profile_name_and_alias() {
    assert_eq!(
        ZensimProfile::PreviewV0_5CrossCodec.name(),
        "zensim-preview-v0.5-cross-codec"
    );
    // The const constructor alias matches the variant.
    assert_eq!(
        ZensimProfile::cross_codec(),
        ZensimProfile::PreviewV0_5CrossCodec
    );
}

#[test]
fn cross_codec_score_in_range() {
    let (src, dst) = make_pair_with_delta(64, 64, 16);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);

    let z = Zensim::new(ZensimProfile::PreviewV0_5CrossCodec).with_parallel(false);
    let r = z.compute(&s, &d).unwrap();

    let score = r.score();
    assert!(
        score.is_finite(),
        "cross-codec score is NaN/Inf: {score} (per-sample-α dispatch may be broken)"
    );
    assert!(
        (0.0..=100.0).contains(&score),
        "cross-codec score out of range: {score}"
    );
    assert_eq!(r.profile(), ZensimProfile::PreviewV0_5CrossCodec);
}

#[test]
fn cross_codec_score_in_range_across_distortion_levels() {
    // Run the bake across several distortion levels and confirm every
    // output is finite + in [0, 100]. No spread / monotonicity check
    // (the bake's per-sample-α output compresses heavily on OOD
    // synthetic gradient patterns; spread-or-monotonicity assertions
    // would be brittle without natural-image fixtures).
    let z = Zensim::new(ZensimProfile::PreviewV0_5CrossCodec).with_parallel(false);
    for delta in (0..50).step_by(5) {
        let (src, dst) = make_pair_with_delta(64, 64, delta as u8);
        let s = RgbSlice::new(&src, 64, 64);
        let d = RgbSlice::new(&dst, 64, 64);
        let score = z.compute(&s, &d).unwrap().score();
        assert!(
            score.is_finite() && (0.0..=100.0).contains(&score),
            "cross-codec score out of range or NaN at delta={delta}: {score}"
        );
    }
}

#[test]
fn cross_codec_differs_from_tuner_on_typical_pair() {
    // PreviewV0_5CrossCodec and PreviewV0_5Tuner share the same
    // architecture (per-sample-α 372→128→128 identity) but were
    // trained with different loss functions (Tuner: MSE+mono on
    // safesyn only; CrossCodec: + cross-codec equivalence pair loss).
    // The weights differ. On a non-trivial pair the scores should
    // differ measurably — sanity check that the two bakes are
    // actually distinct files, not accidentally pointing at the
    // same bytes.
    //
    // NOTE: on OOD synthetic gradients both bakes may compress into
    // a similar score band. We use a moderate distortion level (16)
    // that drives the metric far enough from the saturation region
    // to expose weight differences. If this test flakes in the
    // future, increase the input variation (e.g. random noise) or
    // load a natural-image fixture.
    let (src, dst) = make_pair_with_delta(64, 64, 16);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);

    let z_cc = Zensim::new(ZensimProfile::PreviewV0_5CrossCodec).with_parallel(false);
    let z_t = Zensim::new(ZensimProfile::PreviewV0_5Tuner).with_parallel(false);
    let s_cc = z_cc.compute(&s, &d).unwrap().score();
    let s_t = z_t.compute(&s, &d).unwrap().score();
    assert!(
        s_cc.is_finite() && s_t.is_finite(),
        "non-finite score(s): cross-codec={s_cc} tuner={s_t}"
    );
    assert!(
        (s_cc - s_t).abs() > 0.01,
        "cross-codec and tuner produced near-identical scores ({s_cc} vs {s_t}); \
         the cross-codec bake may not have been wired correctly (different weights expected)"
    );
}
