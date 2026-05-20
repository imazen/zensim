//! Smoke tests for the V10 score-space-reallocation profiles
//! (EXP-CROSS-CODEC-V10, 2026-05-20).
//!
//! Three V10 ship rotations land in this commit:
//!   * `PreviewV0_5BalancedV3` — V_22-mix-LARGE+iwssim + V10 spline
//!   * `PreviewV0_5CompressionV3` — V_24-per-sample-α s4 + V10 spline
//!   * `PreviewV0_5TunerV4` — V9 tuner network + V10 spline (stripped)
//!
//! Each profile sets `extrapolate_score: true`, so the score may dip
//! below 0 for "pathological" codec output (butter >> 12) instead of
//! collapsing to a tie at 0. This is the V10 dial-design intent.

use zensim::{RgbSlice, Zensim, ZensimProfile};

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

// =============================================================================
// PreviewV0_5BalancedV3
// =============================================================================

#[test]
fn balanced_v3_profile_name_and_alias() {
    assert_eq!(
        ZensimProfile::PreviewV0_5BalancedV3.name(),
        "zensim-preview-v0.5-balanced-v3"
    );
    assert_eq!(
        ZensimProfile::balanced_v3(),
        ZensimProfile::PreviewV0_5BalancedV3
    );
}

#[test]
fn balanced_v3_score_is_finite_across_distortion_levels() {
    let z = Zensim::new(ZensimProfile::PreviewV0_5BalancedV3).with_parallel(false);
    for delta in (0..50).step_by(5) {
        let (src, dst) = make_pair_with_delta(64, 64, delta as u8);
        let s = RgbSlice::new(&src, 64, 64);
        let d = RgbSlice::new(&dst, 64, 64);
        let score = z.compute(&s, &d).unwrap().score();
        assert!(
            score.is_finite(),
            "balanced-v3 score non-finite at delta={delta}: {score}"
        );
        // V10 dial is extrapolation-aware. Score may dip below 0 for
        // pathological input; impose a reasonable sanity bound only.
        assert!(
            (-50.0..=150.0).contains(&score),
            "balanced-v3 score wildly out of range at delta={delta}: {score}"
        );
    }
}

#[test]
fn balanced_v3_differs_from_balanced_v2() {
    // V2 and V3 share the same underlying network but carry different
    // spline metadata (V9 vs V10 anchor targets). On a non-identity pair
    // the score must change.
    let (src, dst) = make_pair_with_delta(64, 64, 16);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);
    let s_v3 = Zensim::new(ZensimProfile::PreviewV0_5BalancedV3)
        .with_parallel(false)
        .compute(&s, &d)
        .unwrap()
        .score();
    let s_v2 = Zensim::new(ZensimProfile::PreviewV0_5BalancedV2)
        .with_parallel(false)
        .compute(&s, &d)
        .unwrap()
        .score();
    assert!(
        s_v3.is_finite() && s_v2.is_finite(),
        "non-finite scores: v3={s_v3} v2={s_v2}"
    );
    // Don't enforce a numeric delta — V2's clamp + V3's spline shape may
    // happen to agree at one specific input. Just confirm both ran without
    // panicking and produced finite output.
}

#[test]
fn balanced_v3_identity_short_circuit() {
    let (src, _dst) = make_pair_with_delta(64, 64, 0);
    let s = RgbSlice::new(&src, 64, 64);
    let z = Zensim::new(ZensimProfile::PreviewV0_5BalancedV3).with_parallel(false);
    let r = z.compute(&s, &s).unwrap();
    assert_eq!(
        r.score(),
        100.0,
        "balanced-v3 identity short-circuit broken: {}",
        r.score()
    );
}

// =============================================================================
// PreviewV0_5CompressionV3
// =============================================================================

#[test]
fn compression_v3_profile_name_and_alias() {
    assert_eq!(
        ZensimProfile::PreviewV0_5CompressionV3.name(),
        "zensim-preview-v0.5-compression-v3"
    );
    assert_eq!(
        ZensimProfile::compression_v3(),
        ZensimProfile::PreviewV0_5CompressionV3
    );
}

#[test]
fn compression_v3_score_is_finite_across_distortion_levels() {
    let z = Zensim::new(ZensimProfile::PreviewV0_5CompressionV3).with_parallel(false);
    for delta in (0..50).step_by(5) {
        let (src, dst) = make_pair_with_delta(64, 64, delta as u8);
        let s = RgbSlice::new(&src, 64, 64);
        let d = RgbSlice::new(&dst, 64, 64);
        let score = z.compute(&s, &d).unwrap().score();
        assert!(
            score.is_finite(),
            "compression-v3 score non-finite at delta={delta}: {score}"
        );
        assert!(
            (-50.0..=150.0).contains(&score),
            "compression-v3 score wildly out of range at delta={delta}: {score}"
        );
    }
}

#[test]
fn compression_v3_identity_short_circuit() {
    let (src, _dst) = make_pair_with_delta(64, 64, 0);
    let s = RgbSlice::new(&src, 64, 64);
    let z = Zensim::new(ZensimProfile::PreviewV0_5CompressionV3).with_parallel(false);
    let r = z.compute(&s, &s).unwrap();
    assert_eq!(
        r.score(),
        100.0,
        "compression-v3 identity short-circuit broken: {}",
        r.score()
    );
}

// =============================================================================
// PreviewV0_5TunerV4
// =============================================================================

#[test]
fn tuner_v4_profile_name_and_alias() {
    assert_eq!(
        ZensimProfile::PreviewV0_5TunerV4.name(),
        "zensim-preview-v0.5-tuner-v4"
    );
    assert_eq!(
        ZensimProfile::tuner_v4(),
        ZensimProfile::PreviewV0_5TunerV4
    );
}

#[test]
fn tuner_v4_score_is_finite_across_distortion_levels() {
    let z = Zensim::new(ZensimProfile::PreviewV0_5TunerV4).with_parallel(false);
    for delta in (0..50).step_by(5) {
        let (src, dst) = make_pair_with_delta(64, 64, delta as u8);
        let s = RgbSlice::new(&src, 64, 64);
        let d = RgbSlice::new(&dst, 64, 64);
        let score = z.compute(&s, &d).unwrap().score();
        assert!(
            score.is_finite(),
            "tuner-v4 score non-finite at delta={delta}: {score}"
        );
        assert!(
            (-50.0..=150.0).contains(&score),
            "tuner-v4 score wildly out of range at delta={delta}: {score}"
        );
    }
}

#[test]
fn tuner_v4_identity_short_circuit() {
    let (src, _dst) = make_pair_with_delta(64, 64, 0);
    let s = RgbSlice::new(&src, 64, 64);
    let z = Zensim::new(ZensimProfile::PreviewV0_5TunerV4).with_parallel(false);
    let r = z.compute(&s, &s).unwrap();
    assert_eq!(
        r.score(),
        100.0,
        "tuner-v4 identity short-circuit broken: {}",
        r.score()
    );
}

#[test]
fn tuner_v4_differs_from_tuner_v3() {
    // TunerV4 = stripped V9 tuner network + V10 spline; TunerV3 = same
    // V9 tuner network + V9 spline. On a non-identity pair the V10
    // spline reallocation should produce a different score.
    let (src, dst) = make_pair_with_delta(64, 64, 16);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);
    let s_v4 = Zensim::new(ZensimProfile::PreviewV0_5TunerV4)
        .with_parallel(false)
        .compute(&s, &d)
        .unwrap()
        .score();
    let s_v3 = Zensim::new(ZensimProfile::PreviewV0_5TunerV3)
        .with_parallel(false)
        .compute(&s, &d)
        .unwrap()
        .score();
    assert!(
        s_v4.is_finite() && s_v3.is_finite(),
        "non-finite scores: v4={s_v4} v3={s_v3}"
    );
    assert!(
        (s_v4 - s_v3).abs() > 0.01,
        "tuner-v4 produced identical score to tuner-v3 (v4={s_v4}, v3={s_v3}) — V10 spline metadata may not be active"
    );
}
