//! V0_4 MLP dispatch end-to-end tests.
//!
//! V0_4 ships the trained 228 → 64 LeakyReLU → 1 MLP from
//! `zensim/weights/v0_4_2026-05-07.bin` (val_srocc=0.9547,
//! test_srocc=0.9814). The bake's final layer is flipped to
//! "distance" semantics (0 = identical, 100 = worst), so the runtime
//! `score = 100 - 1·d^1` mapping with `(a=1, b=1)` produces ssim2-scale
//! output (0..100, 100 = identical).
//!
//! These tests assert the dispatch path — profile params → MLP load →
//! forward → mapped score — produces sane outputs in the right range
//! and tracks degradation monotonically.

use zensim::{RgbSlice, Zensim, ZensimProfile};

fn make_test_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
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
        .map(|&[r, g, b]| [r.saturating_add(8), g.saturating_add(4), b])
        .collect();
    (src, dst)
}

#[test]
fn v04_score_is_in_unit_range() {
    let (src, dst) = make_test_pair(64, 64);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);

    let z = Zensim::new(ZensimProfile::PreviewV0_4).with_parallel(false);
    let r = z.compute(&s, &d).unwrap();

    let score = r.score();
    assert!(
        (0.0..=100.0).contains(&score),
        "v0_4 score out of range: {score}"
    );
    assert_eq!(r.profile(), ZensimProfile::PreviewV0_4);
}

#[test]
fn v04_identical_inputs_near_perfect() {
    let (src, _) = make_test_pair(32, 32);
    let s = RgbSlice::new(&src, 32, 32);

    let z = Zensim::new(ZensimProfile::PreviewV0_4).with_parallel(false);
    let r = z.compute(&s, &s).unwrap();

    // Trained MLP + scaler: identical inputs produce all-zero raw
    // features, but the scaler shifts/scales each feature by its
    // train-set mean/std, so the standardized vector is non-zero and
    // the model output is some small finite distance — not exactly 0.
    // The CID22 paper anchors visually-lossless at MCOS≈90, so we
    // require identical inputs to score above 90 with comfortable
    // headroom.
    let score = r.score();
    assert!(
        score >= 90.0,
        "identical inputs should score >= 90 (visually lossless), got {score}"
    );
}

#[test]
fn v04_degraded_scores_lower_than_identical() {
    let (src, dst) = make_test_pair(64, 64);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);

    let z = Zensim::new(ZensimProfile::PreviewV0_4).with_parallel(false);
    let r_self = z.compute(&s, &s).unwrap();
    let r_diff = z.compute(&s, &d).unwrap();

    assert!(
        r_diff.score() < r_self.score(),
        "degraded score {} should be < identical score {}",
        r_diff.score(),
        r_self.score()
    );
}

#[test]
fn v04_compute_with_ref_matches_compute() {
    let (src, dst) = make_test_pair(96, 64);
    let s = RgbSlice::new(&src, 96, 64);
    let d = RgbSlice::new(&dst, 96, 64);

    let z = Zensim::new(ZensimProfile::PreviewV0_4).with_parallel(false);

    let r_direct = z.compute(&s, &d).unwrap();
    let pre = z.precompute_reference(&s).unwrap();
    let r_ref = z.compute_with_ref(&pre, &d).unwrap();

    let raw_diff = (r_direct.raw_distance() - r_ref.raw_distance()).abs();
    assert!(
        raw_diff < 1e-6,
        "compute vs compute_with_ref drift: {} vs {} diff={raw_diff}",
        r_direct.raw_distance(),
        r_ref.raw_distance()
    );
}

#[test]
fn v04_profile_name() {
    assert_eq!(ZensimProfile::PreviewV0_4.name(), "zensim-preview-v0.4");
}
