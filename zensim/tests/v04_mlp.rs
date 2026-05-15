//! V0_4 MLP dispatch end-to-end tests.
//!
//! V0_4 ships the 2026-04-30 trained 228 → 64 LeakyReLU → 1 MLP from
//! `zensim/weights/v0_4_2026-04-30.bin`, trained with synthetic +
//! KADID_train + TID_train mixed supervision. Outputs raw distance
//! (0..90 range, mean 2.8); runtime applies the classic
//! `100 - 18·d^0.7` mapping shared with V0_1 / V0_2.
//!
//! Gated behind `__experimental_versions` to match the profile's
//! feature gate.


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

    let z = Zensim::new(ZensimProfile::PreviewV0_3).with_parallel(false);
    let r = z.compute(&s, &d).unwrap();

    let score = r.score();
    assert!(
        (0.0..=100.0).contains(&score),
        "v0_4 score out of range: {score}"
    );
    assert_eq!(r.profile(), ZensimProfile::PreviewV0_3);
}

#[test]
fn v04_identical_inputs_near_perfect() {
    let (src, _) = make_test_pair(32, 32);
    let s = RgbSlice::new(&src, 32, 32);

    let z = Zensim::new(ZensimProfile::PreviewV0_3).with_parallel(false);
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
fn v04_degraded_does_not_exceed_identical() {
    // Heavy degradation so the prior-trained MLP definitely fires.
    // The V0_4 bake's score range can be saturated at the high end
    // for cleanly-rendered / unusual content, so the strict `<`
    // form is too brittle when degradation is mild — use a noise
    // floor that's clearly outside the training distribution.
    let (src, _) = make_test_pair(64, 64);
    let s = RgbSlice::new(&src, 64, 64);

    // Stronger degradation: half-amplitude inversion on every other
    // pixel pair — guaranteed to drop the score on any reasonable
    // perceptual model.
    let dst: Vec<[u8; 3]> = src
        .iter()
        .enumerate()
        .map(|(i, &[r, g, b])| {
            if i % 2 == 0 {
                [255 - r, 255 - g, 255 - b]
            } else {
                [r, g, b]
            }
        })
        .collect();
    let d = RgbSlice::new(&dst, 64, 64);

    let z = Zensim::new(ZensimProfile::PreviewV0_3).with_parallel(false);
    let r_self = z.compute(&s, &s).unwrap();
    let r_diff = z.compute(&s, &d).unwrap();

    assert!(
        r_diff.score() < r_self.score(),
        "heavily degraded score {} should be < identical score {}",
        r_diff.score(),
        r_self.score()
    );
}

#[test]
fn v04_compute_with_ref_matches_compute() {
    let (src, dst) = make_test_pair(96, 64);
    let s = RgbSlice::new(&src, 96, 64);
    let d = RgbSlice::new(&dst, 96, 64);

    let z = Zensim::new(ZensimProfile::PreviewV0_3).with_parallel(false);

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
    assert_eq!(ZensimProfile::PreviewV0_3.name(), "zensim-preview-v0.3");
}

#[test]
fn v04_profile_name_and_score() {
    // PreviewV0_4 is the D2 α=0.7 multi-bake ensemble (V_18 ship +
    // V_20 IS calibrated). Smoke-test that the profile loads both
    // bakes, scores in the 0..100 range, and produces a different
    // score from PreviewV0_3 (proving the second bake is actually
    // mixed in — if mlp_bytes_b3 were silently ignored, scores would
    // be identical).
    assert_eq!(ZensimProfile::PreviewV0_4.name(), "zensim-preview-v0.4");

    let (src, dst) = make_test_pair(64, 64);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);

    let z3 = Zensim::new(ZensimProfile::PreviewV0_3).with_parallel(false);
    let z4 = Zensim::new(ZensimProfile::PreviewV0_4).with_parallel(false);
    let r3 = z3.compute(&s, &d).unwrap();
    let r4 = z4.compute(&s, &d).unwrap();

    let s3 = r3.score();
    let s4 = r4.score();
    assert!(
        (0.0..=100.0).contains(&s4),
        "v0.4 score out of range: {s4}"
    );
    assert_eq!(r4.profile(), ZensimProfile::PreviewV0_4);
    // Different mix → different output. If mlp_bytes_b3 were ignored,
    // s4 would equal s3 exactly. We want them measurably different
    // to confirm the secondary bake is mixed in.
    assert!(
        (s3 - s4).abs() > 0.01,
        "PreviewV0_3 and PreviewV0_4 produced near-identical scores ({s3} vs {s4}); \
         the D2 secondary bake doesn't appear to be active"
    );
}
