//! V0_4 MLP dispatch end-to-end tests.
//!
//! The V0_4 placeholder is a single-layer 228 → 1 linear MLP whose
//! weights are `WEIGHTS_PREVIEW_V0_2 / num_scales`, so the MLP forward
//! output equals the V0_2 linear scorer's `raw_distance` (after
//! V0_2's post-divide). These tests verify the full dispatch path —
//! profile params → MLP load → forward → score override — produces
//! results matching V0_2 within f32 precision.

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
fn v04_placeholder_reproduces_v02_score() {
    let (src, dst) = make_test_pair(64, 64);
    let s = RgbSlice::new(&src, 64, 64);
    let d = RgbSlice::new(&dst, 64, 64);

    let z_v02 = Zensim::new(ZensimProfile::PreviewV0_2).with_parallel(false);
    let z_v04 = Zensim::new(ZensimProfile::PreviewV0_4).with_parallel(false);

    let r_v02 = z_v02.compute(&s, &d).unwrap();
    let r_v04 = z_v04.compute(&s, &d).unwrap();

    // Placeholder MLP runs the same dot product in f32 instead of
    // f64. Drift is ~1e-4 in raw_distance, ≪0.01 in score.
    let raw_diff = (r_v02.raw_distance() - r_v04.raw_distance()).abs();
    let score_diff = (r_v02.score() - r_v04.score()).abs();
    assert!(
        raw_diff < 1e-3,
        "v0_4 placeholder raw_distance drift: v02={} v04={} diff={raw_diff}",
        r_v02.raw_distance(),
        r_v04.raw_distance()
    );
    assert!(
        score_diff < 1e-2,
        "v0_4 placeholder score drift: v02={} v04={} diff={score_diff}",
        r_v02.score(),
        r_v04.score()
    );
    assert_eq!(r_v04.profile(), ZensimProfile::PreviewV0_4);
}

#[test]
fn v04_identical_images_score_100() {
    let (src, _) = make_test_pair(32, 32);
    let s = RgbSlice::new(&src, 32, 32);

    let z = Zensim::new(ZensimProfile::PreviewV0_4).with_parallel(false);
    let r = z.compute(&s, &s).unwrap();
    assert_eq!(r.score(), 100.0, "identical inputs must score exactly 100");
    assert_eq!(r.raw_distance(), 0.0);
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
