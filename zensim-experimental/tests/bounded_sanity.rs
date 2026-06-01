//! Bounded-by-construction score sanity for the relocated `LinearBounded`
//! profile: identical = 100, blur < 100, heavier blur scores lower, and every
//! score lands in [0, 100].
//!
//! This guards the metric that is DESIGNED to satisfy these axioms on the
//! entire input domain. The shipped `zensim::ZensimProfile::A` MLP does NOT
//! (it can return > 100 / rank heavier blur higher on off-manifold synthetic
//! content — the tracked known-limit asserted in `metric_invariants.rs`).
//! Relocated from `zensim/tests/cross_platform.rs` together with the
//! `LinearBounded` profile (2026-06-01).

mod common;

use common::generators::*;
use zensim::{RgbSlice, Zensim};

struct TestPair {
    name: &'static str,
    source: Vec<[u8; 3]>,
    distorted: Vec<[u8; 3]>,
}

fn generate_test_pairs(w: usize, h: usize) -> Vec<TestPair> {
    let checker = gen_checkerboard(w, h, 8);
    let mandel = gen_mandelbrot(w, h);
    let noise = gen_value_noise(w, h, 42);
    let blocks = gen_color_blocks(w, h);

    vec![
        TestPair {
            name: "checkerboard+blur",
            distorted: distort_blur(&checker, w, h, 3),
            source: checker.clone(),
        },
        TestPair {
            name: "checkerboard+sharpen",
            distorted: distort_sharpen(&checker, w, h),
            source: checker,
        },
        TestPair {
            name: "mandelbrot+blur",
            distorted: distort_blur(&mandel, w, h, 3),
            source: mandel.clone(),
        },
        TestPair {
            name: "mandelbrot+color_shift",
            distorted: distort_color_shift(&mandel, w, h),
            source: mandel,
        },
        TestPair {
            name: "noise+blur",
            distorted: distort_blur(&noise, w, h, 3),
            source: noise.clone(),
        },
        TestPair {
            name: "noise+block_artifacts",
            distorted: distort_block_artifacts(&noise, w, h),
            source: noise,
        },
        TestPair {
            name: "color_blocks+color_shift",
            distorted: distort_color_shift(&blocks, w, h),
            source: blocks.clone(),
        },
        TestPair {
            name: "color_blocks+sharpen",
            distorted: distort_sharpen(&blocks, w, h),
            source: blocks,
        },
    ]
}

#[test]
fn linear_bounded_score_sanity() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(zensim_experimental::linear_bounded());
    let source = gen_mandelbrot(W, H);

    // Identical images must score exactly 100.0
    let src = RgbSlice::new(&source, W, H);
    let identical = z.compute(&src, &src).expect("compute failed");
    println!(
        "  identical: score={:.15} raw_dist={:.15e}",
        identical.score(),
        identical.raw_distance(),
    );
    assert_eq!(
        identical.score(),
        100.0,
        "Identical images must score exactly 100.0, got {:.15} (raw_dist={:.15e})",
        identical.score(),
        identical.raw_distance(),
    );

    // Light blur → < 100
    let light_blur = distort_blur(&source, W, H, 1);
    let dst = RgbSlice::new(&light_blur, W, H);
    let light_result = z.compute(&src, &dst).expect("compute failed");
    assert!(
        light_result.score() < 100.0,
        "Light blur should score < 100, got {}",
        light_result.score(),
    );
    println!("  light blur (r=1): {:.6}", light_result.score());

    // Heavy blur → lower than light blur
    let heavy_blur = distort_blur(&source, W, H, 5);
    let dst = RgbSlice::new(&heavy_blur, W, H);
    let heavy_result = z.compute(&src, &dst).expect("compute failed");
    assert!(
        heavy_result.score() < light_result.score(),
        "Heavy blur ({:.4}) should be lower than light blur ({:.4})",
        heavy_result.score(),
        light_result.score(),
    );
    println!("  heavy blur (r=5): {:.6}", heavy_result.score());

    // LinearBounded guarantees the full [0, 100] range by construction.
    let pairs = generate_test_pairs(W, H);
    for pair in &pairs {
        let src = RgbSlice::new(&pair.source, W, H);
        let dst = RgbSlice::new(&pair.distorted, W, H);
        let result = z.compute(&src, &dst).expect("compute failed");
        assert!(
            (0.0..=100.0).contains(&result.score()),
            "{}: score {:.4} outside [0,100]",
            pair.name,
            result.score(),
        );
    }
    println!("  All scores in [0, 100] (bounded by construction)");
}
