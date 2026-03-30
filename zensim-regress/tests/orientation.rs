//! Integration tests for EXIF orientation handling using codec-corpus
//! Landscape and Portrait JPEG test images (orientations 1–8).
//!
//! These images contain the same visual scene but were independently JPEG-encoded
//! per orientation, resulting in significant pixel differences (mean abs diff ~14,
//! PSNR ~22 dB) even after applying the correct transform. Because of this,
//! zensim scores are negative for all orientation pairs.
//!
//! These tests verify:
//! - Correct dimension detection (same-dim for 1-4, swapped for 5-8)
//! - Correct DimensionMismatchKind categorization
//! - The scoring pipeline doesn't panic on real JPEG images
//!
//! For transform detection correctness with pixel-identical content,
//! see the unit tests in testing.rs (synthetic gradient images).

use image::RgbaImage;
use zensim::{Zensim, ZensimProfile};
use zensim_regress::testing::{
    DimensionMismatchKind, RegressionTolerance, check_regression, check_regression_resized,
};

fn orientation_dir() -> std::path::PathBuf {
    let corpus = codec_corpus::Corpus::new().expect("failed to initialize codec-corpus");
    corpus
        .get("imageflow/test_inputs/orientation")
        .expect("failed to get orientation test images")
}

fn load_rgba(path: &std::path::Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    let rgba: RgbaImage = img.into_rgba8();
    let (w, h) = rgba.dimensions();
    (rgba.into_raw(), w, h)
}

fn px(rgba: &[u8]) -> Vec<[u8; 4]> {
    rgba.chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

#[test]
fn orientation_1_self_comparison_scores_100() {
    let dir = orientation_dir();
    let z = Zensim::new(ZensimProfile::latest());
    let tol = RegressionTolerance::exact();

    let (rgba, w, h) = load_rgba(&dir.join("Landscape_1.jpg"));
    let report = check_regression(
        &z,
        &zensim::RgbaSlice::new(&px(&rgba), w as usize, h as usize),
        &zensim::RgbaSlice::new(&px(&rgba), w as usize, h as usize),
        &tol,
    )
    .unwrap();

    assert_eq!(report.score(), 100.0);
    assert!(report.passed());
}

#[test]
fn same_dim_orientations_have_matching_dimensions() {
    let dir = orientation_dir();
    let (_, rw, rh) = load_rgba(&dir.join("Landscape_1.jpg"));

    for orient in [2, 3, 4] {
        let (_, aw, ah) = load_rgba(&dir.join(format!("Landscape_{orient}.jpg")));
        assert_eq!(
            (rw, rh), (aw, ah),
            "Landscape_{orient} should have same dims as Landscape_1",
        );
    }
}

#[test]
fn swapped_dim_orientations_have_swapped_dimensions() {
    let dir = orientation_dir();
    let (_, rw, rh) = load_rgba(&dir.join("Landscape_1.jpg"));

    for orient in [5, 6, 7, 8] {
        let (_, aw, ah) = load_rgba(&dir.join(format!("Landscape_{orient}.jpg")));
        assert_eq!(
            (rw, rh), (ah, aw),
            "Landscape_{orient} should have swapped dims vs Landscape_1 \
             (expected {rh}x{rw}, got {aw}x{ah})",
        );
    }
}

#[test]
fn swapped_orientations_classified_as_orientation_swap() {
    let dir = orientation_dir();
    let z = Zensim::new(ZensimProfile::latest());
    let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);

    let (ref_rgba, rw, rh) = load_rgba(&dir.join("Landscape_1.jpg"));

    for orient in [5, 6, 7, 8] {
        let (act_rgba, aw, ah) = load_rgba(&dir.join(format!("Landscape_{orient}.jpg")));

        let report =
            check_regression_resized(&z, &ref_rgba, rw, rh, &act_rgba, aw, ah, &tol).unwrap();

        let dim = report.dimension_info().unwrap();
        assert_eq!(
            dim.kind,
            DimensionMismatchKind::OrientationSwap,
            "Landscape_{orient}: expected OrientationSwap, got {:?}",
            dim.kind,
        );
    }
}

#[test]
fn portrait_swapped_orientations_classified_as_orientation_swap() {
    let dir = orientation_dir();
    let z = Zensim::new(ZensimProfile::latest());
    let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);

    let (ref_rgba, rw, rh) = load_rgba(&dir.join("Portrait_1.jpg"));

    for orient in [5, 6, 7, 8] {
        let (act_rgba, aw, ah) = load_rgba(&dir.join(format!("Portrait_{orient}.jpg")));

        let report =
            check_regression_resized(&z, &ref_rgba, rw, rh, &act_rgba, aw, ah, &tol).unwrap();

        let dim = report.dimension_info().unwrap();
        assert_eq!(
            dim.kind,
            DimensionMismatchKind::OrientationSwap,
            "Portrait_{orient}: expected OrientationSwap, got {:?}",
            dim.kind,
        );
    }
}

#[test]
fn all_orientation_pairs_produce_reports_without_panic() {
    let dir = orientation_dir();
    let z = Zensim::new(ZensimProfile::latest());
    let tol = RegressionTolerance::off_by_one().with_min_similarity(0.0);

    for prefix in ["Landscape", "Portrait"] {
        let (ref_rgba, rw, rh) = load_rgba(&dir.join(format!("{prefix}_1.jpg")));
        let ref_px = px(&ref_rgba);

        for orient in 2..=8 {
            let (act_rgba, aw, ah) = load_rgba(&dir.join(format!("{prefix}_{orient}.jpg")));

            if aw == rw && ah == rh {
                // Same dimensions — normal comparison
                let _report = check_regression(
                    &z,
                    &zensim::RgbaSlice::new(&ref_px, rw as usize, rh as usize),
                    &zensim::RgbaSlice::new(&px(&act_rgba), aw as usize, ah as usize),
                    &tol,
                )
                .unwrap();
            } else {
                // Different dimensions — resized comparison
                let _report = check_regression_resized(
                    &z, &ref_rgba, rw, rh, &act_rgba, aw, ah, &tol,
                )
                .unwrap();
            }
        }
    }
}
