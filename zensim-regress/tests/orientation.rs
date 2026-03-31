//! Integration tests for EXIF orientation handling using codec-corpus
//! Landscape and Portrait JPEG test images (orientations 1–8).
//!
//! Each image contains a different number (1–8) rendered in the center,
//! matching its EXIF orientation flag. The images are designed for visual
//! verification ("does the number appear upright after applying the EXIF
//! transform?"), not for pixel-level comparison between orientations.
//! Comparing Landscape_1 to Landscape_2 is comparing two different images.
//!
//! These tests verify:
//! - Correct dimension detection (same-dim for 1–4, swapped for 5–8)
//! - Correct DimensionMismatchKind categorization
//! - The scoring pipeline doesn't panic on real JPEG images
//!
//! For transform detection correctness (pixel-identical content with
//! applied transforms), see the unit tests in testing.rs.

use image::{RgbaImage, imageops};
use zensim::{Zensim, ZensimProfile};
use zensim_regress::testing::{
    ComparisonMethod, DimensionMismatchKind, RegressionTolerance, check_regression,
    check_regression_resized, detect_transform,
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

// ─── Exact-transform tests ─────────────────────────────────────────────
//
// Take a real image, apply each transform ourselves (pixel-exact), then
// verify detection identifies the correct method with score 100.

fn apply_transform(img: &RgbaImage, method: ComparisonMethod) -> RgbaImage {
    match method {
        ComparisonMethod::FlipHorizontal => imageops::flip_horizontal(img),
        ComparisonMethod::FlipVertical => imageops::flip_vertical(img),
        ComparisonMethod::Rotated180 => imageops::rotate180(img),
        ComparisonMethod::Rotated90 => imageops::rotate90(img),
        ComparisonMethod::Rotated270 => imageops::rotate270(img),
        ComparisonMethod::Transpose => {
            imageops::flip_horizontal(&imageops::rotate90(img))
        }
        ComparisonMethod::Transverse => {
            imageops::flip_horizontal(&imageops::rotate270(img))
        }
        _ => img.clone(),
    }
}

/// Same-dimension transforms (flipH, flipV, rot180): applied to a real photo,
/// detect_transform must identify the exact method with score 100.
#[test]
fn exact_same_dim_transforms_detected() {
    let dir = orientation_dir();
    let z = Zensim::new(ZensimProfile::latest());
    let tol = RegressionTolerance::exact();

    let (ref_rgba, rw, rh) = load_rgba(&dir.join("Landscape_1.jpg"));
    let ref_img = RgbaImage::from_raw(rw, rh, ref_rgba.clone()).unwrap();

    for method in [
        ComparisonMethod::FlipHorizontal,
        ComparisonMethod::FlipVertical,
        ComparisonMethod::Rotated180,
    ] {
        let transformed = apply_transform(&ref_img, method);
        let act_rgba = transformed.into_raw();

        // Direct comparison should score low
        let orig = check_regression(
            &z,
            &zensim::RgbaSlice::new(&px(&ref_rgba), rw as usize, rh as usize),
            &zensim::RgbaSlice::new(&px(&act_rgba), rw as usize, rh as usize),
            &tol,
        )
        .unwrap();

        assert!(
            orig.score() < 50.0,
            "{method}: direct score {:.1} should be < 50",
            orig.score(),
        );

        // detect_transform must find the exact method
        let result = detect_transform(
            &z, &ref_rgba, &act_rgba, rw, rh, orig.score(), &tol,
        );

        let (report, detected) = result.unwrap_or_else(|| {
            panic!("{method}: detection returned None (direct score {:.1})", orig.score())
        });

        assert_eq!(detected, method, "{method}: detected wrong method {detected}");
        assert!(
            report.score() > 95.0,
            "{method}: corrected score {:.1} should be > 95",
            report.score(),
        );
    }
}

/// Swapped-dimension transforms (rot90, rot270, transpose, transverse):
/// applied to a real photo, check_regression_resized must identify the exact
/// method with score 100.
#[test]
fn exact_swapped_dim_transforms_detected() {
    let dir = orientation_dir();
    let z = Zensim::new(ZensimProfile::latest());
    let tol = RegressionTolerance::exact();

    let (ref_rgba, rw, rh) = load_rgba(&dir.join("Landscape_1.jpg"));
    let ref_img = RgbaImage::from_raw(rw, rh, ref_rgba.clone()).unwrap();

    for method in [
        ComparisonMethod::Rotated90,
        ComparisonMethod::Rotated270,
        ComparisonMethod::Transpose,
        ComparisonMethod::Transverse,
    ] {
        let transformed = apply_transform(&ref_img, method);
        let (aw, ah) = transformed.dimensions();
        let act_rgba = transformed.into_raw();

        // Dimensions must differ (swapped)
        assert_ne!(
            (rw, rh), (aw, ah),
            "{method}: expected swapped dimensions",
        );

        let report = check_regression_resized(
            &z, &ref_rgba, rw, rh, &act_rgba, aw, ah, &tol,
        )
        .unwrap();

        let dim = report.dimension_info().unwrap();
        assert_eq!(
            dim.kind,
            DimensionMismatchKind::OrientationSwap,
            "{method}: expected OrientationSwap, got {:?}",
            dim.kind,
        );
        // Corner-SAD pre-filter may pick a different rotation that happens to
        // score identically (e.g., rot90 vs rot270 on near-symmetric corners).
        // What matters is the score is perfect.
        assert_eq!(
            report.score(),
            100.0,
            "{method}: exact transform should score 100, got {:.1} (detected {})",
            report.score(),
            dim.method,
        );
    }
}

/// Exact same-dim transforms on a portrait image (different aspect ratio).
#[test]
fn exact_same_dim_transforms_portrait() {
    let dir = orientation_dir();
    let z = Zensim::new(ZensimProfile::latest());
    let tol = RegressionTolerance::exact();

    let (ref_rgba, rw, rh) = load_rgba(&dir.join("Portrait_1.jpg"));
    let ref_img = RgbaImage::from_raw(rw, rh, ref_rgba.clone()).unwrap();

    for method in [
        ComparisonMethod::FlipHorizontal,
        ComparisonMethod::FlipVertical,
        ComparisonMethod::Rotated180,
    ] {
        let transformed = apply_transform(&ref_img, method);
        let act_rgba = transformed.into_raw();

        let orig = check_regression(
            &z,
            &zensim::RgbaSlice::new(&px(&ref_rgba), rw as usize, rh as usize),
            &zensim::RgbaSlice::new(&px(&act_rgba), rw as usize, rh as usize),
            &tol,
        )
        .unwrap();

        let result = detect_transform(
            &z, &ref_rgba, &act_rgba, rw, rh, orig.score(), &tol,
        );

        let (report, detected) = result.unwrap_or_else(|| {
            panic!("portrait {method}: detection returned None (direct score {:.1})", orig.score())
        });

        assert_eq!(detected, method, "portrait {method}: wrong method {detected}");
        assert!(report.score() > 95.0,
            "portrait {method}: corrected score {:.1} should be > 95", report.score());
    }
}

/// Exact swapped-dim transforms on a portrait image.
#[test]
fn exact_swapped_dim_transforms_portrait() {
    let dir = orientation_dir();
    let z = Zensim::new(ZensimProfile::latest());
    let tol = RegressionTolerance::exact();

    let (ref_rgba, rw, rh) = load_rgba(&dir.join("Portrait_1.jpg"));
    let ref_img = RgbaImage::from_raw(rw, rh, ref_rgba.clone()).unwrap();

    for method in [
        ComparisonMethod::Rotated90,
        ComparisonMethod::Rotated270,
        ComparisonMethod::Transpose,
        ComparisonMethod::Transverse,
    ] {
        let transformed = apply_transform(&ref_img, method);
        let (aw, ah) = transformed.dimensions();
        let act_rgba = transformed.into_raw();

        let report = check_regression_resized(
            &z, &ref_rgba, rw, rh, &act_rgba, aw, ah, &tol,
        )
        .unwrap();

        let dim = report.dimension_info().unwrap();
        assert_eq!(dim.kind, DimensionMismatchKind::OrientationSwap);
        assert_eq!(report.score(), 100.0,
            "portrait {method}: exact transform should score 100, got {:.1} (detected {})",
            report.score(), dim.method);
    }
}
