//! Integration tests for zensim under wasm32-wasip1 with SIMD128.
//!
//! These tests verify:
//! 1. The full checksums workflow works under wasip1 (filesystem I/O)
//! 2. zensim wasm128 SIMD produces correct similarity scores
//! 3. Generators and distortions produce deterministic output cross-platform
//!
//! Run with:
//! ```sh
//! RUSTFLAGS='-C target-feature=+simd128' \
//!   cargo test --target wasm32-wasip1 -p zensim-wasm-tests -- --nocapture
//! ```

use zensim::{RgbaSlice, Zensim, ZensimProfile};
use zensim_regress::checksums::ChecksumManager;
use zensim_regress::distortions;
use zensim_regress::generators;
use zensim_regress::testing::{RegressionTolerance, check_regression};
use zensim_regress::tolerance::ToleranceSpec;

const W: u32 = 64;
const H: u32 = 64;

fn mgr() -> ChecksumManager {
    // On native, CARGO_MANIFEST_DIR gives the absolute path.
    // On wasm, wasmtime --dir=. maps the host CWD (crate dir) to "." in the guest.
    let dir = if cfg!(target_arch = "wasm32") {
        std::path::PathBuf::from("./tests/checksums")
    } else {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/checksums")
    };
    ChecksumManager::new(&dir)
}

fn px(rgba: &[u8]) -> Vec<[u8; 4]> {
    rgba.chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

// ---- Checksums workflow tests ----
// These verify that ChecksumManager read/write works under wasip1.

#[test]
fn checksums_mandelbrot() {
    let rgba = generators::mandelbrot(W, H);
    let result = mgr()
        .check_pixels("wasm", "mandelbrot", "64x64", &rgba, W, H, None)
        .expect("check_pixels failed");
    assert!(result.passed(), "mandelbrot: {result}");
}

#[test]
fn checksums_gradient() {
    let rgba = generators::gradient(W, H);
    let result = mgr()
        .check_pixels("wasm", "gradient", "64x64", &rgba, W, H, None)
        .expect("check_pixels failed");
    assert!(result.passed(), "gradient: {result}");
}

#[test]
fn checksums_checkerboard() {
    let rgba = generators::checkerboard(W, H, 8);
    let result = mgr()
        .check_pixels("wasm", "checkerboard", "64x64_f8", &rgba, W, H, None)
        .expect("check_pixels failed");
    assert!(result.passed(), "checkerboard: {result}");
}

#[test]
fn checksums_value_noise() {
    let rgba = generators::value_noise(W, H, 42);
    let result = mgr()
        .check_pixels("wasm", "value_noise", "64x64_s42", &rgba, W, H, None)
        .expect("check_pixels failed");
    assert!(result.passed(), "value_noise: {result}");
}

#[test]
fn checksums_color_blocks() {
    let rgba = generators::color_blocks(W, H);
    let result = mgr()
        .check_pixels("wasm", "color_blocks", "64x64", &rgba, W, H, None)
        .expect("check_pixels failed");
    assert!(result.passed(), "color_blocks: {result}");
}

#[test]
fn checksums_truncate_lsb() {
    let base = generators::gradient(W, H);
    let distorted = distortions::truncate_lsb(&base);
    let tol = ToleranceSpec::off_by_one();
    let result = mgr()
        .check_pixels(
            "wasm",
            "truncate_lsb",
            "gradient_64x64",
            &distorted,
            W,
            H,
            Some(&tol),
        )
        .expect("check_pixels failed");
    assert!(result.passed(), "truncate_lsb: {result}");
}

// ---- Score verification tests ----
// These verify that wasm128 SIMD produces numerically correct scores.

#[test]
fn identical_images_score_100() {
    let z = Zensim::new(ZensimProfile::latest());
    let rgba = generators::mandelbrot(W, H);
    let pixels = px(&rgba);
    let src = RgbaSlice::new(&pixels, W as usize, H as usize);
    let result = z.compute(&src, &src).unwrap();
    assert_eq!(result.score(), 100.0, "identical images must score exactly 100");
}

#[test]
fn slight_distortion_scores_high() {
    let z = Zensim::new(ZensimProfile::latest());
    let base = generators::mandelbrot(128, 128);
    let distorted = distortions::truncate_lsb(&base);
    let base_px = px(&base);
    let dist_px = px(&distorted);
    let src = RgbaSlice::new(&base_px, 128, 128);
    let dst = RgbaSlice::new(&dist_px, 128, 128);
    let result = z.compute(&src, &dst).unwrap();
    assert!(
        result.score() > 90.0,
        "truncate_lsb should score >90, got {:.2}",
        result.score()
    );
}

#[test]
fn channel_swap_scores_low() {
    let z = Zensim::new(ZensimProfile::latest());
    let base = generators::color_blocks(64, 64);
    let swapped = distortions::channel_swap_rb(&base);
    let base_px = px(&base);
    let swap_px = px(&swapped);
    let src = RgbaSlice::new(&base_px, 64, 64);
    let dst = RgbaSlice::new(&swap_px, 64, 64);
    let result = z.compute(&src, &dst).unwrap();
    assert!(
        result.score() < 70.0,
        "channel_swap on color_blocks should score <70, got {:.2}",
        result.score()
    );
}

#[test]
fn inversion_scores_very_low() {
    let z = Zensim::new(ZensimProfile::latest());
    let base = generators::gradient(64, 64);
    let inverted = distortions::invert(&base);
    let base_px = px(&base);
    let inv_px = px(&inverted);
    let src = RgbaSlice::new(&base_px, 64, 64);
    let dst = RgbaSlice::new(&inv_px, 64, 64);
    let result = z.compute(&src, &dst).unwrap();
    assert!(
        result.score() < 30.0,
        "inversion should score <30, got {:.2}",
        result.score()
    );
}

#[test]
fn uniform_shift_scores_proportionally() {
    let z = Zensim::new(ZensimProfile::latest());
    let base = generators::value_noise(128, 128, 7);
    let shift_small = distortions::uniform_shift(&base, 2);
    let shift_large = distortions::uniform_shift(&base, 20);

    let base_px = px(&base);
    let small_px = px(&shift_small);
    let large_px = px(&shift_large);
    let src = RgbaSlice::new(&base_px, 128, 128);
    let dst_small = RgbaSlice::new(&small_px, 128, 128);
    let dst_large = RgbaSlice::new(&large_px, 128, 128);

    let score_small = z.compute(&src, &dst_small).unwrap().score();
    let score_large = z.compute(&src, &dst_large).unwrap().score();

    assert!(
        score_small > score_large,
        "small shift ({:.2}) should score higher than large shift ({:.2})",
        score_small,
        score_large
    );
    assert!(
        score_small > 85.0,
        "shift(2) should score >85, got {:.2}",
        score_small
    );
}

// ---- Regression comparison tests ----
// These exercise the full check_regression path (zensim scoring + tolerance).

#[test]
fn regression_off_by_one_passes() {
    let z = Zensim::new(ZensimProfile::latest());
    let tol = RegressionTolerance::off_by_one();
    let base = generators::mandelbrot(128, 128);
    let distorted = generators::off_by_n(&base, 1, 1);
    let base_px = px(&base);
    let dist_px = px(&distorted);
    let src = RgbaSlice::new(&base_px, 128, 128);
    let dst = RgbaSlice::new(&dist_px, 128, 128);
    let report = check_regression(&z, &src, &dst, &tol).unwrap();
    assert!(report.passed(), "off-by-1 should pass off_by_one tolerance: {report}");
}

#[test]
fn regression_large_delta_fails_exact() {
    let z = Zensim::new(ZensimProfile::latest());
    let tol = RegressionTolerance::exact();
    let base = generators::gradient(64, 64);
    let distorted = distortions::uniform_shift(&base, 10);
    let base_px = px(&base);
    let dist_px = px(&distorted);
    let src = RgbaSlice::new(&base_px, 64, 64);
    let dst = RgbaSlice::new(&dist_px, 64, 64);
    let report = check_regression(&z, &src, &dst, &tol).unwrap();
    assert!(!report.passed(), "uniform_shift(10) should fail exact tolerance");
}

// ---- Multi-scale / larger image tests ----
// Exercise the full 4-scale pyramid on larger images.

#[test]
fn large_image_identical() {
    let z = Zensim::new(ZensimProfile::latest());
    let rgba = generators::mandelbrot(256, 256);
    let pixels = px(&rgba);
    let src = RgbaSlice::new(&pixels, 256, 256);
    let result = z.compute(&src, &src).unwrap();
    assert_eq!(result.score(), 100.0);
}

#[test]
fn large_image_noise_distortion() {
    let z = Zensim::new(ZensimProfile::latest());
    let base = generators::value_noise(256, 256, 99);
    let distorted = distortions::truncate_lsb(&base);
    let base_px = px(&base);
    let dist_px = px(&distorted);
    let src = RgbaSlice::new(&base_px, 256, 256);
    let dst = RgbaSlice::new(&dist_px, 256, 256);
    let result = z.compute(&src, &dst).unwrap();
    assert!(
        result.score() > 90.0,
        "256x256 truncate_lsb should score >90, got {:.2}",
        result.score()
    );
}
