//! Integration tests for ICC color primaries coverage.
//!
//! Exercises all `ColorPrimaries` variants (sRGB, Display P3, BT.2020) across
//! all pixel formats, verifying self-comparison, cross-primaries differences,
//! pixel format equivalence, determinism, and out-of-gamut clamping behavior.
//!
//! All tests use synthetic images — no S3 dependency.
//!
//! Run with: `cargo test -p zensim --test icc_coverage`

mod common;

use common::generators::*;
use zensim::{ColorPrimaries, PixelFormat, StridedBytes, Zensim, ZensimProfile};

fn zensim() -> Zensim {
    Zensim::new(ZensimProfile::latest())
}

/// Helper: create StridedBytes from RGB u8 pixels with given primaries.
fn rgb_source_with_primaries(
    buf: &[u8],
    w: usize,
    h: usize,
    primaries: ColorPrimaries,
) -> StridedBytes<'_> {
    StridedBytes::new(buf, w, h, w * 3, PixelFormat::Srgb8Rgb).with_color_primaries(primaries)
}

// ─── Self-comparison: each primaries variant → must be 100.0 ──────────

#[test]
fn self_comparison_srgb() {
    let (w, h) = (64, 64);
    let pixels = gen_color_blocks(w, h);
    let buf: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let src = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::Srgb);
    let dst = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::Srgb);
    let result = zensim().compute(&src, &dst).unwrap();
    assert_eq!(
        result.score(),
        100.0,
        "sRGB self-comparison should be exactly 100.0, got {}",
        result.score()
    );
}

#[test]
fn self_comparison_display_p3() {
    let (w, h) = (64, 64);
    let pixels = gen_color_blocks(w, h);
    let buf: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let src = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::DisplayP3);
    let dst = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::DisplayP3);
    let result = zensim().compute(&src, &dst).unwrap();
    assert_eq!(
        result.score(),
        100.0,
        "Display P3 self-comparison should be exactly 100.0, got {}",
        result.score()
    );
}

#[test]
fn self_comparison_bt2020() {
    let (w, h) = (64, 64);
    let pixels = gen_color_blocks(w, h);
    let buf: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let src = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::Bt2020);
    let dst = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::Bt2020);
    let result = zensim().compute(&src, &dst).unwrap();
    assert_eq!(
        result.score(),
        100.0,
        "BT.2020 self-comparison should be exactly 100.0, got {}",
        result.score()
    );
}

// ─── Cross-primaries: same pixels, different declared primaries → different scores ──

#[test]
fn p3_vs_srgb_interpretation_differs() {
    let (w, h) = (64, 64);
    let pixels = gen_color_blocks(w, h);
    let buf: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let src = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::DisplayP3);
    let dst = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::Srgb);
    let result = zensim().compute(&src, &dst).unwrap();
    assert!(
        result.score() < 100.0,
        "P3 vs sRGB same pixels should differ, got {}",
        result.score()
    );
    println!(
        "  P3 vs sRGB same-pixels score: {:.4} (expected < 100)",
        result.score()
    );
}

#[test]
fn bt2020_vs_srgb_interpretation_differs() {
    let (w, h) = (64, 64);
    let pixels = gen_color_blocks(w, h);
    let buf: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let src = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::Bt2020);
    let dst = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::Srgb);
    let result = zensim().compute(&src, &dst).unwrap();
    assert!(
        result.score() < 100.0,
        "BT.2020 vs sRGB same pixels should differ, got {}",
        result.score()
    );
    // BT.2020 has a wider gamut than P3, so the difference should be larger
    println!(
        "  BT.2020 vs sRGB same-pixels score: {:.4} (expected < 100)",
        result.score()
    );
}

#[test]
fn bt2020_vs_p3_interpretation_differs() {
    let (w, h) = (64, 64);
    let pixels = gen_color_blocks(w, h);
    let buf: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let src = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::Bt2020);
    let dst = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::DisplayP3);
    let result = zensim().compute(&src, &dst).unwrap();
    assert!(
        result.score() < 100.0,
        "BT.2020 vs P3 same pixels should differ, got {}",
        result.score()
    );
    println!(
        "  BT.2020 vs P3 same-pixels score: {:.4} (expected < 100)",
        result.score()
    );
}

#[test]
fn wider_gamut_produces_larger_difference() {
    let (w, h) = (64, 64);
    let pixels = gen_color_blocks(w, h);
    let buf: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();

    // P3 vs sRGB
    let src_p3 = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::DisplayP3);
    let dst_srgb = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::Srgb);
    let p3_vs_srgb = zensim().compute(&src_p3, &dst_srgb).unwrap().score();

    // BT.2020 vs sRGB
    let src_bt = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::Bt2020);
    let dst_srgb2 = rgb_source_with_primaries(&buf, w, h, ColorPrimaries::Srgb);
    let bt_vs_srgb = zensim().compute(&src_bt, &dst_srgb2).unwrap().score();

    println!("  P3 vs sRGB: {p3_vs_srgb:.4}, BT.2020 vs sRGB: {bt_vs_srgb:.4}");
    // BT.2020→sRGB has more extreme matrix entries, so more distortion, lower score
    assert!(
        bt_vs_srgb < p3_vs_srgb,
        "BT.2020 vs sRGB ({bt_vs_srgb:.4}) should be lower than P3 vs sRGB ({p3_vs_srgb:.4})"
    );
}

// ─── Pixel format × primaries cross-product ──────────────────────────

/// Helper: run the same comparison across multiple pixel formats with given primaries.
/// Returns scores for each format.
fn format_scores_with_primaries(primaries: ColorPrimaries) -> Vec<(String, f64)> {
    let (w, h) = (128, 128);
    let src_pixels = gen_mandelbrot(w, h);
    let dst_pixels = distort_blur(&src_pixels, w, h, 3);
    let z = zensim();

    type Converter = fn(&[[u8; 3]], usize, usize) -> (Vec<u8>, usize);

    struct FmtEntry {
        name: &'static str,
        format: PixelFormat,
        converter: Converter,
    }

    #[allow(unused_mut)]
    let mut formats = vec![
        FmtEntry {
            name: "Srgb8Rgb",
            format: PixelFormat::Srgb8Rgb,
            converter: to_srgb8_rgb,
        },
        FmtEntry {
            name: "Srgb8Rgba",
            format: PixelFormat::Srgb8Rgba,
            converter: to_srgb8_rgba,
        },
        FmtEntry {
            name: "Srgb8Bgra",
            format: PixelFormat::Srgb8Bgra,
            converter: to_srgb8_bgra,
        },
        FmtEntry {
            name: "Srgb16Rgba",
            format: PixelFormat::Srgb16Rgba,
            converter: to_srgb16_rgba,
        },
        FmtEntry {
            name: "LinearF32Rgba",
            format: PixelFormat::LinearF32Rgba,
            converter: to_linear_f32_rgba,
        },
    ];

    let mut results = Vec::new();
    for fmt in &formats {
        let (src_buf, src_stride) = (fmt.converter)(&src_pixels, w, h);
        let (dst_buf, dst_stride) = (fmt.converter)(&dst_pixels, w, h);
        let src = StridedBytes::new(&src_buf, w, h, src_stride, fmt.format)
            .with_color_primaries(primaries);
        let dst = StridedBytes::new(&dst_buf, w, h, dst_stride, fmt.format)
            .with_color_primaries(primaries);
        let result = z.compute(&src, &dst).unwrap();
        results.push((fmt.name.to_string(), result.score()));
    }
    results
}

#[test]
fn pixel_format_equivalence_display_p3() {
    let scores = format_scores_with_primaries(ColorPrimaries::DisplayP3);
    let reference = scores[0].1;
    println!("  Display P3 format equivalence (ref=Srgb8Rgb: {reference:.6}):");

    for (name, score) in &scores {
        let diff = (score - reference).abs();
        println!("    {name:20} score={score:.6}  diff={diff:.6}");
        // Same tolerances as cross_platform.rs: sRGB→f32 ±0.15, same-8bit ±0.01
        let tol = match name.as_str() {
            "LinearF32Rgba" => 0.15,
            _ => 0.01,
        };
        assert!(
            diff <= tol,
            "Display P3 {name}: score {score:.6} differs from reference {reference:.6} by {diff:.6} (>{tol})"
        );
    }
}

#[test]
fn pixel_format_equivalence_bt2020() {
    let scores = format_scores_with_primaries(ColorPrimaries::Bt2020);
    let reference = scores[0].1;
    println!("  BT.2020 format equivalence (ref=Srgb8Rgb: {reference:.6}):");

    for (name, score) in &scores {
        let diff = (score - reference).abs();
        println!("    {name:20} score={score:.6}  diff={diff:.6}");
        let tol = match name.as_str() {
            "LinearF32Rgba" => 0.15,
            _ => 0.01,
        };
        assert!(
            diff <= tol,
            "BT.2020 {name}: score {score:.6} differs from reference {reference:.6} by {diff:.6} (>{tol})"
        );
    }
}

// ─── Determinism ─────────────────────────────────────────────────────

#[test]
fn determinism_display_p3() {
    let (w, h) = (64, 64);
    let src_pixels = gen_checkerboard(w, h, 8);
    let dst_pixels = distort_color_shift(&src_pixels, w, h);
    let src_buf: Vec<u8> = src_pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_buf: Vec<u8> = dst_pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let z = zensim();

    let mut scores = Vec::new();
    for _ in 0..5 {
        let src = rgb_source_with_primaries(&src_buf, w, h, ColorPrimaries::DisplayP3);
        let dst = rgb_source_with_primaries(&dst_buf, w, h, ColorPrimaries::DisplayP3);
        scores.push(z.compute(&src, &dst).unwrap().score());
    }

    for (i, s) in scores.iter().enumerate() {
        assert_eq!(
            *s, scores[0],
            "Display P3 run {i}: {s} != run 0: {} (not bitwise identical)",
            scores[0]
        );
    }
}

#[test]
fn determinism_bt2020() {
    let (w, h) = (64, 64);
    let src_pixels = gen_checkerboard(w, h, 8);
    let dst_pixels = distort_color_shift(&src_pixels, w, h);
    let src_buf: Vec<u8> = src_pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_buf: Vec<u8> = dst_pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let z = zensim();

    let mut scores = Vec::new();
    for _ in 0..5 {
        let src = rgb_source_with_primaries(&src_buf, w, h, ColorPrimaries::Bt2020);
        let dst = rgb_source_with_primaries(&dst_buf, w, h, ColorPrimaries::Bt2020);
        scores.push(z.compute(&src, &dst).unwrap().score());
    }

    for (i, s) in scores.iter().enumerate() {
        assert_eq!(
            *s, scores[0],
            "BT.2020 run {i}: {s} != run 0: {} (not bitwise identical)",
            scores[0]
        );
    }
}

// ─── Out-of-gamut clamping ───────────────────────────────────────────

/// Create an image with a single solid color and compare self with given primaries.
fn solid_color_self_score(r: u8, g: u8, b: u8, primaries: ColorPrimaries) -> f64 {
    let (w, h) = (32, 32);
    let buf = vec![[r, g, b]; w * h];
    let bytes: Vec<u8> = buf.iter().flat_map(|p| p.iter().copied()).collect();
    let src = rgb_source_with_primaries(&bytes, w, h, primaries);
    let dst = rgb_source_with_primaries(&bytes, w, h, primaries);
    zensim().compute(&src, &dst).unwrap().score()
}

#[test]
fn bt2020_saturated_red_self_comparison() {
    // Pure red in BT.2020 is far outside sRGB gamut — tests clamping path
    let score = solid_color_self_score(255, 0, 0, ColorPrimaries::Bt2020);
    assert_eq!(
        score, 100.0,
        "BT.2020 saturated red self-comparison should be 100.0, got {score}"
    );
}

#[test]
fn bt2020_saturated_green_self_comparison() {
    let score = solid_color_self_score(0, 255, 0, ColorPrimaries::Bt2020);
    assert_eq!(
        score, 100.0,
        "BT.2020 saturated green self-comparison should be 100.0, got {score}"
    );
}

#[test]
fn p3_saturated_green_self_comparison() {
    // P3 green is outside sRGB gamut
    let score = solid_color_self_score(0, 255, 0, ColorPrimaries::DisplayP3);
    assert_eq!(
        score, 100.0,
        "P3 saturated green self-comparison should be 100.0, got {score}"
    );
}

#[test]
fn bt2020_saturated_colors_differ_from_srgb() {
    // Saturated red in BT.2020 vs sRGB should produce different XYB
    let (w, h) = (32, 32);
    let buf = vec![[255u8, 0, 0]; w * h];
    let bytes: Vec<u8> = buf.iter().flat_map(|p| p.iter().copied()).collect();

    let src = rgb_source_with_primaries(&bytes, w, h, ColorPrimaries::Bt2020);
    let dst = rgb_source_with_primaries(&bytes, w, h, ColorPrimaries::Srgb);
    let result = zensim().compute(&src, &dst).unwrap();

    assert!(
        result.score() < 100.0,
        "BT.2020 red vs sRGB red should differ, got {:.4}",
        result.score()
    );
    println!(
        "  BT.2020 red vs sRGB red: {:.4} (expected < 100)",
        result.score()
    );
}

#[test]
fn p3_green_outside_srgb_gamut_differs() {
    // Pure green (0,255,0) labeled P3 vs same labeled sRGB
    let (w, h) = (32, 32);
    let buf = vec![[0u8, 255, 0]; w * h];
    let bytes: Vec<u8> = buf.iter().flat_map(|p| p.iter().copied()).collect();

    let src = rgb_source_with_primaries(&bytes, w, h, ColorPrimaries::DisplayP3);
    let dst = rgb_source_with_primaries(&bytes, w, h, ColorPrimaries::Srgb);
    let result = zensim().compute(&src, &dst).unwrap();

    assert!(
        result.score() < 100.0,
        "P3 green vs sRGB green should differ, got {:.4}",
        result.score()
    );
    println!(
        "  P3 green vs sRGB green: {:.4} (expected < 100)",
        result.score()
    );
}

// ─── Distorted comparison with non-sRGB primaries ────────────────────

#[test]
fn distorted_comparison_display_p3() {
    let (w, h) = (128, 128);
    let src_pixels = gen_mandelbrot(w, h);
    let dst_pixels = distort_blur(&src_pixels, w, h, 3);
    let src_buf: Vec<u8> = src_pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_buf: Vec<u8> = dst_pixels.iter().flat_map(|p| p.iter().copied()).collect();

    let src = rgb_source_with_primaries(&src_buf, w, h, ColorPrimaries::DisplayP3);
    let dst = rgb_source_with_primaries(&dst_buf, w, h, ColorPrimaries::DisplayP3);

    let result = zensim().compute(&src, &dst).unwrap();
    assert!(
        result.score() >= 0.0 && result.score() < 100.0,
        "P3 blurred comparison should be in [0, 100), got {}",
        result.score()
    );
    println!("  P3 mandelbrot+blur score: {:.4}", result.score());
}

#[test]
fn distorted_comparison_bt2020() {
    let (w, h) = (128, 128);
    let src_pixels = gen_mandelbrot(w, h);
    let dst_pixels = distort_blur(&src_pixels, w, h, 3);
    let src_buf: Vec<u8> = src_pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_buf: Vec<u8> = dst_pixels.iter().flat_map(|p| p.iter().copied()).collect();

    let src = rgb_source_with_primaries(&src_buf, w, h, ColorPrimaries::Bt2020);
    let dst = rgb_source_with_primaries(&dst_buf, w, h, ColorPrimaries::Bt2020);

    let result = zensim().compute(&src, &dst).unwrap();
    assert!(
        result.score() >= 0.0 && result.score() < 100.0,
        "BT.2020 blurred comparison should be in [0, 100), got {}",
        result.score()
    );
    println!("  BT.2020 mandelbrot+blur score: {:.4}", result.score());
}

// ─── Large image to exercise parallel row processing ─────────────────

#[test]
fn large_image_bt2020_parallel_rows() {
    let (w, h) = (256, 256);
    let src_pixels = gen_value_noise(w, h, 99);
    let dst_pixels = distort_block_artifacts(&src_pixels, w, h);
    let src_buf: Vec<u8> = src_pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_buf: Vec<u8> = dst_pixels.iter().flat_map(|p| p.iter().copied()).collect();

    let src = rgb_source_with_primaries(&src_buf, w, h, ColorPrimaries::Bt2020);
    let dst = rgb_source_with_primaries(&dst_buf, w, h, ColorPrimaries::Bt2020);

    let result = zensim().compute(&src, &dst).unwrap();
    assert!(
        result.score() > 0.0 && result.score() < 100.0,
        "BT.2020 256x256 noise+blocks should be between 0 and 100, got {}",
        result.score()
    );
    println!(
        "  BT.2020 256x256 noise+blocks score: {:.4}",
        result.score()
    );
}

#[test]
fn large_image_display_p3_parallel_rows() {
    let (w, h) = (256, 256);
    let src_pixels = gen_value_noise(w, h, 99);
    let dst_pixels = distort_block_artifacts(&src_pixels, w, h);
    let src_buf: Vec<u8> = src_pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_buf: Vec<u8> = dst_pixels.iter().flat_map(|p| p.iter().copied()).collect();

    let src = rgb_source_with_primaries(&src_buf, w, h, ColorPrimaries::DisplayP3);
    let dst = rgb_source_with_primaries(&dst_buf, w, h, ColorPrimaries::DisplayP3);

    let result = zensim().compute(&src, &dst).unwrap();
    assert!(
        result.score() > 0.0 && result.score() < 100.0,
        "P3 256x256 noise+blocks should be between 0 and 100, got {}",
        result.score()
    );
    println!("  P3 256x256 noise+blocks score: {:.4}", result.score());
}
