//! Integration tests for ICC color primaries coverage.
//!
//! Exercises all `ColorPrimaries` variants (sRGB, Display P3, BT.2020) across
//! all pixel formats, verifying self-comparison, cross-primaries differences,
//! pixel format equivalence, determinism, and out-of-gamut clamping behavior.
//!
//! All tests use synthetic images — no S3 dependency.
//!
//! Run with: `cargo test -p zensim --features custom-profiles --test icc_coverage`
//!
//! Gated on the `custom-profiles` feature: these tests assert metric SANITY
//! (score in [0,100), identical = 100, wider-gamut → lower score) on
//! ICC/gamut-converted content, which requires the correct-by-construction
//! linear-bounded profile (see the `zensim()` helper). CI runs them in the
//! `test-all-features` job. The default MLP profile `A` structurally violates
//! these invariants on off-manifold synthetic content, so it cannot back them.
#![cfg(feature = "custom-profiles")]

mod common;

use common::generators::*;
use zensim::profile::ProfileParams;
use zensim::{ColorPrimaries, GamutMapping, PixelFormat, StridedBytes, Zensim, ZensimProfile};

// These tests assert metric SANITY (score in [0,100), identical = 100,
// wider-gamut → lower score) on ICC/gamut-converted content. The
// gamut→feature pipeline is shared across profiles; only the final
// scoring squash differs. We run them on the correct-by-construction
// linear-bounded `Custom` profile (`100·exp(−(a/100)·d^b)` over the
// non-negative V0_2 linear weights), whose [0,100] boundedness +
// monotonicity hold by construction — the same `LinearBounded` profile
// the `zensim-experimental` crate ships. This tests the gamut path rather
// than the (known-broken on off-manifold synthetic content) MLP squash of
// profile `A`. A's invariant violations are tracked separately in
// `tests/metric_invariants.rs::v39_known_limit_violations`. See
// `docs/METRIC_INVARIANTS_MECHANISM_AND_REDESIGN_2026-05-26.md`.
fn linear_bounded_params() -> &'static ProfileParams {
    use std::sync::OnceLock;
    static P: OnceLock<ProfileParams> = OnceLock::new();
    P.get_or_init(|| ProfileParams::builder().bounded_squash(true).build())
}

fn zensim() -> Zensim {
    Zensim::new(ZensimProfile::Custom {
        params: linear_bounded_params(),
        name: "zensim-linear-bounded",
    })
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

// Saturated-corner gamut tests (issue #17, resolved by `GamutMapping`):
// under the default `Clip` mode, `apply_gamut_matrix` clamps to [0, 1]
// post-conversion, so saturated BT.2020/P3 primaries collapse to the same
// sRGB linear values as their sRGB counterparts — post-display-clamp
// semantics, unchanged. The opt-in `GamutMapping::Preserve` lets
// out-of-gamut values flow into XYB, so these saturated-corner assertions
// now fire. The non-saturated counterpart
// `bt2020_vs_srgb_interpretation_differs` covers the in-gamut path in both
// modes.

#[test]
fn bt2020_saturated_colors_differ_from_srgb() {
    // Saturated red in BT.2020 vs sRGB should produce different XYB
    // under GamutMapping::Preserve (issue #17).
    let (w, h) = (32, 32);
    let buf = vec![[255u8, 0, 0]; w * h];
    let bytes: Vec<u8> = buf.iter().flat_map(|p| p.iter().copied()).collect();

    let src = rgb_source_with_primaries(&bytes, w, h, ColorPrimaries::Bt2020)
        .with_gamut_mapping(GamutMapping::Preserve);
    let dst = rgb_source_with_primaries(&bytes, w, h, ColorPrimaries::Srgb);
    let result = zensim().compute(&src, &dst).unwrap();

    assert!(
        result.score() < 100.0,
        "BT.2020 red vs sRGB red should differ under Preserve, got {:.4}",
        result.score()
    );
    println!(
        "  BT.2020 red vs sRGB red (Preserve): {:.4} (expected < 100)",
        result.score()
    );
}

#[test]
fn p3_green_outside_srgb_gamut_differs() {
    // Pure green (0,255,0) labeled P3 vs same labeled sRGB, under
    // GamutMapping::Preserve (issue #17).
    let (w, h) = (32, 32);
    let buf = vec![[0u8, 255, 0]; w * h];
    let bytes: Vec<u8> = buf.iter().flat_map(|p| p.iter().copied()).collect();

    let src = rgb_source_with_primaries(&bytes, w, h, ColorPrimaries::DisplayP3)
        .with_gamut_mapping(GamutMapping::Preserve);
    let dst = rgb_source_with_primaries(&bytes, w, h, ColorPrimaries::Srgb);
    let result = zensim().compute(&src, &dst).unwrap();

    assert!(
        result.score() < 100.0,
        "P3 green vs sRGB green should differ under Preserve, got {:.4}",
        result.score()
    );
    println!(
        "  P3 green vs sRGB green (Preserve): {:.4} (expected < 100)",
        result.score()
    );
}

/// The issue #17 failure case end-to-end: a faithful wide-gamut encode vs
/// one that destructively clipped to sRGB gamut before encoding, both
/// decoded and scored as BT.2020.
///
/// - Default `Clip` mode: both sides collapse to the same clamped sRGB
///   linear values → the regression is MASKED (score ≈ 100).
/// - `Preserve` mode: the clip is DETECTED (score clearly below the
///   masked score).
#[test]
fn gamut_clip_regression_masked_by_clip_detected_by_preserve() {
    let (w, h) = (32, 32);
    // Faithful output: saturated BT.2020 red, preserved wide-gamut.
    let faithful = vec![[255u8, 0, 0]; w * h];
    // Lossy output: the encoder clipped to sRGB gamut before encoding —
    // sRGB red expressed in BT.2020 coordinates. sRGB-linear (1,0,0) in
    // BT.2020 primaries is ≈ (0.6274, 0.0691, 0.0164) linear; through the
    // sRGB transfer that is ≈ (210, 74, 33) in 8-bit code values.
    let clipped = vec![[210u8, 74, 33]; w * h];

    let f_bytes: Vec<u8> = faithful.iter().flat_map(|p| p.iter().copied()).collect();
    let c_bytes: Vec<u8> = clipped.iter().flat_map(|p| p.iter().copied()).collect();

    // Default Clip mode: masked.
    let refc = rgb_source_with_primaries(&f_bytes, w, h, ColorPrimaries::Bt2020);
    let dstc = rgb_source_with_primaries(&c_bytes, w, h, ColorPrimaries::Bt2020);
    let masked = zensim().compute(&refc, &dstc).unwrap().score();
    assert!(
        masked > 99.0,
        "under Clip the gamut-clip regression should be invisible (≈100), got {masked:.4}"
    );

    // Preserve mode: detected.
    let refp = rgb_source_with_primaries(&f_bytes, w, h, ColorPrimaries::Bt2020)
        .with_gamut_mapping(GamutMapping::Preserve);
    let dstp = rgb_source_with_primaries(&c_bytes, w, h, ColorPrimaries::Bt2020)
        .with_gamut_mapping(GamutMapping::Preserve);
    let detected = zensim().compute(&refp, &dstp).unwrap().score();
    assert!(
        detected < masked - 1.0,
        "under Preserve the gamut clip must be detectable: preserve={detected:.4} vs clip={masked:.4}"
    );
    println!(
        "  gamut clip: Clip mode {masked:.4} (masked), Preserve mode {detected:.4} (detected)"
    );
}

/// `Preserve` must be semantically a no-op for in-gamut wide-gamut
/// content: nothing is clipped, so both modes see the same linear light.
/// Scores are NOT bit-identical, because `Preserve` routes the
/// gamut-converted rows through the unclamped scalar XYB converter
/// (`cbrtf_fast`, ~20-bit) while the default path uses the SIMD kernels
/// (`cbrt_midp`, ~15-bit) — measured delta ≈ 0.045 score points on this
/// fixture. Assert tight agreement, not bit equality.
#[test]
fn preserve_matches_clip_for_in_gamut_content() {
    let (w, h) = (32, 32);
    // Mid-saturation content, comfortably inside sRGB gamut even after
    // the BT.2020→sRGB matrix.
    let a: Vec<[u8; 3]> = (0..w * h)
        .map(|i| {
            let v = 90 + (i % 60) as u8;
            [v, 110, 130]
        })
        .collect();
    let b: Vec<[u8; 3]> = a
        .iter()
        .map(|p| [p[0].saturating_add(2), p[1], p[2]])
        .collect();
    let a_bytes: Vec<u8> = a.iter().flat_map(|p| p.iter().copied()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|p| p.iter().copied()).collect();

    let clip = zensim()
        .compute(
            &rgb_source_with_primaries(&a_bytes, w, h, ColorPrimaries::Bt2020),
            &rgb_source_with_primaries(&b_bytes, w, h, ColorPrimaries::Bt2020),
        )
        .unwrap()
        .score();
    let preserve = zensim()
        .compute(
            &rgb_source_with_primaries(&a_bytes, w, h, ColorPrimaries::Bt2020)
                .with_gamut_mapping(GamutMapping::Preserve),
            &rgb_source_with_primaries(&b_bytes, w, h, ColorPrimaries::Bt2020)
                .with_gamut_mapping(GamutMapping::Preserve),
        )
        .unwrap()
        .score();
    assert!(
        (clip - preserve).abs() < 0.2,
        "in-gamut content must score near-identically in both modes \
         (cbrt-precision difference only): clip={clip} preserve={preserve}"
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
