//! Cross-platform score consistency tests for zensim.
//!
//! Validates that hardcoded reference scores match across all 7 CI platforms,
//! that all 6 PixelFormat variants produce equivalent scores, that all 228
//! features activate on synthetic test images, and that results are deterministic.
//!
//! Run with: `cargo test -p zensim --all-features --test cross_platform`

mod common;

use common::generators::*;
use zensim::{PixelFormat, RgbSlice, RgbaSlice, StridedBytes, Zensim, ZensimError, ZensimProfile};

// ─── Test pair generation ──────────────────────────────────────────────────

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

// ─── Tests ─────────────────────────────────────────────────────────────────

/// Hardcoded reference scores validated across all 7 CI platforms.
///
/// The V0_2 contract is "scores approximately stable across builds" — not
/// "bit-identical." The metric is intrinsically a parallel reduction over
/// millions of f32 lane sums; any change to band partition (rayon worker
/// count, STRIP_INNER tuning, etc.) reorders the f64 cross-band sum, which
/// drifts the final score by a few millipoints even with the exact same
/// per-pixel arithmetic.
///
/// Tolerance: ±1e-2 on the 0-100 scale. That's ~100× below human-
/// perceptible delta (~1 score point) and leaves ~3× headroom over the
/// drift observed when changing STRIP_INNER 16→32 (max 3.66e-3 across
/// the 8 reference pairs below). Tighter would require pinning every
/// reduction-order knob, which makes future perf work in the parallel
/// kernels impossible without burning the V0_2 profile.
///
/// Pinned to `PreviewV0_2` rather than `latest()` because the reference
/// scores below are V0_2-specific (228-weight linear profile, SROCC=0.9942).
/// `latest()` returns the MLP profile `A` in zensim 0.3.x — its scores
/// for the same inputs are unrelated to the V0_2 calibration.
#[test]
fn hardcoded_reference_scores() {
    const W: usize = 128;
    const H: usize = 128;
    const TOLERANCE: f64 = 1e-2;
    let z = Zensim::new(ZensimProfile::PreviewV0_2);
    let pairs = generate_test_pairs(W, H);

    // Reference scores with concordant-trained weights (228 weights, SROCC=0.9942).
    // Uses linear-srgb crate (C0-continuous constants) for sRGB linearization.
    // Cube root via magetypes::cbrt_midp (3 ULP, 2 Halley) since 0038bc36;
    // values shifted from the prior hand-rolled cbrt by ≤1e-2 absolute /
    // ≤2e-4 relative (different Kahan magic constant, identical algorithm).
    #[allow(clippy::excessive_precision)]
    let expected: &[(&str, f64)] = &[
        ("checkerboard+blur", -79.872_577_454_381_457),
        ("checkerboard+sharpen", 29.588_149_295_371_096),
        ("mandelbrot+blur", 8.479_272_203_622_543),
        ("mandelbrot+color_shift", 48.143_677_644_353_694),
        ("noise+blur", 60.014_996_187_146_878),
        ("noise+block_artifacts", 52.798_793_807_864_449),
        ("color_blocks+color_shift", 30.389_337_020_934_434),
        ("color_blocks+sharpen", -5.661_428_862_086_183),
    ];

    let mut failures = Vec::new();
    for (pair, &(name, expected_score)) in pairs.iter().zip(expected.iter()) {
        assert_eq!(pair.name, name, "Test pair order mismatch");
        let src = RgbSlice::new(&pair.source, W, H);
        let dst = RgbSlice::new(&pair.distorted, W, H);
        let result = z.compute(&src, &dst).expect("compute failed");

        let diff = (result.score() - expected_score).abs();
        println!(
            "  {name:30} score={:.15}  expected={expected_score:.15}  diff={diff:.2e}",
            result.score(),
        );
        if diff > TOLERANCE {
            failures.push(format!(
                "{name}: score {:.15} differs from expected {expected_score:.15} by {diff:.2e} (>{TOLERANCE})",
                result.score(),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Score mismatches:\n{}",
        failures.join("\n")
    );
}

/// All features must be non-trivial (max > 1e-6 across all 8 test pairs).
/// Layout: [0..156) scored (13/ch × 3ch × 4), [156..228) peaks (6/ch × 3ch × 4)
#[cfg(feature = "training")]
#[test]
fn feature_coverage() {
    const W: usize = 128;
    const H: usize = 128;
    const NUM_SCORED: usize = 156; // 13 × 3 × 4
    const NUM_PEAKS: usize = 72; // 6 × 3 × 4
    const NUM_FEATURES: usize = NUM_SCORED + NUM_PEAKS; // 228
    // This test validates the base 228-feature scored+peaks layout, so it
    // must run on a 228-feature profile. `A` is the 372-feature MLP profile
    // (extended + IW-pool blocks) — use the linear `PreviewV0_2` instead.
    let z = Zensim::new(ZensimProfile::PreviewV0_2);
    let pairs = generate_test_pairs(W, H);

    let mut max_per_feature = vec![0.0f64; NUM_FEATURES];

    for pair in &pairs {
        let src = RgbSlice::new(&pair.source, W, H);
        let dst = RgbSlice::new(&pair.distorted, W, H);
        let result = z
            .compute_all_features(&src, &dst)
            .expect("compute_all_features failed");

        assert_eq!(
            result.features().len(),
            NUM_FEATURES,
            "Expected {NUM_FEATURES} features, got {}",
            result.features().len(),
        );

        for (i, &f) in result.features().iter().enumerate() {
            max_per_feature[i] = max_per_feature[i].max(f.abs());
        }
    }

    let scored_names = [
        "ssim_mean",
        "ssim_4th",
        "ssim_2nd",
        "art_mean",
        "art_4th",
        "art_2nd",
        "det_mean",
        "det_4th",
        "det_2nd",
        "mse",
        "hf_energy_loss",
        "hf_mag_loss",
        "hf_energy_gain",
    ];
    let peak_names = [
        "ssim_max", "art_max", "det_max", "ssim_p95", "art_p95", "det_p95",
    ];

    let mut dead_features = Vec::new();
    for (i, &max_val) in max_per_feature.iter().enumerate() {
        let (scale, ch_name, f_name) = if i < NUM_SCORED {
            let scale = i / 39;
            let within = i % 39;
            let ch = within / 13;
            let fi = within % 13;
            (scale, ["X", "Y", "B"][ch], scored_names[fi])
        } else {
            let pi = i - NUM_SCORED;
            let scale = pi / 18;
            let within = pi % 18;
            let ch = within / 6;
            let fi = within % 6;
            (scale, ["X", "Y", "B"][ch], peak_names[fi])
        };

        if max_val <= 1e-6 {
            dead_features.push(format!(
                "  feat[{i:3}] s{scale} {ch_name} {f_name:16} max={max_val:.2e}"
            ));
        }
    }

    if !dead_features.is_empty() {
        panic!(
            "{} of {NUM_FEATURES} features never exceeded 1e-6:\n{}",
            dead_features.len(),
            dead_features.join("\n"),
        );
    }

    println!("  All {NUM_FEATURES} features activated (max > 1e-6)");
}

/// Tests all 4 generators with separately-allocated copies (not same pointer).
///
/// Pinned to `PreviewV0_2` rather than `latest()` because the
/// "raw_distance == 0.0 for identical input" invariant is V0_2-specific:
/// V0_2 is a linear-weighted sum of feature distances, so all-zero
/// features → 0 distance → score 100 by the `100 − 18·d^0.7` mapping.
/// The MLP profile `A` evaluates a LeakyReLU forward pass
/// on the (all-zero) feature vector; its biases produce a non-zero raw
/// output, which the runtime clamps to 100 at the score level.
/// Different invariant, different test surface.
#[test]
fn identical_images_score_100() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::PreviewV0_2);

    let images: &[(&str, Vec<[u8; 3]>)] = &[
        ("checkerboard", gen_checkerboard(W, H, 8)),
        ("mandelbrot", gen_mandelbrot(W, H)),
        ("value_noise", gen_value_noise(W, H, 42)),
        ("color_blocks", gen_color_blocks(W, H)),
    ];

    for (name, pixels) in images {
        // Separate copy so we're not relying on pointer identity
        let copy = pixels.clone();
        let src = RgbSlice::new(pixels, W, H);
        let dst = RgbSlice::new(&copy, W, H);
        let result = z.compute(&src, &dst).expect("compute failed");

        println!(
            "  {name:20} score={:.15}  raw_dist={:.2e}  max_feat={:.2e}",
            result.score(),
            result.raw_distance(),
            result
                .features()
                .iter()
                .map(|f| f.abs())
                .fold(0.0f64, f64::max),
        );
        assert_eq!(
            result.score(),
            100.0,
            "{name}: identical images must score exactly 100.0, got {:.15}",
            result.score(),
        );
        assert_eq!(
            result.raw_distance(),
            0.0,
            "{name}: identical images must have raw_distance=0.0, got {:.2e}",
            result.raw_distance(),
        );
        assert!(
            result.features().iter().all(|&f| f == 0.0),
            "{name}: identical images must have all-zero features",
        );
        assert_eq!(
            result.mean_offset(),
            [0.0, 0.0, 0.0],
            "{name}: identical images must have zero mean_offset",
        );
    }
}

/// `PreviewV0_1` is the restored 0.2.x-compatible linear profile. Guard its
/// basic contract: correct name, identical images score exactly 100, and a
/// distorted pair scores below 100 (the profile is wired and non-trivial).
#[test]
fn preview_v0_1_compat_profile() {
    const W: usize = 128;
    const H: usize = 128;
    assert_eq!(ZensimProfile::PreviewV0_1.name(), "zensim-preview-v0.1");

    let z = Zensim::new(ZensimProfile::PreviewV0_1);
    let pixels = gen_mandelbrot(W, H);
    let copy = pixels.clone();
    let src = RgbSlice::new(&pixels, W, H);
    let dst = RgbSlice::new(&copy, W, H);
    let identical = z.compute(&src, &dst).expect("compute failed");
    assert_eq!(
        identical.score(),
        100.0,
        "PreviewV0_1: identical images must score 100.0, got {:.6}",
        identical.score(),
    );

    let distorted_px = distort_blur(&pixels, W, H, 3);
    let dist = RgbSlice::new(&distorted_px, W, H);
    let blurred = z.compute(&src, &dist).expect("compute failed");
    assert!(
        blurred.score() < 100.0,
        "PreviewV0_1: blurred pair must score below 100, got {:.6}",
        blurred.score(),
    );
}

/// All PixelFormat variants produce equivalent scores.
/// sRGB↔f32: ±0.15, same-encoding reorder: ±0.01.
#[test]
fn pixel_format_equivalence() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::A);
    let src_pixels = gen_mandelbrot(W, H);
    let dst_pixels = distort_blur(&src_pixels, W, H, 3);

    // Reference: RgbSlice
    let ref_src = RgbSlice::new(&src_pixels, W, H);
    let ref_dst = RgbSlice::new(&dst_pixels, W, H);
    let ref_result = z.compute(&ref_src, &ref_dst).expect("ref compute failed");

    // Format converters and their tolerances
    type Converter = fn(&[[u8; 3]], usize, usize) -> (Vec<u8>, usize);

    struct FormatTest {
        name: &'static str,
        format: PixelFormat,
        converter: Converter,
        tolerance: f64,
    }

    let mut formats = vec![
        FormatTest {
            name: "Srgb8Rgb",
            format: PixelFormat::Srgb8Rgb,
            converter: to_srgb8_rgb,
            tolerance: 0.01,
        },
        FormatTest {
            name: "Srgb8Rgba",
            format: PixelFormat::Srgb8Rgba,
            converter: to_srgb8_rgba,
            tolerance: 0.01,
        },
        FormatTest {
            name: "Srgb8Bgra",
            format: PixelFormat::Srgb8Bgra,
            converter: to_srgb8_bgra,
            tolerance: 0.01,
        },
        FormatTest {
            name: "Srgb16Rgba",
            format: PixelFormat::Srgb16Rgba,
            converter: to_srgb16_rgba,
            tolerance: 0.01,
        },
    ];
    formats.push(FormatTest {
        name: "LinearF32Rgba",
        format: PixelFormat::LinearF32Rgba,
        converter: to_linear_f32_rgba,
        tolerance: 0.15,
    });

    println!("  Reference (RgbSlice): score={:.6}", ref_result.score());

    for fmt in &formats {
        let (src_buf, src_stride) = (fmt.converter)(&src_pixels, W, H);
        let (dst_buf, dst_stride) = (fmt.converter)(&dst_pixels, W, H);
        let src = StridedBytes::new(&src_buf, W, H, src_stride, fmt.format);
        let dst = StridedBytes::new(&dst_buf, W, H, dst_stride, fmt.format);
        let result = z.compute(&src, &dst).expect("compute failed");

        let diff = (result.score() - ref_result.score()).abs();
        println!(
            "  {:20} score={:.6}  diff={diff:.6}  (tol={:.2})",
            fmt.name,
            result.score(),
            fmt.tolerance,
        );
        assert!(
            diff <= fmt.tolerance,
            "{}: score {:.6} differs from reference {:.6} by {diff:.6} (>{:.2})",
            fmt.name,
            result.score(),
            ref_result.score(),
            fmt.tolerance,
        );
    }
}

/// Same computation 3x → bit-exact score, raw_distance, and features.
#[test]
fn determinism_same_platform() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::A);
    let pairs = generate_test_pairs(W, H);

    for pair in &pairs {
        let src = RgbSlice::new(&pair.source, W, H);
        let dst = RgbSlice::new(&pair.distorted, W, H);

        let r1 = z.compute(&src, &dst).expect("compute 1 failed");
        let r2 = z.compute(&src, &dst).expect("compute 2 failed");
        let r3 = z.compute(&src, &dst).expect("compute 3 failed");

        // Scores must be bit-exact
        assert_eq!(
            r1.score().to_bits(),
            r2.score().to_bits(),
            "{}: score not deterministic (run 1 vs 2): {} vs {}",
            pair.name,
            r1.score(),
            r2.score(),
        );
        assert_eq!(
            r1.score().to_bits(),
            r3.score().to_bits(),
            "{}: score not deterministic (run 1 vs 3): {} vs {}",
            pair.name,
            r1.score(),
            r3.score(),
        );

        // raw_distance must be bit-exact
        assert_eq!(
            r1.raw_distance().to_bits(),
            r2.raw_distance().to_bits(),
            "{}: raw_distance not deterministic",
            pair.name,
        );

        // mean_offset must be bit-exact
        for c in 0..3 {
            assert_eq!(
                r1.mean_offset()[c].to_bits(),
                r2.mean_offset()[c].to_bits(),
                "{}: mean_offset[{c}] not deterministic (run 1 vs 2)",
                pair.name,
            );
            assert_eq!(
                r1.mean_offset()[c].to_bits(),
                r3.mean_offset()[c].to_bits(),
                "{}: mean_offset[{c}] not deterministic (run 1 vs 3)",
                pair.name,
            );
        }

        // All features must be bit-exact
        for (i, ((f1, f2), f3)) in r1
            .features()
            .iter()
            .zip(r2.features().iter())
            .zip(r3.features().iter())
            .enumerate()
        {
            assert_eq!(
                f1.to_bits(),
                f2.to_bits(),
                "{}: feature[{i}] not deterministic (run 1 vs 2)",
                pair.name,
            );
            assert_eq!(
                f1.to_bits(),
                f3.to_bits(),
                "{}: feature[{i}] not deterministic (run 1 vs 3)",
                pair.name,
            );
        }
    }
    println!("  All 8 pairs × 3 runs bit-exact");
}

/// mean_offset must reflect XYB channel shifts for color-shifted images.
/// Identical images must have exactly [0, 0, 0].
#[test]
fn mean_offset_color_shift() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::A);
    let source = gen_mandelbrot(W, H);
    let shifted = distort_color_shift(&source, W, H);

    let src = RgbSlice::new(&source, W, H);
    let dst = RgbSlice::new(&shifted, W, H);
    let result = z.compute(&src, &dst).expect("compute failed");

    println!(
        "  mean_offset: X={:.6}, Y={:.6}, B={:.6}",
        result.mean_offset()[0],
        result.mean_offset()[1],
        result.mean_offset()[2],
    );

    // Color shift adds R+20, subtracts G-15, adds B+30.
    // In XYB space, Y channel (luminance) should show a non-trivial offset.
    // All three channels should have non-zero offsets.
    for (c, name) in result.mean_offset().iter().zip(["X", "Y", "B"]) {
        assert!(
            c.abs() > 1e-4,
            "mean_offset {name} should be non-trivial for color-shifted images, got {c:.6e}",
        );
    }
}

/// mean_offset via precomputed reference must match direct computation.
#[test]
fn mean_offset_precomputed_ref() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::A);
    let source = gen_mandelbrot(W, H);
    let shifted = distort_color_shift(&source, W, H);

    let src = RgbSlice::new(&source, W, H);
    let dst = RgbSlice::new(&shifted, W, H);

    let direct = z.compute(&src, &dst).expect("direct compute failed");
    let precomputed = z.precompute_reference(&src).expect("precompute failed");
    let with_ref = z
        .compute_with_ref(&precomputed, &dst)
        .expect("compute_with_ref failed");

    for c in 0..3 {
        let diff = (direct.mean_offset()[c] - with_ref.mean_offset()[c]).abs();
        assert!(
            diff < 1e-10,
            "mean_offset[{c}] mismatch: direct={:.10}, with_ref={:.10}, diff={diff:.2e}",
            direct.mean_offset()[c],
            with_ref.mean_offset()[c],
        );
    }
    println!(
        "  direct vs precomputed: max diff = {:.2e}",
        (0..3)
            .map(|c| (direct.mean_offset()[c] - with_ref.mean_offset()[c]).abs())
            .fold(0.0f64, f64::max),
    );
}

// ─── Error condition tests ───────────────────────────────────────────────────

#[test]
fn small_images_score_via_reflect_pad() {
    // 2026-06-06: sub-64px images are reflect(mirror)-padded to the 4-scale
    // pyramid minimum inside `compute`, so they now SCORE (down to 1×1)
    // rather than erroring. See tests/size_invariance.rs for the invariance
    // gate. The precomputed-reference and strip-streaming paths now reflect-pad
    // sub-64px inputs too (consistent with the direct `compute` path).
    let z = Zensim::new(ZensimProfile::A);
    let small = vec![[128u8; 3]; 4 * 4];
    let src = RgbSlice::new(&small, 4, 4);
    let dst = RgbSlice::new(&small, 4, 4);
    let s = z
        .compute(&src, &dst)
        .expect("4x4 now scores via reflect-pad")
        .score();
    assert!(s.is_finite() && (0.0..=100.0).contains(&s), "4x4 score {s}");
    // Empty images are still unscoreable.
    let empty: Vec<[u8; 3]> = Vec::new();
    assert_eq!(
        z.compute(&RgbSlice::new(&empty, 0, 0), &RgbSlice::new(&empty, 0, 0))
            .unwrap_err(),
        ZensimError::ImageTooSmall
    );
    // Precomputed-reference reuse now ALSO reflect-pads sub-64px sources to the
    // pyramid minimum (consistent with the direct `compute` path), so the whole
    // streaming family scores small images.
    let pref = z
        .precompute_reference(&src)
        .expect("4x4 reference now reflect-pads");
    let rs = z
        .compute_with_ref(&pref, &dst)
        .expect("4x4 compute_with_ref now scores")
        .score();
    assert!(
        rs.is_finite() && (0.0..=100.0).contains(&rs),
        "4x4 ref score {rs}"
    );
    // ...and the strip-streaming paths too.
    assert!(
        z.compute_streaming_strips(&src, &dst, 256, 128).is_ok(),
        "4x4 strip-streaming now scores"
    );
}

#[test]
fn error_dimension_mismatch() {
    let z = Zensim::new(ZensimProfile::A);
    let a = vec![[128u8; 3]; 16 * 16];
    let b = vec![[128u8; 3]; 32 * 8];
    let src = RgbSlice::new(&a, 16, 16);
    let dst = RgbSlice::new(&b, 32, 8);
    assert_eq!(
        z.compute(&src, &dst).unwrap_err(),
        ZensimError::DimensionMismatch
    );
}

#[test]
fn error_invalid_data_length_rgb() {
    // 15 pixels for a 4×4 image (should be 16)
    let short = vec![[128u8; 3]; 15];
    let result = RgbSlice::try_new(&short, 4, 4);
    assert_eq!(result.unwrap_err(), ZensimError::InvalidDataLength);
}

#[test]
fn error_invalid_data_length_rgba() {
    let short = vec![[128u8; 4]; 15];
    let result = RgbaSlice::try_new(&short, 4, 4);
    assert_eq!(result.unwrap_err(), ZensimError::InvalidDataLength);
}

#[test]
fn error_invalid_stride() {
    // stride of 10 bytes for 4-pixel-wide RGB (needs 12)
    let data = vec![0u8; 100];
    let result = StridedBytes::try_new(&data, 4, 4, 10, PixelFormat::Srgb8Rgb);
    assert_eq!(result.unwrap_err(), ZensimError::InvalidStride);
}

#[test]
fn error_invalid_data_length_strided() {
    // stride 24, 4 rows = 96 bytes needed, only 80 provided
    let data = vec![0u8; 80];
    let result = StridedBytes::try_new(&data, 4, 4, 24, PixelFormat::Srgb8Rgb);
    assert_eq!(result.unwrap_err(), ZensimError::InvalidDataLength);
}
