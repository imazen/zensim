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
/// Tolerance: ±1e-5 on the 0-100 scale (~55× headroom over observed x86↔ARM divergence).
#[test]
fn hardcoded_reference_scores() {
    const W: usize = 128;
    const H: usize = 128;
    const TOLERANCE: f64 = 1e-5;
    let z = Zensim::new(ZensimProfile::latest());
    let pairs = generate_test_pairs(W, H);

    // Reference scores with concordant-trained weights (228 weights, SROCC=0.9942).
    // Uses linear-srgb crate (C0-continuous constants) for sRGB linearization.
    #[allow(clippy::excessive_precision)]
    let expected: &[(&str, f64)] = &[
        ("checkerboard+blur", -79.872_537_867_843_306),
        ("checkerboard+sharpen", 29.587_543_251_423_142),
        ("mandelbrot+blur", 8.478_409_174_126_597),
        ("mandelbrot+color_shift", 48.138_848_249_261_251),
        ("noise+blur", 60.015_600_223_264_798),
        ("noise+block_artifacts", 52.799_117_798_200_086),
        ("color_blocks+color_shift", 30.398_914_843_637_783),
        ("color_blocks+sharpen", -5.656_076_020_014_737),
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

/// All PixelFormat variants produce equivalent scores.
/// sRGB↔f32: ±0.15, same-encoding reorder: ±0.01.
#[test]
fn pixel_format_equivalence() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::latest());
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
    let z = Zensim::new(ZensimProfile::latest());
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

/// Basic sanity: identical=100, blur<100, heavier blur=lower score, all in [0,100].
#[test]
fn score_sanity_checks() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::latest());
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

    // All scores <= 100, can go negative for extreme distortions
    let pairs = generate_test_pairs(W, H);
    for pair in &pairs {
        let src = RgbSlice::new(&pair.source, W, H);
        let dst = RgbSlice::new(&pair.distorted, W, H);
        let result = z.compute(&src, &dst).expect("compute failed");
        assert!(
            result.score() <= 100.0,
            "{}: score {:.4} above 100",
            pair.name,
            result.score(),
        );
    }
    println!("  All scores <= 100 (sub-zero allowed for extreme distortions)");
}

/// Same computation 3x → bit-exact score, raw_distance, and features.
#[test]
fn determinism_same_platform() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::latest());
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

/// Identical images must score exactly 100.0 with raw_distance=0.0 and all-zero features.
/// Tests all 4 generators with separately-allocated copies (not same pointer).
#[test]
fn identical_images_score_100() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::latest());

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

/// mean_offset must reflect XYB channel shifts for color-shifted images.
/// Identical images must have exactly [0, 0, 0].
#[test]
fn mean_offset_color_shift() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::latest());
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
    let z = Zensim::new(ZensimProfile::latest());
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
fn error_image_too_small() {
    let z = Zensim::new(ZensimProfile::latest());
    // 4×4 is below 8×8 minimum
    let small = vec![[128u8; 3]; 4 * 4];
    let src = RgbSlice::new(&small, 4, 4);
    let dst = RgbSlice::new(&small, 4, 4);
    assert_eq!(
        z.compute(&src, &dst).unwrap_err(),
        ZensimError::ImageTooSmall
    );
    assert!(matches!(
        z.precompute_reference(&src),
        Err(ZensimError::ImageTooSmall)
    ));
}

#[test]
fn error_dimension_mismatch() {
    let z = Zensim::new(ZensimProfile::latest());
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
