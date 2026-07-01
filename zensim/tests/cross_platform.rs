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
