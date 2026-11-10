//! Cross-platform score consistency tests for zensim.
//!
//! Validates that hardcoded reference scores match across all 7 CI platforms,
//! that all 6 PixelFormat variants produce equivalent scores, that all 228
//! features activate on synthetic test images, and that results are deterministic.
//!
//! Run with: `cargo test -p zensim --all-features --test cross_platform`

#![allow(deprecated)] // exercises the adopted `ZensimProfile::A`

mod common;

use common::generators::*;
use zensim::{
    DiffmapOptions, PixelFormat, RgbSlice, RgbaSlice, StridedBytes, Zensim, ZensimError,
    ZensimProfile,
};

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

#[cfg(feature = "legacy-models")]
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

/// Hardcoded profile-A reference scores — pinned **bit-exact**, unlike the
/// V0_2 table above.
///
/// A (revision 7) is designed for cross-platform bit-exactness: every
/// transcendental in the score path goes through `PowForm::PureRust`
/// (pure-Rust libm, no platform libc variance), the canonical kernels are
/// bit-identical across SIMD tiers (the maxNum-safe clamp forms; verified by
/// `cross_tier` and `nan_repro` on every CI architecture), and the f64
/// reductions are thread-count invariant (parallel and serial compute the
/// same bits).
///
/// Verified bit-identical on x86-64 (Zen4, v4x/v3 tiers) and Apple M6
/// (AArch64 NEON): all 8 scores below match to_bits() on both platforms.
/// A failure here means some backend produced different bits — the exact
/// class of bug the 2026-10-03 REV4VEC scalar-fallback hid.
#[test]
fn hardcoded_reference_scores_a() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::A);
    let pairs = generate_test_pairs(W, H);

    #[allow(clippy::excessive_precision)]
    let expected: &[(&str, u64)] = &[
        ("checkerboard+blur", 0xc04a71132b781358),
        ("checkerboard+sharpen", 0x4051eaec883df850),
        ("mandelbrot+blur", 0xc0324213b1d90bb0),
        ("mandelbrot+color_shift", 0x4029cbcd84d43acc),
        ("noise+blur", 0x40491b6b0aed6865),
        ("noise+block_artifacts", 0x403df7d44f23d45d),
        ("color_blocks+color_shift", 0x40413b80fb2969e6),
        ("color_blocks+sharpen", 0x4031223d84002ebe),
    ];

    let mut failures = Vec::new();
    for (pair, &(name, expected_bits)) in pairs.iter().zip(expected.iter()) {
        assert_eq!(pair.name, name, "Test pair order mismatch");
        let src = RgbSlice::new(&pair.source, W, H);
        let dst = RgbSlice::new(&pair.distorted, W, H);
        let result = z.compute(&src, &dst).expect("compute failed");
        let bits = result.score().to_bits();
        println!(
            "  {name:30} score={:.15}  bits={bits:016x}  expected={expected_bits:016x}",
            result.score(),
        );
        if bits != expected_bits {
            failures.push(format!(
                "{name}: score bits {bits:016x} != expected {expected_bits:016x} ({:.15} vs {:.15})",
                result.score(),
                f64::from_bits(expected_bits),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Profile A is not bit-exact on this platform:\n{}",
        failures.join("\n")
    );
}

/// Profile-A output goldens across the full representative distortion set:
/// per-pixel diffmap (XXH3-128), feature vector (XXH3-128) and score bits —
/// all pinned **exact** across platforms.
///
/// Every distortion class the codec-facing metric sees is covered at least
/// once per generated source: blur (lowpass), sharpen (HF gain), color shift
/// (chroma), JPEG-style block artifacts, additive noise and posterize
/// (banding/quantization), plus an identity control.
///
/// Why exact hashes are legitimate: Rev7 score math uses `PowForm::PureRust`
/// (no platform libm), the canonical kernels are bit-identical across SIMD
/// tiers (maxNum-safe clamps), and the reductions are thread-count invariant.
/// The diffmap additionally exercises the multi-scale fused map, edge/MSE/HF
/// accumulators, contrast masking and the final sqrt — hashed over canonical
/// little-endian f32 bytes, so the golden is endianness-independent.
///
/// Verified bit-identical on x86-64 (Zen4) and Apple M6 (AArch64 NEON).
/// Regenerate the table with `ZENSIM_GOLDEN_HARVEST=1 cargo test -p zensim
/// --release --test cross_platform map_feature_goldens_a -- --nocapture`.
#[test]
fn map_feature_goldens_a() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::A);
    // Exercise the full map machinery: SSIM + edge/MSE + HF accumulators,
    // contrast masking and the final sqrt.
    let options = DiffmapOptions {
        include_edge_mse: true,
        include_hf: true,
        masking_strength: Some(4.0),
        sqrt: true,
        ..Default::default()
    };

    let checker = gen_checkerboard(W, H, 8);
    let mandel = gen_mandelbrot(W, H);
    let noise = gen_value_noise(W, H, 42);
    let blocks = gen_color_blocks(W, H);
    let pairs: Vec<(&str, Vec<[u8; 3]>, Vec<[u8; 3]>)> = vec![
        (
            "checkerboard+blur",
            checker.clone(),
            distort_blur(&checker, W, H, 3),
        ),
        (
            "checkerboard+sharpen",
            checker.clone(),
            distort_sharpen(&checker, W, H),
        ),
        (
            "checkerboard+posterize",
            checker.clone(),
            distort_posterize(&checker, W, H),
        ),
        (
            "mandelbrot+blur",
            mandel.clone(),
            distort_blur(&mandel, W, H, 3),
        ),
        (
            "mandelbrot+color_shift",
            mandel.clone(),
            distort_color_shift(&mandel, W, H),
        ),
        (
            "mandelbrot+noise",
            mandel.clone(),
            distort_noise(&mandel, W, H, 10, 0x5EED_0001),
        ),
        ("mandelbrot+identity", mandel.clone(), mandel.clone()),
        ("noise+blur", noise.clone(), distort_blur(&noise, W, H, 3)),
        (
            "noise+block_artifacts",
            noise.clone(),
            distort_block_artifacts(&noise, W, H),
        ),
        (
            "noise+posterize",
            noise.clone(),
            distort_posterize(&noise, W, H),
        ),
        (
            "color_blocks+color_shift",
            blocks.clone(),
            distort_color_shift(&blocks, W, H),
        ),
        (
            "color_blocks+sharpen",
            blocks.clone(),
            distort_sharpen(&blocks, W, H),
        ),
        (
            "color_blocks+noise",
            blocks.clone(),
            distort_noise(&blocks, W, H, 10, 0x5EED_0002),
        ),
    ];

    let le_bytes = |v: &[f64]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
    let le_bytes32 = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };

    // (name, score_bits, features_xxh3_128, diffmap_xxh3_128)
    let expected: &[(&str, u64, u128, u128)] = &[
        (
            "checkerboard+blur",
            0xc04a71132b781358,
            0xb6e78a5c84308e9b92860bc2c828ee6d,
            0xc33bca8b398d0f2bb98d60922b34c606,
        ),
        (
            "checkerboard+sharpen",
            0x4051eaec883df850,
            0x19a3d15f7e8e49df267445d88514dffd,
            0xe174681bcba287ba133b5780a771c1dc,
        ),
        (
            "checkerboard+posterize",
            0x4052b58923c27b22,
            0xf40eb381de092409fb690b462da6a019,
            0x4e9eb4f5fe7a7af2027328a984fec78b,
        ),
        (
            "mandelbrot+blur",
            0xc0324213b1d90bb0,
            0x4a99b5851b71b028e980e6f108e4f37d,
            0x25ee677c7bd9d6166bbdc5f2628ac397,
        ),
        (
            "mandelbrot+color_shift",
            0x4029cbcd84d43acc,
            0xd7c10743f2c7afcab039fb87aeb1c333,
            0xd4e21adae47c9fbe6dbab1036cee6007,
        ),
        (
            "mandelbrot+noise",
            0x404df7eeb7fc8632,
            0x70ec26b762bdace3d461a7e22fa31d0f,
            0x921dc839f3fb8e3452256656e4298b8f,
        ),
        (
            "mandelbrot+identity",
            0x4059000000000000,
            0xde4886cdae5cc3cc9d9af848e275ad75,
            0xfd5ee061c8433a0f33b202d302b65caa,
        ),
        (
            "noise+blur",
            0x40491b6b0aed6865,
            0x3cf5413296dc40d616a5fa457fdc1b0e,
            0x399345a3070de0ffe93513691934be83,
        ),
        (
            "noise+block_artifacts",
            0x403df7d44f23d45d,
            0x8c8e783807a4b1e78b237c08e3481c19,
            0xa8cc04f74e8c45ee00e087491b8f0227,
        ),
        (
            "noise+posterize",
            0x405336b23e4937a6,
            0x1814f7e982e2ab440af46dae1483c495,
            0x56497f6fb14ff180f9892cac5e2b9950,
        ),
        (
            "color_blocks+color_shift",
            0x40413b80fb2969e6,
            0x3b60d29b00916dbee4908ab1e86e1311,
            0x761bdf1bdd43dad4cad93e806665ee3d,
        ),
        (
            "color_blocks+sharpen",
            0x4031223d84002ebe,
            0x0294cd69cfd0131e9b15bcf80fac25bd,
            0x3f49b6a6b04a1c39bb7fbac41f6e4eed,
        ),
        (
            "color_blocks+noise",
            0x404d15118458c341,
            0x72defb8a37829bcd2ff8221d42fca45d,
            0x8520cd85151c985e02407308faa02907,
        ),
    ];

    let harvest = std::env::var_os("ZENSIM_GOLDEN_HARVEST").is_some();
    let mut failures = Vec::new();
    for ((name, src, dst), &(exp_name, exp_score, exp_feat, exp_map)) in
        pairs.iter().zip(expected.iter())
    {
        assert_eq!(*name, exp_name, "Test pair order mismatch");
        let r = z
            .compute_with_diffmap(
                &RgbSlice::new(src, W, H),
                &RgbSlice::new(dst, W, H),
                options,
            )
            .expect("compute_with_diffmap failed");
        let score_bits = r.score().to_bits();
        let feat_hash = zenhash::xxh3_128(&le_bytes(r.result().features()));
        let map_hash = zenhash::xxh3_128(&le_bytes32(r.diffmap()));
        println!("  ({name:?}, 0x{score_bits:016x}, 0x{feat_hash:032x}, 0x{map_hash:032x}),",);
        if harvest {
            continue;
        }
        if score_bits != exp_score || feat_hash != exp_feat || map_hash != exp_map {
            failures.push(format!(
                "{name}: got (score=0x{score_bits:016x}, feat=0x{feat_hash:032x}, map=0x{map_hash:032x}) \
                 expected (0x{exp_score:016x}, 0x{exp_feat:032x}, 0x{exp_map:032x})",
            ));
        }
    }
    if harvest {
        panic!("harvest mode: re-run without ZENSIM_GOLDEN_HARVEST to assert");
    }
    assert!(
        failures.is_empty(),
        "Profile A outputs are not bit-exact on this platform:\n{}",
        failures.join("\n")
    );
}

#[cfg(feature = "legacy-models")]
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
    // must run on the linear `PreviewV0_2` profile.
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

#[cfg(feature = "legacy-models")]
/// Tests all 4 generators with separately-allocated copies (not same pointer).
///
/// Pinned to `PreviewV0_2` rather than `latest()` because the
/// "raw_distance == 0.0 for identical input" invariant is V0_2-specific:
/// V0_2 is a linear-weighted sum of feature distances, so all-zero
/// features → 0 distance → score 100 by the `100 − 18·d^0.7` mapping.
/// The MLP profile `A` evaluates a network forward pass
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

#[cfg(feature = "legacy-models")]
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

#[cfg(feature = "legacy-models")]
/// mean_offset must reflect XYB channel shifts for color-shifted images.
/// Identical images must have exactly [0, 0, 0].
#[test]
fn mean_offset_color_shift() {
    const W: usize = 128;
    const H: usize = 128;
    let z = Zensim::new(ZensimProfile::PreviewV0_2);
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
