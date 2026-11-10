//! Reproduce NaN in zensim diffmap when fed out-of-range linear RGB.
//!
//! The encoder's XYB→RGB inverse transform can produce values outside [0,1]
//! due to quantization error. When these are passed to zensim's planar linear
//! RGB API, the SSIM statistics can produce NaN via division-by-near-zero.

#![allow(deprecated)] // exercises the adopted `ZensimProfile::A`

#[test]
fn diffmap_nan_from_negative_linear_rgb() {
    let width = 64;
    let height = 64;
    let n = width * height;

    // Source: normal linear RGB in [0, 1]
    let src_r: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 0.8 + 0.1).collect();
    let src_g = src_r.clone();
    let src_b = src_r.clone();

    // Distorted: some values out of range (simulating XYB→RGB inverse error)
    let mut dst_r = src_r.clone();
    let mut dst_g = src_g.clone();
    let mut dst_b = src_b.clone();

    // Introduce out-of-range values like the encoder produces
    for i in 0..n / 4 {
        dst_r[i] = -0.3; // negative
        dst_g[i] = 1.5; // above 1.0
        dst_b[i] = -0.1;
    }

    let z = zensim::Zensim::new(zensim::ZensimProfile::A).with_parallel(false);
    let precomputed = z
        .precompute_reference_linear_planar([&src_r, &src_g, &src_b], width, height, width)
        .unwrap();

    let opts = zensim::DiffmapOptions {
        weighting: zensim::DiffmapWeighting::Trained,
        include_edge_mse: true,
        ..Default::default()
    };

    let result = z
        .compute_with_ref_and_diffmap_linear_planar(
            &precomputed,
            [&dst_r, &dst_g, &dst_b],
            width,
            height,
            width,
            opts,
        )
        .unwrap();

    let diffmap = result.diffmap();
    let nan_count = diffmap.iter().filter(|v| v.is_nan()).count();
    let inf_count = diffmap.iter().filter(|v| v.is_infinite()).count();

    eprintln!(
        "diffmap: len={} nan={} inf={} score={:.6} approx_bfly={:.6}",
        diffmap.len(),
        nan_count,
        inf_count,
        result.score(),
        result.result().approx_butteraugli(),
    );

    assert_eq!(
        nan_count, 0,
        "{nan_count} NaN in diffmap from out-of-range linear RGB"
    );
    assert_eq!(
        inf_count, 0,
        "{inf_count} Inf in diffmap from out-of-range linear RGB"
    );
    assert!(result.score().is_finite(), "score is not finite");
}

#[test]
fn diffmap_nan_from_realistic_encoder_output() {
    // Simulate what the encoder actually produces: mostly in-range values
    // with a few slightly out-of-range pixels from quantization error.
    let width = 128;
    let height = 128;
    let n = width * height;

    // Source: smooth gradient
    let src_r: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            (x * 0.7 + y * 0.2 + 0.05).clamp(0.0, 1.0)
        })
        .collect();
    let src_g: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            (x * 0.3 + y * 0.5 + 0.1).clamp(0.0, 1.0)
        })
        .collect();
    let src_b: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            (x * 0.1 + y * 0.6 + 0.15).clamp(0.0, 1.0)
        })
        .collect();

    // Distorted: add small quantization noise, with occasional out-of-range
    let mut dst_r = src_r.clone();
    let mut dst_g = src_g.clone();
    let mut dst_b = src_b.clone();

    // Add realistic noise pattern
    for i in 0..n {
        let noise = ((i * 7 + 13) % 100) as f32 / 1000.0 - 0.05; // [-0.05, 0.05]
        dst_r[i] += noise;
        dst_g[i] += noise * 0.8;
        dst_b[i] += noise * 1.2;
    }
    // A few strongly out-of-range pixels (dark areas where XYB→RGB overshoots)
    for i in (0..n).step_by(37) {
        dst_r[i] = -0.18;
        dst_b[i] = -0.05;
    }
    for i in (0..n).step_by(53) {
        dst_g[i] = 1.3;
    }

    let z = zensim::Zensim::new(zensim::ZensimProfile::A).with_parallel(false);
    let precomputed = z
        .precompute_reference_linear_planar([&src_r, &src_g, &src_b], width, height, width)
        .unwrap();

    let opts = zensim::DiffmapOptions {
        weighting: zensim::DiffmapWeighting::Trained,
        include_edge_mse: true,
        ..Default::default()
    };

    let result = z
        .compute_with_ref_and_diffmap_linear_planar(
            &precomputed,
            [&dst_r, &dst_g, &dst_b],
            width,
            height,
            width,
            opts,
        )
        .unwrap();

    let diffmap = result.diffmap();
    let nan_count = diffmap.iter().filter(|v| v.is_nan()).count();
    let inf_count = diffmap.iter().filter(|v| v.is_infinite()).count();
    let score = result.score();
    let bfly = result.result().approx_butteraugli();

    eprintln!(
        "realistic: len={} nan={} inf={} score={:.6} bfly≈{:.6}",
        diffmap.len(),
        nan_count,
        inf_count,
        score,
        bfly
    );

    assert_eq!(
        nan_count, 0,
        "{nan_count} NaN in diffmap from realistic encoder output"
    );
    assert_eq!(
        inf_count, 0,
        "{inf_count} Inf in diffmap from realistic encoder output"
    );
    assert!(score.is_finite(), "score is not finite: {score}");
    assert!(bfly.is_finite(), "approx_butteraugli is not finite: {bfly}");
}

/// The REV4VEC gate: every SIMD tier must suppress internally generated NaN
/// IDENTICALLY — aarch64 NEON regressed here on 2026-10-03 because vector
/// `max`/`min` propagate NaN where `f32::max`'s maxNum drops it
/// (imazen/archmage#145); the canonical kernels ran scalar on aarch64 until
/// the clamps were rewritten as ordered-compare + blend. Finite adversarial
/// inputs still produce NaN mid-kernel (extreme moments overflow to `inf`,
/// `inf/inf` yields NaN, and the canon's `.max(0)` clamps are the
/// suppressors), so every dispatch permutation must agree bit-for-bit on
/// score, features and diffmap.
#[test]
fn nan_suppression_is_tier_invariant() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    let width = 97;
    let height = 65;
    let n = width * height;

    // Source: normal linear RGB in [0, 1]
    let src: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 0.8 + 0.1).collect();
    let (src_r, src_g, src_b) = (src.clone(), src.clone(), src);

    // Distorted: finite but adversarial — out-of-range values, extreme
    // magnitudes (moment sums overflow to `inf`, `inf/inf` yields NaN
    // mid-kernel) and subnormals, spread so lanes and tails all see them.
    let mut dst_r = src_r.clone();
    let mut dst_g = src_g.clone();
    let mut dst_b = src_b.clone();
    for i in (0..n).step_by(7) {
        dst_r[i] = -0.3;
        dst_g[i] = 1.5;
        dst_b[i] = -0.1;
    }
    for i in (0..n).step_by(211) {
        dst_r[i] = 1e30;
        dst_g[i] = -1e30;
    }
    for i in (0..n).step_by(503) {
        dst_b[i] = 1e-40;
    }

    let z = zensim::Zensim::new(zensim::ZensimProfile::A).with_parallel(false);
    let mut runs: Vec<(String, f64, Vec<f64>, Vec<f32>)> = Vec::new();
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let precomputed = z
            .precompute_reference_linear_planar([&src_r, &src_g, &src_b], width, height, width)
            .unwrap();
        let result = z
            .compute_with_ref_and_diffmap_linear_planar(
                &precomputed,
                [&dst_r, &dst_g, &dst_b],
                width,
                height,
                width,
                zensim::DiffmapOptions {
                    weighting: zensim::DiffmapWeighting::Trained,
                    include_edge_mse: true,
                    ..Default::default()
                },
            )
            .unwrap();
        runs.push((
            perm.label.clone(),
            result.score(),
            result.result().features().to_vec(),
            result.diffmap().to_vec(),
        ));
    });

    let (ref_label, ref_score, ref_feats, ref_map) = &runs[0];
    eprintln!(
        "nan tier-invariance: {} permutations ({} executed)",
        runs.len(),
        report.permutations_run
    );
    assert!(
        ref_score.is_finite(),
        "score not finite on {ref_label}: {ref_score}"
    );
    assert!(
        ref_map.iter().all(|v| v.is_finite()),
        "diffmap not finite on {ref_label}"
    );
    for (label, score, feats, map) in &runs[1..] {
        assert_eq!(
            score.to_bits(),
            ref_score.to_bits(),
            "score diverges: {ref_label} vs {label}"
        );
        assert_eq!(feats.len(), ref_feats.len());
        for (i, (a, b)) in feats.iter().zip(ref_feats.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "feature[{i}] diverges: {ref_label} vs {label}"
            );
        }
        assert_eq!(map.len(), ref_map.len());
        for (i, (a, b)) in map.iter().zip(ref_map.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "diffmap[{i}] diverges: {ref_label} vs {label}"
            );
        }
    }
}
