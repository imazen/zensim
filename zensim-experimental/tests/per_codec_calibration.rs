//! Runtime test for EXP-CROSS-CODEC-V11-E per-codec post-spline affine
//! calibration (task #186, 2026-05-20).
//!
//! This test guards the per-codec calibration runtime dispatch:
//!
//! 1. The `*_per_codec_2026-05-20.bin` bakes load without panic.
//! 2. Calling `Zensim::compute()` (no codec hint) produces a score
//!    bit-exact to the un-calibrated parent bake — the per-codec
//!    affine is opt-in via `compute_with_codec_hint`.
//! 3. Calling `compute_with_codec_hint(_, _, Some("unknown_codec"))`
//!    also produces the identical baseline — unknown codecs fall back
//!    to identity by lookup.
//! 4. Calling `compute_with_codec_hint(_, _, Some("jpeg"))` produces
//!    a DIFFERENT score from baseline (per-codec affine fires). The
//!    delta is small but nonzero (the fit's α was on the order of
//!    a few score units).
//! 5. The same hint always produces the same score (deterministic).
//!
//! Together these guard the wiring without depending on the specific
//! fit numbers (which can change across substrate revisions).

use zensim::{RgbSlice, Zensim, ZensimProfile};

fn make_pair_with_delta(w: usize, h: usize, delta: u8) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = w * h;
    let src: Vec<[u8; 3]> = (0..n)
        .map(|i| {
            let x = ((i % w) * 255 / w) as u8;
            let y = ((i / w) * 255 / h) as u8;
            [x, y, x.wrapping_add(y)]
        })
        .collect();
    let dst: Vec<[u8; 3]> = src
        .iter()
        .map(|&[r, g, b]| {
            [
                r.saturating_add(delta),
                g.saturating_add(delta.saturating_sub(2)),
                b.saturating_add(delta.saturating_sub(1)),
            ]
        })
        .collect();
    (src, dst)
}

fn score_with_profile(
    profile: ZensimProfile,
    w: usize,
    h: usize,
    delta: u8,
    codec_hint: Option<&str>,
) -> f64 {
    let zen = Zensim::new(profile).with_parallel(false);
    let (src, dst) = make_pair_with_delta(w, h, delta);
    let src_slice = RgbSlice::new(&src, w, h);
    let dst_slice = RgbSlice::new(&dst, w, h);
    let result = zen
        .compute_with_codec_hint(&src_slice, &dst_slice, codec_hint)
        .unwrap();
    result.score()
}

/// Per-codec calibration tests run against the three V10 ship variants
/// that carry the `zentrain.per_codec_calibration` metadata.
///
/// Note: at the time of writing, the per-codec affine fit was a near-
/// no-op for the V10 ships (the cross-codec stddev tightening was
/// <5% on the held-out substrate; see
/// `benchmarks/v11_e_per_codec_falsification_2026-05-20.md`). The
/// metadata + runtime are nevertheless guarded so future bakes that
/// benefit from per-codec affine can ship without re-implementing the
/// dispatch.
#[test]
fn per_codec_calibration_dispatch_identity_without_hint() {
    // No codec hint → bake's per-codec affine is silently skipped.
    // The score equals what the un-calibrated bake would produce.
    let profile = zensim_experimental::preview_v0_5_tuner_v4_calibrated();
    let s_no_hint = score_with_profile(profile, 64, 64, 30, None);
    let s_unknown = score_with_profile(profile, 64, 64, 30, Some("not_a_codec"));
    let diff = (s_no_hint - s_unknown).abs();
    assert!(
        diff < 1e-9,
        "unknown codec hint must produce identity score; \
         no-hint={s_no_hint:.6}  unknown={s_unknown:.6}  diff={diff:.3e}"
    );
}

#[test]
fn per_codec_calibration_calibrated_matches_base_without_hint() {
    // The *_Calibrated profile + no hint must produce a score
    // identical to the un-calibrated parent profile. This is the
    // load-bearing invariant: bake_verdict and any other tool that
    // doesn't supply a codec hint sees the same SROCC.
    let s_base = score_with_profile(
        zensim_experimental::preview_v0_5_tuner_v4(),
        64,
        64,
        30,
        None,
    );
    let s_calib = score_with_profile(
        zensim_experimental::preview_v0_5_tuner_v4_calibrated(),
        64,
        64,
        30,
        None,
    );
    assert!(
        (s_base - s_calib).abs() < 1e-9,
        "calibrated bake without hint MUST match un-calibrated; \
         base={s_base:.6}  calib={s_calib:.6}"
    );
}

#[test]
fn per_codec_calibration_dispatch_jpeg_hint_changes_score_tuner() {
    // The TunerV4Calibrated bake carries `zentrain.per_codec_calibration`
    // metadata. A JPEG hint must produce a different score than no-hint
    // baseline. The fit's α was ~-5 on TunerV4 (post-spline raw output
    // is ~5 score units higher than the ssim2 target on average), so the
    // calibrated score is lower.
    let profile = zensim_experimental::preview_v0_5_tuner_v4_calibrated();
    let s_no_hint = score_with_profile(profile, 64, 64, 30, None);
    let s_jpeg = score_with_profile(profile, 64, 64, 30, Some("jpeg"));
    let s_jpeg_2 = score_with_profile(profile, 64, 64, 30, Some("zenjpeg"));
    let s_webp = score_with_profile(profile, 64, 64, 30, Some("webp"));
    // jpeg / zenjpeg alias to the same codec entry.
    assert!(
        (s_jpeg - s_jpeg_2).abs() < 1e-9,
        "jpeg vs zenjpeg should alias; jpeg={s_jpeg:.6}  zenjpeg={s_jpeg_2:.6}"
    );
    // Any codec hint should produce a different score from no-hint.
    assert!(
        (s_no_hint - s_jpeg).abs() > 1e-6,
        "jpeg hint must change score (per-codec affine fires); \
         no-hint={s_no_hint:.6}  jpeg={s_jpeg:.6}"
    );
    // jpeg and webp should produce different scores (different fits).
    assert!(
        (s_jpeg - s_webp).abs() > 1e-6,
        "jpeg and webp must produce different scores; \
         jpeg={s_jpeg:.6}  webp={s_webp:.6}"
    );
}

#[test]
fn per_codec_calibration_dispatch_balanced_v3_jpeg_changes_score() {
    let profile = zensim_experimental::preview_v0_5_balanced_v3_calibrated();
    let s_no_hint = score_with_profile(profile, 64, 64, 30, None);
    let s_jpeg = score_with_profile(profile, 64, 64, 30, Some("jpeg"));
    assert!(
        (s_no_hint - s_jpeg).abs() > 1e-6,
        "BalancedV3Calibrated + per-codec metadata: jpeg hint must change score; \
         no-hint={s_no_hint:.6}  jpeg={s_jpeg:.6}"
    );
}

#[test]
fn per_codec_calibration_dispatch_compression_v3_jpeg_changes_score() {
    let profile = zensim_experimental::preview_v0_5_compression_v3_calibrated();
    let s_no_hint = score_with_profile(profile, 64, 64, 30, None);
    let s_jpeg = score_with_profile(profile, 64, 64, 30, Some("jpeg"));
    assert!(
        (s_no_hint - s_jpeg).abs() > 1e-6,
        "CompressionV3Calibrated + per-codec metadata: jpeg hint must change score; \
         no-hint={s_no_hint:.6}  jpeg={s_jpeg:.6}"
    );
}

#[test]
fn per_codec_calibration_dispatch_deterministic() {
    // Same hint, same input → identical score across calls.
    let profile = zensim_experimental::preview_v0_5_tuner_v4_calibrated();
    let s1 = score_with_profile(profile, 64, 64, 30, Some("jpeg"));
    let s2 = score_with_profile(profile, 64, 64, 30, Some("jpeg"));
    let s3 = score_with_profile(profile, 64, 64, 30, Some("JPEG"));
    let s4 = score_with_profile(profile, 64, 64, 30, Some("mozjpeg"));
    assert_eq!(s1, s2, "deterministic — same call");
    assert_eq!(s1, s3, "deterministic — case-insensitive");
    assert_eq!(s1, s4, "deterministic — alias mozjpeg → jpeg");
}

#[test]
fn per_codec_calibration_no_metadata_profile_ignores_hint() {
    // A profile WITHOUT the `zentrain.per_codec_calibration` metadata
    // (e.g. plain PreviewV0_5TunerV4) must produce identical scores
    // with or without a codec hint — the runtime parses the metadata
    // and only applies the affine when the metadata + matching codec
    // are BOTH present.
    let profile = zensim_experimental::preview_v0_5_tuner_v4();
    let s_no_hint = score_with_profile(profile, 64, 64, 30, None);
    let s_jpeg = score_with_profile(profile, 64, 64, 30, Some("jpeg"));
    let s_webp = score_with_profile(profile, 64, 64, 30, Some("webp"));
    assert!(
        (s_no_hint - s_jpeg).abs() < 1e-9,
        "TunerV4 (no per-codec metadata) must ignore hint; \
         no-hint={s_no_hint:.6}  jpeg={s_jpeg:.6}"
    );
    assert!(
        (s_no_hint - s_webp).abs() < 1e-9,
        "TunerV4 (no per-codec metadata) must ignore hint; \
         no-hint={s_no_hint:.6}  webp={s_webp:.6}"
    );
}

#[test]
fn per_codec_calibration_identity_image_short_circuit() {
    // The identity-image short-circuit guard in `apply_mlp_scoring`
    // must fire BEFORE the MLP forward path (including per-codec
    // affine). When source == distorted, score = 100.0 regardless
    // of codec hint or per-codec metadata.
    let zen =
        Zensim::new(zensim_experimental::preview_v0_5_tuner_v4_calibrated()).with_parallel(false);
    let (src, _) = make_pair_with_delta(64, 64, 0);
    let src_slice = RgbSlice::new(&src, 64, 64);
    let result_no_hint = zen
        .compute_with_codec_hint(&src_slice, &src_slice, None)
        .unwrap();
    let result_jpeg = zen
        .compute_with_codec_hint(&src_slice, &src_slice, Some("jpeg"))
        .unwrap();
    assert_eq!(
        result_no_hint.score(),
        100.0,
        "identity image must score 100.0 (no hint)"
    );
    assert_eq!(
        result_jpeg.score(),
        100.0,
        "identity image must score 100.0 (jpeg hint)"
    );
}
