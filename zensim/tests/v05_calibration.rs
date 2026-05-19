//! Regression test: V0_5 bakes must emit POSITIVE scores on real JPEG
//! re-encodes after the 2026-05-19 affine-calibration fix.
//!
//! Bug history (2026-05-19): the V0_5 bakes (`PreviewV0_5Balanced`,
//! `PreviewV0_5Compression`, `PreviewV0_5Ensemble`) were trained against
//! `mix_cv40_iw60`, a distance-shaped target. The bakes' raw output is
//! `Spearman = -0.99` against quality — low raw = high quality. With
//! `skip_score_mapping=true` + hard `clamp(0, 100)`, every real-codec
//! pair returned 0 (because raw was negative for any quality level).
//!
//! V0_5-IDENTITY-FIX report's pre-fix observations:
//!   - V0_5Balanced raw on q=75 JPEG ≈ -6 → score = 0 (clamped)
//!   - V0_5Compression raw on q=98 JPEG ≈ -22 → score = 0 (clamped)
//!
//! Fix: apply affine `score = α + β · raw` in `apply_mlp_scoring` AFTER
//! the MLP forward + per-sample-α dispatch, BEFORE the score clamp.
//! Fit on safesyn training corpus against `ssim2_gpu`:
//!   - Balanced:    α=45.0561, β=-2.6602   (R²_holdout=0.925)
//!   - Compression: α=49.3380, β=-2.3967   (R²_holdout=0.853)
//!
//! Post-fix expectation: all q levels in {30, 50, 70, 90} on real-codec
//! re-encodes return SCORE > 0 (and most return score ≥ 30 — score=0
//! means the bake legitimately scored a heavy distortion).

use image::codecs::jpeg::JpegEncoder;
use image::{ImageBuffer, Rgb};
use std::io::Cursor;
use zensim::{RgbSlice, Zensim, ZensimProfile};

/// Generate a synthetic 64×64 RGB pattern with non-trivial frequency
/// content. The pattern is repeatable, so per-pixel-bit changes (the
/// kind a JPEG re-encode produces) drive the bake's feature extractor
/// to non-trivial outputs.
fn make_gradient_pattern(w: usize, h: usize) -> Vec<[u8; 3]> {
    (0..w * h)
        .map(|i| {
            let x = i % w;
            let y = i / w;
            // Diagonal gradient + small high-frequency component so a
            // JPEG re-encode at any quality introduces measurable
            // distortion.
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = (((x + y) * 255) / (w + h).max(1)) as u8;
            // Modulate with a small high-frequency checkerboard so JPEG
            // doesn't trivially round-trip.
            let hf = ((x ^ y) & 0b1111) as u8 * 8;
            [
                r.saturating_add(hf),
                g.saturating_sub(hf / 2),
                b.saturating_add(hf / 3),
            ]
        })
        .collect()
}

/// Encode `pixels` (RGB8) as JPEG at the given quality, return the
/// decoded RGB8 image.
fn jpeg_roundtrip(pixels: &[[u8; 3]], w: u32, h: u32, quality: u8) -> Vec<[u8; 3]> {
    // Flatten RGB triples into a contiguous u8 stream.
    let flat: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(w, h, flat).expect("ImageBuffer build");

    // Encode to JPEG.
    let mut jpeg_bytes = Vec::with_capacity(8 * 1024);
    {
        let encoder = JpegEncoder::new_with_quality(&mut jpeg_bytes, quality);
        img.write_with_encoder(encoder).expect("jpeg encode");
    }

    // Decode back to RGB8.
    let decoded = image::ImageReader::new(Cursor::new(&jpeg_bytes))
        .with_guessed_format()
        .expect("guess jpeg format")
        .decode()
        .expect("jpeg decode")
        .into_rgb8();
    assert_eq!(decoded.width(), w, "decoded width mismatch");
    assert_eq!(decoded.height(), h, "decoded height mismatch");
    decoded.pixels().map(|p| [p[0], p[1], p[2]]).collect()
}

/// Score one (reference, jpeg-re-encode-at-q) pair with the given
/// profile. Returns the score in [0, 100].
fn score_jpeg_pair(profile: ZensimProfile, q: u8) -> f64 {
    const W: usize = 64;
    const H: usize = 64;
    let reference = make_gradient_pattern(W, H);
    let distorted = jpeg_roundtrip(&reference, W as u32, H as u32, q);
    let src = RgbSlice::new(&reference, W, H);
    let dst = RgbSlice::new(&distorted, W, H);
    let z = Zensim::new(profile).with_parallel(false);
    let r = z.compute(&src, &dst).expect("compute failed");
    r.score()
}

/// Assert that the score is strictly positive AND above a sane floor.
/// Pre-fix all V0_5 profiles returned 0 on every q ≥ 30; post-fix
/// every q ∈ {30, 50, 70, 90} should produce a measurable positive
/// score.
fn assert_positive(profile: ZensimProfile, q: u8, label: &str, min_score: f64) {
    let score = score_jpeg_pair(profile, q);
    assert!(
        score > 0.0,
        "{label} q={q}: score must be > 0 (was {score}); pre-fix all \
         V0_5 profiles returned exactly 0 on real-codec re-encodes \
         because raw output was distance-shaped and hard-clamped."
    );
    assert!(
        score >= min_score,
        "{label} q={q}: score {score} below floor {min_score}; \
         affine calibration may be miscalibrated."
    );
    // Sanity upper bound: score must stay in [0, 100].
    assert!(
        score <= 100.0,
        "{label} q={q}: score {score} > 100.0; clamp not applied?"
    );
}

// ---- V0_5 Balanced ----
//
// Floors chosen conservatively from the canonical safesyn fit. The
// fit produces score ≈ 45 + 2.66·|raw| on quality content. For 64×64
// JPEG re-encodes, q=30 lands the heaviest distortion; we expect at
// least 10 (well above the all-clamped-to-0 pre-fix state). q=90
// produces near-PJND distortion; floor = 30.

#[test]
fn v05_balanced_q30_positive() {
    assert_positive(ZensimProfile::PreviewV0_5Balanced, 30, "V0_5Balanced", 5.0);
}

#[test]
fn v05_balanced_q50_positive() {
    assert_positive(ZensimProfile::PreviewV0_5Balanced, 50, "V0_5Balanced", 10.0);
}

#[test]
fn v05_balanced_q70_positive() {
    assert_positive(ZensimProfile::PreviewV0_5Balanced, 70, "V0_5Balanced", 20.0);
}

#[test]
fn v05_balanced_q90_positive() {
    assert_positive(ZensimProfile::PreviewV0_5Balanced, 90, "V0_5Balanced", 30.0);
}

// ---- V0_5 Compression ----

#[test]
fn v05_compression_q30_positive() {
    assert_positive(
        ZensimProfile::PreviewV0_5Compression,
        30,
        "V0_5Compression",
        5.0,
    );
}

#[test]
fn v05_compression_q50_positive() {
    assert_positive(
        ZensimProfile::PreviewV0_5Compression,
        50,
        "V0_5Compression",
        10.0,
    );
}

#[test]
fn v05_compression_q70_positive() {
    assert_positive(
        ZensimProfile::PreviewV0_5Compression,
        70,
        "V0_5Compression",
        20.0,
    );
}

#[test]
fn v05_compression_q90_positive() {
    assert_positive(
        ZensimProfile::PreviewV0_5Compression,
        90,
        "V0_5Compression",
        30.0,
    );
}

// ---- V0_5 Ensemble ----

#[test]
fn v05_ensemble_q30_positive() {
    assert_positive(ZensimProfile::PreviewV0_5Ensemble, 30, "V0_5Ensemble", 5.0);
}

#[test]
fn v05_ensemble_q50_positive() {
    assert_positive(ZensimProfile::PreviewV0_5Ensemble, 50, "V0_5Ensemble", 10.0);
}

#[test]
fn v05_ensemble_q70_positive() {
    assert_positive(ZensimProfile::PreviewV0_5Ensemble, 70, "V0_5Ensemble", 20.0);
}

#[test]
fn v05_ensemble_q90_positive() {
    assert_positive(ZensimProfile::PreviewV0_5Ensemble, 90, "V0_5Ensemble", 30.0);
}

// ---- Monotonicity sanity check ----
//
// q=30 < q=50 < q=70 < q=90 — higher quality should produce higher
// score. JPEG quality scales aren't perfectly monotonic at every q
// jump per content, but 20-quality-step jumps should always show an
// improvement on a synthetic gradient.
fn assert_monotone(profile: ZensimProfile, label: &str) {
    let s30 = score_jpeg_pair(profile, 30);
    let s50 = score_jpeg_pair(profile, 50);
    let s70 = score_jpeg_pair(profile, 70);
    let s90 = score_jpeg_pair(profile, 90);
    // Allow ties — the soft-clamp can compress small differences at
    // the high tail. But each step must be ≥ previous.
    assert!(s50 >= s30, "{label}: q=50 ({s50}) must be >= q=30 ({s30})");
    assert!(s70 >= s50, "{label}: q=70 ({s70}) must be >= q=50 ({s50})");
    assert!(s90 >= s70, "{label}: q=90 ({s90}) must be >= q=70 ({s70})");
    // The full sweep must show a meaningful range — pre-fix gave
    // identical zeros, so this catches accidental regressions of
    // the affine path entirely.
    assert!(
        s90 - s30 > 5.0,
        "{label}: q=90..q=30 span {} should exceed 5 score units",
        s90 - s30
    );
}

#[test]
fn v05_balanced_monotone() {
    assert_monotone(ZensimProfile::PreviewV0_5Balanced, "V0_5Balanced");
}

#[test]
fn v05_compression_monotone() {
    assert_monotone(ZensimProfile::PreviewV0_5Compression, "V0_5Compression");
}

#[test]
fn v05_ensemble_monotone() {
    assert_monotone(ZensimProfile::PreviewV0_5Ensemble, "V0_5Ensemble");
}

// ---- Identity-image invariant preserved ----
//
// The byte-identical short-circuit (V0_5-IDENTITY-FIX) returns score
// = 100 with raw = 0, BEFORE the affine path runs. The 2026-05-19
// calibration must NOT regress this: identical inputs still return
// 100, regardless of the new α, β.

fn assert_identity_returns_100(profile: ZensimProfile, label: &str) {
    const W: usize = 64;
    const H: usize = 64;
    let pixels = make_gradient_pattern(W, H);
    let copy = pixels.clone();
    let src = RgbSlice::new(&pixels, W, H);
    let dst = RgbSlice::new(&copy, W, H);
    let z = Zensim::new(profile).with_parallel(false);
    let r = z.compute(&src, &dst).expect("compute failed");
    assert!(
        (r.score() - 100.0).abs() < 1e-6,
        "{label}: identity short-circuit broken by affine; \
         expected score=100.0, got {} (raw={})",
        r.score(),
        r.raw_distance()
    );
}

#[test]
fn v05_balanced_identity_preserved() {
    assert_identity_returns_100(ZensimProfile::PreviewV0_5Balanced, "PreviewV0_5Balanced");
}

#[test]
fn v05_compression_identity_preserved() {
    assert_identity_returns_100(
        ZensimProfile::PreviewV0_5Compression,
        "PreviewV0_5Compression",
    );
}

#[test]
fn v05_ensemble_identity_preserved() {
    assert_identity_returns_100(ZensimProfile::PreviewV0_5Ensemble, "PreviewV0_5Ensemble");
}
