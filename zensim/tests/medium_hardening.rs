//! Regression tests for the M1-M4 hardening pass (audit 2026-05-06).
//!
//! Each test maps to one audit finding:
//! - M1: cross-image dim mismatch in `compute_with_ref*` returns Err.
//! - M2: 32-bit `width * height` / stride overflow rejected at construction.
//! - M3: `Zensim::with_max_pixels` cap fires before allocation.
//! - M4: `try_score_from_features` returns Err on length mismatch.

use zensim::{PixelFormat, RgbSlice, RgbaSlice, StridedBytes, Zensim, ZensimError, ZensimProfile};

fn z() -> Zensim {
    Zensim::new(ZensimProfile::latest())
}

// ─── M1: cross-image dim mismatch in compute_with_ref* ─────────────────────

#[test]
fn m1_compute_with_ref_dim_mismatch_returns_err() {
    let big_pixels = vec![[100u8, 110, 120]; 32 * 16];
    let small_pixels = vec![[100u8, 110, 120]; 16 * 8];
    let big = RgbSlice::new(&big_pixels, 32, 16);
    let small = RgbSlice::new(&small_pixels, 16, 8);

    let zensim = z();
    let pre = zensim.precompute_reference(&big).unwrap();
    assert_eq!(pre.width(), 32);
    assert_eq!(pre.height(), 16);

    let err = zensim.compute_with_ref(&pre, &small).unwrap_err();
    assert_eq!(err, ZensimError::DimensionMismatch);
}

#[test]
fn m1_compute_with_ref_into_dim_mismatch_returns_err() {
    let big_pixels = vec![[200u8, 50, 0]; 24 * 24];
    let small_pixels = vec![[200u8, 50, 0]; 16 * 16];
    let big = RgbSlice::new(&big_pixels, 24, 24);
    let small = RgbSlice::new(&small_pixels, 16, 16);

    let zensim = z();
    let pre = zensim.precompute_reference(&big).unwrap();

    let mut scratch = zensim::ZensimScratch::new();
    let err = zensim
        .compute_with_ref_into(&pre, &small, &mut scratch)
        .unwrap_err();
    assert_eq!(err, ZensimError::DimensionMismatch);
}

#[test]
fn m1_compute_with_ref_matching_dims_succeeds() {
    let pixels = vec![[100u8, 100, 100]; 16 * 16];
    let img = RgbSlice::new(&pixels, 16, 16);
    let zensim = z();
    let pre = zensim.precompute_reference(&img).unwrap();
    let result = zensim.compute_with_ref(&pre, &img).unwrap();
    // Identical input → score must round to ~100.
    assert!(result.score() > 99.0, "score {}", result.score());
}

#[test]
fn m1_diffmap_with_ref_dim_mismatch_returns_err() {
    use zensim::DiffmapOptions;
    let big_pixels = vec![[10u8, 20, 30]; 32 * 16];
    let small_pixels = vec![[10u8, 20, 30]; 16 * 8];
    let big = RgbSlice::new(&big_pixels, 32, 16);
    let small = RgbSlice::new(&small_pixels, 16, 8);

    let zensim = z();
    let pre = zensim.precompute_reference(&big).unwrap();

    match zensim.compute_with_ref_and_diffmap(&pre, &small, DiffmapOptions::default()) {
        Ok(_) => panic!("expected DimensionMismatch"),
        Err(e) => assert_eq!(e, ZensimError::DimensionMismatch),
    }
}

// ─── M2: integer overflow rejection ────────────────────────────────────────

#[test]
fn m2_rgb_slice_overflow_rejected() {
    // Pick dimensions whose product overflows usize on 32-bit AND 64-bit.
    // On 64-bit, usize::MAX / 2 fits but usize::MAX × 2 doesn't.
    let huge = usize::MAX / 4 + 1;
    let pixels: Vec<[u8; 3]> = vec![[0; 3]; 16];
    let err = RgbSlice::try_new(&pixels, huge, huge).unwrap_err();
    // Either overflow (ImageTooLarge) or length mismatch — both are valid;
    // the important thing is we don't panic.
    assert!(
        matches!(
            err,
            ZensimError::ImageTooLarge | ZensimError::InvalidDataLength
        ),
        "unexpected {:?}",
        err
    );

    // Also try a pair that wraps to small on 32-bit only. On 64-bit this
    // doesn't overflow, but the data-length check still rejects it.
    let w_32 = 1usize << 30;
    let h_32 = 8usize;
    let err = RgbSlice::try_new(&pixels, w_32, h_32).unwrap_err();
    // On 64-bit width * height = 8 GB which doesn't overflow, so we should
    // get InvalidDataLength. On 32-bit we should get ImageTooLarge.
    assert!(matches!(
        err,
        ZensimError::ImageTooLarge | ZensimError::InvalidDataLength
    ));
}

#[test]
fn m2_rgba_slice_overflow_rejected() {
    let huge = usize::MAX / 4 + 1;
    let pixels: Vec<[u8; 4]> = vec![[0; 4]; 16];
    let err = RgbaSlice::try_new(&pixels, huge, huge).unwrap_err();
    assert!(matches!(
        err,
        ZensimError::ImageTooLarge | ZensimError::InvalidDataLength
    ));
}

#[test]
fn m2_strided_bytes_min_stride_overflow_rejected() {
    // width × bpp overflow path: width close to usize::MAX with bpp=16
    // (LinearF32Rgba). Should produce ImageTooLarge before any stride or
    // length check fires.
    let bytes = vec![0u8; 256];
    let huge_w = usize::MAX / 8 + 1; // width × 16 wraps
    let err = StridedBytes::try_new(&bytes, huge_w, 1, 16, PixelFormat::LinearF32Rgba).unwrap_err();
    assert_eq!(err, ZensimError::ImageTooLarge);
}

#[test]
fn m2_strided_bytes_required_overflow_rejected() {
    // (height-1) * stride + min_stride overflow path
    let bytes = vec![0u8; 64];
    // Pick width=4, bpp=4 (Srgb8Rgba) → min_stride=16 fits. Then height
    // huge enough that (height-1) * stride wraps.
    let stride = 16usize;
    let huge_h = usize::MAX / 8 + 1;
    let err = StridedBytes::try_new(&bytes, 4, huge_h, stride, PixelFormat::Srgb8Rgba).unwrap_err();
    assert_eq!(err, ZensimError::ImageTooLarge);
}

#[test]
fn m2_strided_bytes_normal_dims_still_work() {
    let bytes = vec![0u8; 16 * 4 * 8];
    let _ = StridedBytes::try_new(&bytes, 16, 8, 16 * 4, PixelFormat::Srgb8Rgba)
        .expect("normal dims must still work");
}

// ─── M3: max_pixels cap fires ──────────────────────────────────────────────

#[test]
fn m3_max_pixels_cap_fires_on_compute() {
    let pixels = vec![[120u8, 130, 140]; 32 * 32];
    let img = RgbSlice::new(&pixels, 32, 32);

    // Cap at 100 pixels — image is 1024 pixels.
    let zensim = z().with_max_pixels(100);
    let err = zensim.compute(&img, &img).unwrap_err();
    assert_eq!(err, ZensimError::ImageTooLarge);
    assert_eq!(zensim.max_pixels(), Some(100));
}

#[test]
fn m3_max_pixels_cap_fires_on_precompute() {
    let pixels = vec![[100u8, 100, 100]; 32 * 32];
    let img = RgbSlice::new(&pixels, 32, 32);

    let zensim = z().with_max_pixels(100);
    match zensim.precompute_reference(&img) {
        Ok(_) => panic!("expected ImageTooLarge"),
        Err(e) => assert_eq!(e, ZensimError::ImageTooLarge),
    }
}

#[test]
fn m3_max_pixels_cap_fires_on_compute_with_ref() {
    let pixels = vec![[100u8, 100, 100]; 32 * 32];
    let img = RgbSlice::new(&pixels, 32, 32);

    // Build the precomputed reference under a permissive Zensim, then
    // re-test with a stricter cap. The strict instance must reject before
    // running compute.
    let pre = z().precompute_reference(&img).unwrap();

    let strict = z().with_max_pixels(100);
    let err = strict.compute_with_ref(&pre, &img).unwrap_err();
    assert_eq!(err, ZensimError::ImageTooLarge);
}

#[test]
fn m3_no_cap_by_default() {
    let zensim = z();
    assert_eq!(zensim.max_pixels(), None);
    let pixels = vec![[100u8, 100, 100]; 32 * 32];
    let img = RgbSlice::new(&pixels, 32, 32);
    // Default Zensim with no cap should accept reasonable images.
    let _ = zensim.compute(&img, &img).expect("default Zensim accepts");
}

#[test]
fn m3_generous_cap_does_not_block() {
    let pixels = vec![[100u8, 100, 100]; 16 * 16];
    let img = RgbSlice::new(&pixels, 16, 16);
    let zensim = z().with_max_pixels(usize::MAX);
    let _ = zensim
        .compute(&img, &img)
        .expect("usize::MAX cap accepts everything");
}

// ─── M4: try_score_from_features length mismatch ───────────────────────────

#[cfg(feature = "training")]
#[test]
fn m4_try_score_from_features_length_mismatch_returns_err() {
    let features = vec![0.5_f64; 39];
    let weights = vec![0.5_f64; 13];
    let err = zensim::try_score_from_features(&features, &weights).unwrap_err();
    assert_eq!(err, ZensimError::FeatureWeightsLengthMismatch);
}

#[cfg(feature = "training")]
#[test]
fn m4_try_score_from_features_matched_lengths_ok() {
    let features = vec![0.0_f64; 39];
    let weights = vec![1.0_f64; 39];
    let (score, raw) = zensim::try_score_from_features(&features, &weights).unwrap();
    assert!(score.is_finite());
    assert!(raw.is_finite());
}
