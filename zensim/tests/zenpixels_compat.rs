#![cfg(feature = "zenpixels")]
//! End-to-end integration tests for the zenpixels ImageSource adapter.
//!
//! Validates that `ZenpixelsSource` produces identical scores to the
//! equivalent native zensim types (RgbSlice, RgbaSlice, StridedBytes).
//!
//! Run with: `cargo test -p zensim --features zenpixels --test zenpixels_compat`

use zenpixels::{AlphaMode as ZpAlpha, PixelBuffer, PixelDescriptor, PixelSlice, TransferFunction};
use zensim::{
    AlphaMode, ImageSource, RgbSlice, RgbaSlice, StridedBytes, ZenpixelsSource, Zensim,
    ZensimProfile,
};

fn zensim() -> Zensim {
    Zensim::new(ZensimProfile::latest())
}

/// Generate deterministic 16x16 test pixels (gradient pattern).
fn test_pixels_rgb() -> Vec<[u8; 3]> {
    (0..16 * 16)
        .map(|i| {
            let x = (i % 16) as u8;
            let y = (i / 16) as u8;
            [
                x.wrapping_mul(17),
                y.wrapping_mul(17),
                (x ^ y).wrapping_mul(11),
            ]
        })
        .collect()
}

fn test_pixels_rgba() -> Vec<[u8; 4]> {
    test_pixels_rgb()
        .into_iter()
        .map(|[r, g, b]| [r, g, b, 255])
        .collect()
}

fn test_pixels_rgba_with_alpha() -> Vec<[u8; 4]> {
    (0..16 * 16)
        .map(|i| {
            let x = (i % 16) as u8;
            let y = (i / 16) as u8;
            let a = if x < 8 { 255 } else { 128 };
            [
                x.wrapping_mul(17),
                y.wrapping_mul(17),
                (x ^ y).wrapping_mul(11),
                a,
            ]
        })
        .collect()
}

fn distorted_pixels_rgb() -> Vec<[u8; 3]> {
    test_pixels_rgb()
        .into_iter()
        .map(|[r, g, b]| [r.wrapping_add(5), g, b.wrapping_sub(3)])
        .collect()
}

fn distorted_pixels_rgba() -> Vec<[u8; 4]> {
    distorted_pixels_rgb()
        .into_iter()
        .map(|[r, g, b]| [r, g, b, 255])
        .collect()
}

#[test]
fn rgb8_matches_rgbslice() {
    let z = zensim();
    let src = test_pixels_rgb();
    let dst = distorted_pixels_rgb();

    // Native
    let native = z
        .compute(&RgbSlice::new(&src, 16, 16), &RgbSlice::new(&dst, 16, 16))
        .unwrap();

    // Via zenpixels
    let src_bytes: Vec<u8> = src.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_bytes: Vec<u8> = dst.iter().flat_map(|p| p.iter().copied()).collect();
    let src_slice =
        PixelSlice::new(&src_bytes, 16, 16, 16 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let dst_slice =
        PixelSlice::new(&dst_bytes, 16, 16, 16 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let zp_src = ZenpixelsSource::try_from_slice(&src_slice).unwrap();
    let zp_dst = ZenpixelsSource::try_from_slice(&dst_slice).unwrap();
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    assert!(
        (native.score() - zp_result.score()).abs() < 0.001,
        "RGB8 score mismatch: native={}, zenpixels={}",
        native.score(),
        zp_result.score()
    );
}

#[test]
fn rgba8_matches_rgbaslice() {
    let z = zensim();
    let src = test_pixels_rgba();
    let dst = distorted_pixels_rgba();

    let native = z
        .compute(&RgbaSlice::new(&src, 16, 16), &RgbaSlice::new(&dst, 16, 16))
        .unwrap();

    let src_bytes: Vec<u8> = src.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_bytes: Vec<u8> = dst.iter().flat_map(|p| p.iter().copied()).collect();
    let src_slice =
        PixelSlice::new(&src_bytes, 16, 16, 16 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let dst_slice =
        PixelSlice::new(&dst_bytes, 16, 16, 16 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let zp_src = ZenpixelsSource::try_from_slice(&src_slice).unwrap();
    let zp_dst = ZenpixelsSource::try_from_slice(&dst_slice).unwrap();
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    assert!(
        (native.score() - zp_result.score()).abs() < 0.001,
        "RGBA8 score mismatch: native={}, zenpixels={}",
        native.score(),
        zp_result.score()
    );
}

#[test]
fn rgbx8_treated_as_opaque() {
    let z = zensim();
    let src = test_pixels_rgba();
    let dst = distorted_pixels_rgba();

    // Native with explicit Opaque
    let native = z
        .compute(
            &RgbaSlice::with_alpha_mode(&src, 16, 16, AlphaMode::Opaque),
            &RgbaSlice::with_alpha_mode(&dst, 16, 16, AlphaMode::Opaque),
        )
        .unwrap();

    // RGBX8 — padding byte should be ignored
    let src_bytes: Vec<u8> = src.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_bytes: Vec<u8> = dst.iter().flat_map(|p| p.iter().copied()).collect();
    let src_slice =
        PixelSlice::new(&src_bytes, 16, 16, 16 * 4, PixelDescriptor::RGBX8_SRGB).unwrap();
    let dst_slice =
        PixelSlice::new(&dst_bytes, 16, 16, 16 * 4, PixelDescriptor::RGBX8_SRGB).unwrap();
    let zp_src = ZenpixelsSource::try_from_slice(&src_slice).unwrap();
    let zp_dst = ZenpixelsSource::try_from_slice(&dst_slice).unwrap();

    assert_eq!(zp_src.alpha_mode(), AlphaMode::Opaque);
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    assert!(
        (native.score() - zp_result.score()).abs() < 0.001,
        "RGBX8 score mismatch: native={}, zenpixels={}",
        native.score(),
        zp_result.score()
    );
}

#[test]
fn bgra8_matches_stridedbytes() {
    let z = zensim();
    // BGRA layout: [B, G, R, A]
    let src_bgra: Vec<u8> = test_pixels_rgba()
        .iter()
        .flat_map(|[r, g, b, a]| [*b, *g, *r, *a])
        .collect();
    let dst_bgra: Vec<u8> = distorted_pixels_rgba()
        .iter()
        .flat_map(|[r, g, b, a]| [*b, *g, *r, *a])
        .collect();

    let native_src = StridedBytes::with_alpha_mode(
        &src_bgra,
        16,
        16,
        16 * 4,
        zensim::PixelFormat::Srgb8Bgra,
        AlphaMode::Straight,
    );
    let native_dst = StridedBytes::with_alpha_mode(
        &dst_bgra,
        16,
        16,
        16 * 4,
        zensim::PixelFormat::Srgb8Bgra,
        AlphaMode::Straight,
    );
    let native = z.compute(&native_src, &native_dst).unwrap();

    let src_slice =
        PixelSlice::new(&src_bgra, 16, 16, 16 * 4, PixelDescriptor::BGRA8_SRGB).unwrap();
    let dst_slice =
        PixelSlice::new(&dst_bgra, 16, 16, 16 * 4, PixelDescriptor::BGRA8_SRGB).unwrap();
    let zp_src = ZenpixelsSource::try_from_slice(&src_slice).unwrap();
    let zp_dst = ZenpixelsSource::try_from_slice(&dst_slice).unwrap();
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    assert!(
        (native.score() - zp_result.score()).abs() < 0.001,
        "BGRA8 score mismatch: native={}, zenpixels={}",
        native.score(),
        zp_result.score()
    );
}

#[test]
fn identical_images_score_100() {
    let z = zensim();
    let px = test_pixels_rgba();
    let bytes: Vec<u8> = px.iter().flat_map(|p| p.iter().copied()).collect();
    let slice = PixelSlice::new(&bytes, 16, 16, 16 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let src = ZenpixelsSource::try_from_slice(&slice).unwrap();

    let slice2 = PixelSlice::new(&bytes, 16, 16, 16 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let dst = ZenpixelsSource::try_from_slice(&slice2).unwrap();

    let result = z.compute(&src, &dst).unwrap();
    assert_eq!(result.score(), 100.0);
}

#[test]
fn different_images_score_below_100() {
    let z = zensim();
    let src_bytes: Vec<u8> = test_pixels_rgba()
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect();
    let dst_bytes: Vec<u8> = distorted_pixels_rgba()
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect();
    let s = PixelSlice::new(&src_bytes, 16, 16, 16 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let d = PixelSlice::new(&dst_bytes, 16, 16, 16 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let src = ZenpixelsSource::try_from_slice(&s).unwrap();
    let dst = ZenpixelsSource::try_from_slice(&d).unwrap();
    let result = z.compute(&src, &dst).unwrap();
    assert!(
        result.score() < 100.0,
        "different images should score < 100"
    );
}

#[test]
fn premultiplied_matches_straight() {
    let z = zensim();
    let straight = test_pixels_rgba_with_alpha();

    // Create premultiplied version
    let premul: Vec<u8> = straight
        .iter()
        .flat_map(|[r, g, b, a]| {
            let af = *a as f32 / 255.0;
            [
                (*r as f32 * af) as u8,
                (*g as f32 * af) as u8,
                (*b as f32 * af) as u8,
                *a,
            ]
        })
        .collect();

    let straight_bytes: Vec<u8> = straight.iter().flat_map(|p| p.iter().copied()).collect();

    // Premultiplied via zenpixels adapter (should un-premultiply internally)
    let premul_desc = PixelDescriptor::RGBA8_SRGB.with_alpha(Some(ZpAlpha::Premultiplied));
    let p_slice = PixelSlice::new(&premul, 16, 16, 16 * 4, premul_desc).unwrap();
    let zp_src = ZenpixelsSource::try_from_slice(&p_slice).unwrap();

    // Compare premul source against straight source
    let s_slice =
        PixelSlice::new(&straight_bytes, 16, 16, 16 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let zp_dst = ZenpixelsSource::try_from_slice(&s_slice).unwrap();
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    // Un-premultiply has rounding, so allow some tolerance
    assert!(
        zp_result.score() > 95.0,
        "premul un-premultiply should produce near-identical result, got {}",
        zp_result.score()
    );
}

#[test]
fn hdr_transfer_rejected() {
    let desc = PixelDescriptor::RGBA8_SRGB.with_transfer(TransferFunction::Pq);
    let data = vec![0u8; 16 * 16 * 4];
    let slice = PixelSlice::new(&data, 16, 16, 16 * 4, desc).unwrap();
    let result = ZenpixelsSource::try_from_slice(&slice);
    assert!(result.is_err());
}

#[test]
fn grayscale_rejected() {
    let data = vec![128u8; 16 * 16];
    let slice = PixelSlice::new(&data, 16, 16, 16, PixelDescriptor::GRAY8_SRGB).unwrap();
    let result = ZenpixelsSource::try_from_slice(&slice);
    assert!(result.is_err());
}

#[test]
fn pixel_buffer_end_to_end() {
    let z = zensim();
    let px: Vec<u8> = test_pixels_rgba()
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect();
    let buf = PixelBuffer::from_vec(px.clone(), 16, 16, PixelDescriptor::RGBA8_SRGB).unwrap();
    let src = ZenpixelsSource::try_from_buffer(&buf).unwrap();

    let buf2 = PixelBuffer::from_vec(px, 16, 16, PixelDescriptor::RGBA8_SRGB).unwrap();
    let dst = ZenpixelsSource::try_from_buffer(&buf2).unwrap();

    let result = z.compute(&src, &dst).unwrap();
    assert_eq!(result.score(), 100.0);
}
