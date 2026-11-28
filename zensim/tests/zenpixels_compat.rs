#![cfg(feature = "zenpixels")]
#![allow(deprecated)]
// exercises the adopted `ZensimProfile::A`
//! End-to-end integration tests for the zenpixels 0.5 ImageSource adapter.
//!
//! Validates that `ZenpixelsSource` produces identical scores to the
//! equivalent native zensim types (`RgbSlice`, `RgbaSlice`, `StridedBytes`).
//!
//! Run with: `cargo test -p zensim --features zenpixels --test zenpixels_compat`

use zenpixels::{
    AlphaMode as ZpAlpha, ColorSpace, PixelBuf, PixelFormat as ZpPixelFormat, PixelView,
    Primaries as ZpPrimaries, SampleLayout, SignalRange, Transfer, formats,
};
use zensim::{
    AlphaMode, ImageSource, RgbSlice, RgbaSlice, StridedBytes, ZenpixelsSource, Zensim,
    ZensimProfile,
};

fn zensim() -> Zensim {
    Zensim::new(ZensimProfile::A)
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

    let native = z
        .compute(&RgbSlice::new(&src, 16, 16), &RgbSlice::new(&dst, 16, 16))
        .unwrap();

    let src_bytes: Vec<u8> = src.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_bytes: Vec<u8> = dst.iter().flat_map(|p| p.iter().copied()).collect();
    let src_view = PixelView::new(&src_bytes, 16, 16, 16 * 3, &formats::RGB8_SRGB).unwrap();
    let dst_view = PixelView::new(&dst_bytes, 16, 16, 16 * 3, &formats::RGB8_SRGB).unwrap();
    let zp_src = ZenpixelsSource::try_from_view(&src_view).unwrap();
    let zp_dst = ZenpixelsSource::try_from_view(&dst_view).unwrap();
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    assert_eq!(
        native.score().to_bits(),
        zp_result.score().to_bits(),
        "RGB8 score mismatch: native={}, zenpixels={}",
        native.score(),
        zp_result.score()
    );
}

#[test]
fn strided_rgb8_matches_packed_bit_exact() {
    let z = zensim();
    let src = test_pixels_rgb();
    let dst = distorted_pixels_rgb();

    let native = z
        .compute(&RgbSlice::new(&src, 16, 16), &RgbSlice::new(&dst, 16, 16))
        .unwrap();

    // Add 13 poisoned padding bytes per row.
    let stride = 16 * 3 + 13;
    let pack_strided = |px: &[[u8; 3]]| {
        let mut out = vec![0xAAu8; stride * 16];
        for y in 0..16 {
            for x in 0..16 {
                out[y * stride + x * 3..y * stride + x * 3 + 3].copy_from_slice(&px[y * 16 + x]);
            }
        }
        out
    };
    let src_bytes = pack_strided(&src);
    let dst_bytes = pack_strided(&dst);
    let src_view = PixelView::new(&src_bytes, 16, 16, stride, &formats::RGB8_SRGB).unwrap();
    let dst_view = PixelView::new(&dst_bytes, 16, 16, stride, &formats::RGB8_SRGB).unwrap();
    let zp_src = ZenpixelsSource::try_from_view(&src_view).unwrap();
    let zp_dst = ZenpixelsSource::try_from_view(&dst_view).unwrap();
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    assert_eq!(native.score().to_bits(), zp_result.score().to_bits());
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
    let src_view = PixelView::new(&src_bytes, 16, 16, 16 * 4, &formats::RGBA8_SRGB).unwrap();
    let dst_view = PixelView::new(&dst_bytes, 16, 16, 16 * 4, &formats::RGBA8_SRGB).unwrap();
    let zp_src = ZenpixelsSource::try_from_view(&src_view).unwrap();
    let zp_dst = ZenpixelsSource::try_from_view(&dst_view).unwrap();
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    assert_eq!(native.score().to_bits(), zp_result.score().to_bits());
}

#[test]
fn rgbx8_treated_as_opaque() {
    let z = zensim();
    let src = test_pixels_rgba();
    let dst = distorted_pixels_rgba();

    let native = z
        .compute(
            &RgbaSlice::with_alpha_mode(&src, 16, 16, AlphaMode::Opaque),
            &RgbaSlice::with_alpha_mode(&dst, 16, 16, AlphaMode::Opaque),
        )
        .unwrap();

    let src_bytes: Vec<u8> = src.iter().flat_map(|p| p.iter().copied()).collect();
    let dst_bytes: Vec<u8> = dst.iter().flat_map(|p| p.iter().copied()).collect();
    let src_view = PixelView::new(&src_bytes, 16, 16, 16 * 4, &formats::RGBX8_SRGB).unwrap();
    let dst_view = PixelView::new(&dst_bytes, 16, 16, 16 * 4, &formats::RGBX8_SRGB).unwrap();
    let zp_src = ZenpixelsSource::try_from_view(&src_view).unwrap();
    let zp_dst = ZenpixelsSource::try_from_view(&dst_view).unwrap();

    assert_eq!(zp_src.alpha_mode(), AlphaMode::Opaque);
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    assert_eq!(native.score().to_bits(), zp_result.score().to_bits());
}

#[test]
fn bgra8_matches_stridedbytes() {
    let z = zensim();
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

    let src_view = PixelView::new(&src_bgra, 16, 16, 16 * 4, &formats::BGRA8_SRGB).unwrap();
    let dst_view = PixelView::new(&dst_bgra, 16, 16, 16 * 4, &formats::BGRA8_SRGB).unwrap();
    let zp_src = ZenpixelsSource::try_from_view(&src_view).unwrap();
    let zp_dst = ZenpixelsSource::try_from_view(&dst_view).unwrap();
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();

    assert_eq!(native.score().to_bits(), zp_result.score().to_bits());
}

#[test]
fn identical_images_score_100() {
    let z = zensim();
    let px = test_pixels_rgba();
    let bytes: Vec<u8> = px.iter().flat_map(|p| p.iter().copied()).collect();
    let view1 = PixelView::new(&bytes, 16, 16, 16 * 4, &formats::RGBA8_SRGB).unwrap();
    let src = ZenpixelsSource::try_from_view(&view1).unwrap();

    let view2 = PixelView::new(&bytes, 16, 16, 16 * 4, &formats::RGBA8_SRGB).unwrap();
    let dst = ZenpixelsSource::try_from_view(&view2).unwrap();

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
    let s = PixelView::new(&src_bytes, 16, 16, 16 * 4, &formats::RGBA8_SRGB).unwrap();
    let d = PixelView::new(&dst_bytes, 16, 16, 16 * 4, &formats::RGBA8_SRGB).unwrap();
    let src = ZenpixelsSource::try_from_view(&s).unwrap();
    let dst = ZenpixelsSource::try_from_view(&d).unwrap();
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

    let premul_fmt = ZpPixelFormat::new(
        SampleLayout::RGBA8,
        ColorSpace::Rgb {
            primaries: ZpPrimaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: ZpAlpha::Premultiplied,
            white: None,
        },
    );
    let p_view = PixelView::new(&premul, 16, 16, 16 * 4, &premul_fmt).unwrap();
    let zp_src = ZenpixelsSource::try_from_view(&p_view).unwrap();

    // (1) Pixel-exact adapter contract: recovery must match round-to-nearest un-premultiply byte-for-byte.
    let reference: Vec<u8> = premul
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|px| {
            let a = px[3];
            if a == 0 {
                [0, 0, 0, a]
            } else if a < 255 {
                let inv = 255.0 / a as f32;
                [
                    (px[0] as f32 * inv).round().min(255.0) as u8,
                    (px[1] as f32 * inv).round().min(255.0) as u8,
                    (px[2] as f32 * inv).round().min(255.0) as u8,
                    a,
                ]
            } else {
                [px[0], px[1], px[2], a]
            }
        })
        .collect();
    let r_view = PixelView::new(&reference, 16, 16, 16 * 4, &formats::RGBA8_SRGB).unwrap();
    let r_src = ZenpixelsSource::try_from_view(&r_view).unwrap();
    let exact = z.compute(&zp_src, &r_src).unwrap();
    assert_eq!(
        exact.score(),
        100.0,
        "adapter un-premultiply must match the round-to-nearest reference exactly"
    );

    // (2) Catastrophic canary vs the straight original.
    let s_view = PixelView::new(&straight_bytes, 16, 16, 16 * 4, &formats::RGBA8_SRGB).unwrap();
    let zp_dst = ZenpixelsSource::try_from_view(&s_view).unwrap();
    let zp_result = z.compute(&zp_src, &zp_dst).unwrap();
    assert!(
        zp_result.score() > 90.0,
        "premul un-premultiply should produce near-identical result, got {}",
        zp_result.score()
    );
}

#[test]
fn hdr_transfer_rejected() {
    let fmt = ZpPixelFormat::new(
        SampleLayout::RGBA16,
        ColorSpace::Rgb {
            primaries: ZpPrimaries::Bt2020,
            transfer: Transfer::Pq,
            range: SignalRange::Full,
            alpha: ZpAlpha::Opaque,
            white: None,
        },
    );
    let data = vec![0u8; 16 * 16 * 8];
    let view = PixelView::new(&data, 16, 16, 16 * 8, &fmt).unwrap();
    assert!(ZenpixelsSource::try_from_view(&view).is_err());
}

#[test]
fn grayscale_rejected() {
    let data = vec![128u8; 16 * 16];
    let view = PixelView::new(&data, 16, 16, 16, &formats::GRAY8_SRGB).unwrap();
    assert!(ZenpixelsSource::try_from_view(&view).is_err());
}

#[test]
fn f32_nan_and_out_of_range_rejected() {
    let mut px = vec![[0.5f32, 0.25, 0.75, 1.0]; 16 * 16];
    px[10][1] = f32::NAN;
    let view = PixelView::from_typed(&px, 16, 16, &formats::RGBAF32_LINEAR).unwrap();
    assert!(ZenpixelsSource::try_from_view(&view).is_err());

    px[10][1] = 1.5;
    let view2 = PixelView::from_typed(&px, 16, 16, &formats::RGBAF32_LINEAR).unwrap();
    assert!(ZenpixelsSource::try_from_view(&view2).is_err());

    // Opaque alpha lane is logically ignored even if out of [0, 1].
    let opaque_fmt = ZpPixelFormat::new(
        SampleLayout::RGBAF32,
        ColorSpace::Rgb {
            primaries: ZpPrimaries::Bt709,
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: ZpAlpha::Opaque,
            white: None,
        },
    );
    let mut px_opaque = vec![[0.5f32, 0.25, 0.75, f32::NAN]; 16 * 16];
    let view3 = PixelView::from_typed(&px_opaque, 16, 16, &opaque_fmt).unwrap();
    assert!(ZenpixelsSource::try_from_view(&view3).is_ok());

    // RGB still checked even when alpha is Opaque.
    px_opaque[3][0] = -0.1;
    let view4 = PixelView::from_typed(&px_opaque, 16, 16, &opaque_fmt).unwrap();
    assert!(ZenpixelsSource::try_from_view(&view4).is_err());
}

#[test]
fn pixel_buf_end_to_end() {
    let z = zensim();
    let px: Vec<u8> = test_pixels_rgba()
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect();
    let buf = PixelBuf::from_bytes(px.clone(), 16, 16, ZpPixelFormat::RGBA8_SRGB).unwrap();
    let src = ZenpixelsSource::try_from_buf(&buf).unwrap();

    let buf2 = PixelBuf::from_bytes(px, 16, 16, ZpPixelFormat::RGBA8_SRGB).unwrap();
    let dst = ZenpixelsSource::try_from_buf(&buf2).unwrap();

    let result = z.compute(&src, &dst).unwrap();
    assert_eq!(result.score(), 100.0);
}
