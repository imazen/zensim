//! Adapter for `zenpixels 0.5` pixel views and buffers.
//!
//! Provides [`ZenpixelsSource`], a validated wrapper that implements
//! [`ImageSource`] for any [`zenpixels::PixelView`] or [`zenpixels::PixelBuf`].
//!
//! # Supported SDR formats
//!
//! | zenpixels layout | transfer | alpha mode | zensim format / alpha |
//! |------------------|----------|------------|-----------------------|
//! | `RGB8` | `Srgb` | `None` | `Srgb8Rgb`, `Opaque` |
//! | `RGBA8` | `Srgb` | `Opaque` / `Straight` (`Premultiplied` via [`ZenpixelsSource`]) | `Srgb8Rgba` |
//! | `BGRA8` | `Srgb` | `Opaque` / `Straight` (`Premultiplied` via [`ZenpixelsSource`]) | `Srgb8Bgra` |
//! | `RGBX8` | `Srgb` | `Undefined` | `Srgb8Rgba`, `Opaque` |
//! | `BGRX8` | `Srgb` | `Undefined` | `Srgb8Bgra`, `Opaque` |
//! | `RGBA16` | `Srgb` | `Opaque` / `Straight` (`Premultiplied` via [`ZenpixelsSource`]) | `Srgb16Rgba` |
//! | `RGBAF32` | `Linear` | `Opaque` / `Straight` (`Premultiplied` via [`ZenpixelsSource`]) | `LinearF32Rgba` |
//!
//! Supported primaries: `Bt709` (sRGB), `DisplayP3`, `Bt2020`.
//!
//! # Rejected formats
//!
//! - Non-identity [`Orientation`](zenpixels::Orientation) (apply orientation before scoring)
//! - Non-full [`BitPacking`](zenpixels::BitPacking) or non-full [`SignalRange`](zenpixels::SignalRange)
//! - HDR transfers (`Pq`, `Hlg`), [`DiffuseWhite`](zenpixels::DiffuseWhite), or non-empty
//!   [`HdrMetadata`](zenpixels::HdrMetadata) — SDR ingress never silently scores HDR
//! - `Transfer::Bt709` or other non-sRGB/non-linear transfers (convert explicitly)
//! - Unknown or unsupported primaries (never silently aliased to sRGB)
//! - Grayscale, ICC, Oklab, XYZ, CMYK, YCbCr, or mismatched layout/alpha bindings
//! - Non-finite or out-of-`[0.0, 1.0]` `f32` samples (checked unless proven by `SampleFacts`)

use std::borrow::Cow;

use zenpixels::{
    AlphaMode as ZpAlpha, BitPacking, ChannelOrder, ColorSpace, Orientation, PixelBuf,
    PixelFormat as ZpPixelFormat, PixelView, Primaries as ZpPrimaries, SampleType, SignalRange,
    Transfer,
};

use crate::error::UnsupportedFormat;
use crate::source::{AlphaMode, ColorPrimaries, ImageSource, PixelFormat};

/// Reason why a `PixelView` was refused by strict SDR ingress.
#[cfg(feature = "zenpsy-api")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StrictIngressError {
    /// Unsupported format, color space, primaries, transfer, orientation, HDR, or alpha mode.
    Unsupported(&'static str),
    /// Well-formed format with invalid sample values (NaN, infinity, or out-of-SDR-range `f32`).
    InvalidSamples(&'static str),
    /// Cancelled during sample validation.
    Stopped(enough::StopReason),
}

/// Validated adapter wrapping `zenpixels 0.5` pixel data for use with zensim.
///
/// Created via [`try_from_view`](Self::try_from_view) or
/// [`try_from_buf`](Self::try_from_buf). Validates the [`zenpixels::PixelFormat`],
/// orientation, HDR metadata, and (for unproven `f32` views) sample range.
///
/// For premultiplied alpha, [`try_from_view`](Self::try_from_view) and
/// [`try_from_buf`](Self::try_from_buf) un-premultiply into an internal buffer
/// (one-time allocation). For all other formats, active strided bytes are
/// borrowed with zero copies.
pub struct ZenpixelsSource<'a> {
    data: Cow<'a, [u8]>,
    width: usize,
    height: usize,
    stride: usize,
    pixel_format: PixelFormat,
    alpha_mode: AlphaMode,
    color_primaries: ColorPrimaries,
}

impl<'a> ZenpixelsSource<'a> {
    /// Create from a borrowed [`PixelView<'a>`].
    ///
    /// Borrows the backing storage for `'a` with zero copies (preserving arbitrary
    /// row strides) unless the input uses [`zenpixels::AlphaMode::Premultiplied`],
    /// which is un-premultiplied into an owned buffer.
    pub fn try_from_view(view: &PixelView<'a>) -> Result<Self, UnsupportedFormat> {
        Self::try_from_view_inner(view, true, None).map_err(|e| match e {
            IngressFailure::Unsupported(msg) | IngressFailure::InvalidSamples(msg) => {
                UnsupportedFormat(msg)
            }
            IngressFailure::Stopped(_) => UnsupportedFormat("cancelled"),
        })
    }

    /// Create from an owned [`PixelBuf`], borrowing its pixel storage for `'a`.
    ///
    /// Handles both contiguous and strided buffers transparently via
    /// [`PixelBuf::as_view`].
    pub fn try_from_buf(buf: &'a PixelBuf) -> Result<Self, UnsupportedFormat> {
        let view = buf.as_view();
        Self::try_from_view(&view)
    }

    /// Strict SDR ingress for `zenpsy-api`: rejects premultiplied alpha explicitly,
    /// polls `stop` during float validation, and distinguishes unsupported formats
    /// from invalid float sample values.
    #[cfg(feature = "zenpsy-api")]
    pub(crate) fn try_from_view_strict_stoppable(
        view: &PixelView<'a>,
        stop: Option<&dyn enough::Stop>,
    ) -> Result<Self, StrictIngressError> {
        Self::try_from_view_inner(view, false, stop).map_err(|e| match e {
            IngressFailure::Unsupported(msg) => StrictIngressError::Unsupported(msg),
            IngressFailure::InvalidSamples(msg) => StrictIngressError::InvalidSamples(msg),
            IngressFailure::Stopped(reason) => StrictIngressError::Stopped(reason),
        })
    }

    fn try_from_view_inner(
        view: &PixelView<'a>,
        allow_premultiplied: bool,
        stop: Option<&dyn enough::Stop>,
    ) -> Result<Self, IngressFailure> {
        if let Some(s) = stop {
            s.check().map_err(IngressFailure::Stopped)?;
        }
        if view.orientation() != Orientation::Identity {
            return Err(IngressFailure::Unsupported(
                "pending EXIF orientation is not supported — apply orientation before scoring",
            ));
        }
        if !view.hdr().is_empty() {
            return Err(IngressFailure::Unsupported(
                "HDR container metadata is not supported by the SDR zensim pipeline",
            ));
        }

        let fmt = view.format();
        let (pixel_format, alpha_mode, color_primaries, is_premul) =
            map_format(fmt, allow_premultiplied)?;

        let width = view.width() as usize;
        let height = view.height() as usize;
        let stride = view.stride_bytes();
        let raw = view.as_strided_bytes();

        if pixel_format == PixelFormat::LinearF32Rgba && !view.facts().float.is_unit_interval() {
            let check_alpha = !matches!(alpha_mode, AlphaMode::Opaque) || is_premul;
            validate_sdr_f32_rgba(raw, width, height, stride, check_alpha, stop)?;
        }

        let (data, out_stride) = if is_premul {
            let bpp = pixel_format.bytes_per_pixel();
            let packed_stride = width * bpp;
            (
                Cow::Owned(unpremultiply_to_packed(
                    raw,
                    width,
                    height,
                    stride,
                    pixel_format,
                )),
                packed_stride,
            )
        } else {
            (Cow::Borrowed(raw), stride)
        };

        Ok(Self {
            data,
            width,
            height,
            stride: out_stride,
            pixel_format,
            alpha_mode,
            color_primaries,
        })
    }
}

impl ImageSource for ZenpixelsSource<'_> {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }
    #[inline]
    fn height(&self) -> usize {
        self.height
    }
    #[inline]
    fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }
    #[inline]
    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }
    #[inline]
    fn color_primaries(&self) -> ColorPrimaries {
        self.color_primaries
    }
    #[inline]
    fn row_bytes(&self, y: usize) -> &[u8] {
        let bpp = self.pixel_format.bytes_per_pixel();
        let start = y * self.stride;
        &self.data[start..start + self.width * bpp]
    }
}

#[derive(Clone, Copy, Debug)]
enum IngressFailure {
    Unsupported(&'static str),
    InvalidSamples(&'static str),
    Stopped(enough::StopReason),
}

/// Validate a `zenpixels 0.5` [`PixelFormat`] against zensim's SDR ingress contract.
fn map_format(
    fmt: &ZpPixelFormat,
    allow_premultiplied: bool,
) -> Result<(PixelFormat, AlphaMode, ColorPrimaries, bool), IngressFailure> {
    if fmt.layout.packing != BitPacking::Full {
        return Err(IngressFailure::Unsupported(
            "bit-packed pixel layouts are not supported",
        ));
    }

    let (primaries, transfer, range, alpha, white) = match fmt.color {
        ColorSpace::Rgb {
            primaries,
            transfer,
            range,
            alpha,
            white,
        } => (primaries, transfer, range, alpha, white),
        ColorSpace::Gray { .. } => {
            return Err(IngressFailure::Unsupported(
                "grayscale formats are not supported",
            ));
        }
        _ => {
            return Err(IngressFailure::Unsupported(
                "only RGB ColorSpace is supported by SDR zensim — convert via zenpixels-convert first",
            ));
        }
    };

    if white.is_some() {
        return Err(IngressFailure::Unsupported(
            "explicit DiffuseWhite / HDR reference luminance is not supported by the SDR zensim pipeline",
        ));
    }

    if range != SignalRange::Full {
        return Err(IngressFailure::Unsupported(
            "narrow/limited signal range is not supported — expand to full range first",
        ));
    }

    let color_primaries = match primaries {
        ZpPrimaries::Bt709 => ColorPrimaries::Srgb,
        ZpPrimaries::DisplayP3 => ColorPrimaries::DisplayP3,
        ZpPrimaries::Bt2020 => ColorPrimaries::Bt2020,
        _ => {
            return Err(IngressFailure::Unsupported(
                "unsupported or unknown color primaries — convert to BT.709, Display P3, or BT.2020 first",
            ));
        }
    };

    match transfer {
        Transfer::Srgb | Transfer::Linear => {}
        Transfer::Pq | Transfer::Hlg(_) => {
            return Err(IngressFailure::Unsupported(
                "HDR transfers (PQ, HLG) are not supported by the SDR zensim pipeline",
            ));
        }
        Transfer::Bt709 => {
            return Err(IngressFailure::Unsupported(
                "BT.709 transfer is not aliased to sRGB — convert to sRGB or linear RGB first",
            ));
        }
        _ => {
            return Err(IngressFailure::Unsupported(
                "unsupported or unknown transfer function",
            ));
        }
    }

    let (pixel_format, alpha_mode, is_premul) =
        match (fmt.layout.sample, fmt.layout.order, transfer) {
            (SampleType::U8, ChannelOrder::Rgb, Transfer::Srgb) => match alpha {
                ZpAlpha::None => (PixelFormat::Srgb8Rgb, AlphaMode::Opaque, false),
                _ => {
                    return Err(IngressFailure::Unsupported(
                        "RGB8 layout requires AlphaMode::None",
                    ));
                }
            },
            (SampleType::U8, ChannelOrder::Rgbx, Transfer::Srgb) => match alpha {
                ZpAlpha::Undefined => (PixelFormat::Srgb8Rgba, AlphaMode::Opaque, false),
                _ => {
                    return Err(IngressFailure::Unsupported(
                        "RGBX8 layout requires AlphaMode::Undefined",
                    ));
                }
            },
            (SampleType::U8, ChannelOrder::Bgrx, Transfer::Srgb) => match alpha {
                ZpAlpha::Undefined => (PixelFormat::Srgb8Bgra, AlphaMode::Opaque, false),
                _ => {
                    return Err(IngressFailure::Unsupported(
                        "BGRX8 layout requires AlphaMode::Undefined",
                    ));
                }
            },
            (SampleType::U8, ChannelOrder::Rgba, Transfer::Srgb) => {
                let (am, premul) = map_rgba_alpha(alpha, allow_premultiplied)?;
                (PixelFormat::Srgb8Rgba, am, premul)
            }
            (SampleType::U8, ChannelOrder::Bgra, Transfer::Srgb) => {
                let (am, premul) = map_rgba_alpha(alpha, allow_premultiplied)?;
                (PixelFormat::Srgb8Bgra, am, premul)
            }
            (SampleType::U16, ChannelOrder::Rgba, Transfer::Srgb) => {
                let (am, premul) = map_rgba_alpha(alpha, allow_premultiplied)?;
                (PixelFormat::Srgb16Rgba, am, premul)
            }
            (SampleType::F32, ChannelOrder::Rgba, Transfer::Linear) => {
                let (am, premul) = map_rgba_alpha(alpha, allow_premultiplied)?;
                (PixelFormat::LinearF32Rgba, am, premul)
            }
            (_, _, Transfer::Linear) => {
                return Err(IngressFailure::Unsupported(
                    "linear transfer requires RGBAF32 layout",
                ));
            }
            _ => {
                return Err(IngressFailure::Unsupported(
                    "unsupported pixel layout for zensim",
                ));
            }
        };

    Ok((pixel_format, alpha_mode, color_primaries, is_premul))
}

fn map_rgba_alpha(
    alpha: ZpAlpha,
    allow_premultiplied: bool,
) -> Result<(AlphaMode, bool), IngressFailure> {
    match alpha {
        ZpAlpha::Opaque => Ok((AlphaMode::Opaque, false)),
        ZpAlpha::Straight => Ok((AlphaMode::Straight, false)),
        ZpAlpha::Premultiplied if allow_premultiplied => Ok((AlphaMode::Straight, true)),
        ZpAlpha::Premultiplied => Err(IngressFailure::Unsupported(
            "premultiplied alpha is not accepted directly — convert to straight alpha first",
        )),
        _ => Err(IngressFailure::Unsupported(
            "RGBA/BGRA layout requires Opaque or Straight alpha mode",
        )),
    }
}

/// Validate that all active RGB (and meaningful alpha) `f32` samples are finite
/// and inside the SDR unit interval `[0.0, 1.0]`. Logical `Opaque` alpha ignores
/// its stored 4th lane.
fn validate_sdr_f32_rgba(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    check_alpha: bool,
    stop: Option<&dyn enough::Stop>,
) -> Result<(), IngressFailure> {
    let channels = if check_alpha { 4 } else { 3 };
    for y in 0..height {
        if y & 63 == 0
            && let Some(s) = stop
        {
            s.check().map_err(IngressFailure::Stopped)?;
        }
        let row_start = y * stride;
        let row = &data[row_start..row_start + width * 16];
        let pixels = crate::source::packed_row::<[f32; 4]>(row, width);
        for px in pixels.iter() {
            for &v in &px[..channels] {
                if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                    return Err(IngressFailure::InvalidSamples(
                        "linear f32 SDR input contains non-finite or out-of-[0, 1] sample",
                    ));
                }
            }
        }
    }
    if let Some(s) = stop {
        s.check().map_err(IngressFailure::Stopped)?;
    }
    Ok(())
}

/// Un-premultiply active strided rows into a tightly packed buffer with straight alpha.
///
/// Divides each RGB channel by alpha in the native domain (sRGB for u8/u16,
/// linear for f32). Fully transparent pixels (A=0) get RGB zeroed.
fn unpremultiply_to_packed(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    pixel_format: PixelFormat,
) -> Vec<u8> {
    let bpp = pixel_format.bytes_per_pixel();
    let packed_stride = width * bpp;
    let mut out = vec![0u8; packed_stride * height];
    for y in 0..height {
        let src_start = y * stride;
        let dst_start = y * packed_stride;
        out[dst_start..dst_start + packed_stride]
            .copy_from_slice(&data[src_start..src_start + packed_stride]);
    }

    match pixel_format {
        PixelFormat::Srgb8Rgba | PixelFormat::Srgb8Bgra => {
            for y in 0..height {
                let row_start = y * packed_stride;
                for x in 0..width {
                    let off = row_start + x * 4;
                    let a = out[off + 3];
                    if a == 0 {
                        out[off] = 0;
                        out[off + 1] = 0;
                        out[off + 2] = 0;
                    } else if a < 255 {
                        let inv = 255.0 / a as f32;
                        out[off] = (out[off] as f32 * inv).round().min(255.0) as u8;
                        out[off + 1] = (out[off + 1] as f32 * inv).round().min(255.0) as u8;
                        out[off + 2] = (out[off + 2] as f32 * inv).round().min(255.0) as u8;
                    }
                }
            }
        }
        PixelFormat::Srgb16Rgba => {
            for y in 0..height {
                let row_start = y * packed_stride;
                for x in 0..width {
                    let off = row_start + x * 8;
                    let a = u16::from_ne_bytes([out[off + 6], out[off + 7]]);
                    if a == 0 {
                        out[off..off + 6].fill(0);
                    } else if a < 65535 {
                        let inv = 65535.0 / a as f32;
                        for c in 0..3 {
                            let co = off + c * 2;
                            let v = u16::from_ne_bytes([out[co], out[co + 1]]);
                            let unpremul = (v as f32 * inv).round().min(65535.0) as u16;
                            out[co..co + 2].copy_from_slice(&unpremul.to_ne_bytes());
                        }
                    }
                }
            }
        }
        PixelFormat::LinearF32Rgba => {
            for y in 0..height {
                let row_start = y * packed_stride;
                for x in 0..width {
                    let off = row_start + x * 16;
                    let a = f32::from_ne_bytes([
                        out[off + 12],
                        out[off + 13],
                        out[off + 14],
                        out[off + 15],
                    ]);
                    if a <= 0.0 {
                        out[off..off + 12].fill(0);
                    } else if a < 1.0 {
                        for c in 0..3 {
                            let co = off + c * 4;
                            let v = f32::from_ne_bytes([
                                out[co],
                                out[co + 1],
                                out[co + 2],
                                out[co + 3],
                            ]);
                            // Divide directly rather than multiplying by `1.0 / a`:
                            // for subnormal `a` (e.g. `f32::from_bits(1)`), `1.0 / a`
                            // overflows to +inf and `0.0 * inf` produces NaN.
                            let straight = if v <= 0.0 {
                                0.0f32
                            } else {
                                (v / a).clamp(0.0, 1.0)
                            };
                            out[co..co + 4].copy_from_slice(&straight.to_ne_bytes());
                        }
                    }
                }
            }
        }
        _ => {}
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenpixels::{ContentLightLevel, DiffuseWhite, HdrMetadata, SampleLayout, formats};

    #[test]
    fn rgbx_maps_to_opaque() {
        let (fmt, alpha, _, premul) = map_format(&formats::RGBX8_SRGB, true).unwrap();
        assert_eq!(fmt, PixelFormat::Srgb8Rgba);
        assert_eq!(alpha, AlphaMode::Opaque);
        assert!(!premul);
    }

    #[test]
    fn bgrx_maps_to_opaque() {
        let (fmt, alpha, _, premul) = map_format(&formats::BGRX8_SRGB, true).unwrap();
        assert_eq!(fmt, PixelFormat::Srgb8Bgra);
        assert_eq!(alpha, AlphaMode::Opaque);
        assert!(!premul);
    }

    #[test]
    fn rgba_straight_maps_correctly() {
        let (fmt, alpha, _, premul) = map_format(&formats::RGBA8_SRGB, true).unwrap();
        assert_eq!(fmt, PixelFormat::Srgb8Rgba);
        assert_eq!(alpha, AlphaMode::Straight);
        assert!(!premul);
    }

    #[test]
    fn hdr_rejected() {
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
        assert!(map_format(&fmt, true).is_err());
    }

    #[test]
    fn bt709_transfer_rejected() {
        let fmt = ZpPixelFormat::new(
            SampleLayout::RGB8,
            ColorSpace::Rgb {
                primaries: ZpPrimaries::Bt709,
                transfer: Transfer::Bt709,
                range: SignalRange::Full,
                alpha: ZpAlpha::None,
                white: None,
            },
        );
        assert!(map_format(&fmt, true).is_err());
    }

    #[test]
    fn unknown_primaries_rejected() {
        let fmt = ZpPixelFormat::new(
            SampleLayout::RGB8,
            ColorSpace::Rgb {
                primaries: ZpPrimaries::Unknown,
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha: ZpAlpha::None,
                white: None,
            },
        );
        assert!(map_format(&fmt, true).is_err());
    }

    #[test]
    fn diffuse_white_rejected() {
        let fmt = ZpPixelFormat::new(
            SampleLayout::RGBAF32,
            ColorSpace::Rgb {
                primaries: ZpPrimaries::Bt709,
                transfer: Transfer::Linear,
                range: SignalRange::Full,
                alpha: ZpAlpha::Opaque,
                white: Some(DiffuseWhite::BT2408),
            },
        );
        assert!(map_format(&fmt, true).is_err());
    }

    #[test]
    fn grayscale_rejected() {
        assert!(map_format(&formats::GRAY8_SRGB, true).is_err());
    }

    #[test]
    fn p3_primaries_mapped() {
        let fmt = ZpPixelFormat::new(
            SampleLayout::RGBA8,
            ColorSpace::Rgb {
                primaries: ZpPrimaries::DisplayP3,
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha: ZpAlpha::Straight,
                white: None,
            },
        );
        let (_, _, primaries, _) = map_format(&fmt, true).unwrap();
        assert_eq!(primaries, ColorPrimaries::DisplayP3);
    }

    #[test]
    fn pending_orientation_and_hdr_metadata_rejected() {
        let data = vec![128u8; 4 * 4 * 3];
        let view = PixelView::new(&data, 4, 4, 12, &formats::RGB8_SRGB)
            .unwrap()
            .with_orientation(Orientation::Rotate90);
        assert!(ZenpixelsSource::try_from_view(&view).is_err());

        let hdr_view = PixelView::new(&data, 4, 4, 12, &formats::RGB8_SRGB)
            .unwrap()
            .with_hdr(HdrMetadata {
                content_light_level: Some(ContentLightLevel::new(1000, 400)),
                mastering: None,
            });
        assert!(ZenpixelsSource::try_from_view(&hdr_view).is_err());
    }

    #[test]
    fn unpremultiply_u8_roundtrip() {
        let data = vec![100u8, 50, 25, 200];
        let result = unpremultiply_to_packed(&data, 1, 1, 4, PixelFormat::Srgb8Rgba);
        assert_eq!(result[3], 200);
        assert_eq!(result[0], 128);
        assert_eq!(result[1], 64);
    }

    #[test]
    fn unpremultiply_a0_clears_rgb() {
        let data = vec![255u8, 128, 64, 0];
        let result = unpremultiply_to_packed(&data, 1, 1, 4, PixelFormat::Srgb8Rgba);
        assert_eq!(result, [0, 0, 0, 0]);
    }

    #[test]
    fn unpremultiply_f32_subnormal_alpha_never_produces_nan() {
        let tiny = f32::from_bits(1);
        let px = [[0.0f32, 0.0, 0.0, tiny], [tiny, tiny * 0.5, 0.0, tiny]];
        let bytes: &[u8] = bytemuck::cast_slice(&px);
        let out = unpremultiply_to_packed(bytes, 2, 1, 32, PixelFormat::LinearF32Rgba);
        let straight: &[[f32; 4]] = bytemuck::cast_slice(&out);
        for p in straight {
            for &ch in p {
                assert!(
                    ch.is_finite() && (0.0..=1.0).contains(&ch),
                    "bad sample {ch:?}"
                );
            }
        }
        assert_eq!(straight[0][0], 0.0);
        assert_eq!(straight[0][1], 0.0);
        assert_eq!(straight[0][2], 0.0);
        assert_eq!(straight[1][0], 1.0);
    }
}
