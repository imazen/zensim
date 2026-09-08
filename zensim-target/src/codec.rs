//! Per-codec encode + decode adapters used by the target-search loop.
//!
//! Each backend takes an RGB8 packed buffer and a quality knob, then
//! returns `(encoded_bytes, decoded_rgb8_bytes)` for scoring against the
//! original.

use anyhow::{Result, bail};

use crate::CodecKind;

/// Codec-side adapter. Implementations encode the reference RGB8 buffer
/// at a chosen quality knob and round-trip through the codec's decoder.
pub trait CodecBackend {
    /// The (inclusive) range of the codec's native quality knob.
    /// Most codecs use 0..=100; zenjxl uses distance ~0.0..=15.0.
    fn quality_range(&self) -> (f32, f32);

    /// `true` for codecs where a LOWER knob value yields a HIGHER
    /// perceived score (zenjxl distance). `false` for q-based codecs
    /// where quality↑ → score↑.
    fn lower_quality_means_higher_score(&self) -> bool {
        false
    }

    /// Encode at `knob`, then decode back to packed RGB8. Returns
    /// `(encoded_bytes, decoded_rgb8)`. Decoded buffer must be exactly
    /// `width * height * 3` bytes.
    fn encode_decode(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        knob: f32,
    ) -> Result<(Vec<u8>, Vec<u8>)>;
}

/// Resolve the backend for a codec kind. Panics in builds where the
/// codec's feature flag is disabled — callers must check the build.
pub fn backend_for(codec: CodecKind) -> Box<dyn CodecBackend> {
    match codec {
        #[cfg(feature = "zenjpeg")]
        CodecKind::Jpeg => Box::new(jpeg::Jpeg),
        #[cfg(feature = "zenwebp")]
        CodecKind::Webp => Box::new(webp::Webp),
        #[cfg(feature = "zenavif")]
        CodecKind::Avif => Box::new(avif::Avif),
        #[cfg(feature = "zenjxl")]
        CodecKind::Jxl => Box::new(jxl::Jxl),
        #[cfg(feature = "zenpng")]
        CodecKind::Png => Box::new(png::Png),

        #[allow(unreachable_patterns)]
        other => panic!(
            "codec {other:?} not enabled in this build (feature flag missing). \
             Rebuild zensim-target with the corresponding feature."
        ),
    }
}

// --------------------------------------------------------------------
// zenjpeg backend
// --------------------------------------------------------------------

#[cfg(feature = "zenjpeg")]
pub mod jpeg {
    use super::*;

    pub struct Jpeg;

    impl CodecBackend for Jpeg {
        fn quality_range(&self) -> (f32, f32) {
            // Full documented ApproxJpegli range; low-q settings may
            // saturate to byte-identical encodes, measured by the harness.
            (0.0, 100.0)
        }

        fn encode_decode(
            &self,
            rgb: &[u8],
            width: u32,
            height: u32,
            knob: f32,
        ) -> Result<(Vec<u8>, Vec<u8>)> {
            use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

            let q = knob.clamp(0.0, 100.0);
            let config = EncoderConfig::ycbcr(Quality::ApproxJpegli(q), ChromaSubsampling::Quarter);
            let mut enc = config
                .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
                .map_err(|e| anyhow::anyhow!("zenjpeg setup: {e}"))?;
            enc.push_packed(rgb, enough::Unstoppable)
                .map_err(|e| anyhow::anyhow!("zenjpeg push: {e}"))?;
            let encoded = enc
                .finish()
                .map_err(|e| anyhow::anyhow!("zenjpeg finish: {e}"))?;

            let decoded = zenjpeg::decoder::Decoder::new()
                .output_format(zenjpeg::decoder::PixelFormat::Rgb)
                .decode(&encoded, enough::Unstoppable)
                .map_err(|e| anyhow::anyhow!("zenjpeg decode: {e}"))?;
            if decoded.width() != width || decoded.height() != height {
                bail!("decoded JPEG dimensions differ from source");
            }
            let rgb_out = decoded
                .pixels_u8()
                .ok_or_else(|| anyhow::anyhow!("zenjpeg decoder did not return RGB8"))?
                .to_vec();
            Ok((encoded, rgb_out))
        }
    }
}

// --------------------------------------------------------------------
// zenwebp backend
// --------------------------------------------------------------------

#[cfg(feature = "zenwebp")]
pub mod webp {
    use super::*;

    pub struct Webp;

    impl CodecBackend for Webp {
        fn quality_range(&self) -> (f32, f32) {
            (0.0, 100.0)
        }

        fn encode_decode(
            &self,
            rgb: &[u8],
            width: u32,
            height: u32,
            knob: f32,
        ) -> Result<(Vec<u8>, Vec<u8>)> {
            use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};

            let q = knob.clamp(0.0, 100.0);
            let cfg = LossyConfig::new().with_quality(q).with_method(4);
            let encoded = EncodeRequest::lossy(&cfg, rgb, PixelLayout::Rgb8, width, height)
                .encode()
                .map_err(|e| anyhow::anyhow!("zenwebp encode: {e:?}"))?;

            let (decoded, dw, dh) = zenwebp::oneshot::decode_rgb(&encoded)
                .map_err(|e| anyhow::anyhow!("zenwebp decode: {e:?}"))?;
            if dw != width || dh != height {
                bail!("decoded webp dimensions {dw}x{dh} != source {width}x{height}");
            }
            Ok((encoded, decoded))
        }
    }
}

// --------------------------------------------------------------------
// zenavif backend
// --------------------------------------------------------------------

#[cfg(feature = "zenavif")]
pub mod avif {
    use super::*;
    use rgb::Rgb;
    use zenpixels::PixelDescriptor;

    pub struct Avif;

    impl CodecBackend for Avif {
        fn quality_range(&self) -> (f32, f32) {
            // ravif quality scale 1..=100 (encoder rejects exactly 0).
            (1.0, 100.0)
        }

        fn encode_decode(
            &self,
            rgb: &[u8],
            width: u32,
            height: u32,
            knob: f32,
        ) -> Result<(Vec<u8>, Vec<u8>)> {
            let q = knob.clamp(1.0, 100.0);

            // rgb::Rgb<u8> is Pod via the `bytemuck` feature; safe cast
            // from packed u8 RGB.
            let rgb_pixels: &[Rgb<u8>] = bytemuck::cast_slice(rgb);
            let img = imgref::ImgRef::new(rgb_pixels, width as usize, height as usize);

            let config = zenavif::EncoderConfig::default().quality(q).speed(6);
            let stop = almost_enough::StopToken::new(enough::Unstoppable);
            let encoded = zenavif::encode_rgb8(img, &config, stop)
                .map_err(|e| anyhow::anyhow!("zenavif encode: {e:?}"))?;

            let pb = zenavif::decode(&encoded.avif_file)
                .map_err(|e| anyhow::anyhow!("zenavif decode: {e:?}"))?;
            let dw = pb.width();
            let dh = pb.height();
            if dw != width || dh != height {
                bail!("decoded avif dimensions {dw}x{dh} != source {width}x{height}");
            }
            let decoded_rgb = pixelbuffer_to_rgb8(&pb)?;
            Ok((encoded.avif_file, decoded_rgb))
        }
    }

    fn pixelbuffer_to_rgb8(pb: &zenavif::PixelBuffer) -> Result<Vec<u8>> {
        let desc = pb.descriptor();
        let w = pb.width() as usize;
        let h = pb.height() as usize;
        let slice = pb.as_slice();
        let stride = slice.stride();
        let data = slice.as_strided_bytes();

        if desc.layout_compatible(PixelDescriptor::RGB8) {
            let bpr = w * 3;
            let mut out = Vec::with_capacity(bpr * h);
            for row in 0..h {
                let start = row * stride;
                out.extend_from_slice(&data[start..start + bpr]);
            }
            Ok(out)
        } else if desc.layout_compatible(PixelDescriptor::RGBA8) {
            let bpr_in = w * 4;
            let bpr_out = w * 3;
            let mut out = Vec::with_capacity(bpr_out * h);
            for row in 0..h {
                let start = row * stride;
                let row_slice = &data[start..start + bpr_in];
                for px in row_slice.as_chunks::<4>().0 {
                    out.extend_from_slice(&px[..3]);
                }
            }
            Ok(out)
        } else {
            bail!("zenavif decoded pixel descriptor {desc:?} not RGB8 or RGBA8");
        }
    }
}

// --------------------------------------------------------------------
// zenjxl backend (fixed VarDCT distance family; separate from native map loops)
// --------------------------------------------------------------------

#[cfg(feature = "zenjxl")]
pub mod jxl {
    use super::*;
    use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};
    use zenjxl::JxlEncoderConfig;
    use zenpixels::{PixelDescriptor, PixelSlice};

    pub struct Jxl;

    impl CodecBackend for Jxl {
        fn quality_range(&self) -> (f32, f32) {
            // jxl distance: 0.0 ≈ mathematically lossless, ~15 = very lossy.
            // Floor above 0 because jxl-encoder 0.3 panics at exactly 0.0
            // (divide-by-zero in vardct/ac_context.rs).
            (0.01, 25.0)
        }

        fn lower_quality_means_higher_score(&self) -> bool {
            true
        }

        fn encode_decode(
            &self,
            rgb: &[u8],
            width: u32,
            height: u32,
            knob: f32,
        ) -> Result<(Vec<u8>, Vec<u8>)> {
            let distance = knob.clamp(0.01, 25.0);
            let cfg = JxlEncoderConfig::new().with_distance(distance);
            let stride = width as usize * 3;
            let slice = PixelSlice::new(rgb, width, height, stride, PixelDescriptor::RGB8_SRGB)
                .map_err(|e| anyhow::anyhow!("zenjxl slice: {e}"))?;
            let encoder = cfg
                .job()
                .encoder()
                .map_err(|e| anyhow::anyhow!("zenjxl ctor: {e}"))?;
            let output = encoder
                .encode(slice)
                .map_err(|e| anyhow::anyhow!("zenjxl encode: {e}"))?;
            let encoded = output.into_vec();

            let decoded = zenjxl::decode(&encoded, None, &[])
                .map_err(|e| anyhow::anyhow!("zenjxl decode: {e:?}"))?;
            let pb = &decoded.pixels;
            if pb.width() != width || pb.height() != height {
                bail!(
                    "decoded jxl dimensions {}x{} != source {width}x{height}",
                    pb.width(),
                    pb.height()
                );
            }
            let decoded_rgb = jxl_pixelbuffer_to_rgb8(pb)?;
            Ok((encoded, decoded_rgb))
        }
    }

    fn jxl_pixelbuffer_to_rgb8(pb: &zenpixels::PixelBuffer) -> Result<Vec<u8>> {
        let desc = pb.descriptor();
        let w = pb.width() as usize;
        let h = pb.height() as usize;
        let slice = pb.as_slice();
        let stride = slice.stride();
        let data = slice.as_strided_bytes();
        if desc.layout_compatible(PixelDescriptor::RGB8_SRGB)
            || desc.layout_compatible(PixelDescriptor::RGB8)
        {
            let bpr = w * 3;
            let mut out = Vec::with_capacity(bpr * h);
            for row in 0..h {
                let start = row * stride;
                out.extend_from_slice(&data[start..start + bpr]);
            }
            Ok(out)
        } else if desc.layout_compatible(PixelDescriptor::RGBA8_SRGB)
            || desc.layout_compatible(PixelDescriptor::RGBA8)
        {
            let bpr_in = w * 4;
            let bpr_out = w * 3;
            let mut out = Vec::with_capacity(bpr_out * h);
            for row in 0..h {
                let start = row * stride;
                let row_slice = &data[start..start + bpr_in];
                for px in row_slice.as_chunks::<4>().0 {
                    out.extend_from_slice(&px[..3]);
                }
            }
            Ok(out)
        } else {
            bail!("zenjxl decoded pixel descriptor {desc:?} not RGB8 or RGBA8");
        }
    }
}

// --------------------------------------------------------------------
// zenpng backend (lossless — single probe)
// --------------------------------------------------------------------

#[cfg(feature = "zenpng")]
pub mod png {
    use super::*;
    use enough::Unstoppable;
    use rgb::Rgb;

    pub struct Png;

    impl CodecBackend for Png {
        fn quality_range(&self) -> (f32, f32) {
            (100.0, 100.0)
        }

        fn encode_decode(
            &self,
            rgb: &[u8],
            width: u32,
            height: u32,
            _knob: f32,
        ) -> Result<(Vec<u8>, Vec<u8>)> {
            let rgb_pixels: &[Rgb<u8>] = bytemuck::cast_slice(rgb);
            let img = imgref::ImgRef::new(rgb_pixels, width as usize, height as usize);
            let cfg = zenpng::EncodeConfig::default();
            let encoded = zenpng::encode_rgb8(img, None, &cfg, &Unstoppable, &Unstoppable)
                .map_err(|e| anyhow::anyhow!("zenpng encode: {e:?}"))?;

            let dec = zenpng::decode(&encoded, &zenpng::PngDecodeConfig::strict(), &Unstoppable)?;
            let pb = dec.pixels;
            if pb.width() != width || pb.height() != height {
                bail!("decoded PNG dimensions differ from source");
            }
            let view = pb.as_slice();
            let data = view.as_strided_bytes();
            let desc = pb.descriptor();
            let gray = desc.layout_compatible(zenpixels::PixelDescriptor::GRAY8);
            let channels = if gray {
                1
            } else if desc.layout_compatible(zenpixels::PixelDescriptor::RGB8) {
                3
            } else if desc.layout_compatible(zenpixels::PixelDescriptor::RGBA8) {
                4
            } else {
                bail!("unexpected PNG reconstruction format {desc:?}");
            };
            let mut decoded = Vec::with_capacity(rgb.len());
            for row in 0..height as usize {
                let start = row * view.stride();
                for px in data[start..start + width as usize * channels].chunks_exact(channels) {
                    if gray {
                        decoded.extend_from_slice(&[px[0]; 3]);
                    } else {
                        decoded.extend_from_slice(&px[..3]);
                    }
                }
            }
            Ok((encoded, decoded))
        }
    }
}
