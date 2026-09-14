//! Versioned input interpretation for the existing extractor and its audit.
//! Color arithmetic belongs to zenpixels-convert; scoring belongs to zensim.
#![allow(dead_code)]

use super::zen_decode::{DecodedNative, NativeMetadata, decode_native_bytes, decode_rgb8_path};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;
use zenpixels::{ColorPrimaries, PixelDescriptor, TransferFunction};
use zenpixels_convert::{RowConverter, cms::PluggableCms, policy::ConvertOptions};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum InputContract {
    #[default]
    LegacyRgb8,
    SdrNativeClipV1,
}

impl InputContract {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "legacy-rgb8" => Ok(Self::LegacyRgb8),
            "sdr-native-clip-v1" => Ok(Self::SdrNativeClipV1),
            _ => Err(format!("unknown input contract: {value}")),
        }
    }
}

enum Pixels {
    Rgb8(Vec<u8>),
    LinearRgba(Vec<[f32; 4]>, ColorPrimaries),
    SrgbRgba16(Vec<[u16; 4]>, ColorPrimaries),
}

pub(crate) struct ScoreInput {
    pub width: u32,
    pub height: u32,
    pixels: Pixels,
    pub receipt: Option<Value>,
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}

fn known_sdr(desc: PixelDescriptor) -> bool {
    matches!(
        desc.transfer(),
        TransferFunction::Srgb | TransferFunction::Linear
    ) && matches!(
        desc.primaries,
        ColorPrimaries::Bt709 | ColorPrimaries::DisplayP3 | ColorPrimaries::Bt2020
    ) && desc.signal_range == zenpixels::SignalRange::Full
}

impl ScoreInput {
    pub fn decode(path: &Path, contract: InputContract) -> Result<Self, String> {
        if contract == InputContract::LegacyRgb8 {
            let value = decode_rgb8_path(path).map_err(|e| e.to_string())?;
            return Ok(Self {
                width: value.width,
                height: value.height,
                pixels: Pixels::Rgb8(value.pixels),
                receipt: None,
            });
        }
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let native =
            decode_native_bytes(&bytes, &path.display().to_string()).map_err(|e| e.to_string())?;
        Self::from_native(native).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn from_native(native: DecodedNative) -> Result<Self, String> {
        let buffer = &native.pixels;
        let original = buffer.descriptor();
        let mut interpreted = original;
        let mut profile: Option<&[u8]> = None;
        let interpretation;
        let (icc, cicp, depth, orientation) = match &native.metadata {
            NativeMetadata::Png(info) => {
                if info.content_light_level.is_some() || info.mastering_display.is_some() {
                    return Err("SDR contract refuses HDR mastering/light metadata".into());
                }
                if let Some(c) = info.cicp {
                    if info.icc_profile.is_some() || info.srgb_intent.is_some() {
                        return Err("conflicting PNG color authorities".into());
                    }
                    if !c.full_range || c.matrix_coefficients != 0 {
                        return Err("PNG requires full-range RGB CICP".into());
                    }
                    let primaries = match c.color_primaries {
                        1 => ColorPrimaries::Bt709,
                        9 => ColorPrimaries::Bt2020,
                        12 => ColorPrimaries::DisplayP3,
                        _ => return Err("unsupported PNG CICP primaries".into()),
                    };
                    let transfer = match c.transfer_characteristics {
                        13 => TransferFunction::Srgb,
                        8 => TransferFunction::Linear,
                        _ => return Err("SDR contract refuses HDR/unsupported PNG transfer".into()),
                    };
                    interpreted = original.with_primaries(primaries).with_transfer(transfer);
                    interpretation = "PNG CICP";
                } else if let Some(icc) = info.icc_profile.as_deref() {
                    if info.srgb_intent.is_some() {
                        return Err("conflicting PNG ICC and sRGB authorities".into());
                    }
                    profile = Some(icc);
                    interpretation = "PNG ICC through full CMS";
                } else if info.srgb_intent.is_some() {
                    interpreted = original
                        .with_primaries(ColorPrimaries::Bt709)
                        .with_transfer(TransferFunction::Srgb);
                    interpretation = "PNG sRGB chunk";
                } else {
                    if info.source_gamma.is_some() || info.chromaticities.is_some() {
                        return Err("PNG gamma/chromaticities require an explicit supported color interpretation".into());
                    }
                    interpreted = original
                        .with_primaries(ColorPrimaries::Bt709)
                        .with_transfer(TransferFunction::Srgb);
                    interpretation = "untagged PNG: contract assumes sRGB";
                }
                (
                    info.icc_profile.as_deref(),
                    info.cicp,
                    Some(info.bit_depth),
                    json!({"policy":"stored raster; EXIF not applied", "exif_sha256":info.exif.as_deref().map(sha)}),
                )
            }
            NativeMetadata::Codec(info) => {
                let sc = &info.source_color;
                if sc.content_light_level.is_some()
                    || sc.mastering_display.is_some()
                    || sc
                        .cicp
                        .is_some_and(|c| matches!(c.transfer_characteristics, 16 | 18))
                {
                    return Err("SDR contract refuses HDR metadata/transfer".into());
                }
                // A codec may already have inverted XYB or changed its output
                // encoding. A known decoded descriptor describes THESE pixels;
                // a source ICC is not permission to transform them a second time.
                if known_sdr(original) {
                    interpretation = "codec decoded descriptor";
                } else if matches!(
                    original.transfer(),
                    TransferFunction::Pq | TransferFunction::Hlg
                ) {
                    return Err("SDR contract refuses decoded HDR transfer".into());
                } else if let Some(icc) = sc.icc_profile.as_deref() {
                    profile = Some(icc);
                    interpretation = "unconverted codec ICC through full CMS";
                } else {
                    return Err(format!(
                        "unknown/unsupported decoded color interpretation: {original:?}"
                    ));
                }
                (
                    sc.icc_profile.as_deref(),
                    sc.cicp,
                    sc.bit_depth,
                    json!({"policy":"stored decoded raster", "reported":format!("{:?}",info.orientation)}),
                )
            }
        };
        if let Some(icc) = profile
            && zenpixels::icc::extract_cicp(icc)
                .is_some_and(|c| matches!(c.transfer_characteristics, 16 | 18))
        {
            return Err("SDR contract refuses HDR ICC CICP".into());
        }
        let width = buffer.width();
        let height = buffer.height();
        let count = (width as usize)
            .checked_mul(height as usize)
            .ok_or("pixel count overflow")?;
        let options = ConvertOptions::permissive().with_clip_out_of_gamut(false);
        let target = PixelDescriptor::RGBAF32_LINEAR;
        let neutral = original
            .with_transfer(TransferFunction::Linear)
            .with_primaries(ColorPrimaries::Bt709);
        let (pixels, presented) = if let Some(icc) = profile {
            // CMS accepts f32 codes, not integer byte patterns interpreted as
            // floats. Normalize depth/layout without guessing the ICC transfer.
            let mut pack =
                RowConverter::new_explicit(neutral, target, &options).map_err(|e| e.to_string())?;
            let mut transform = zenpixels_convert::MoxCms
                .build_source_transform(
                    zenpixels::ColorProfileSource::Icc(icc),
                    zenpixels::ColorProfileSource::PrimariesTransferPair {
                        primaries: ColorPrimaries::Bt709,
                        transfer: TransferFunction::Linear,
                    },
                    zenpixels::PixelFormat::RgbaF32,
                    zenpixels::PixelFormat::RgbaF32,
                    &options,
                )
                .ok_or("CMS declined source ICC")?
                .map_err(|e| e.to_string())?;
            let mut values = Vec::new();
            values.try_reserve_exact(count).map_err(|e| e.to_string())?;
            values.resize(count, [0f32; 4]);
            let mut row = vec![[0f32; 4]; width as usize];
            for y in 0..height {
                pack.convert_row(
                    buffer.as_slice().row(y),
                    bytemuck::cast_slice_mut(&mut row),
                    width,
                );
                let dst =
                    &mut values[y as usize * width as usize..(y as usize + 1) * width as usize];
                transform.transform_row(
                    bytemuck::cast_slice(&row),
                    bytemuck::cast_slice_mut(dst),
                    width,
                );
            }
            if !values.iter().flatten().all(|v| v.is_finite()) {
                return Err("nonfinite color-converted samples".into());
            }
            (
                Pixels::LinearRgba(values, ColorPrimaries::Bt709),
                "linear sRGB RGBA f32, straight alpha",
            )
        } else {
            if !known_sdr(interpreted) {
                return Err(format!("unsupported SDR descriptor: {interpreted:?}"));
            }
            // Let the public scoring owner linearize and convert named primaries.
            // Retain every u16 code; do not route through an approximate f32 TRC
            // or an ICC lookup table when the source encoding is already known.
            if interpreted.transfer() == TransferFunction::Srgb {
                if !matches!(
                    original.channel_type(),
                    zenpixels::ChannelType::U8 | zenpixels::ChannelType::U16
                ) {
                    return Err(
                        "SDR float encoded-sRGB needs an explicit native float transfer contract"
                            .into(),
                    );
                }
                let mut pack = RowConverter::new_explicit(
                    neutral,
                    PixelDescriptor::RGBA16_SRGB.with_transfer(TransferFunction::Linear),
                    &options,
                )
                .map_err(|e| e.to_string())?;
                let mut values = Vec::new();
                values.try_reserve_exact(count).map_err(|e| e.to_string())?;
                values.resize(count, [0u16; 4]);
                for y in 0..height {
                    let dst =
                        &mut values[y as usize * width as usize..(y as usize + 1) * width as usize];
                    pack.convert_row(
                        buffer.as_slice().row(y),
                        bytemuck::cast_slice_mut(dst),
                        width,
                    );
                }
                (
                    Pixels::SrgbRgba16(values, interpreted.primaries),
                    "sRGB-transfer RGBA u16, declared primaries, straight alpha",
                )
            } else {
                let mut pack = RowConverter::new_explicit(neutral, target, &options)
                    .map_err(|e| e.to_string())?;
                let mut values = Vec::new();
                values.try_reserve_exact(count).map_err(|e| e.to_string())?;
                values.resize(count, [0f32; 4]);
                for y in 0..height {
                    let dst =
                        &mut values[y as usize * width as usize..(y as usize + 1) * width as usize];
                    pack.convert_row(
                        buffer.as_slice().row(y),
                        bytemuck::cast_slice_mut(dst),
                        width,
                    );
                }
                if !values.iter().flatten().all(|v| v.is_finite()) {
                    return Err("nonfinite linear input".into());
                }
                (
                    Pixels::LinearRgba(values, interpreted.primaries),
                    "linear RGBA f32, declared primaries, straight alpha",
                )
            }
        };
        let receipt = json!({"contract":"sdr-native-clip-v1", "interpretation":interpretation,
            "source_icc_sha256":icc.map(sha), "source_cicp":cicp.map(|c|format!("{c:?}")),
            "source_bit_depth":depth,"decoded_descriptor":format!("{original:?}"),
            "native_active_rows_sha256":sha(&buffer.copy_to_contiguous_bytes()),
            "native_stride_bytes":buffer.stride(),"orientation":orientation,
            "presented_format":presented,
            "endianness":if cfg!(target_endian="little") {"little"} else {"big"},
            "display":"public SDR sRGB clipping; existing alpha compositing",
            "converter":"zenpixels-convert 0.2.16", "full_cms":profile.is_some(), "source_icc_used":profile.is_some()});
        let mut result = Self {
            width,
            height,
            pixels,
            receipt: Some(receipt),
        };
        // Record what the public scorer actually receives, independently of
        // source ICC/CICP and of a codec's original decoded descriptor.
        use zensim::ImageSource;
        let source = result.source();
        let identity = json!({
            "pixel_format":format!("{:?}",source.pixel_format()),
            "primaries":format!("{:?}",source.color_primaries()),
            "alpha":format!("{:?}",source.alpha_mode()),
            "gamut":format!("{:?}",source.gamut_mapping()),
            "width":source.width(), "height":source.height(),
            "endianness":if cfg!(target_endian="little") {"little"} else {"big"}
        });
        result.receipt.as_mut().unwrap()["scoring_identity"] = identity;
        Ok(result)
    }

    pub fn bytes(&self) -> &[u8] {
        match &self.pixels {
            Pixels::Rgb8(v) => v,
            Pixels::LinearRgba(v, _) => bytemuck::cast_slice(v),
            Pixels::SrgbRgba16(v, _) => bytemuck::cast_slice(v),
        }
    }

    /// Cached identity includes the interpretation of the samples. Equal RGB
    /// codes tagged P3 and sRGB are not the same input to the public scorer.
    pub fn is_identical_to(&self, other: &Self) -> bool {
        use zensim::ImageSource;
        let a = self.source();
        let b = other.source();
        self.width == other.width
            && self.height == other.height
            && a.pixel_format() == b.pixel_format()
            && a.color_primaries() == b.color_primaries()
            && a.alpha_mode() == b.alpha_mode()
            && a.gamut_mapping() == b.gamut_mapping()
            && self.bytes() == other.bytes()
    }

    pub fn source(&self) -> zensim::StridedBytes<'_> {
        let (format, primaries) = match &self.pixels {
            Pixels::Rgb8(_) => (zensim::PixelFormat::Srgb8Rgb, ColorPrimaries::Bt709),
            Pixels::LinearRgba(_, p) => (zensim::PixelFormat::LinearF32Rgba, *p),
            Pixels::SrgbRgba16(_, p) => (zensim::PixelFormat::Srgb16Rgba, *p),
        };
        let primaries = match primaries {
            ColorPrimaries::Bt709 => zensim::ColorPrimaries::Srgb,
            ColorPrimaries::DisplayP3 => zensim::ColorPrimaries::DisplayP3,
            ColorPrimaries::Bt2020 => zensim::ColorPrimaries::Bt2020,
            _ => unreachable!("validated SDR primaries"),
        };
        zensim::StridedBytes::new(
            self.bytes(),
            self.width as usize,
            self.height as usize,
            self.width as usize * format.bytes_per_pixel(),
            format,
        )
        .with_color_primaries(primaries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png16(pixels: &[rgb::Rgb<u16>], metadata: Option<&zencodec::Metadata>) -> DecodedNative {
        let bytes = zenpng::encode_rgb16(
            imgref::Img::new(pixels, 16, 16),
            metadata,
            &zenpng::EncodeConfig::default(),
            &enough::Unstoppable,
            &enough::Unstoppable,
        )
        .unwrap();
        decode_native_bytes(&bytes, "synthetic.png").unwrap()
    }

    #[test]
    fn native_sdr_preserves_every_u16_code_and_opaque_alpha() {
        let pixels: Vec<_> = (0..256)
            .map(|i| rgb::Rgb::new(0x1200 + i, 0x4567, 0xabcd))
            .collect();
        let input = ScoreInput::from_native(png16(&pixels, None)).unwrap();
        let packed: &[[u16; 4]] = bytemuck::cast_slice(input.bytes());
        for (p, actual) in pixels.iter().zip(packed) {
            assert_eq!(*actual, [p.r, p.g, p.b, 65535]);
        }
        assert!(packed.windows(2).all(|p| p[1][0] > p[0][0]));
    }

    #[test]
    fn full_icc_transform_agrees_with_independent_p3_colorimetry() {
        let pixels: Vec<_> = (0..256)
            .map(|i| rgb::Rgb::new(28000 + i * 20, 22000 + i * 13, 33000 - i * 9))
            .collect();
        let icc =
            zencodec::Metadata::none().with_icc(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
        let a = ScoreInput::from_native(png16(&pixels, Some(&icc))).unwrap();
        let av: &[[f32; 4]] = bytemuck::cast_slice(a.bytes());
        // Independent f64 sRGB EOTF and published D65 P3 -> sRGB matrix.
        // ICC s15Fixed16 colorants/TRC and CMS LUT precision are bounded here;
        // this is deliberately separate from exact native-code retention.
        let decode = |v: u16| {
            let v = f64::from(v) / 65535.;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        };
        let m = [
            [1.2249401763, -0.2249401763, 0.0],
            [-0.0420569547, 1.0420569547, 0.0],
            [-0.0196375546, -0.0786360456, 1.0982736001],
        ];
        let mut max_delta = 0f64;
        for (p, a) in pixels.iter().zip(av) {
            let linear = [decode(p.r), decode(p.g), decode(p.b)];
            for c in 0..3 {
                let expected: f64 = m[c].iter().zip(linear).map(|(a, b)| a * b).sum();
                max_delta = max_delta.max((f64::from(a[c]) - expected).abs());
            }
            assert_eq!(a[3], 1.0);
        }
        assert!(max_delta < 0.0005, "ICC/reference mismatch {max_delta}");
        assert_eq!(a.receipt.as_ref().unwrap()["full_cms"], true);
    }

    #[test]
    fn public_scoring_observes_errors_hidden_by_rgb8_projection() {
        let a = vec![rgb::Rgb::new(0x1200, 0x4567, 0xabcd); 256];
        let mut b = a.clone();
        for (i, p) in b.iter_mut().enumerate() {
            p.r += if i % 2 == 0 { 8 } else { 16 };
        }
        let an = png16(&a, None);
        let bn = png16(&b, None);
        assert_eq!(
            an.to_rgb8("a").unwrap().pixels,
            bn.to_rgb8("b").unwrap().pixels
        );
        let a = ScoreInput::from_native(an).unwrap();
        let b = ScoreInput::from_native(bn).unwrap();
        let metric = zensim::Zensim::new(zensim::ZensimProfile::B);
        let different = metric.compute(&a.source(), &b.source()).unwrap();
        let identity = metric.compute(&a.source(), &a.source()).unwrap();
        assert_ne!(different.features(), identity.features());
        assert!(different.features().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn native_and_icc_paths_preserve_translucent_alpha() {
        let pixels: Vec<_> = (0..256)
            .map(|i| rgb::Rgba::new(28000u16, 22000, 33000, i * 257))
            .collect();
        let icc =
            zencodec::Metadata::none().with_icc(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
        for metadata in [None, Some(&icc)] {
            let bytes = zenpng::encode_rgba16(
                imgref::Img::new(pixels.as_slice(), 16, 16),
                metadata,
                &zenpng::EncodeConfig::default(),
                &enough::Unstoppable,
                &enough::Unstoppable,
            )
            .unwrap();
            let native = decode_native_bytes(&bytes, "translucent.png").unwrap();
            let input = ScoreInput::from_native(native).unwrap();
            if metadata.is_some() {
                let values: &[[f32; 4]] = bytemuck::cast_slice(input.bytes());
                for (p, v) in pixels.iter().zip(values) {
                    assert!((v[3] - f32::from(p.a) / 65535.).abs() < 1e-7);
                }
            } else {
                let values: &[[u16; 4]] = bytemuck::cast_slice(input.bytes());
                for (p, v) in pixels.iter().zip(values) {
                    assert_eq!(*v, [p.r, p.g, p.b, p.a]);
                }
            }
        }
    }

    #[test]
    fn identity_requires_matching_color_interpretation() {
        let pixels = vec![rgb::Rgb::new(40000, 15000, 30000); 256];
        let p3 = zencodec::Metadata::none().with_cicp(zenpixels::Cicp::new(12, 13, 0, true));
        let a = ScoreInput::from_native(png16(&pixels, None)).unwrap();
        let b = ScoreInput::from_native(png16(&pixels, Some(&p3))).unwrap();
        assert_eq!(a.bytes(), b.bytes());
        assert!(a.is_identical_to(&a));
        assert!(!a.is_identical_to(&b));
        let ai = &a.receipt.as_ref().unwrap()["scoring_identity"];
        let bi = &b.receipt.as_ref().unwrap()["scoring_identity"];
        assert_eq!(ai["primaries"], "Srgb");
        assert_eq!(bi["primaries"], "DisplayP3");
        assert_ne!(ai, bi);
        let metric = zensim::Zensim::new(zensim::ZensimProfile::B);
        assert!(metric.compute(&a.source(), &b.source()).unwrap().score() < 99.9);
    }

    #[test]
    fn native_sdr_refuses_hdr_unknown_and_invalid_profiles() {
        let pixels = vec![rgb::Rgb::new(12345, 23456, 34567); 256];
        for cicp in [
            zenpixels::Cicp::new(9, 16, 0, true),
            zenpixels::Cicp::new(9, 18, 0, true),
            zenpixels::Cicp::new(255, 13, 0, true),
            zenpixels::Cicp::new(1, 13, 0, false),
        ] {
            let meta = zencodec::Metadata::none().with_cicp(cicp);
            assert!(ScoreInput::from_native(png16(&pixels, Some(&meta))).is_err());
        }
        let meta = zencodec::Metadata::none().with_icc(b"invalid ICC".as_slice());
        assert!(ScoreInput::from_native(png16(&pixels, Some(&meta))).is_err());
        assert!(InputContract::parse("native-ish").is_err());
    }
}
