//! Exposure-normalized content screen. This is not an HDR quality transform.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
use zenpixels::{ColorPrimaries, PixelDescriptor, TransferFunction};

use super::dhash_bits;

pub(super) fn fingerprint(path: &Path, assume_untagged_srgb: bool) -> Result<Value> {
    let bytes = fs::read(path).with_context(|| path.display().to_string())?;
    let (pixels, metadata, untagged_png) = if bytes.starts_with(&[0x76, 0x2f, 0x31, 0x01]) {
        let decoded = zenexr::ExrDecoderConfig::new()
            .decode(&bytes, &enough::Unstoppable)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        let metadata = format!("{:?}", decoded.header());
        (decoded.into_pixels(), metadata, false)
    } else {
        let decoded = zenpng::decode(
            &bytes,
            &zenpng::PngDecodeConfig::default(),
            &enough::Unstoppable,
        )
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        let untagged = decoded.info.icc_profile.is_none()
            && decoded.info.source_gamma.is_none()
            && decoded.info.srgb_intent.is_none()
            && decoded.info.cicp.is_none();
        (decoded.pixels, format!("{:?}", decoded.info), untagged)
    };
    let (width, height) = (pixels.width(), pixels.height());
    ensure!(
        width >= 9 && height >= 8,
        "native linear audit requires at least 9x8"
    );
    let source_descriptor = pixels.descriptor();
    let mut descriptor = source_descriptor;
    let applied_srgb_assumption =
        untagged_png && assume_untagged_srgb && descriptor.transfer == TransferFunction::Unknown;
    if applied_srgb_assumption {
        descriptor.transfer = TransferFunction::Srgb;
    }
    ensure!(
        descriptor.primaries == ColorPrimaries::Bt709,
        "unknown/non-BT709 primaries"
    );
    ensure!(
        descriptor.alpha.is_none(),
        "alpha needs an explicit audit compositing policy"
    );
    ensure!(
        matches!(
            descriptor.transfer,
            TransferFunction::Srgb | TransferFunction::Linear
        ),
        "unsupported transfer: {descriptor:?}"
    );
    let mut converter =
        zenpixels_convert::RowConverter::new(descriptor, PixelDescriptor::RGBF32_LINEAR)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut row = vec![0u8; width as usize * 12];
    let mut luma = Vec::with_capacity(width as usize * height as usize);
    let view = pixels.as_slice();
    for y in 0..height {
        converter.convert_row(view.row(y), &mut row, width);
        for pixel in row.as_chunks::<12>().0 {
            let channels = [0, 1, 2].map(|i| f32::from_ne_bytes(pixel.as_chunks::<4>().0[i]));
            ensure!(
                channels.iter().all(|v| v.is_finite()),
                "nonfinite RGB sample"
            );
            luma.push(0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]);
        }
    }
    let (hash, peak) = fingerprint_luma(&mut luma, width, height)?;
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect();
    Ok(
        json!({"path":path,"sha256":digest,"width":width,"height":height,
        "dhash":hash,"positive_luminance_peak":peak,"descriptor":format!("{descriptor:?}"),
        "source_descriptor":format!("{source_descriptor:?}"),
        "applied_untagged_srgb_assumption":applied_srgb_assumption,"source_metadata":metadata}),
    )
}

fn fingerprint_luma(luma: &mut [f32], width: u32, height: u32) -> Result<(u64, f32)> {
    ensure!(
        width >= 9 && height >= 8 && luma.len() == width as usize * height as usize,
        "luma geometry"
    );
    ensure!(luma.iter().all(|v| v.is_finite()), "nonfinite luminance");
    let peak = luma.iter().copied().fold(0.0f32, f32::max);
    for value in &mut *luma {
        *value = if peak > 0.0 {
            (255.0 * (value.max(0.0) / peak)).ln_1p()
        } else {
            0.0
        };
    }
    let config = zenresize::ResizeConfig::builder(width, height, 9, 8)
        .filter(zenresize::Filter::Lanczos)
        .format(PixelDescriptor::GRAYF32_LINEAR)
        .build();
    let small = zenresize::Resizer::new(&config).resize_f32(luma);
    ensure!(small.iter().all(|v| v.is_finite()), "nonfinite resample");
    Ok((
        dhash_bits::from_luma9x8(small.as_slice().try_into().context("linear 9x8")?),
        peak,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposure_invariance_and_unrelated_control() {
        let base: Vec<f32> = (0..64 * 64)
            .map(|i| (((i / 64) * 13 + (i % 64) * 7) % 257) as f32)
            .collect();
        let expected = fingerprint_luma(&mut base.clone(), 64, 64).unwrap().0;
        for exposure in [0.125, 1.0, 8.0, 1024.0] {
            let mut scaled: Vec<_> = base.iter().map(|v| v * exposure).collect();
            assert_eq!(fingerprint_luma(&mut scaled, 64, 64).unwrap().0, expected);
        }
        let mut other: Vec<_> = base.iter().rev().copied().collect();
        assert!((fingerprint_luma(&mut other, 64, 64).unwrap().0 ^ expected).count_ones() > 16);
    }

    #[test]
    fn black_is_defined_and_nonfinite_or_wrong_geometry_fails() {
        assert_eq!(fingerprint_luma(&mut [0.0; 72], 9, 8).unwrap(), (0, 0.0));
        assert!(fingerprint_luma(&mut [f32::NAN; 72], 9, 8).is_err());
        assert!(fingerprint_luma(&mut [f32::INFINITY; 72], 9, 8).is_err());
        assert!(fingerprint_luma(&mut [0.0; 72], 8, 9).is_err());
    }
}
