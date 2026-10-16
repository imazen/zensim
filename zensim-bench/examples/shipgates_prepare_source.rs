//! Explicit source preparation by a lossless all-opaque crop.
#![forbid(unsafe_code)]
use rgb::FromSlice;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf};
use zenpixels::{
    PixelDescriptor,
    policy::{AlphaPolicy, ConvertOptions},
};
use zenpixels_convert::{PixelBufferConvertTypedExt, RowConverter};

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 4 || args[1] != "--crop-opaque" {
        return Err("expected --crop-opaque input PNG and fresh output PNG; matte conversion is not admitted".into());
    }
    let input = PathBuf::from(&args[2]);
    let output = PathBuf::from(&args[3]);
    let bytes = fs::read(&input)?;
    let image = zenpng::decode(
        &bytes,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )?;
    let info = &image.info;
    if info.bit_depth != 8
        || info.sequence.is_animation()
        || info.icc_profile.is_some()
        || info.exif.is_some()
        || info.cicp.is_some()
        || info.chromaticities.is_some()
        || info.content_light_level.is_some()
        || info.mastering_display.is_some()
        || info.source_gamma.is_some_and(|v| v != 45455)
    {
        return Err("outside static sRGB8 source contract".into());
    }
    let rgba = image.pixels.to_rgba8().copy_to_contiguous_bytes();
    let rect = largest_opaque(imgref::ImgRef::new(
        rgba.as_rgba(),
        info.width as usize,
        info.height as usize,
    ))
    .ok_or("source has no opaque rectangle")?;
    let (width, height) = (rect.width as u32, rect.height as u32);
    let mut rgb = vec![0u8; width as usize * height as usize * 3];
    let options = ConvertOptions::forbid_lossy().with_alpha_policy(AlphaPolicy::DiscardIfOpaque);
    let mut converter = RowConverter::new_explicit(
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::RGB8_SRGB,
        &options,
    )?;
    converter.convert_rows(
        &rgba[(rect.y * info.width as usize + rect.x) * 4..],
        info.width as usize * 4,
        &mut rgb,
        width as usize * 3,
        width,
        height,
    )?;
    let png = zenpng::encode_rgb8(
        imgref::ImgRef::new(rgb.as_rgb(), width as usize, height as usize),
        None,
        &zenpng::EncodeConfig::default().with_compression(zenpng::Compression::Fastest),
        &enough::Unstoppable,
        &enough::Unstoppable,
    )?;
    let back = zenpng::decode(
        &png,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )?;
    let check = back.pixels.to_rgb8().copy_to_contiguous_bytes();
    if check != rgb || back.info.width != width || back.info.height != height {
        return Err("prepared PNG readback differs".into());
    }
    fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output)?
        .write_all(&png)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
        "input":input,"input_sha256":sha(&bytes),"input_rgba_sha256":sha(&rgba),
        "output":output,"output_sha256":sha(&png),"output_rgb_sha256":sha(&rgb),
        "width":width,"height":height,"crop":{"x":rect.x,"y":rect.y,"width":rect.width,"height":rect.height},
        "recipe":"maximal opaque rectangle; alpha discard verified by RowConverter; zenpng RGB8",
        "driver_sha256":sha(&fs::read(std::env::current_exe()?)?)}))?
    );
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

/// Largest all-opaque rectangle, with deterministic top/left tie breaking.
/// Row iteration preserves arbitrary source strides.
fn largest_opaque(pixels: imgref::ImgRef<'_, rgb::RGBA8>) -> Option<Rect> {
    let mut heights = vec![0; pixels.width()];
    let mut best: Option<Rect> = None;
    for (y, row) in pixels.rows().enumerate() {
        for (height, pixel) in heights.iter_mut().zip(row) {
            *height = if pixel.a == 255 { *height + 1 } else { 0 };
        }
        let mut stack: Vec<(usize, usize)> = Vec::with_capacity(pixels.width());
        for x in 0..=pixels.width() {
            let height = heights.get(x).copied().unwrap_or(0);
            let mut start = x;
            while stack.last().is_some_and(|&(_, h)| h > height) {
                let (left, h) = stack.pop().expect("nonempty histogram stack");
                let candidate = Rect {
                    x: left,
                    y: y + 1 - h,
                    width: x - left,
                    height: h,
                };
                if best.is_none_or(|b| {
                    candidate.width * candidate.height > b.width * b.height
                        || (candidate.width * candidate.height == b.width * b.height
                            && (candidate.y, candidate.x, candidate.height, candidate.width)
                                < (b.y, b.x, b.height, b.width))
                }) {
                    best = Some(candidate);
                }
                start = left;
            }
            if height > 0 && stack.last().is_none_or(|&(_, h)| h < height) {
                stack.push((start, height));
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_three_by_three_alpha_mask_matches_exhaustive_strided_oracle() {
        for mask in 0..512 {
            let mut pixels = vec![rgb::RGBA8::new(0, 0, 0, 0); 15];
            for y in 0..3 {
                for x in 0..3 {
                    pixels[y * 5 + x].a = if mask & (1 << (y * 3 + x)) != 0 {
                        255
                    } else {
                        0
                    };
                }
            }
            let found = largest_opaque(imgref::ImgRef::new_stride(&pixels, 3, 3, 5));
            let mut maximum = 0;
            for y0 in 0..3 {
                for y1 in y0 + 1..=3 {
                    for x0 in 0..3 {
                        for x1 in x0 + 1..=3 {
                            if (y0..y1).all(|y| (x0..x1).all(|x| pixels[y * 5 + x].a == 255)) {
                                maximum = maximum.max((y1 - y0) * (x1 - x0));
                            }
                        }
                    }
                }
            }
            assert_eq!(
                found.map_or(0, |r| r.width * r.height),
                maximum,
                "mask{mask}"
            );
            if let Some(r) = found {
                assert!(
                    (r.y..r.y + r.height)
                        .all(|y| (r.x..r.x + r.width).all(|x| pixels[y * 5 + x].a == 255))
                );
            }
        }
    }
}
