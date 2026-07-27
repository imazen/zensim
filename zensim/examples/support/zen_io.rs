//! Shared decode/resize/encode helpers for phase-2 example tooling.
//!
//! Uses zen* crates (zenpng, zenjpeg, zenresize) instead of the `image`
//! crate for all NEW example/bench tooling, per project convention. Lives
//! under `examples/support/` (not directly in `examples/`) specifically so
//! Cargo's example auto-discovery does NOT treat it as its own example
//! binary -- only `examples/*.rs` (not subdirectories) are auto-discovered.
//! Included into each consumer via `#[path = "support/zen_io.rs"] mod zen_io;`
//! (examples can't `use` each other directly; this is the standard
//! same-crate example-sharing pattern).
//!
//! `#[allow(dead_code)]`: not every example uses every helper.
#![allow(dead_code)]

use enough::Unstoppable;

/// Decode a PNG or JPEG file (by extension) to packed RGB8 + dimensions.
/// Alpha (if present) is dropped -- all phase-2 tooling here works in RGB8.
pub fn decode_rgb8(path: &std::path::Path) -> (Vec<[u8; 3]>, usize, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" => decode_png_rgb8(&bytes),
        "jpg" | "jpeg" => decode_jpeg_rgb8(&bytes),
        "bmp" => decode_bmp_rgb8(&bytes),
        other => panic!("unsupported extension {other:?} for {path:?}"),
    }
}

fn decode_bmp_rgb8(bytes: &[u8]) -> (Vec<[u8; 3]>, usize, usize) {
    // BMP (the LIVE-R2 corpus) via the `image` crate — already a dev-dep and
    // used by the other examples. Deliberately NOT a zen crate here: the zen
    // BMP codec (zenbitmaps) is an unpublished sibling, and a path/git dep on
    // it breaks the manifest on CI (siblings aren't checked out). This is a
    // research example, so the pragmatic, CI-safe decode wins.
    let img = image::load_from_memory(bytes)
        .expect("image decode bmp")
        .to_rgb8();
    let (w, h) = img.dimensions();
    let rgb: Vec<[u8; 3]> = img.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    (rgb, w as usize, h as usize)
}

fn decode_png_rgb8(bytes: &[u8]) -> (Vec<[u8; 3]>, usize, usize) {
    use zenpixels::ChannelType;
    let cfg = zenpng::PngDecodeConfig::default();
    let out = zenpng::decode(bytes, &cfg, &Unstoppable).expect("zenpng decode");
    let (w, h) = (out.info.width as usize, out.info.height as usize);
    let desc = out.pixels.descriptor();
    let channels = desc.channels() as usize;
    let has_alpha = desc.has_alpha();
    let slice = out.pixels.as_slice();
    let samples_per_row = w * channels;
    // Flatten rows to one u8 sample buffer. 16-bit PNGs (e.g. the CID22
    // validation set) narrow each native-endian u16 sample by rounded
    // v*255/65535 — the same convention `image::open().to_rgb8()` applies
    // on the v1-arm extraction path.
    let mut samples: Vec<u8> = Vec::with_capacity(h * samples_per_row);
    match desc.channel_type() {
        ChannelType::U8 => {
            for y in 0..h as u32 {
                samples.extend_from_slice(&slice.row(y)[..samples_per_row]);
            }
        }
        ChannelType::U16 => {
            for y in 0..h as u32 {
                for pair in slice.row(y).chunks_exact(2).take(samples_per_row) {
                    let v = u16::from_ne_bytes([pair[0], pair[1]]) as u32;
                    samples.push(((v * 255 + 32767) / 65535) as u8);
                }
            }
        }
        other => panic!("unsupported PNG channel type {other:?}"),
    }
    let mut rgb = Vec::with_capacity(w * h);
    match (channels, has_alpha) {
        (4, true) => {
            for px in samples.chunks_exact(4) {
                rgb.push([px[0], px[1], px[2]]);
            }
        }
        (3, false) => {
            for px in samples.chunks_exact(3) {
                rgb.push([px[0], px[1], px[2]]);
            }
        }
        (1, false) => {
            for &g in &samples {
                rgb.push([g, g, g]);
            }
        }
        (2, true) => {
            for px in samples.chunks_exact(2) {
                rgb.push([px[0], px[0], px[0]]);
            }
        }
        other => panic!("unsupported PNG channel layout {other:?}"),
    }
    (rgb, w, h)
}

fn decode_jpeg_rgb8(bytes: &[u8]) -> (Vec<[u8; 3]>, usize, usize) {
    use zenjpeg::decoder::Decoder;
    let result = Decoder::new()
        .decode(bytes, Unstoppable)
        .expect("zenjpeg decode");
    let (w, h) = result.dimensions();
    let (w, h) = (w as usize, h as usize);
    let px = result.pixels_u8().expect("u8 jpeg output");
    let rgb: Vec<[u8; 3]> = px.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    (rgb, w, h)
}

/// Resize packed RGB8 to `(target_w, target_h)` via zenresize's Lanczos
/// kernel (a principled kernel per CLAUDE.md's resampling guidance --
/// "Lanczos when sharpness matters more"; used here because several target
/// sizes UPSCALE the source, where Lanczos's sharper response is the more
/// defensible default for a synthetic timing/bounds fixture).
pub fn resize_rgb8(
    pixels: &[[u8; 3]],
    w: usize,
    h: usize,
    target_w: usize,
    target_h: usize,
) -> Vec<[u8; 3]> {
    if w == target_w && h == target_h {
        return pixels.to_vec();
    }
    use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};
    let flat: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let config = ResizeConfig::builder(w as u32, h as u32, target_w as u32, target_h as u32)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGB8_SRGB)
        .build();
    let out = Resizer::new(&config).resize(&flat);
    out.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Encode packed RGB8 to JPEG at `quality` (0-100) with 4:2:0 chroma
/// subsampling (the aggressive default -- deliberately chosen over 4:4:4
/// since chroma-subsampling artifacts at hard chroma edges are exactly the
/// D1/masked/IW pathology this tooling probes for).
pub fn encode_jpeg_q(pixels: &[[u8; 3]], w: usize, h: usize, quality: u8) -> Vec<u8> {
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
    let flat: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("zenjpeg encoder init");
    enc.push_packed(&flat, Unstoppable).expect("zenjpeg push");
    enc.finish().expect("zenjpeg finish")
}

/// Decode a PNG to packed RGB u16 (native 16-bit samples preserved — for
/// PQ/HLG code-value containers like the kadis-hdr cICP-spliced PNGs);
/// 8-bit inputs widen by `v * 257`. Alpha dropped. Panics on failure.
pub fn decode_rgb16(path: &std::path::Path) -> (Vec<[u16; 3]>, usize, usize) {
    use zenpixels::ChannelType;
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let cfg = zenpng::PngDecodeConfig::default();
    let out = zenpng::decode(&bytes, &cfg, &Unstoppable).expect("zenpng decode");
    let (w, h) = (out.info.width as usize, out.info.height as usize);
    let desc = out.pixels.descriptor();
    let channels = desc.channels() as usize;
    let slice = out.pixels.as_slice();
    let samples_per_row = w * channels;
    let mut samples: Vec<u16> = Vec::with_capacity(h * samples_per_row);
    match desc.channel_type() {
        ChannelType::U8 => {
            for y in 0..h as u32 {
                for &v in &slice.row(y)[..samples_per_row] {
                    samples.push(v as u16 * 257);
                }
            }
        }
        ChannelType::U16 => {
            for y in 0..h as u32 {
                for pair in slice.row(y).chunks_exact(2).take(samples_per_row) {
                    samples.push(u16::from_ne_bytes([pair[0], pair[1]]));
                }
            }
        }
        other => panic!("unsupported PNG channel type {other:?}"),
    }
    let mut rgb = Vec::with_capacity(w * h);
    match channels {
        4 => {
            for px in samples.chunks_exact(4) {
                rgb.push([px[0], px[1], px[2]]);
            }
        }
        3 => {
            for px in samples.chunks_exact(3) {
                rgb.push([px[0], px[1], px[2]]);
            }
        }
        1 => {
            for &g in &samples {
                rgb.push([g, g, g]);
            }
        }
        2 => {
            for px in samples.chunks_exact(2) {
                rgb.push([px[0], px[0], px[0]]);
            }
        }
        other => panic!("unsupported channel count {other}"),
    }
    (rgb, w, h)
}
