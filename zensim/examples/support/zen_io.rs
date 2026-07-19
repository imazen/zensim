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
        other => panic!("unsupported extension {other:?} for {path:?}"),
    }
}

fn decode_png_rgb8(bytes: &[u8]) -> (Vec<[u8; 3]>, usize, usize) {
    use zenpixels::ChannelType;
    let cfg = zenpng::PngDecodeConfig::default();
    let out = zenpng::decode(bytes, &cfg, &Unstoppable).expect("zenpng decode");
    let (w, h) = (out.info.width as usize, out.info.height as usize);
    let desc = out.pixels.descriptor();
    assert_eq!(
        desc.channel_type(),
        ChannelType::U8,
        "only 8-bit PNG supported by this helper"
    );
    let slice = out.pixels.as_slice();
    let channels = desc.channels();
    let has_alpha = desc.has_alpha();
    let mut rgb = Vec::with_capacity(w * h);
    for y in 0..h as u32 {
        let row = slice.row(y);
        match (channels, has_alpha) {
            (4, true) => {
                for px in row.chunks_exact(4).take(w) {
                    rgb.push([px[0], px[1], px[2]]);
                }
            }
            (3, false) => {
                for px in row.chunks_exact(3).take(w) {
                    rgb.push([px[0], px[1], px[2]]);
                }
            }
            (1, false) => {
                for &g in row.iter().take(w) {
                    rgb.push([g, g, g]);
                }
            }
            (2, true) => {
                for px in row.chunks_exact(2).take(w) {
                    rgb.push([px[0], px[0], px[0]]);
                }
            }
            other => panic!("unsupported PNG channel layout {other:?}"),
        }
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
