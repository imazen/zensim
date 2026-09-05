//! JXL-floor-ladder tile generator — decode + crop/downscale + re-encode
//! through imazen owners only, for the D-peaks presentation lane
//! (`benchmarks/d_peaks_jxl_ladders_2026-09-05.md`).
//!
//! # Why this exists
//!
//! The presentation page needs PNG tiles (a full-frame downscale for context
//! and a native-resolution crop so compression artifacts are actually
//! visible) built from the dial-grid's reference and JXL-distorted PNGs. Per
//! the USER RULE **"IMAZEN-ONLY IMAGING/CODEC SOFTWARE"**
//! (`~/work/zen/CLAUDE.md`, 2026-09-02), that decode/resize step must not
//! shell ImageMagick or any non-imazen tool — only `zenpng` (decode/encode)
//! and `zenresize` (Mitchell downscale) do the pixel work here. The orchestrating
//! script (`scripts/dpeaks_jxl_ladders_page.py`) calls this binary once per
//! tile and only composes the *already-decoded* PNGs into an HTML page —
//! no ImageMagick anywhere in this lane.
//!
//! This is a sibling of `m3_fixture_gen.rs` (same decode/encode/downscale
//! primitives, same `m3-fixtures` feature gate) with a `crop` mode added: a
//! native 1:1 pixel-exact crop, no resampling, so a detail region shows the
//! codec's actual output bytes rather than a resample of them.
//!
//! # Usage
//!
//! ```text
//! ladder_tile_gen full --in <ref.png> --out <small.png> --max <N>
//! ladder_tile_gen crop --in <ref.png> --out <crop.png> --x X --y Y --w W --h H
//! ```
//!
//! `full` fits inside an `N x N` box preserving aspect ratio (Mitchell,
//! B=C=1/3) and refuses to upscale. `crop` extracts `[x, x+w) x [y, y+h)`
//! verbatim (no resampling) and fails loud if the window falls outside the
//! decoded image bounds — a silently-clamped crop would misreport its own
//! origin, and the crop origin is part of what the page reports.
//!
//! # Build
//!
//! ```text
//! cargo build --release -p zensim-bench --example ladder_tile_gen \
//!     --features m3-fixtures
//! ```

use std::path::PathBuf;

use enough::Unstoppable;
use zenpng::{EncodeConfig, PngDecodeConfig, decode, encode_rgb8};
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// An RGB8 image: packed `w * h * 3` bytes.
struct Rgb8 {
    w: u32,
    h: u32,
    px: Vec<u8>,
}

/// Decode a PNG to packed RGB8 through the imazen owner (`zenpng`), normalising
/// whatever native colour type the file carries via `zenpixels-convert` —
/// identical helper to `m3_fixture_gen::read_png_rgb8` (this example does not
/// import that one back since examples are separate binaries; see that
/// file's own doc comment for the stride-correctness rationale, unchanged
/// here).
fn read_png_rgb8(path: &PathBuf) -> Res<Rgb8> {
    use zenpixels_convert::PixelBufferConvertTypedExt;
    let bytes = std::fs::read(path)?;
    let out = decode(&bytes, &PngDecodeConfig::default(), &Unstoppable)?;
    let (w, h) = (out.info.width, out.info.height);
    let rgb = out.pixels.to_rgb8();
    let view = rgb.as_imgref();
    let mut px: Vec<u8> = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for row in view.rows() {
        for p in row {
            px.extend_from_slice(&[p.r, p.g, p.b]);
        }
    }
    let want = (w as usize) * (h as usize) * 3;
    if px.len() != want {
        return Err(format!(
            "{}: decoded {} bytes, expected {w}x{h}x3 = {want}",
            path.display(),
            px.len()
        )
        .into());
    }
    Ok(Rgb8 { w, h, px })
}

fn write_png_rgb8(path: &PathBuf, img: &Rgb8) -> Res<()> {
    let view = imgref::ImgRef::new(
        rgb::FromSlice::as_rgb(&img.px[..]),
        img.w as usize,
        img.h as usize,
    );
    let bytes = encode_rgb8(
        view,
        None,
        &EncodeConfig::default(),
        &Unstoppable,
        &Unstoppable,
    )?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Mitchell-Netravali downscale to fit inside `max x max`, aspect preserved.
/// Refuses to upscale (every ladder tile source here is already >= 512px on
/// its long side, so this is expected to always fire, not a corner case).
fn mitchell_fit(src: &Rgb8, max: u32) -> Res<Rgb8> {
    let scale = f64::from(max) / f64::from(src.w.max(src.h));
    if scale >= 1.0 {
        return Err(format!(
            "refusing to upscale: source is {}x{}, target box {max}x{max}",
            src.w, src.h
        )
        .into());
    }
    let out_w = ((f64::from(src.w) * scale).round() as u32).max(1);
    let out_h = ((f64::from(src.h) * scale).round() as u32).max(1);

    let mut rgba = Vec::with_capacity(src.px.len() / 3 * 4);
    for p in src.px.chunks_exact(3) {
        rgba.extend_from_slice(&[p[0], p[1], p[2], 255]);
    }
    let cfg = ResizeConfig::builder(src.w, src.h, out_w, out_h)
        .filter(Filter::Mitchell)
        .format(PixelDescriptor::RGBA8_SRGB)
        .build();
    let out = Resizer::new(&cfg).resize(&rgba);

    let px: Vec<u8> = out
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    Ok(Rgb8 {
        w: out_w,
        h: out_h,
        px,
    })
}

/// Native 1:1 pixel crop, no resampling. Fails loud on an out-of-bounds
/// window rather than clamping — a silently-shrunk crop would make the
/// reported crop origin/size wrong.
fn crop_native(src: &Rgb8, x: u32, y: u32, w: u32, h: u32) -> Res<Rgb8> {
    if x.saturating_add(w) > src.w || y.saturating_add(h) > src.h {
        return Err(format!(
            "crop window ({x},{y})+{w}x{h} exceeds source bounds {}x{}",
            src.w, src.h
        )
        .into());
    }
    let mut px = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for row in y..(y + h) {
        let row_start = (row as usize) * (src.w as usize) * 3 + (x as usize) * 3;
        let row_end = row_start + (w as usize) * 3;
        px.extend_from_slice(&src.px[row_start..row_end]);
    }
    Ok(Rgb8 { w, h, px })
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn need(args: &[String], key: &str) -> Res<String> {
    arg(args, key).ok_or_else(|| format!("missing required argument {key}").into())
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("");
    let inp = PathBuf::from(need(&args, "--in")?);
    let outp = PathBuf::from(need(&args, "--out")?);

    match mode {
        "full" => {
            let max: u32 = need(&args, "--max")?.parse()?;
            let src = read_png_rgb8(&inp)?;
            let dst = mitchell_fit(&src, max)?;
            write_png_rgb8(&outp, &dst)?;
            eprintln!(
                "ladder_tile_gen full: {}x{} -> {}x{} (Mitchell, zenresize) -> {}",
                src.w,
                src.h,
                dst.w,
                dst.h,
                outp.display()
            );
        }
        "crop" => {
            let x: u32 = need(&args, "--x")?.parse()?;
            let y: u32 = need(&args, "--y")?.parse()?;
            let w: u32 = need(&args, "--w")?.parse()?;
            let h: u32 = need(&args, "--h")?.parse()?;
            let src = read_png_rgb8(&inp)?;
            let dst = crop_native(&src, x, y, w, h)?;
            write_png_rgb8(&outp, &dst)?;
            eprintln!(
                "ladder_tile_gen crop: {}x{} native crop ({x},{y})+{w}x{h} -> {}",
                src.w,
                src.h,
                outp.display()
            );
        }
        other => {
            return Err(format!(
                "unknown mode {other:?}; expected `full` or `crop`.\n\
                 usage: ladder_tile_gen full --in <ref.png> --out <small.png> --max <N>\n\
                        ladder_tile_gen crop --in <ref.png> --out <crop.png> --x X --y Y --w W --h H"
            )
            .into());
        }
    }
    Ok(())
}
