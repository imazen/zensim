//! M3/M3a coherence-fixture generator — the IMAZEN owner for the size × quality
//! grid `scripts/run_full_eval.sh` measures diffmap coherence on.
//!
//! # Why this exists
//!
//! Until 2026-09-02 `run_full_eval.sh` shelled **ImageMagick** for both halves
//! of this grid: `-filter Mitchell -resize NxN` for the size axis and
//! `-quality Q` (ImageMagick's bundled libjpeg) for the quality axis. Under the
//! USER RULE **"IMAZEN-ONLY IMAGING/CODEC SOFTWARE"** (`~/work/zen/CLAUDE.md`,
//! 2026-09-02) that is a rule violation in the worst possible place: M3a is a
//! first-class **model-selection** input (`docs/WAVE_PLAYBOOK.md` step 6, the
//! `freeze_check --select` tie-break), so a foreign JPEG encoder was sitting
//! inside the loop that picks which zensim model ships.
//!
//! Both halves now run on the imazen owners:
//!
//! | axis | owner |
//! |---|---|
//! | PNG decode / encode | `zenpng` |
//! | Mitchell downscale | `zenresize` (`Filter::Mitchell`, B=C=1/3) |
//! | JPEG encode | `zenjpeg` |
//!
//! # ⚠ ERA HAZARD — read before regenerating anything
//!
//! **This binary does NOT reproduce the ImageMagick-era fixtures, and must not
//! be pointed at a directory that holds them.** A zenjpeg q50 encode is not
//! ImageMagick-libjpeg's q50 — different quant tables, different chroma
//! downsampling, different trellis — so an M3a number measured on a fixture
//! this tool made is **not comparable** to one measured on the old fixtures.
//!
//! The 48 fixtures under
//! `/mnt/v/output/zensim/diffmap-coherence-2026-07-18/` are ImageMagick-era and
//! are deliberately **left alone**: every M3/M3a value in the record, on the
//! board, and in `benchmarks/` was measured against them. Regenerating in place
//! would silently re-base the entire M3a axis.
//!
//! So the fixture root is **era-stamped**, exactly as the 372 feature roots are
//! (`2026-05-15-full-features` vs `2026-08-30-full-features-372`). A new era
//! gets a new directory; the caller names it. `run_full_eval.sh` keeps pointing
//! at the ImageMagick-era root by default, so no published number moves on this
//! commit.
//!
//! # Usage
//!
//! ```text
//! m3_fixture_gen resize --in <ref.png> --out <small.png> --max <N>
//! m3_fixture_gen jpeg   --in <ref.png> --out <out.jpg>    --quality <Q>
//! ```
//!
//! `resize` fits INSIDE an `N x N` box preserving aspect ratio, and **refuses to
//! upscale** (per CLAUDE.md's dense-sampling rule: "Skip upscaling. Synthetic
//! upscale features mislead any model that conditions on edge density").
//!
//! # Build
//!
//! ```text
//! cargo build --release -p zensim-bench --example m3_fixture_gen \
//!     --features m3-fixtures
//! ```

use std::path::PathBuf;

use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
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
/// whatever native colour type the file carries (gray / indexed / 16-bit / RGBA)
/// via `zenpixels-convert` — the same converter zenmetrics' decode path funnels
/// through, so a fixture matches what the fleet extractor would have seen.
fn read_png_rgb8(path: &PathBuf) -> Res<Rgb8> {
    use zenpixels_convert::PixelBufferConvertTypedExt;
    let bytes = std::fs::read(path)?;
    let out = decode(&bytes, &PngDecodeConfig::default(), &Unstoppable)?;
    let (w, h) = (out.info.width, out.info.height);
    let rgb = out.pixels.to_rgb8();
    // Iterate ROWS via `ImgRef`, never a flat buffer: a `PixelBuffer` may be
    // strided (SIMD-aligned padding), and `.rows()` is the stride-correct read.
    // Per CLAUDE.md's pixel-buffer rule, the tight case is just the case where
    // every row happens to abut.
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
///
/// Mitchell (B=C=1/3) is the filter the ImageMagick call named, and is what
/// CLAUDE.md's dense-sampling rule prescribes for a balanced blur/ringing
/// tradeoff. `zenresize` carries it as `Filter::Mitchell` (`filter.rs:52`).
///
/// The resize runs on RGBA8 because that is `zenresize`'s packed-8-bit
/// descriptor; the opaque alpha added here is dropped again on the way out, so
/// the result is exactly a 3-channel Mitchell downscale.
fn mitchell_fit(src: &Rgb8, max: u32) -> Res<Rgb8> {
    let scale = f64::from(max) / f64::from(src.w.max(src.h));
    if scale >= 1.0 {
        return Err(format!(
            "refusing to upscale: source is {}x{}, target box {max}x{max}. Upscaled fixtures \
             are over-smooth and mislead any model conditioning on edge density (CLAUDE.md, \
             dense-sampling rule)",
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

    let px: Vec<u8> = out.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    Ok(Rgb8 {
        w: out_w,
        h: out_h,
        px,
    })
}

/// Encode packed RGB8 to JPEG through `zenjpeg`.
///
/// `ChromaSubsampling::Quarter` is 4:2:0 — the web default, and what the
/// ImageMagick call it replaces also produced at these qualities. The M3 grid's
/// point is a realistic codec distortion at q20/50/75, not a 4:4:4 laboratory
/// encode.
fn encode_jpeg(img: &Rgb8, quality: f32) -> Res<Vec<u8>> {
    let cfg = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
    let mut enc = cfg.encode_from_bytes(img.w, img.h, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(&img.px, Unstoppable)?;
    Ok(enc.finish()?)
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
        "resize" => {
            let max: u32 = need(&args, "--max")?.parse()?;
            let src = read_png_rgb8(&inp)?;
            let dst = mitchell_fit(&src, max)?;
            write_png_rgb8(&outp, &dst)?;
            eprintln!(
                "m3_fixture_gen resize: {}x{} -> {}x{} (Mitchell, zenresize) -> {}",
                src.w,
                src.h,
                dst.w,
                dst.h,
                outp.display()
            );
        }
        "jpeg" => {
            let q: f32 = need(&args, "--quality")?.parse()?;
            if !(1.0..=100.0).contains(&q) {
                return Err(format!("--quality {q} out of range 1..=100").into());
            }
            let src = read_png_rgb8(&inp)?;
            let bytes = encode_jpeg(&src, q)?;
            std::fs::write(&outp, &bytes)?;
            eprintln!(
                "m3_fixture_gen jpeg: {}x{} q{q} (zenjpeg 4:2:0) -> {} ({} B)",
                src.w,
                src.h,
                outp.display(),
                bytes.len()
            );
        }
        other => {
            return Err(format!(
                "unknown mode {other:?}; expected `resize` or `jpeg`.\n\
                 usage: m3_fixture_gen resize --in <ref.png> --out <small.png> --max <N>\n\
                        m3_fixture_gen jpeg   --in <ref.png> --out <out.jpg>    --quality <Q>"
            )
            .into());
        }
    }
    Ok(())
}
