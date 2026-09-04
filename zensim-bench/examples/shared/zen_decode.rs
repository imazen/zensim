//! `zen_decode` — THE decode owner for zensim-bench's corpus extractors.
//!
//! One implementation of "bytes on disk → packed RGB8", built entirely from
//! imazen codecs, shared by every example that reads a corpus image. Per the
//! workspace rule *IMAZEN-ONLY IMAGING/CODEC SOFTWARE* (`~/work/zen/CLAUDE.md`)
//! and *NO DUPLICATE IMPLEMENTATIONS* (`zensim/CLAUDE.md`): nothing here may
//! call a third-party imaging crate, and no caller may re-implement any of it.
//!
//! ## Why this module exists (2026-09-04)
//!
//! `extract_features_372col` decoded with `image::open(..).ok()?` — a
//! third-party crate, behind a `?` that turned every decode failure into a
//! silently dropped row. Two measured consequences:
//!
//! 1. **Silent format drop.** `image` 0.25's default features carry no AVIF and
//!    no JXL decoder, so `zenavif-*` (34,001 rows) and `zenjxl-*` (26,362 rows)
//!    — **30.8 % of the safesyn corpus** — vanished from any extraction without
//!    a word. That is the "graceful skip" the repo bans outright.
//! 2. **Wrong pixels, no error.** `image` decodes an **XYB** JPEG as an ordinary
//!    YCbCr JPEG and never applies the inverse XYB→sRGB transform, so the
//!    `zenjpeg-420-xyb-*` families decoded to garbage that still looked like a
//!    valid image (a probe cell read `0.659` stored vs `2875.0` fresh). zenjpeg
//!    detects XYB from the RGB component IDs and runs the inverse transform
//!    (`zenjpeg/src/decode/parser/mod.rs:1109`), so routing through it is not a
//!    style preference — it is the difference between right and wrong pixels.
//!
//! ## Design
//!
//! * **Format detection has one owner**: `zencodec::ImageFormatRegistry::common()`
//!   sniffs magic bytes. The file extension is *never* trusted for dispatch; it
//!   is used only to enrich an error message. (A `.png` written by the decode
//!   cache and a `.png` that is really a JPEG both decode correctly.)
//! * **Decode has one owner per format**: the `zencodec`
//!   `DecoderConfig → job() → decoder() → decode()` trait path, exactly as
//!   `verify_bitstream_decode` uses it, so a corpus row extracted here sees the
//!   same pixels that tool verifies.
//! * **Failure is loud.** Every entry point returns `Result`; there is no
//!   `Option` and no fallback decoder. An unsupported or corrupt input is an
//!   error the caller must handle explicitly — it can never become a skipped
//!   row.
//!
//! ## Why `zencodecs` (plural) is not the dependency
//!
//! `zencodecs` is the workspace's unified detect+dispatch registry and would be
//! the natural single call. It is not used *here* because its dependency set
//! (`zenpng 0.2`, `zenjxl-decoder 0.4`, a rev-pinned `zenavif-parse`, `heic`,
//! `zentiff`, …) does not resolve against zensim-bench's existing
//! `[patch.crates-io]` table, which pins `zenpng 0.1.4` and the sibling
//! codecs this repo already builds. This module is the same dispatch over the
//! same `zencodec` traits, scoped to the five formats the corpora contain.
//! If zensim-bench's patch table is ever aligned with zenpipe's, replace the
//! body of [`decode_rgb8_bytes`] with a `zencodecs::DecodeRequest` call and
//! delete the per-format arms — the signature is deliberately the same shape.

#![allow(dead_code)] // each example uses a different subset of the surface

use std::borrow::Cow;
use std::fmt;
use std::path::Path;

use zencodec::decode::{Decode, DecodeJob, DecoderConfig};
use zencodec::{ImageFormat, ImageFormatRegistry};
use zenpixels::{PixelBuffer, PixelDescriptor};

/// True when this build has an imazen decoder wired for `format`.
///
/// AVIF / JXL / WebP are behind cargo features because their dependency graphs
/// (rav1d, the JXL decoder, libwebp-free zenwebp) are heavy and not every
/// consumer needs them. A format that is *detected but not built* returns
/// [`DecodeError::UnsupportedFormat`] — it is never treated as a decode
/// failure, and never silently skipped.
pub fn is_supported(format: ImageFormat) -> bool {
    match format {
        ImageFormat::Jpeg | ImageFormat::Png => true,
        ImageFormat::WebP => cfg!(feature = "verify-webp"),
        ImageFormat::Avif => cfg!(feature = "verify-avif"),
        ImageFormat::Jxl => cfg!(feature = "verify-jxl"),
        _ => false,
    }
}

/// A decoded image: dimensions plus tightly packed RGB8 bytes
/// (`len == width * height * 3`).
#[derive(Clone)]
pub struct DecodedRgb8 {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Every way decoding can fail. All variants are hard errors — none of them is
/// a reason to skip a row.
#[derive(Debug)]
pub enum DecodeError {
    /// The file could not be read.
    Io { path: String, source: std::io::Error },
    /// `zencodec`'s magic-byte registry did not recognise the leading bytes.
    Undetectable { path: String, ext: String, head: String },
    /// The format was detected, but this build has no imazen decoder wired for
    /// it. Distinct from `Undetectable` so a missing arm is never mistaken for
    /// a corrupt file.
    UnsupportedFormat { path: String, format: &'static str },
    /// The codec rejected the bitstream.
    Codec { path: String, format: &'static str, message: String },
    /// Decode succeeded but the pixel buffer is in a layout this module cannot
    /// flatten to RGB8.
    PixelLayout { path: String, message: String },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Io { path, source } => write!(f, "read {path}: {source}"),
            DecodeError::Undetectable { path, ext, head } => write!(
                f,
                "{path}: zencodec could not detect an image format from the magic bytes \
                 (extension {ext:?}, first bytes {head})"
            ),
            DecodeError::UnsupportedFormat { path, format } => write!(
                f,
                "{path}: detected {format}, which has no imazen decoder wired into \
                 zen_decode in this build — add the arm, never fall back to a \
                 third-party decoder"
            ),
            DecodeError::Codec { path, format, message } => {
                write!(f, "{path}: {format} decode failed: {message}")
            }
            DecodeError::PixelLayout { path, message } => {
                write!(f, "{path}: unsupported pixel layout: {message}")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Deliberately terse: `expect_err` on a decode requires `Debug` on the OK
/// side, and printing a megabyte of pixels into a test failure helps nobody.
impl fmt::Debug for DecodedRgb8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DecodedRgb8({}x{}, {} bytes)", self.width, self.height, self.pixels.len())
    }
}

/// Decode the file at `path` to packed RGB8 through imazen codecs.
///
/// Detection is by magic bytes, not by extension. Fails loudly on any error.
pub fn decode_rgb8_path(path: &Path) -> Result<DecodedRgb8, DecodeError> {
    let bytes = std::fs::read(path).map_err(|e| DecodeError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    decode_rgb8_bytes(&bytes, &path.display().to_string())
}

/// Decode `bytes` to packed RGB8. `label` is used only in error messages.
pub fn decode_rgb8_bytes(bytes: &[u8], label: &str) -> Result<DecodedRgb8, DecodeError> {
    let format = ImageFormatRegistry::common().detect(bytes).ok_or_else(|| {
        let head: String = bytes
            .iter()
            .take(8)
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" ");
        DecodeError::Undetectable {
            path: label.to_string(),
            ext: Path::new(label)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_string(),
            head,
        }
    })?;

    match format {
        // zenjpeg is XYB-aware: it detects the XYB component IDs and runs the
        // inverse XYB→sRGB transform in its output stage. This is the arm the
        // `image` crate got silently wrong.
        ImageFormat::Jpeg => decode_jpeg(bytes, label),
        ImageFormat::Png => decode_png(bytes, label),
        ImageFormat::WebP => decode_webp(bytes, label),
        ImageFormat::Avif => decode_avif(bytes, label),
        ImageFormat::Jxl => decode_jxl(bytes, label),
        other => Err(DecodeError::UnsupportedFormat {
            path: label.to_string(),
            format: format_name(other),
        }),
    }
}

fn format_name(f: ImageFormat) -> &'static str {
    match f {
        ImageFormat::Jpeg => "JPEG",
        ImageFormat::Png => "PNG",
        ImageFormat::Gif => "GIF",
        ImageFormat::WebP => "WebP",
        ImageFormat::Avif => "AVIF",
        ImageFormat::Jxl => "JXL",
        ImageFormat::Heic => "HEIC",
        ImageFormat::Bmp => "BMP",
        ImageFormat::Tiff => "TIFF",
        ImageFormat::Ico => "ICO",
        ImageFormat::Pnm => "PNM",
        ImageFormat::Farbfeld => "farbfeld",
        ImageFormat::Qoi => "QOI",
        ImageFormat::Pdf => "PDF",
        ImageFormat::Exr => "EXR",
        ImageFormat::Hdr => "Radiance HDR",
        ImageFormat::Tga => "TGA",
        ImageFormat::Jp2 => "JPEG 2000",
        ImageFormat::Dng => "DNG",
        ImageFormat::Raw => "camera RAW",
        ImageFormat::Svg => "SVG",
        _ => "unknown/custom",
    }
}

macro_rules! zc_decode {
    ($cfg:expr, $bytes:expr, $label:expr, $fmt:expr) => {{
        let out = $cfg
            .job()
            .decoder(Cow::Borrowed($bytes), &[])
            .map_err(|e| DecodeError::Codec {
                path: $label.to_string(),
                format: $fmt,
                message: format!("job: {e}"),
            })?
            .decode()
            .map_err(|e| DecodeError::Codec {
                path: $label.to_string(),
                format: $fmt,
                message: format!("decode: {e}"),
            })?;
        let pb = out.into_buffer();
        let (w, h) = (pb.width(), pb.height());
        let pixels = pixelbuffer_to_rgb8(&pb).map_err(|message| DecodeError::PixelLayout {
            path: $label.to_string(),
            message,
        })?;
        Ok(DecodedRgb8 { width: w, height: h, pixels })
    }};
}

fn decode_jpeg(bytes: &[u8], label: &str) -> Result<DecodedRgb8, DecodeError> {
    zc_decode!(zenjpeg::JpegDecoderConfig::new(), bytes, label, "JPEG")
}

#[cfg(feature = "verify-webp")]
fn decode_webp(bytes: &[u8], label: &str) -> Result<DecodedRgb8, DecodeError> {
    zc_decode!(zenwebp::zencodec::WebpDecoderConfig::new(), bytes, label, "WebP")
}
#[cfg(not(feature = "verify-webp"))]
fn decode_webp(_bytes: &[u8], label: &str) -> Result<DecodedRgb8, DecodeError> {
    Err(DecodeError::UnsupportedFormat { path: label.to_string(), format: "WebP" })
}

#[cfg(feature = "verify-avif")]
fn decode_avif(bytes: &[u8], label: &str) -> Result<DecodedRgb8, DecodeError> {
    zc_decode!(zenavif::AvifDecoderConfig::new(), bytes, label, "AVIF")
}
#[cfg(not(feature = "verify-avif"))]
fn decode_avif(_bytes: &[u8], label: &str) -> Result<DecodedRgb8, DecodeError> {
    Err(DecodeError::UnsupportedFormat { path: label.to_string(), format: "AVIF" })
}

#[cfg(feature = "verify-jxl")]
fn decode_jxl(bytes: &[u8], label: &str) -> Result<DecodedRgb8, DecodeError> {
    zc_decode!(zenjxl::JxlDecoderConfig::new(), bytes, label, "JXL")
}
#[cfg(not(feature = "verify-jxl"))]
fn decode_jxl(_bytes: &[u8], label: &str) -> Result<DecodedRgb8, DecodeError> {
    Err(DecodeError::UnsupportedFormat { path: label.to_string(), format: "JXL" })
}

/// PNG goes through `zenpng`'s native entry point rather than the `zencodec`
/// trait path: this repo pins `zenpng 0.1.4`, whose `zencodec` adapter landed
/// in 0.2. Same decoder either way.
fn decode_png(bytes: &[u8], label: &str) -> Result<DecodedRgb8, DecodeError> {
    let out = zenpng::decode(bytes, &zenpng::PngDecodeConfig::default(), &enough::Unstoppable)
        .map_err(|e| DecodeError::Codec {
            path: label.to_string(),
            format: "PNG",
            message: format!("decode: {e}"),
        })?;
    let (w, h) = (out.info.width, out.info.height);
    let pixels = pixelbuffer_to_rgb8(&out.pixels).map_err(|message| DecodeError::PixelLayout {
        path: label.to_string(),
        message,
    })?;
    Ok(DecodedRgb8 { width: w, height: h, pixels })
}

/// Flatten a possibly-strided `PixelBuffer` (RGB8 or RGBA8) to tightly packed
/// RGB8, dropping alpha. Rows are read via the buffer's stride — never as one
/// flat slice — per the workspace's strided-buffer rule.
fn pixelbuffer_to_rgb8(pb: &PixelBuffer) -> Result<Vec<u8>, String> {
    let desc = pb.descriptor();
    let w = pb.width() as usize;
    let h = pb.height() as usize;
    let slice = pb.as_slice();
    let stride = slice.stride();
    let data = slice.as_strided_bytes();

    if desc.layout_compatible(PixelDescriptor::RGB8)
        || desc.layout_compatible(PixelDescriptor::RGB8_SRGB)
    {
        let bpr = w * 3;
        let mut out = Vec::with_capacity(bpr * h);
        for row in 0..h {
            let start = row * stride;
            out.extend_from_slice(&data[start..start + bpr]);
        }
        Ok(out)
    } else if desc.layout_compatible(PixelDescriptor::RGBA8)
        || desc.layout_compatible(PixelDescriptor::RGBA8_SRGB)
    {
        let mut out = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            let start = row * stride;
            let src = &data[start..start + w * 4];
            for px in src.chunks_exact(4) {
                out.extend_from_slice(&px[..3]);
            }
        }
        Ok(out)
    } else {
        // Everything else — 10/12-bit AVIF (`Rgb16`), gray, f32, premultiplied,
        // channel-reordered — goes through the CANONICAL pixel-format owner,
        // `zenpixels_convert::RowConverter` → RGB8_SRGB. That is the exact path
        // zenmetrics' `decode.rs` and `verify_bitstream_decode` use, so a
        // bitstream decoded here matches what the fleet extractor saw; a
        // hand-rolled `v >> 8` here would be a second, divergent converter.
        //
        // MEASURED NEED (2026-09-04): the safesyn `zenavif-s5-e6` family decodes
        // to `Rgb16` on every probed row — 60 of 60 — so without this branch the
        // whole AVIF family is an error, not a row.
        use zenpixels_convert::converter::RowConverter;
        let dst_stride = w * 3;
        let mut out = vec![0u8; dst_stride * h];
        let mut conv = RowConverter::new(desc, PixelDescriptor::RGB8_SRGB)
            .map_err(|e| format!("cannot plan {desc:?} -> RGB8_SRGB: {e}"))?;
        conv.convert_rows(data, stride, &mut out, dst_stride, w as u32, h as u32)
            .map_err(|e| format!("row conversion {desc:?} -> RGB8_SRGB: {e}"))?;
        Ok(out)
    }
}
