//! Pixel operations — `image`-crate-free RGBA8 canvas + ops.
//!
//! Provides:
//! - [`Bitmap`] — packed RGBA8 buffer (`Vec<u8>`, row-major, no padding).
//!   Public API of `zensim-regress` returns this type; consumers can
//!   inspect raw bytes via [`Bitmap::as_raw`] / [`Bitmap::into_raw`]
//!   or load/save PNG via [`Bitmap::open`] / [`Bitmap::save`].
//! - [`overlay`] — straight-alpha source-over composite.
//! - [`resize`] / [`resize_gray`] — `zenresize`-backed resampling.
//! - [`crop`], [`flip_horizontal`], [`flip_vertical`], [`rotate90`],
//!   [`rotate180`], [`rotate270`] — hand-rolled, byte-level.
//! - [`fill_rect`], [`draw_rect_border`] — drawing primitives.
//! - [`PngError`] — PNG IO error type, hides `zenpng` internals.
//!
//! **Privacy:** [`overlay`] and the basic ops are `pub(crate)`; only
//! [`Bitmap`] and [`PngError`] are `pub` (the public canvas types
//! exposed via `MontageOptions::render` etc.). Per the migration
//! contract: NO `zen*` crate types appear in the public surface.

use std::fs;
use std::io;
use std::path::Path;

use enough::Unstoppable;
use rgb::Rgba;
use zenpixels::PixelBuffer;
use zenpng::{EncodeConfig, PngDecodeConfig};

/// Resampling filter for [`resize`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ResampleFilter {
    /// Nearest-neighbor — pixelated, fastest.
    #[allow(dead_code)]
    Nearest,
    /// Bilinear — fast, soft.
    Triangle,
    /// Lanczos-3 — sharp, slower.
    Lanczos3,
}

impl ResampleFilter {
    fn to_zenresize(self) -> zenresize::Filter {
        match self {
            ResampleFilter::Nearest => zenresize::Filter::Box,
            ResampleFilter::Triangle => zenresize::Filter::Triangle,
            ResampleFilter::Lanczos3 => zenresize::Filter::Lanczos,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Bitmap — public RGBA8 canvas type
// ════════════════════════════════════════════════════════════════════════

/// Packed-RGBA8 bitmap. Row-major, no padding.
///
/// This is the canvas type returned by [`crate::diff_image::MontageOptions::render`]
/// and friends. To interop with other crates, use [`Bitmap::as_raw`]
/// to inspect the underlying RGBA bytes, or [`Bitmap::open`] /
/// [`Bitmap::save`] for PNG IO.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bitmap {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Bitmap {
    /// Create a new bitmap zero-filled (transparent black).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0u8; (width as usize) * (height as usize) * 4],
        }
    }

    /// Create a bitmap filled with a single color.
    pub fn from_pixel(width: u32, height: u32, color: [u8; 4]) -> Self {
        let n = (width as usize) * (height as usize);
        let mut pixels = Vec::with_capacity(n * 4);
        for _ in 0..n {
            pixels.extend_from_slice(&color);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Create a bitmap from raw RGBA8 bytes. Returns `None` if
    /// `pixels.len() != width * height * 4`.
    pub fn from_raw(width: u32, height: u32, pixels: Vec<u8>) -> Option<Self> {
        if pixels.len() != (width as usize) * (height as usize) * 4 {
            return None;
        }
        Some(Self {
            width,
            height,
            pixels,
        })
    }

    /// Create a bitmap from a function `f(x, y) -> [r, g, b, a]`.
    pub fn from_fn<F>(width: u32, height: u32, f: F) -> Self
    where
        F: Fn(u32, u32) -> [u8; 4],
    {
        let mut pixels = Vec::with_capacity((width as usize) * (height as usize) * 4);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&f(x, y));
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// `(width, height)` in pixels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Borrow the underlying RGBA8 byte buffer.
    pub fn as_raw(&self) -> &[u8] {
        &self.pixels
    }

    /// Mutable borrow of the underlying RGBA8 byte buffer.
    pub fn as_raw_mut(&mut self) -> &mut [u8] {
        &mut self.pixels
    }

    /// Take ownership of the underlying RGBA8 byte buffer.
    pub fn into_raw(self) -> Vec<u8> {
        self.pixels
    }

    /// Read pixel `(x, y)`. Returns transparent black if out of bounds.
    pub fn get_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.width || y >= self.height {
            return [0, 0, 0, 0];
        }
        let off = ((y * self.width + x) * 4) as usize;
        [
            self.pixels[off],
            self.pixels[off + 1],
            self.pixels[off + 2],
            self.pixels[off + 3],
        ]
    }

    /// Write pixel `(x, y)`. No-op if out of bounds.
    pub fn put_pixel(&mut self, x: u32, y: u32, p: [u8; 4]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let off = ((y * self.width + x) * 4) as usize;
        self.pixels[off..off + 4].copy_from_slice(&p);
    }

    /// Decode a PNG file into an RGBA8 [`Bitmap`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, PngError> {
        let bytes = fs::read(path.as_ref()).map_err(PngError::Io)?;
        Self::from_png_bytes(&bytes)
    }

    /// Decode a PNG byte buffer into an RGBA8 [`Bitmap`].
    pub fn from_png_bytes(bytes: &[u8]) -> Result<Self, PngError> {
        let cfg = PngDecodeConfig::default();
        let out = zenpng::decode(bytes, &cfg, &Unstoppable)
            .map_err(|e| PngError::Decode(format!("{e:?}")))?;
        let (w, h) = (out.info.width, out.info.height);
        let rgba = pixel_buffer_to_rgba8(&out.pixels, w, h)?;
        Self::from_raw(w, h, rgba)
            .ok_or_else(|| PngError::Decode("unexpected dimension mismatch".into()))
    }

    /// Encode this bitmap as PNG and write it to `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PngError> {
        let bytes = self.to_png_bytes()?;
        fs::write(path.as_ref(), bytes).map_err(PngError::Io)
    }

    /// Encode this bitmap as PNG bytes (no IO).
    pub fn to_png_bytes(&self) -> Result<Vec<u8>, PngError> {
        let pixels: &[Rgba<u8>] = bytemuck::cast_slice(&self.pixels);
        let img = imgref::Img::new(pixels, self.width as usize, self.height as usize);
        zenpng::encode_rgba8(
            img,
            None,
            &EncodeConfig::default(),
            &Unstoppable,
            &Unstoppable,
        )
        .map_err(|e| PngError::Encode(format!("{e:?}")))
    }
}

/// PNG encode/decode error. Hides `zenpng` internals from the public
/// API.
#[derive(Debug)]
pub enum PngError {
    /// Underlying file IO error.
    Io(io::Error),
    /// PNG decode error (malformed, unsupported, etc.).
    Decode(String),
    /// PNG encode error (resource limit, etc.).
    Encode(String),
}

impl std::fmt::Display for PngError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PngError::Io(e) => write!(f, "PNG IO error: {e}"),
            PngError::Decode(s) => write!(f, "PNG decode error: {s}"),
            PngError::Encode(s) => write!(f, "PNG encode error: {s}"),
        }
    }
}

impl std::error::Error for PngError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PngError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for PngError {
    fn from(e: io::Error) -> Self {
        PngError::Io(e)
    }
}

/// Convert zenpng's erased `PixelBuffer` into RGBA8 packed bytes.
/// Handles RGBA8, RGB8, Gray8, GrayAlpha8 (the formats PNG can yield
/// for typical assets). 16-bit and float formats are not supported.
fn pixel_buffer_to_rgba8(pixels: &PixelBuffer, w: u32, h: u32) -> Result<Vec<u8>, PngError> {
    use zenpixels::ChannelType;
    let desc = pixels.descriptor();
    if desc.channel_type() != ChannelType::U8 {
        return Err(PngError::Decode(format!(
            "unsupported channel type {:?}; only U8 supported",
            desc.channel_type()
        )));
    }
    let n = (w as usize) * (h as usize);
    let slice = pixels.as_slice();
    let mut out = Vec::with_capacity(n * 4);
    let channels = desc.channels();
    let has_alpha = desc.has_alpha();
    for y in 0..h {
        let row = slice.row(y);
        match (channels, has_alpha) {
            // Rgba
            (4, true) => out.extend_from_slice(&row[..(w as usize) * 4]),
            // Rgb
            (3, false) => {
                for px in row.chunks_exact(3).take(w as usize) {
                    out.extend_from_slice(&[px[0], px[1], px[2], 255]);
                }
            }
            // Gray
            (1, false) => {
                for &g in row.iter().take(w as usize) {
                    out.extend_from_slice(&[g, g, g, 255]);
                }
            }
            // GrayAlpha
            (2, true) => {
                for px in row.chunks_exact(2).take(w as usize) {
                    out.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
                }
            }
            other => {
                return Err(PngError::Decode(format!(
                    "unsupported channel layout (channels={}, has_alpha={})",
                    other.0, other.1
                )));
            }
        }
    }
    Ok(out)
}

// ════════════════════════════════════════════════════════════════════════
// GrayBitmap — single-channel u8 (used for the embedded font strip)
// ════════════════════════════════════════════════════════════════════════

/// Packed-Gray8 bitmap. Used for the embedded font glyph strip.
#[derive(Clone, Debug)]
pub(crate) struct GrayBitmap {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Vec<u8>,
}

impl GrayBitmap {
    /// Decode a PNG byte buffer into a single-channel [`GrayBitmap`].
    /// Used by [`crate::font`] for the embedded glyph strip.
    pub(crate) fn from_png_bytes(bytes: &[u8]) -> Result<Self, PngError> {
        let cfg = PngDecodeConfig::default();
        let out = zenpng::decode(bytes, &cfg, &Unstoppable)
            .map_err(|e| PngError::Decode(format!("{e:?}")))?;
        let (w, h) = (out.info.width, out.info.height);
        let pixels = pixel_buffer_to_gray8(&out.pixels)?;
        Ok(Self {
            width: w,
            height: h,
            pixels,
        })
    }

    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    #[allow(dead_code)] // used in tests only
    pub(crate) fn height(&self) -> u32 {
        self.height
    }

    pub(crate) fn get_pixel(&self, x: u32, y: u32) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.pixels[(y * self.width + x) as usize]
    }
}

fn pixel_buffer_to_gray8(pixels: &PixelBuffer) -> Result<Vec<u8>, PngError> {
    use zenpixels::ChannelType;
    let desc = pixels.descriptor();
    if desc.channel_type() != ChannelType::U8 {
        return Err(PngError::Decode(format!(
            "expected U8 channels, got {:?}",
            desc.channel_type()
        )));
    }
    let slice = pixels.as_slice();
    let w = slice.width() as usize;
    let h = slice.rows() as usize;
    let channels = desc.channels();
    let has_alpha = desc.has_alpha();
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h as u32 {
        let row = slice.row(y);
        match (channels, has_alpha) {
            (1, false) => out.extend_from_slice(&row[..w]),
            (2, true) => {
                for px in row.chunks_exact(2).take(w) {
                    out.push(px[0]);
                }
            }
            (3, false) => {
                for px in row.chunks_exact(3).take(w) {
                    out.push(px[1]); // green as luminance proxy
                }
            }
            (4, true) => {
                for px in row.chunks_exact(4).take(w) {
                    out.push(px[1]);
                }
            }
            other => {
                return Err(PngError::Decode(format!(
                    "unsupported gray decode channel layout ({}, alpha={})",
                    other.0, other.1
                )));
            }
        }
    }
    Ok(out)
}

// ════════════════════════════════════════════════════════════════════════
// Resize — backed by zenresize
// ════════════════════════════════════════════════════════════════════════

/// Resample `src` to `(w, h)` using `filter`. Returns a fresh buffer.
pub(crate) fn resize(src: &Bitmap, w: u32, h: u32, filter: ResampleFilter) -> Bitmap {
    if src.width == 0 || src.height == 0 || w == 0 || h == 0 {
        return Bitmap::new(w, h);
    }
    if (src.width, src.height) == (w, h) {
        return src.clone();
    }
    let config = zenresize::ResizeConfig::builder(src.width, src.height, w, h)
        .filter(filter.to_zenresize())
        .input(zenresize::PixelDescriptor::RGBA8_SRGB)
        .build();
    let pixels = zenresize::Resizer::new(&config).resize(&src.pixels);
    Bitmap {
        width: w,
        height: h,
        pixels,
    }
}

/// Resample a [`GrayBitmap`]. Used by the font glyph-strip downscale.
pub(crate) fn resize_gray(src: &GrayBitmap, w: u32, h: u32, filter: ResampleFilter) -> GrayBitmap {
    if src.width == 0 || src.height == 0 || w == 0 || h == 0 {
        return GrayBitmap {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h) as usize],
        };
    }
    if (src.width, src.height) == (w, h) {
        return src.clone();
    }
    let config = zenresize::ResizeConfig::builder(src.width, src.height, w, h)
        .filter(filter.to_zenresize())
        .input(zenresize::PixelDescriptor::GRAY8_SRGB)
        .build();
    let pixels = zenresize::Resizer::new(&config).resize(&src.pixels);
    GrayBitmap {
        width: w,
        height: h,
        pixels,
    }
}

// ════════════════════════════════════════════════════════════════════════
// Overlay (straight-alpha source-over) + crop + flips + rotates
// ════════════════════════════════════════════════════════════════════════

/// Source-over composite of `src` onto `canvas` at signed `(dx, dy)`.
/// Both straight-alpha sRGB-u8. Out-of-bounds pixels are clipped.
pub(crate) fn overlay(canvas: &mut Bitmap, src: &Bitmap, dx: i64, dy: i64) {
    let cw = canvas.width as i64;
    let ch = canvas.height as i64;
    let sw = src.width as i64;
    let sh = src.height as i64;
    if dx >= cw || dy >= ch || dx + sw <= 0 || dy + sh <= 0 {
        return;
    }

    // Clipped destination + matching source offset.
    let x_start = dx.max(0);
    let y_start = dy.max(0);
    let x_end = (dx + sw).min(cw);
    let y_end = (dy + sh).min(ch);
    let sx_start = (x_start - dx) as u32;
    let sy_start = (y_start - dy) as u32;
    let cw_u = canvas.width as usize;
    let sw_u = src.width as usize;

    for y_dst in y_start..y_end {
        let y_src = sy_start + (y_dst - y_start) as u32;
        let dst_row_off = (y_dst as usize) * cw_u * 4;
        let src_row_off = (y_src as usize) * sw_u * 4;
        for x_dst in x_start..x_end {
            let x_src = sx_start + (x_dst - x_start) as u32;
            let d_off = dst_row_off + (x_dst as usize) * 4;
            let s_off = src_row_off + (x_src as usize) * 4;

            let sa = src.pixels[s_off + 3];
            if sa == 0 {
                continue;
            }
            if sa == 255 {
                canvas.pixels[d_off..d_off + 4].copy_from_slice(&src.pixels[s_off..s_off + 4]);
                continue;
            }
            // Straight-alpha source-over.
            let sa_u = sa as u32;
            let inv_sa = 255 - sa_u;
            let da = canvas.pixels[d_off + 3] as u32;
            // out_a = sa + da * (1 - sa) / 255
            let out_a = (sa_u + (da * inv_sa).div_ceil(255)).min(255);
            for c in 0..3 {
                let s = src.pixels[s_off + c] as u32;
                let d = canvas.pixels[d_off + c] as u32;
                // Blend in straight alpha:
                //   out_c = (s * sa + d * da * (1 - sa) / 255) / out_a
                let num = s * sa_u + (d * da * inv_sa) / (255 * 255);
                let v = num.checked_div(out_a).unwrap_or(0);
                canvas.pixels[d_off + c] = v.min(255) as u8;
            }
            canvas.pixels[d_off + 3] = out_a as u8;
        }
    }
}

/// Crop `src` to `(x, y, w, h)`. Returns a fresh buffer; does not
/// modify `src`. Out-of-bounds requests yield a partial / empty crop.
pub(crate) fn crop(src: &Bitmap, x: u32, y: u32, w: u32, h: u32) -> Bitmap {
    let x_end = x.saturating_add(w).min(src.width);
    let y_end = y.saturating_add(h).min(src.height);
    let cw = x_end.saturating_sub(x);
    let ch = y_end.saturating_sub(y);
    let mut out = Bitmap::new(w, h);
    let row_stride_src = (src.width as usize) * 4;
    let row_stride_dst = (w as usize) * 4;
    let copy_bytes = (cw as usize) * 4;
    for row in 0..ch {
        let src_off = ((y + row) as usize) * row_stride_src + (x as usize) * 4;
        let dst_off = (row as usize) * row_stride_dst;
        out.pixels[dst_off..dst_off + copy_bytes]
            .copy_from_slice(&src.pixels[src_off..src_off + copy_bytes]);
    }
    out
}

/// Mirror `src` horizontally (left-right swap).
pub(crate) fn flip_horizontal(src: &Bitmap) -> Bitmap {
    let mut out = Bitmap::new(src.width, src.height);
    let stride = (src.width as usize) * 4;
    for y in 0..src.height as usize {
        let row_off = y * stride;
        for x in 0..src.width as usize {
            let s = row_off + x * 4;
            let d = row_off + (src.width as usize - 1 - x) * 4;
            out.pixels[d..d + 4].copy_from_slice(&src.pixels[s..s + 4]);
        }
    }
    out
}

/// Mirror `src` vertically (top-bottom swap).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn flip_vertical(src: &Bitmap) -> Bitmap {
    let mut out = Bitmap::new(src.width, src.height);
    let stride = (src.width as usize) * 4;
    for y in 0..src.height as usize {
        let s = y * stride;
        let d = (src.height as usize - 1 - y) * stride;
        out.pixels[d..d + stride].copy_from_slice(&src.pixels[s..s + stride]);
    }
    out
}

/// Rotate `src` by 90° clockwise.
pub(crate) fn rotate90(src: &Bitmap) -> Bitmap {
    let (sw, sh) = (src.width as usize, src.height as usize);
    let mut out = Bitmap::new(src.height, src.width);
    let dst_stride = sh * 4;
    let src_stride = sw * 4;
    for y in 0..sh {
        for x in 0..sw {
            let s = y * src_stride + x * 4;
            // (x, y) → (sh-1-y, x) in the rotated frame.
            let d = x * dst_stride + (sh - 1 - y) * 4;
            out.pixels[d..d + 4].copy_from_slice(&src.pixels[s..s + 4]);
        }
    }
    out
}

/// Rotate `src` by 180°.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn rotate180(src: &Bitmap) -> Bitmap {
    let (sw, sh) = (src.width as usize, src.height as usize);
    let mut out = Bitmap::new(src.width, src.height);
    let stride = sw * 4;
    for y in 0..sh {
        for x in 0..sw {
            let s = y * stride + x * 4;
            let d = (sh - 1 - y) * stride + (sw - 1 - x) * 4;
            out.pixels[d..d + 4].copy_from_slice(&src.pixels[s..s + 4]);
        }
    }
    out
}

/// Rotate `src` by 270° clockwise (≡ 90° counter-clockwise).
pub(crate) fn rotate270(src: &Bitmap) -> Bitmap {
    let (sw, sh) = (src.width as usize, src.height as usize);
    let mut out = Bitmap::new(src.height, src.width);
    let dst_stride = sh * 4;
    let src_stride = sw * 4;
    for y in 0..sh {
        for x in 0..sw {
            let s = y * src_stride + x * 4;
            // (x, y) → (y, sw-1-x).
            let d = (sw - 1 - x) * dst_stride + y * 4;
            out.pixels[d..d + 4].copy_from_slice(&src.pixels[s..s + 4]);
        }
    }
    out
}

// ════════════════════════════════════════════════════════════════════════
// Drawing primitives
// ════════════════════════════════════════════════════════════════════════

/// Fill the rectangle `(x, y, w, h)` of `img` with `color`. Clips.
pub(crate) fn fill_rect(img: &mut Bitmap, x: u32, y: u32, w: u32, h: u32, color: [u8; 4]) {
    let img_w = img.width;
    let img_h = img.height;
    let x_end = x.saturating_add(w).min(img_w);
    let y_end = y.saturating_add(h).min(img_h);
    if x >= x_end || y >= y_end {
        return;
    }
    let stride = (img_w as usize) * 4;
    // Build one row of fill bytes once.
    let row_w = (x_end - x) as usize;
    let mut row = Vec::with_capacity(row_w * 4);
    for _ in 0..row_w {
        row.extend_from_slice(&color);
    }
    for yy in y..y_end {
        let off = (yy as usize) * stride + (x as usize) * 4;
        img.pixels[off..off + row_w * 4].copy_from_slice(&row);
    }
}

/// Draw a 1-px outline at the rectangle `(x, y, w, h)`.
pub(crate) fn draw_rect_border(img: &mut Bitmap, x: u32, y: u32, w: u32, h: u32, color: [u8; 4]) {
    if w == 0 || h == 0 {
        return;
    }
    let img_w = img.width;
    let img_h = img.height;
    let x_end = x.saturating_add(w).min(img_w);
    let y_end = y.saturating_add(h).min(img_h);
    let bot = y_end.saturating_sub(1);
    let right = x_end.saturating_sub(1);
    // Top + bottom.
    for xx in x..x_end {
        if y < img_h {
            img.put_pixel(xx, y, color);
        }
        if bot < img_h && bot > y {
            img.put_pixel(xx, bot, color);
        }
    }
    // Left + right.
    for yy in y..y_end {
        if x < img_w {
            img.put_pixel(x, yy, color);
        }
        if right < img_w && right > x {
            img.put_pixel(right, yy, color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [u8; 4]) -> Bitmap {
        Bitmap::from_pixel(w, h, c)
    }

    #[test]
    fn bitmap_new_is_zero_filled() {
        let b = Bitmap::new(2, 2);
        assert_eq!(b.dimensions(), (2, 2));
        assert_eq!(b.as_raw(), &[0u8; 16]);
    }

    #[test]
    fn bitmap_from_pixel_is_solid() {
        let b = Bitmap::from_pixel(2, 2, [10, 20, 30, 40]);
        assert_eq!(b.get_pixel(0, 0), [10, 20, 30, 40]);
        assert_eq!(b.get_pixel(1, 1), [10, 20, 30, 40]);
    }

    #[test]
    fn bitmap_from_raw_validates_size() {
        assert!(Bitmap::from_raw(2, 2, vec![0u8; 16]).is_some());
        assert!(Bitmap::from_raw(2, 2, vec![0u8; 12]).is_none());
    }

    #[test]
    fn bitmap_from_fn() {
        let b = Bitmap::from_fn(2, 2, |x, y| [x as u8, y as u8, 0, 255]);
        assert_eq!(b.get_pixel(1, 0), [1, 0, 0, 255]);
        assert_eq!(b.get_pixel(0, 1), [0, 1, 0, 255]);
    }

    #[test]
    fn put_pixel_round_trips() {
        let mut b = Bitmap::new(4, 4);
        b.put_pixel(2, 1, [255, 128, 64, 200]);
        assert_eq!(b.get_pixel(2, 1), [255, 128, 64, 200]);
    }

    #[test]
    fn overlay_blits_opaque() {
        let mut canvas = solid(10, 10, [0, 0, 0, 255]);
        let src = solid(4, 4, [255, 0, 0, 255]);
        overlay(&mut canvas, &src, 2, 3);
        assert_eq!(canvas.get_pixel(2, 3), [255, 0, 0, 255]);
        assert_eq!(canvas.get_pixel(5, 6), [255, 0, 0, 255]);
        assert_eq!(canvas.get_pixel(0, 0), [0, 0, 0, 255]);
    }

    #[test]
    fn overlay_skips_transparent_source() {
        let mut canvas = solid(4, 4, [0, 0, 0, 255]);
        let src = solid(2, 2, [255, 0, 0, 0]);
        overlay(&mut canvas, &src, 1, 1);
        assert_eq!(canvas.get_pixel(1, 1), [0, 0, 0, 255]);
    }

    #[test]
    fn overlay_clips_negative_offsets() {
        let mut canvas = solid(4, 4, [0, 0, 0, 255]);
        let src = solid(4, 4, [255, 0, 0, 255]);
        overlay(&mut canvas, &src, -2, -2);
        assert_eq!(canvas.get_pixel(0, 0), [255, 0, 0, 255]);
        assert_eq!(canvas.get_pixel(1, 1), [255, 0, 0, 255]);
        assert_eq!(canvas.get_pixel(3, 3), [0, 0, 0, 255]);
    }

    #[test]
    fn resize_same_dim_returns_clone() {
        let b = solid(8, 8, [128; 4]);
        let r = resize(&b, 8, 8, ResampleFilter::Triangle);
        assert_eq!(r.as_raw(), b.as_raw());
    }

    #[test]
    fn resize_changes_dimensions() {
        let b = solid(4, 4, [128, 128, 128, 255]);
        let r = resize(&b, 8, 8, ResampleFilter::Lanczos3);
        assert_eq!(r.dimensions(), (8, 8));
    }

    #[test]
    fn crop_extracts_region() {
        let mut src = solid(8, 8, [0, 0, 0, 255]);
        src.put_pixel(3, 3, [255, 255, 255, 255]);
        let cropped = crop(&src, 2, 2, 4, 4);
        assert_eq!(cropped.dimensions(), (4, 4));
        assert_eq!(cropped.get_pixel(1, 1), [255, 255, 255, 255]);
    }

    #[test]
    fn flip_horizontal_swaps_columns() {
        let mut src = solid(4, 1, [0, 0, 0, 255]);
        src.put_pixel(0, 0, [255, 0, 0, 255]);
        let flipped = flip_horizontal(&src);
        assert_eq!(flipped.get_pixel(3, 0), [255, 0, 0, 255]);
    }

    #[test]
    fn flip_vertical_swaps_rows() {
        let mut src = solid(1, 4, [0, 0, 0, 255]);
        src.put_pixel(0, 0, [255, 0, 0, 255]);
        let flipped = flip_vertical(&src);
        assert_eq!(flipped.get_pixel(0, 3), [255, 0, 0, 255]);
    }

    #[test]
    fn rotate90_then_270_is_identity() {
        let mut src = Bitmap::from_fn(3, 5, |x, y| [x as u8 * 60, y as u8 * 50, 128, 255]);
        src.put_pixel(0, 0, [255, 128, 64, 255]);
        let r = rotate90(&rotate270(&src));
        assert_eq!(r.dimensions(), src.dimensions());
        assert_eq!(r.get_pixel(0, 0), src.get_pixel(0, 0));
        assert_eq!(r.get_pixel(2, 4), src.get_pixel(2, 4));
    }

    #[test]
    fn rotate180_inverts() {
        let src = Bitmap::from_fn(4, 3, |x, y| [x as u8 * 30, y as u8 * 50, 0, 255]);
        let r = rotate180(&src);
        assert_eq!(r.get_pixel(0, 0), src.get_pixel(3, 2));
        assert_eq!(r.get_pixel(3, 2), src.get_pixel(0, 0));
    }

    #[test]
    fn fill_rect_clips_to_bounds() {
        let mut img = solid(10, 10, [0, 0, 0, 255]);
        fill_rect(&mut img, 2, 2, 4, 4, [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(3, 3), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(0, 0), [0, 0, 0, 255]);
        assert_eq!(img.get_pixel(6, 6), [0, 0, 0, 255]);
    }

    #[test]
    fn draw_rect_border_outlines_only() {
        let mut img = solid(10, 10, [0, 0, 0, 255]);
        draw_rect_border(&mut img, 1, 1, 8, 8, [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(1, 1), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(8, 8), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(5, 5), [0, 0, 0, 255]);
    }

    #[test]
    fn png_roundtrip() {
        let src = Bitmap::from_fn(8, 8, |x, y| [x as u8 * 32, y as u8 * 32, 128, 255]);
        let bytes = src.to_png_bytes().expect("encode");
        let decoded = Bitmap::from_png_bytes(&bytes).expect("decode");
        assert_eq!(decoded.dimensions(), src.dimensions());
        assert_eq!(decoded.as_raw(), src.as_raw());
    }
}
