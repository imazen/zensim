//! Sixel terminal display for images.
//!
//! Pure-Rust sixel encoder — no libsixel dependency. Encodes images using
//! a 216-color RGB cube (6×6×6) plus 24 grayscale levels (240 total).
//! Writes directly to stdout and flushes.
//!
//! # Sixel protocol overview
//!
//! Sixel is a bitmap graphics format for terminals. Each character encodes
//! a 1-pixel-wide, 6-pixel-tall column. Colors are defined in a palette
//! and selected per-band. The sequence is:
//!
//! ```text
//! ESC P q            DCS (Device Control String) with sixel mode
//! "1;1;W;H           Raster attributes: aspect 1:1, width, height
//! #N;2;R%;G%;B%      Define color N as RGB (0-100%)
//! #N <sixel data>    Select color N, emit sixel characters
//! $                  Graphics carriage return (same band)
//! -                  Graphics new line (next 6-row band)
//! ESC \              ST (String Terminator)
//! ```

use std::fmt::Write as FmtWrite;
use std::io::Write;

use image::RgbaImage;
use image::imageops::{self, FilterType};

/// Encode an image as sixel bytes.
///
/// If `max_width` is set and the image is wider, it's downscaled to fit
/// while preserving aspect ratio.
///
/// Uses a 240-color palette: 216-color 6×6×6 RGB cube + 24 grayscale levels.
pub fn sixel_encode(img: &RgbaImage, max_width: Option<u32>) -> Vec<u8> {
    let img = maybe_resize(img, max_width);
    let (w, h) = img.dimensions();

    // Build palette and quantize
    let palette = build_palette_240();
    let indices = quantize_image(&img, &palette);

    let mut out = Vec::with_capacity((w * h) as usize);

    // DCS q
    out.extend_from_slice(b"\x1bPq");

    // Raster attributes: aspect 1:1, pixel dimensions
    write_fmt(&mut out, format_args!("\"1;1;{};{}", w, h));

    // Define palette colors (RGB as 0-100%)
    for (i, &[r, g, b]) in palette.iter().enumerate() {
        write_fmt(
            &mut out,
            format_args!(
                "#{};2;{};{};{}",
                i,
                r as u32 * 100 / 255,
                g as u32 * 100 / 255,
                b as u32 * 100 / 255,
            ),
        );
    }

    // Encode sixel bands (6 rows each)
    let bands = h.div_ceil(6);
    for band in 0..bands {
        let y_start = band * 6;
        let y_end = (y_start + 6).min(h);

        // Find colors used in this band
        let mut colors_in_band: Vec<u8> = Vec::new();
        {
            let mut seen = [false; 240];
            for y in y_start..y_end {
                for x in 0..w {
                    let idx = indices[(y * w + x) as usize];
                    if !seen[idx as usize] {
                        seen[idx as usize] = true;
                        colors_in_band.push(idx);
                    }
                }
            }
        }

        let mut first_color = true;
        for &color_idx in &colors_in_band {
            if !first_color {
                out.push(b'$'); // Graphics CR: back to start of band
            }
            first_color = false;

            // Select color
            write_fmt(&mut out, format_args!("#{}", color_idx));

            // Build sixel characters with RLE
            let mut prev_ch: u8 = 0;
            let mut run_len: u32 = 0;

            for x in 0..w {
                let mut bits: u8 = 0;
                for dy in 0..6u32 {
                    let y = y_start + dy;
                    if y < h && indices[(y * w + x) as usize] == color_idx {
                        bits |= 1 << dy;
                    }
                }
                let ch = bits + 63; // Sixel char: 63 ('?') for no bits

                if run_len > 0 && ch == prev_ch {
                    run_len += 1;
                } else {
                    flush_run(&mut out, prev_ch, run_len);
                    prev_ch = ch;
                    run_len = 1;
                }
            }
            flush_run(&mut out, prev_ch, run_len);
        }

        if band + 1 < bands {
            out.push(b'-'); // Graphics NL: next 6-row band
        }
    }

    // ST (String Terminator)
    out.extend_from_slice(b"\x1b\\");
    out
}

/// Print an image to stdout as sixels, then flush.
///
/// If `max_width` is set, the image is downscaled to fit.
pub fn print_image(img: &RgbaImage, max_width: Option<u32>) {
    let bytes = sixel_encode(img, max_width);
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    let _ = handle.write_all(&bytes);
    let _ = handle.flush();
}

/// Print a 3-panel comparison (Expected | Actual | Diff) as sixels.
///
/// Labels are printed as text before the sixel image.
/// The diff uses the given amplification factor (default: 10).
pub fn print_comparison(
    expected: &RgbaImage,
    actual: &RgbaImage,
    amplification: u8,
    max_width: Option<u32>,
) {
    let montage =
        crate::diff_image::create_comparison_montage(expected, actual, amplification, 2);

    // Print text labels above the panels
    let panel_w = expected.width() as usize;
    let gap = 2;
    let label_expected = "Expected";
    let label_actual = "Actual";
    let label_diff = format!("Diff x{amplification}");

    // Center labels in their panels (approximate: terminal chars ≠ pixels,
    // but reasonable when image is close to terminal scale)
    let pad_e = panel_w.saturating_sub(label_expected.len()) / 2;
    let pad_a = panel_w.saturating_sub(label_actual.len()) / 2;
    let pad_d = panel_w.saturating_sub(label_diff.len()) / 2;

    println!(
        "{:>pad_e$}{}{:>gap$}{:>pad_a$}{}{:>gap$}{:>pad_d$}{}",
        "",
        label_expected,
        "",
        "",
        label_actual,
        "",
        "",
        label_diff,
        pad_e = pad_e,
        pad_a = pad_a,
        pad_d = pad_d,
        gap = gap,
    );

    print_image(&montage, max_width);
    println!();
}

// ─── Palette ─────────────────────────────────────────────────────────────

/// Build a 240-color palette: 216-color 6×6×6 RGB cube + 24 grayscale levels.
fn build_palette_240() -> Vec<[u8; 3]> {
    let mut palette = Vec::with_capacity(240);

    // 216-color cube: indices 0-215
    for r in 0u8..6 {
        for g in 0u8..6 {
            for b in 0u8..6 {
                palette.push([r * 51, g * 51, b * 51]);
            }
        }
    }

    // 24 grayscale levels: indices 216-239
    // Evenly spaced from 8 to 238 (avoiding pure black/white which are in the cube)
    for i in 0u8..24 {
        let v = 8 + i * 10;
        palette.push([v, v, v]);
    }

    palette
}

/// Quantize a pixel to the nearest palette index.
fn quantize_pixel(r: u8, g: u8, b: u8, palette: &[[u8; 3]]) -> u8 {
    // Fast path: try the RGB cube first
    let ri = ((r as u16 * 5 + 127) / 255) as u8;
    let gi = ((g as u16 * 5 + 127) / 255) as u8;
    let bi = ((b as u16 * 5 + 127) / 255) as u8;
    let cube_idx = (ri * 36 + gi * 6 + bi) as usize;

    let cube_color = palette[cube_idx];
    let cube_dist = color_dist_sq(r, g, b, cube_color[0], cube_color[1], cube_color[2]);

    // Check if a grayscale entry is closer (important for near-gray pixels)
    if is_near_gray(r, g, b) {
        let avg = ((r as u16 + g as u16 + b as u16) / 3) as u8;
        // Find closest grayscale entry
        let mut best_idx = cube_idx;
        let mut best_dist = cube_dist;
        for (i, entry) in palette.iter().enumerate().take(240).skip(216) {
            let v = entry[0];
            let d = color_dist_sq(r, g, b, v, v, v);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        let _ = avg;
        best_idx as u8
    } else {
        cube_idx as u8
    }
}

/// Check if a color is near-gray (all channels within 20 of each other).
fn is_near_gray(r: u8, g: u8, b: u8) -> bool {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    (max - min) <= 20
}

/// Squared distance between two colors (perceptual weighting).
fn color_dist_sq(r1: u8, g1: u8, b1: u8, r2: u8, g2: u8, b2: u8) -> u32 {
    let dr = r1 as i32 - r2 as i32;
    let dg = g1 as i32 - g2 as i32;
    let db = b1 as i32 - b2 as i32;
    // Weighted: green is most perceptually important
    (dr * dr * 2 + dg * dg * 4 + db * db) as u32
}

/// Quantize an entire image to palette indices.
fn quantize_image(img: &RgbaImage, palette: &[[u8; 3]]) -> Vec<u8> {
    let (w, h) = img.dimensions();
    let mut indices = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y);
            // Alpha-blend against dark gray (24,24,24) for transparent pixels
            let a = p[3] as u16;
            let bg = 24u16;
            let r = ((p[0] as u16 * a + bg * (255 - a)) / 255) as u8;
            let g = ((p[1] as u16 * a + bg * (255 - a)) / 255) as u8;
            let b = ((p[2] as u16 * a + bg * (255 - a)) / 255) as u8;
            indices.push(quantize_pixel(r, g, b, palette));
        }
    }
    indices
}

// ─── Helpers ─────────────────────────────────────────────────────────────

/// Resize the image if it exceeds max_width, preserving aspect ratio.
fn maybe_resize(img: &RgbaImage, max_width: Option<u32>) -> RgbaImage {
    match max_width {
        Some(max_w) if img.width() > max_w => {
            let scale = max_w as f64 / img.width() as f64;
            let new_h = (img.height() as f64 * scale).round() as u32;
            imageops::resize(img, max_w, new_h.max(1), FilterType::Lanczos3)
        }
        _ => img.clone(),
    }
}

/// Write formatted text into a byte buffer.
fn write_fmt(buf: &mut Vec<u8>, args: std::fmt::Arguments<'_>) {
    let mut s = String::new();
    s.write_fmt(args).unwrap();
    buf.extend_from_slice(s.as_bytes());
}

/// Flush a run-length encoded sixel run.
fn flush_run(out: &mut Vec<u8>, ch: u8, len: u32) {
    if len == 0 {
        return;
    }
    if len <= 3 {
        for _ in 0..len {
            out.push(ch);
        }
    } else {
        write_fmt(out, format_args!("!{}{}", len, ch as char));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn palette_has_240_entries() {
        let palette = build_palette_240();
        assert_eq!(palette.len(), 240);
    }

    #[test]
    fn palette_starts_with_black() {
        let palette = build_palette_240();
        assert_eq!(palette[0], [0, 0, 0]);
    }

    #[test]
    fn palette_has_white() {
        let palette = build_palette_240();
        assert_eq!(palette[215], [255, 255, 255]); // 5*36 + 5*6 + 5 = 215
    }

    #[test]
    fn quantize_black() {
        let palette = build_palette_240();
        assert_eq!(quantize_pixel(0, 0, 0, &palette), 0);
    }

    #[test]
    fn quantize_white() {
        let palette = build_palette_240();
        assert_eq!(quantize_pixel(255, 255, 255, &palette), 215);
    }

    #[test]
    fn quantize_pure_red() {
        let palette = build_palette_240();
        let idx = quantize_pixel(255, 0, 0, &palette);
        // Should be (5, 0, 0) → 5*36 = 180
        assert_eq!(idx, 180);
    }

    #[test]
    fn sixel_starts_with_dcs() {
        let img = RgbaImage::from_pixel(4, 4, Rgba([128, 128, 128, 255]));
        let bytes = sixel_encode(&img, None);
        assert!(bytes.starts_with(b"\x1bPq"));
    }

    #[test]
    fn sixel_ends_with_st() {
        let img = RgbaImage::from_pixel(4, 4, Rgba([128, 128, 128, 255]));
        let bytes = sixel_encode(&img, None);
        assert!(bytes.ends_with(b"\x1b\\"));
    }

    #[test]
    fn sixel_contains_raster_attrs() {
        let img = RgbaImage::from_pixel(16, 12, Rgba([64, 64, 64, 255]));
        let bytes = sixel_encode(&img, None);
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("\"1;1;16;12"), "missing raster attributes: {s}");
    }

    #[test]
    fn sixel_respects_max_width() {
        let img = RgbaImage::from_pixel(100, 50, Rgba([128, 128, 128, 255]));
        let bytes = sixel_encode(&img, Some(40));
        let s = String::from_utf8_lossy(&bytes);
        // Should contain resized dimensions
        assert!(s.contains("\"1;1;40;20"), "should be resized to 40x20: {s}");
    }

    #[test]
    fn sixel_single_color_image() {
        // A uniform image should produce very compact sixel output
        let img = RgbaImage::from_pixel(100, 6, Rgba([128, 128, 128, 255]));
        let bytes = sixel_encode(&img, None);
        let s = String::from_utf8_lossy(&bytes);
        // Should use RLE for the uniform run
        assert!(s.contains("!100"), "should use RLE for 100 identical columns");
    }

    #[test]
    fn sixel_multi_band() {
        // 12 rows = 2 bands of 6
        let img = RgbaImage::from_pixel(4, 12, Rgba([200, 100, 50, 255]));
        let bytes = sixel_encode(&img, None);
        let s = String::from_utf8_lossy(&bytes);
        // Should contain a band separator
        assert!(s.contains('-'), "should have band separator for 12-row image");
    }

    #[test]
    fn resize_preserves_aspect() {
        let img = RgbaImage::new(200, 100);
        let resized = maybe_resize(&img, Some(100));
        assert_eq!(resized.width(), 100);
        assert_eq!(resized.height(), 50);
    }

    #[test]
    fn resize_noop_when_smaller() {
        let img = RgbaImage::new(50, 50);
        let resized = maybe_resize(&img, Some(100));
        assert_eq!(resized.width(), 50);
        assert_eq!(resized.height(), 50);
    }
}
