//! Monospace font rendering for annotating diff images.
//!
//! Embeds a DejaVu Sans Mono glyph strip (20px, ASCII 32–126) as a PNG,
//! decoded once on first use. Renders text at any size by scaling the
//! base glyphs with the `image` crate's Lanczos3 resampler.
//!
//! No external font files, no text shaping — just `render_text()` to
//! stamp stats onto diff montages.

use std::sync::OnceLock;

use image::{GrayImage, imageops};

/// Base glyph width in the embedded strip (before scaling).
const BASE_CHAR_W: u32 = 12;
/// Base glyph height in the embedded strip (before scaling).
const BASE_CHAR_H: u32 = 24;
/// Number of printable ASCII characters (0x20..=0x7E).
const CHAR_COUNT: u32 = 95;
/// First printable ASCII codepoint.
const FIRST_CHAR: u32 = 0x20;

/// The embedded font strip PNG (DejaVu Sans Mono, 20px, grayscale).
/// 95 characters × 12px wide × 24px tall = 1140×24 pixels, ~7KB PNG.
static FONT_PNG: &[u8] = include_bytes!("font_strip.png");

/// Decoded font strip (grayscale, decoded once).
fn font_strip() -> &'static GrayImage {
    static STRIP: OnceLock<GrayImage> = OnceLock::new();
    STRIP.get_or_init(|| {
        image::load_from_memory(FONT_PNG)
            .expect("embedded font strip PNG is invalid")
            .to_luma8()
    })
}

/// Render a string into an RGBA pixel buffer at the base font size (12×24 per char).
///
/// Returns `(pixels, width, height)` where pixels is `width * height * 4` bytes.
/// Newlines are supported.
pub fn render_text(text: &str, fg: [u8; 4], bg: [u8; 4]) -> (Vec<u8>, u32, u32) {
    render_text_scaled(text, fg, bg, 1)
}

/// Render a string at a given pixel height.
///
/// The font is scaled so each character is `target_char_h` pixels tall.
/// Character width scales proportionally to maintain the monospace aspect ratio.
///
/// Returns `(pixels, width, height)` where pixels is `width * height * 4` bytes.
pub fn render_text_height(
    text: &str,
    fg: [u8; 4],
    bg: [u8; 4],
    target_char_h: u32,
) -> (Vec<u8>, u32, u32) {
    if target_char_h == 0 {
        return (vec![], 0, 0);
    }

    let lines: Vec<&str> = text.lines().collect();
    let max_cols = lines.iter().map(|l| l.len()).max().unwrap_or(0) as u32;
    let num_lines = lines.len() as u32;

    if max_cols == 0 || num_lines == 0 {
        return (vec![], 0, 0);
    }

    // Scale the strip to target height
    let strip = font_strip();
    let scale_numer = target_char_h;
    let scale_denom = BASE_CHAR_H;
    let scaled_char_w = (BASE_CHAR_W * scale_numer + scale_denom / 2) / scale_denom;
    let scaled_char_h = target_char_h;

    // Scale the entire strip at once (one resize, not per-char)
    let scaled_strip_w = scaled_char_w * CHAR_COUNT;
    let scaled_strip = imageops::resize(
        strip,
        scaled_strip_w,
        scaled_char_h,
        imageops::FilterType::Lanczos3,
    );

    let out_w = max_cols * scaled_char_w;
    let out_h = num_lines * scaled_char_h;

    let mut buf = vec![0u8; (out_w * out_h * 4) as usize];

    // Fill background
    for pixel in buf.chunks_exact_mut(4) {
        pixel.copy_from_slice(&bg);
    }

    // Blit characters
    for (line_idx, line) in lines.iter().enumerate() {
        let y_base = line_idx as u32 * scaled_char_h;
        for (col, ch) in line.chars().enumerate() {
            let x_base = col as u32 * scaled_char_w;
            let glyph_idx = char_index(ch);
            let src_x = glyph_idx * scaled_char_w;

            for gy in 0..scaled_char_h {
                for gx in 0..scaled_char_w {
                    let sx = src_x + gx;
                    if sx >= scaled_strip.width() {
                        continue;
                    }
                    let alpha = scaled_strip.get_pixel(sx, gy)[0];
                    if alpha == 0 {
                        continue;
                    }

                    let px = x_base + gx;
                    let py = y_base + gy;
                    if px >= out_w || py >= out_h {
                        continue;
                    }

                    let off = ((py * out_w + px) * 4) as usize;
                    if alpha == 255 {
                        buf[off..off + 4].copy_from_slice(&fg);
                    } else {
                        // Alpha blend fg over bg
                        let a = alpha as u16;
                        let inv_a = 255 - a;
                        buf[off] = ((fg[0] as u16 * a + bg[0] as u16 * inv_a) / 255) as u8;
                        buf[off + 1] = ((fg[1] as u16 * a + bg[1] as u16 * inv_a) / 255) as u8;
                        buf[off + 2] = ((fg[2] as u16 * a + bg[2] as u16 * inv_a) / 255) as u8;
                        buf[off + 3] = ((fg[3] as u16 * a + bg[3] as u16 * inv_a) / 255) as u8;
                    }
                }
            }
        }
    }

    (buf, out_w, out_h)
}

/// Render a string at an integer multiple of the base font size.
///
/// `scale` of 1 is 12×24 per char. Scale 2 is 24×48, etc.
pub fn render_text_scaled(
    text: &str,
    fg: [u8; 4],
    bg: [u8; 4],
    scale: u32,
) -> (Vec<u8>, u32, u32) {
    let target_h = BASE_CHAR_H * scale.max(1);
    render_text_height(text, fg, bg, target_h)
}

/// Map a character to its index in the font strip (0–94).
/// Unknown characters map to space (index 0).
fn char_index(ch: char) -> u32 {
    let code = ch as u32;
    if (FIRST_CHAR..FIRST_CHAR + CHAR_COUNT).contains(&code) {
        code - FIRST_CHAR
    } else {
        0 // space
    }
}

/// Base character width in pixels (before scaling).
pub const GLYPH_W: u32 = BASE_CHAR_W;
/// Base character height in pixels (before scaling).
pub const GLYPH_H: u32 = BASE_CHAR_H;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_strip_loads() {
        let strip = font_strip();
        assert_eq!(strip.width(), BASE_CHAR_W * CHAR_COUNT);
        assert_eq!(strip.height(), BASE_CHAR_H);
    }

    #[test]
    fn render_empty_string() {
        let (buf, w, h) = render_text("", [255; 4], [0; 4]);
        assert_eq!(w, 0);
        assert_eq!(h, 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn render_single_char() {
        let (buf, w, h) = render_text("A", [255, 255, 255, 255], [0, 0, 0, 255]);
        assert_eq!(w, BASE_CHAR_W);
        assert_eq!(h, BASE_CHAR_H);
        assert_eq!(buf.len(), (w * h * 4) as usize);
        let white_count = buf.chunks(4).filter(|px| px[0] > 128).count();
        assert!(white_count > 10, "A glyph should have lit pixels, got {white_count}");
    }

    #[test]
    fn render_multiline() {
        let (buf, w, h) = render_text("AB\nCD", [255; 4], [0; 4]);
        assert_eq!(w, 2 * BASE_CHAR_W);
        assert_eq!(h, 2 * BASE_CHAR_H);
        assert_eq!(buf.len(), (w * h * 4) as usize);
    }

    #[test]
    fn render_stats_line() {
        let text = "score:87.2 delta:[12,8,3] 34.2%";
        let (buf, w, h) = render_text(text, [255; 4], [0; 4]);
        assert_eq!(w, text.len() as u32 * BASE_CHAR_W);
        assert_eq!(h, BASE_CHAR_H);
        assert_eq!(buf.len(), (w * h * 4) as usize);
    }

    #[test]
    fn render_scaled_2x() {
        let (_, w1, h1) = render_text("A", [255; 4], [0; 4]);
        let (_, w2, h2) = render_text_scaled("A", [255; 4], [0; 4], 2);
        assert_eq!(w2, w1 * 2);
        assert_eq!(h2, h1 * 2);
    }

    #[test]
    fn render_height_custom() {
        let target_h = 40;
        let (buf, w, h) = render_text_height("AB", [255; 4], [0; 4], target_h);
        assert_eq!(h, target_h);
        assert!(w > 0);
        assert_eq!(buf.len(), (w * h * 4) as usize);
    }

    #[test]
    fn digits_have_pixels() {
        let strip = font_strip();
        for d in b'0'..=b'9' {
            let idx = char_index(d as char);
            let x0 = idx * BASE_CHAR_W;
            let mut lit = 0u32;
            for y in 0..BASE_CHAR_H {
                for x in x0..x0 + BASE_CHAR_W {
                    if strip.get_pixel(x, y)[0] > 128 {
                        lit += 1;
                    }
                }
            }
            assert!(lit > 10, "digit '{}' should have lit pixels, got {lit}", d as char);
        }
    }
}
