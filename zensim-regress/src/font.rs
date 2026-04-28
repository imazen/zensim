//! Monospace font rendering for annotating diff images.
//!
//! Embeds a Consolas glyph strip (48px, ASCII 32–126) as a PNG, decoded
//! once on first use. Renders text at any size by downscaling with the
//! `image` crate's Lanczos3 resampler — the base is larger than any
//! render target, so we always downsample (crisp edges, no upscale blur).
//!
//! Word-wraps on spaces when a max width is specified.

use std::sync::OnceLock;

use crate::pixel_ops::{self, GrayBitmap, ResampleFilter};

/// Base glyph width in the embedded strip (before scaling).
const BASE_CHAR_W: u32 = 26;
/// Base glyph height in the embedded strip (before scaling).
const BASE_CHAR_H: u32 = 54;
/// Total characters in the strip: ASCII 32-126 (95) + extras (Δ).
const CHAR_COUNT: u32 = 96;
/// First printable ASCII codepoint.
const FIRST_CHAR: u32 = 0x20;

/// The embedded font strip PNG (Consolas 48px, grayscale).
static FONT_PNG: &[u8] = include_bytes!("font_strip.png");

/// Decoded font strip (grayscale, decoded once).
fn font_strip() -> &'static GrayBitmap {
    static STRIP: OnceLock<GrayBitmap> = OnceLock::new();
    STRIP.get_or_init(|| {
        GrayBitmap::from_png_bytes(FONT_PNG).expect("embedded font strip PNG is invalid")
    })
}

/// Render a string into an RGBA pixel buffer at the base font size.
///
/// Returns `(pixels, width, height)`. Newlines are supported.
pub fn render_text(text: &str, fg: [u8; 4], bg: [u8; 4]) -> (Vec<u8>, u32, u32) {
    render_text_height(text, fg, bg, BASE_CHAR_H)
}

/// Render text word-wrapped to fit within `max_width_px` pixels.
///
/// Lines are broken on spaces. If a single word is wider than `max_width_px`,
/// it is placed on its own line (not split mid-word). Lines stack tightly
/// — for typographic leading, see [`render_text_wrapped_lh`].
///
/// Returns `(pixels, width, height)`.
pub fn render_text_wrapped(
    text: &str,
    fg: [u8; 4],
    bg: [u8; 4],
    target_char_h: u32,
    max_width_px: u32,
) -> (Vec<u8>, u32, u32) {
    render_text_wrapped_lh(text, fg, bg, target_char_h, max_width_px, 1.0)
}

/// Same as [`render_text_wrapped`], with a CSS-`line-height` ratio
/// applied to the inter-line stride.
pub fn render_text_wrapped_lh(
    text: &str,
    fg: [u8; 4],
    bg: [u8; 4],
    target_char_h: u32,
    max_width_px: u32,
    line_height: f32,
) -> (Vec<u8>, u32, u32) {
    if target_char_h == 0 || max_width_px == 0 {
        return (vec![], 0, 0);
    }

    let scaled_char_w = char_width_for_height(target_char_h);
    let max_cols = (max_width_px / scaled_char_w.max(1)) as usize;
    let wrapped = wrap_text(text, max_cols.max(1));
    render_text_height_lh(&wrapped, fg, bg, target_char_h, line_height)
}

/// Render a string at a given pixel height.
///
/// The font is scaled so each character is `target_char_h` pixels tall.
/// Character width scales proportionally to maintain the monospace aspect ratio.
/// Multi-line input stacks lines tightly (no inter-line leading) — for a
/// typographic gap, use [`render_text_height_lh`] with `line_height > 1.0`.
///
/// Returns `(pixels, width, height)`.
pub fn render_text_height(
    text: &str,
    fg: [u8; 4],
    bg: [u8; 4],
    target_char_h: u32,
) -> (Vec<u8>, u32, u32) {
    render_text_height_lh(text, fg, bg, target_char_h, 1.0)
}

/// Same as [`render_text_height`], but stacks multi-line input with a
/// CSS-`line-height` ratio (`line_height = 1.2` ≈ default browser
/// leading). The first line still occupies `target_char_h` pixels;
/// each subsequent line advances by `round(target_char_h *
/// line_height)`, clamped to ≥ 1px. The output buffer height is
/// `(num_lines - 1) * line_advance + target_char_h` — no trailing
/// leading.
pub fn render_text_height_lh(
    text: &str,
    fg: [u8; 4],
    bg: [u8; 4],
    target_char_h: u32,
    line_height: f32,
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

    let strip = font_strip();
    let scaled_char_w = char_width_for_height(target_char_h);
    let scaled_char_h = target_char_h;
    let line_advance = line_advance_px(scaled_char_h, line_height);

    // Scale the entire strip at once (one resize, not per-char).
    // Lanczos3 — sharp downsampling for crisp glyph edges.
    let scaled_strip_w = scaled_char_w * CHAR_COUNT;
    let scaled_strip = pixel_ops::resize_gray(
        strip,
        scaled_strip_w,
        scaled_char_h,
        ResampleFilter::Lanczos3,
    );

    let out_w = max_cols * scaled_char_w;
    let out_h = stacked_height_px(scaled_char_h, line_advance, num_lines);

    let mut buf = vec![0u8; (out_w * out_h * 4) as usize];

    // Fill background
    for pixel in buf.chunks_exact_mut(4) {
        pixel.copy_from_slice(&bg);
    }

    // Blit characters
    for (line_idx, line) in lines.iter().enumerate() {
        let y_base = line_idx as u32 * line_advance;
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
                    let alpha = scaled_strip.get_pixel(sx, gy);
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
pub fn render_text_scaled(text: &str, fg: [u8; 4], bg: [u8; 4], scale: u32) -> (Vec<u8>, u32, u32) {
    let target_h = BASE_CHAR_H * scale.max(1);
    render_text_height(text, fg, bg, target_h)
}

// ─── Word wrap ──────────────────────────────────────────────────────────

/// Word-wrap text to fit within `max_cols` characters per line.
///
/// Breaks on spaces. Explicit newlines in the input are preserved.
/// Words longer than `max_cols` are placed on their own line (not split).
fn wrap_text(text: &str, max_cols: usize) -> String {
    let mut result = String::new();

    for input_line in text.lines() {
        if !result.is_empty() {
            result.push('\n');
        }

        let mut col = 0usize;
        for (i, word) in input_line.split(' ').enumerate() {
            let wlen = word.len();

            if i == 0 {
                // First word on this line — always emit
                result.push_str(word);
                col = wlen;
            } else if col + 1 + wlen <= max_cols {
                // Fits on current line with a space
                result.push(' ');
                result.push_str(word);
                col += 1 + wlen;
            } else {
                // Wrap to next line
                result.push('\n');
                result.push_str(word);
                col = wlen;
            }
        }
    }

    result
}

// ─── Helpers ────────────────────────────────────────────────────────────

/// Compute the character width for a given target character height.
fn char_width_for_height(target_char_h: u32) -> u32 {
    (BASE_CHAR_W * target_char_h + BASE_CHAR_H / 2) / BASE_CHAR_H
}

/// Vertical stride between consecutive lines (`char_h * line_height`,
/// clamped to ≥ 1px and ≥ char_h so lines never overlap). A
/// non-finite, non-positive, or NaN `line_height` is treated as 1.0.
fn line_advance_px(char_h: u32, line_height: f32) -> u32 {
    if char_h == 0 {
        return 0;
    }
    let lh = if line_height.is_finite() && line_height > 0.0 {
        line_height
    } else {
        1.0
    };
    let advance = ((char_h as f32) * lh).round() as u32;
    advance.max(char_h.max(1))
}

/// Total stacked height for `n` lines: `(n - 1) * line_advance + char_h`.
/// No trailing leading; a single line is exactly `char_h` tall.
fn stacked_height_px(char_h: u32, line_advance: u32, n_lines: u32) -> u32 {
    if n_lines == 0 {
        return 0;
    }
    n_lines
        .saturating_sub(1)
        .saturating_mul(line_advance)
        .saturating_add(char_h)
}

/// Map a character to its index in the font strip.
/// ASCII 32-126 at indices 0-94, then extras: Δ at 95.
fn char_index(ch: char) -> u32 {
    let code = ch as u32;
    if (FIRST_CHAR..FIRST_CHAR + 95).contains(&code) {
        code - FIRST_CHAR
    } else {
        match ch {
            '\u{0394}' => 95, // Δ (Greek capital delta)
            _ => 0,           // space for unknown
        }
    }
}

/// Render multiple lines, each with its own color, at a char height that
/// makes the longest line fit within `max_width_px`.
///
/// Returns `(pixels, width, height)`. No word-wrapping — each line is
/// rendered as-is. The font size is computed from the longest line.
/// Lines stack tightly; for typographic leading, see
/// [`render_lines_fitted_lh`].
pub fn render_lines_fitted(
    lines: &[(&str, [u8; 4])],
    bg: [u8; 4],
    max_width_px: u32,
) -> (Vec<u8>, u32, u32) {
    render_lines_fitted_lh(lines, bg, max_width_px, 1.0)
}

/// Same as [`render_lines_fitted`], with a CSS-`line-height` ratio
/// applied to the inter-line stride.
pub fn render_lines_fitted_lh(
    lines: &[(&str, [u8; 4])],
    bg: [u8; 4],
    max_width_px: u32,
    line_height: f32,
) -> (Vec<u8>, u32, u32) {
    if lines.is_empty() || max_width_px == 0 {
        return (vec![], 0, 0);
    }
    let (char_w, char_h) = fit_char_h_for_lines(lines, max_width_px);
    if char_w == 0 || char_h == 0 {
        return (vec![], 0, 0);
    }
    let line_advance = line_advance_px(char_h, line_height);
    let longest = lines.iter().map(|(s, _)| s.len()).max().unwrap_or(0) as u32;

    let out_w = longest * char_w;
    let out_h = stacked_height_px(char_h, line_advance, lines.len() as u32);

    if out_w == 0 || out_h == 0 {
        return (vec![], 0, 0);
    }

    // Scale strip once.
    let strip = font_strip();
    let scaled_strip_w = char_w * CHAR_COUNT;
    let scaled_strip =
        pixel_ops::resize_gray(strip, scaled_strip_w, char_h, ResampleFilter::Lanczos3);

    let mut buf = vec![0u8; (out_w * out_h * 4) as usize];
    for pixel in buf.chunks_exact_mut(4) {
        pixel.copy_from_slice(&bg);
    }

    for (line_idx, (text, fg)) in lines.iter().enumerate() {
        let y_base = line_idx as u32 * line_advance;
        // Center each line within the output width
        let line_w = text.len() as u32 * char_w;
        let x_offset = (out_w.saturating_sub(line_w)) / 2;
        for (col, ch) in text.chars().enumerate() {
            let x_base = x_offset + col as u32 * char_w;
            let glyph_idx = char_index(ch);
            let src_x = glyph_idx * char_w;

            for gy in 0..char_h {
                for gx in 0..char_w {
                    let sx = src_x + gx;
                    if sx >= scaled_strip.width() {
                        continue;
                    }
                    let alpha = scaled_strip.get_pixel(sx, gy);
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
                        buf[off..off + 4].copy_from_slice(fg);
                    } else {
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

/// Base character width in pixels (before scaling).
pub const GLYPH_W: u32 = BASE_CHAR_W;
/// Base character height in pixels (before scaling).
pub const GLYPH_H: u32 = BASE_CHAR_H;

/// Typographic vertical-centering adjustment as a fraction of char
/// height. Empirically derived from the embedded Consolas strip: the
/// inked cap-mid (cap-top + baseline)/2 sits at ~42.6% of cell height,
/// while the geometric cell mid is 50%. To make capitals appear
/// visually centered when an enclosing layout uses geometric
/// centering, shift the cell down by this fraction.
pub const TYPO_CAP_MID_OFFSET: f32 = 0.074;

// ─── Cheap measurement (no rasterization) ──────────────────────────────

/// Return the `(w, h)` a [`render_text_height`] call would produce —
/// without actually rasterizing. Useful for two-pass layout where the
/// measure pass needs sizes but the render pass will rasterize anyway.
pub fn measure_text_height(text: &str, target_char_h: u32) -> (u32, u32) {
    measure_text_height_lh(text, target_char_h, 1.0)
}

/// Same as [`measure_text_height`], with a CSS-`line-height` ratio
/// applied to the inter-line stride.
pub fn measure_text_height_lh(text: &str, target_char_h: u32, line_height: f32) -> (u32, u32) {
    if target_char_h == 0 {
        return (0, 0);
    }
    let lines: Vec<&str> = text.lines().collect();
    let max_cols = lines.iter().map(|l| l.len()).max().unwrap_or(0) as u32;
    let num_lines = lines.len() as u32;
    if max_cols == 0 || num_lines == 0 {
        return (0, 0);
    }
    let scaled_char_w = char_width_for_height(target_char_h);
    let line_advance = line_advance_px(target_char_h, line_height);
    (
        max_cols * scaled_char_w,
        stacked_height_px(target_char_h, line_advance, num_lines),
    )
}

/// Return the `(w, h)` a [`render_text_wrapped`] call would produce.
pub fn measure_text_wrapped(text: &str, target_char_h: u32, max_width_px: u32) -> (u32, u32) {
    measure_text_wrapped_lh(text, target_char_h, max_width_px, 1.0)
}

/// Same as [`measure_text_wrapped`], with a CSS-`line-height` ratio
/// applied to the inter-line stride.
pub fn measure_text_wrapped_lh(
    text: &str,
    target_char_h: u32,
    max_width_px: u32,
    line_height: f32,
) -> (u32, u32) {
    if target_char_h == 0 || max_width_px == 0 {
        return (0, 0);
    }
    let scaled_char_w = char_width_for_height(target_char_h);
    let max_cols = (max_width_px / scaled_char_w.max(1)) as usize;
    let wrapped = wrap_text(text, max_cols.max(1));
    measure_text_height_lh(&wrapped, target_char_h, line_height)
}

/// Return the `(w, h)` a [`render_lines_fitted`] call would produce.
pub fn measure_lines_fitted(lines: &[(&str, [u8; 4])], max_width_px: u32) -> (u32, u32) {
    measure_lines_fitted_lh(lines, max_width_px, 1.0)
}

/// Same as [`measure_lines_fitted`], with a CSS-`line-height` ratio
/// applied to the inter-line stride.
pub fn measure_lines_fitted_lh(
    lines: &[(&str, [u8; 4])],
    max_width_px: u32,
    line_height: f32,
) -> (u32, u32) {
    let (char_w, char_h) = fit_char_h_for_lines(lines, max_width_px);
    if char_w == 0 || char_h == 0 {
        return (0, 0);
    }
    let longest = lines.iter().map(|(s, _)| s.len()).max().unwrap_or(0) as u32;
    let line_advance = line_advance_px(char_h, line_height);
    (
        longest * char_w,
        stacked_height_px(char_h, line_advance, lines.len() as u32),
    )
}

/// Internal: derive `(char_w, char_h)` such that
/// `longest * char_w ≤ max_width_px`. `char_w` is computed via
/// [`char_width_for_height`] which rounds half-up, so the naive formula
/// `floor(max_width_px / longest * BASE_H/BASE_W)` can overshoot by 1
/// when char_w rounds up. We post-correct by decrementing `char_h`
/// until `longest * char_w ≤ max_width_px`.
pub(crate) fn fit_char_h_for_lines(lines: &[(&str, [u8; 4])], max_width_px: u32) -> (u32, u32) {
    if lines.is_empty() || max_width_px == 0 {
        return (0, 0);
    }
    let longest = lines.iter().map(|(s, _)| s.len()).max().unwrap_or(1) as u32;
    if longest == 0 {
        return (0, 0);
    }
    let mut char_h = (max_width_px as f32 / longest as f32
        * (BASE_CHAR_H as f32 / BASE_CHAR_W as f32))
        .floor() as u32;
    char_h = char_h.clamp(BASE_CHAR_H / 4, BASE_CHAR_H);
    // Post-correct: char_width_for_height rounds half-up, so the
    // initial char_h can produce `longest * char_w > max_width_px` by
    // 1 character pixel. Decrement until it fits or we hit the floor.
    while char_h > BASE_CHAR_H / 4 && longest * char_width_for_height(char_h) > max_width_px {
        char_h -= 1;
    }
    (char_width_for_height(char_h), char_h)
}

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
        assert!(
            white_count > 10,
            "A glyph should have lit pixels, got {white_count}"
        );
    }

    #[test]
    fn render_multiline() {
        let (buf, w, h) = render_text("AB\nCD", [255; 4], [0; 4]);
        assert_eq!(w, 2 * BASE_CHAR_W);
        assert_eq!(h, 2 * BASE_CHAR_H);
        assert_eq!(buf.len(), (w * h * 4) as usize);
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
                    if strip.get_pixel(x, y) > 128 {
                        lit += 1;
                    }
                }
            }
            assert!(
                lit > 10,
                "digit '{}' should have lit pixels, got {lit}",
                d as char
            );
        }
    }

    #[test]
    fn wrap_short_text_unchanged() {
        assert_eq!(wrap_text("hello world", 80), "hello world");
    }

    #[test]
    fn wrap_breaks_on_space() {
        assert_eq!(wrap_text("aaa bbb ccc", 7), "aaa bbb\nccc");
    }

    #[test]
    fn wrap_preserves_newlines() {
        assert_eq!(wrap_text("aaa\nbbb ccc", 7), "aaa\nbbb ccc");
    }

    #[test]
    fn wrap_long_word_own_line() {
        assert_eq!(wrap_text("hi verylongword ok", 8), "hi\nverylongword\nok");
    }

    #[test]
    fn wrapped_render_produces_more_lines() {
        let text = "zdsim:0.13 delta:[80,40,0] 23.4% differ category:perceptual";
        let narrow = render_text_wrapped(text, [255; 4], [0; 4], 20, 200);
        let wide = render_text_wrapped(text, [255; 4], [0; 4], 20, 2000);
        // Narrow should be taller (more lines)
        assert!(
            narrow.2 > wide.2,
            "narrow {} should be taller than wide {}",
            narrow.2,
            wide.2
        );
    }
}
