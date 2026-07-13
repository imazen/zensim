//! Character → glyph mapping with automapping and a hex-in-box notdef.
//!
//! Applied by every text composer in [`crate::font`] (both the bitmap
//! strip and `sdf-font` paths share it). Pipeline per character:
//!
//! 1. **Direct atlas hit** — ASCII 32–126 plus Δ (the strip's glyph 95).
//! 2. **Format-character skip** — variation selectors (VS16 etc.), ZWJ /
//!    ZWNJ / ZWSP, BOM, skin-tone modifiers: rendered as *nothing*, no
//!    cell advance. Without this every emoji grows a tofu tail
//!    (measured: VS16 in 70 / ZWJ in 15 of 3,466 READMEs).
//! 3. **Fullwidth fold** — U+FF01–FF5E are ASCII + 0xFEE0 (algorithmic);
//!    ideographic 。、「」 map to `. , "`.
//! 4. **Semantic twin table** — emoji-class symbols map to the
//!    monochrome glyph carrying the same meaning (✅→✓, ❌→✗, 🔴→●,
//!    ⭐→★ …). A twin only applies when the atlas actually covers the
//!    target, so entries targeting future charset tiers (Greek/symbol
//!    atlases) light up automatically when coverage grows.
//! 5. **Hex-in-box notdef** — everything else draws a bordered box with
//!    the codepoint's hex digits (Firefox-style): `🚀` → `[1F680]`.
//!    Legible to humans and LLMs, and unlike the previous clamp-to-Δ
//!    behavior it cannot masquerade as real report data.
//!
//! Coverage math and corpus provenance:
//! `benchmarks/sdf_font_atlas_exploration_2026-07-13.md` (README-corpus
//! audit + automap addenda).

/// Outcome of mapping one character.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Mapped {
    /// Renderable via the glyph strip at this index.
    Glyph(u32),
    /// Zero-width: consume no cell, draw nothing.
    Skip,
    /// No rendering available — draw the hex-in-box notdef for this
    /// codepoint (one cell wide).
    NotDef(u32),
}

/// Direct atlas coverage: ASCII 32–126 at 0–94, Δ at 95. The single
/// source of truth the twin table resolves against — extend here when
/// the atlas grows a tier.
fn direct_index(ch: char) -> Option<u32> {
    let code = ch as u32;
    if (0x20..0x20 + 95).contains(&code) {
        Some(code - 0x20)
    } else if ch == '\u{0394}' {
        Some(95) // Δ
    } else {
        None
    }
}

/// Format / invisible characters: render nothing, advance nothing.
fn is_skip(cp: u32) -> bool {
    matches!(cp,
        0xFE00..=0xFE0F      // variation selectors (VS1–VS16)
        | 0x200B..=0x200D    // ZWSP, ZWNJ, ZWJ
        | 0x2060             // word joiner
        | 0xFEFF             // BOM / ZWNBSP
        | 0xAD               // soft hyphen (render nothing when not breaking)
        | 0x1F3FB..=0x1F3FF  // emoji skin-tone modifiers
        | 0xE0020..=0xE007F  // tag characters (flag sequences)
    )
}

/// Fullwidth / ideographic / lookalike forms → ASCII-range equivalents.
///
/// Space folds matter more than they look: NBSP is the second-most
/// common non-ASCII character on the web (22/36 pages, 7.5 k
/// occurrences in the 2026-07-13 site scan) and would otherwise
/// hex-box in every rendered-HTML-derived string.
fn fold_width(cp: u32) -> Option<char> {
    match cp {
        0xFF01..=0xFF5E => char::from_u32(cp - 0xFEE0),
        0xA0 => Some(' '),            // no-break space
        0x2000..=0x200A => Some(' '), // en/em/thin/hair… spaces
        0x202F => Some(' '),          // narrow no-break space (fr)
        0x205F => Some(' '),          // medium mathematical space
        0x2010 | 0x2011 => Some('-'), // hyphen, non-breaking hyphen
        0x2012 => Some('-'),          // figure dash
        0x3001 => Some(','),          // ideographic comma
        0x3002 => Some('.'),          // ideographic full stop
        0x300C | 0x300D => Some('"'), // corner brackets
        0x3000 => Some(' '),          // ideographic space
        _ => None,
    }
}

/// Emoji-class → monochrome semantic twin. Targets that the atlas
/// doesn't cover *yet* (arrows, ★, ●, ❤, ⊘ live in the planned
/// console-lean tiers) fall through to notdef today and resolve
/// automatically once coverage grows. Sorted by codepoint.
const TWINS: &[(u32, char)] = &[
    (0x26AA, '○'), // medium white circle
    (0x26AB, '●'), // medium black circle
    (0x26D4, '⊘'), // no entry
    (0x2705, '✓'), // white heavy check mark
    (0x2716, '×'), // heavy multiplication x
    (0x274C, '✗'), // cross mark
    (0x274E, '✗'), // negative squared cross mark
    (0x2753, '?'), // black question mark ornament
    (0x2757, '!'), // heavy exclamation mark
    (0x2795, '+'), // heavy plus
    (0x2796, '-'), // heavy minus
    (0x2B05, '←'), // leftwards black arrow
    (0x2B06, '↑'),
    (0x2B07, '↓'),
    (0x2B1B, '■'), // black large square
    (0x2B1C, '□'),
    (0x2B50, '★'), // white medium star
    (0x2B51, '★'),
    (0x2B95, '→'),  // rightwards black arrow
    (0x1F499, '❤'), // colored hearts
    (0x1F49A, '❤'),
    (0x1F49B, '❤'),
    (0x1F49C, '❤'),
    (0x1F534, '●'), // large red circle
    (0x1F535, '●'),
    (0x1F5A4, '❤'), // black heart
    (0x1F6AB, '⊘'), // no entry sign
    (0x1F788, '●'), // very heavy white circle
    (0x1F7E0, '●'), // colored circles
    (0x1F7E1, '●'),
    (0x1F7E2, '●'),
    (0x1F7E3, '●'),
    (0x1F7E4, '●'),
    (0x1F9E1, '❤'), // orange heart
];

/// Map one character to its rendering outcome.
pub(crate) fn map_char(ch: char) -> Mapped {
    if let Some(idx) = direct_index(ch) {
        return Mapped::Glyph(idx);
    }
    let cp = ch as u32;
    if is_skip(cp) {
        return Mapped::Skip;
    }
    if let Some(folded) = fold_width(cp)
        && let Some(idx) = direct_index(folded)
    {
        return Mapped::Glyph(idx);
    }
    if let Ok(i) = TWINS.binary_search_by_key(&cp, |&(c, _)| c)
        && let Some(idx) = direct_index(TWINS[i].1)
    {
        return Mapped::Glyph(idx);
    }
    Mapped::NotDef(cp)
}

/// Number of cells a string occupies after mapping (skips are
/// zero-width; everything else, including notdef boxes, is one cell).
pub(crate) fn cell_count(text: &str) -> u32 {
    text.chars()
        .filter(|&c| map_char(c) != Mapped::Skip)
        .count() as u32
}

// ─── Hex-in-box notdef rendering ────────────────────────────────────────

/// 3×5 hex digit micro-font, rows top→bottom, 3 bits per row (MSB =
/// left pixel). 30 bytes for the whole set.
const HEX3X5: [u16; 16] = [
    0b111_101_101_101_111, // 0
    0b010_110_010_010_111, // 1
    0b111_001_111_100_111, // 2
    0b111_001_111_001_111, // 3
    0b101_101_111_001_001, // 4
    0b111_100_111_001_111, // 5
    0b111_100_111_101_111, // 6
    0b111_001_001_001_001, // 7
    0b111_101_111_101_111, // 8
    0b111_101_111_001_111, // 9
    0b111_101_111_101_101, // A
    0b110_101_110_101_110, // B
    0b111_100_100_100_111, // C
    0b110_101_101_101_110, // D
    0b111_100_111_100_111, // E
    0b111_100_111_100_100, // F
];

/// Draw the hex-in-box notdef for `cp` into an RGBA buffer cell at
/// `(x_base, y_base)` with cell size `char_w × char_h`. Border and
/// digits are drawn in solid `fg` (the cell's background was already
/// filled by the composer). Digits: 4 hex digits in a 2-col × 2-row
/// grid for BMP codepoints, 6 in 2-col × 3-row above that (column-major
/// suits the tall cell). Cells too small for legible digits draw the
/// border only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_notdef(
    buf: &mut [u8],
    out_w: u32,
    out_h: u32,
    x_base: u32,
    y_base: u32,
    char_w: u32,
    char_h: u32,
    cp: u32,
    fg: [u8; 4],
) {
    let mut set = |x: u32, y: u32| {
        let (px, py) = (x_base + x, y_base + y);
        if px < out_w && py < out_h {
            let off = ((py * out_w + px) * 4) as usize;
            buf[off..off + 4].copy_from_slice(&fg);
        }
    };

    if char_w < 3 || char_h < 3 {
        for y in 0..char_h {
            for x in 0..char_w {
                set(x, y);
            }
        }
        return;
    }

    // Box border, inset one pixel, thickness scaled to cell size.
    let t = (char_h / 16).max(1);
    let (bx0, by0) = (1u32, 1u32);
    let (bx1, by1) = (char_w - 2, char_h - 2);
    for y in by0..=by1 {
        for x in bx0..=bx1 {
            if x < bx0 + t || x > bx1 - t || y < by0 + t || y > by1 - t {
                set(x, y);
            }
        }
    }

    // Digits: 2 columns × (2 or 3) rows of 3×5 micro-glyphs — column-
    // major keeps the grid narrow enough for the tall monospace cell.
    let digits: Vec<u32> = {
        let n = if cp > 0xFFFF { 6 } else { 4 };
        (0..n).rev().map(|i| (cp >> (i * 4)) & 0xF).collect()
    };
    let cols = 2u32;
    let rows = digits.len() as u32 / 2;
    let inner_x = bx0 + t + 1;
    let inner_y = by0 + t + 1;
    let inner_w = (bx1 - t).saturating_sub(inner_x);
    let inner_h = (by1 - t).saturating_sub(inner_y);
    // Per-digit slot incl. 1-unit gap; need >= 3x5 scaled by >= 1.
    let slot_w = inner_w / cols;
    let slot_h = inner_h / rows;
    let scale = ((slot_w.saturating_sub(1)) / 3).min((slot_h.saturating_sub(1)) / 5);
    if scale == 0 {
        return; // border-only tofu at tiny sizes
    }
    let (dw, dh) = (3 * scale, 5 * scale);
    let grid_w = cols * dw + (cols - 1) * scale;
    let grid_h = rows * dh + (rows - 1) * scale;
    let gx0 = inner_x + (inner_w.saturating_sub(grid_w)) / 2;
    let gy0 = inner_y + (inner_h.saturating_sub(grid_h)) / 2;

    for (i, &d) in digits.iter().enumerate() {
        let (row, col) = ((i as u32) / cols, (i as u32) % cols);
        let ox = gx0 + col * (dw + scale);
        let oy = gy0 + row * (dh + scale);
        let bits = HEX3X5[d as usize];
        for ry in 0..5u32 {
            for rx in 0..3u32 {
                if bits >> ((4 - ry) * 3 + (2 - rx)) & 1 == 1 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            set(ox + rx * scale + sx, oy + ry * scale + sy);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_and_delta_hit_atlas() {
        assert_eq!(map_char('A'), Mapped::Glyph(('A' as u32) - 0x20));
        assert_eq!(map_char(' '), Mapped::Glyph(0));
        assert_eq!(map_char('~'), Mapped::Glyph(94));
        assert_eq!(map_char('Δ'), Mapped::Glyph(95));
    }

    #[test]
    fn format_chars_skip() {
        for c in ['\u{FE0F}', '\u{200D}', '\u{200B}', '\u{FEFF}', '\u{1F3FB}'] {
            assert_eq!(map_char(c), Mapped::Skip, "{c:?}");
        }
    }

    #[test]
    fn fullwidth_folds_to_ascii() {
        assert_eq!(map_char('（'), map_char('('));
        assert_eq!(map_char('ｘ'), map_char('x'));
        assert_eq!(map_char('，'), map_char(','));
        assert_eq!(map_char('。'), map_char('.'));
    }

    #[test]
    fn spaces_and_hyphen_lookalikes_fold() {
        for c in ['\u{A0}', '\u{2009}', '\u{202F}', '\u{2003}'] {
            assert_eq!(map_char(c), map_char(' '), "{c:?} should fold to space");
        }
        for c in ['\u{2010}', '\u{2011}', '\u{2012}'] {
            assert_eq!(map_char(c), map_char('-'), "{c:?} should fold to hyphen");
        }
        assert_eq!(
            map_char('\u{AD}'),
            Mapped::Skip,
            "soft hyphen is zero-width"
        );
    }

    #[test]
    fn twins_resolve_when_atlas_covers_target() {
        // ASCII-target twins resolve today.
        assert_eq!(map_char('❓'), map_char('?'));
        assert_eq!(map_char('❗'), map_char('!'));
        assert_eq!(map_char('➖'), map_char('-'));
        // Symbol-target twins (✓ ● ★ …) await the console-lean atlas.
        assert_eq!(map_char('✅'), Mapped::NotDef(0x2705));
        assert_eq!(map_char('🔴'), Mapped::NotDef(0x1F534));
    }

    #[test]
    fn twin_table_is_sorted_for_binary_search() {
        assert!(TWINS.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn unknown_is_notdef_not_clamp() {
        assert_eq!(map_char('🚀'), Mapped::NotDef(0x1F680));
        assert_eq!(map_char('中'), Mapped::NotDef(0x4E2D));
    }

    #[test]
    fn cell_count_ignores_skips() {
        assert_eq!(cell_count("ab"), 2);
        assert_eq!(cell_count("✔\u{FE0F}"), 1);
        assert_eq!(cell_count("a\u{200D}b"), 2);
        assert_eq!(cell_count(""), 0);
    }

    fn notdef_cell(cp: u32, w: u32, h: u32) -> Vec<u8> {
        let mut buf = vec![0u8; (w * h * 4) as usize];
        draw_notdef(&mut buf, w, h, 0, 0, w, h, cp, [255; 4]);
        buf
    }

    #[test]
    fn notdef_draws_border_and_distinct_digits() {
        let a = notdef_cell(0x1F680, 26, 54);
        let b = notdef_cell(0x1F389, 26, 54);
        assert!(a.contains(&255), "border/digits drawn");
        assert_ne!(a, b, "different codepoints render differently");
        // Border pixel present near the top-left inset corner.
        let off = ((2 * 26 + 2) * 4) as usize;
        assert_eq!(a[off], 255);
    }

    #[test]
    fn notdef_tiny_cell_does_not_panic() {
        for (w, h) in [(1, 1), (2, 3), (3, 5), (5, 8)] {
            let _ = notdef_cell(0x1F680, w, h);
        }
    }
}
