//! Text leaf — wraps the [`crate::font`] entry points with a measure /
//! rasterize split so the layout module's measure pass can compute
//! sizes without rasterizing.
//!
//! `bg` is pre-blended into glyph alpha by the font rasterizer, so set
//! it to match the background behind the text for clean edges.

use super::color::Color;
use super::geom::Size;
use super::safety;
use crate::font;

#[derive(Clone, Debug)]
pub enum TextStyle {
    /// One or more lines, each with its own color, sized so the longest
    /// fits within the constraint width. [`font::render_lines_fitted`].
    Lines(Vec<(String, Color)>),
    /// Single line at a fixed character pixel-height. [`font::render_text_height`].
    Fixed {
        text: String,
        fg: Color,
        char_h: u32,
    },
    /// Word-wrapped text at a fixed character pixel-height.
    /// [`font::render_text_wrapped`].
    Wrapped {
        text: String,
        fg: Color,
        char_h: u32,
    },
}

#[derive(Clone, Debug)]
pub struct TextSpec {
    pub style: TextStyle,
    /// Background color the glyph rasterizer alpha-blends against — set
    /// to match the strip behind the text for clean edges.
    pub bg: Color,
}

impl TextSpec {
    pub fn lines(lines: Vec<(impl Into<String>, Color)>, bg: Color) -> Self {
        Self {
            style: TextStyle::Lines(lines.into_iter().map(|(s, c)| (s.into(), c)).collect()),
            bg,
        }
    }
    pub fn fixed(text: impl Into<String>, fg: Color, bg: Color, char_h: u32) -> Self {
        Self {
            style: TextStyle::Fixed {
                text: text.into(),
                fg,
                char_h,
            },
            bg,
        }
    }
    pub fn wrapped(text: impl Into<String>, fg: Color, bg: Color, char_h: u32) -> Self {
        Self {
            style: TextStyle::Wrapped {
                text: text.into(),
                fg,
                char_h,
            },
            bg,
        }
    }

    /// Multiply explicit `char_h` (in `Fixed` / `Wrapped` variants) by
    /// `scale`. `Lines` is auto-fit and unchanged.
    pub(super) fn scaled(self, scale: f32) -> Self {
        let scale_h = |h: u32| ((h as f32) * scale).round() as u32;
        Self {
            bg: self.bg,
            style: match self.style {
                TextStyle::Lines(lines) => TextStyle::Lines(lines),
                TextStyle::Fixed { text, fg, char_h } => TextStyle::Fixed {
                    text,
                    fg,
                    char_h: scale_h(char_h),
                },
                TextStyle::Wrapped { text, fg, char_h } => TextStyle::Wrapped {
                    text,
                    fg,
                    char_h: scale_h(char_h),
                },
            },
        }
    }

    /// Rasterize against `(max_w, max_h)` — produces the pixel buffer
    /// used at paint. Char height shrinks to fit *either* axis (so a
    /// short string in a tall-narrow rect doesn't overflow downward).
    /// `max_h == 0` means unbounded height; `max_w == 0` returns empty.
    pub(super) fn rasterize(&self, max_w: u32, max_h: u32) -> (Vec<u8>, u32, u32) {
        let max_w = safety::clamp_dim(max_w);
        let max_h = safety::clamp_dim(max_h);
        if max_w == 0 {
            return (vec![], 0, 0);
        }
        let result = match &self.style {
            TextStyle::Lines(lines) => {
                let refs: Vec<(&str, Color)> =
                    lines.iter().map(|(s, c)| (s.as_str(), *c)).collect();
                let effective_max_w = lines_height_bounded_max_w(&refs, max_w, max_h);
                font::render_lines_fitted(&refs, self.bg, effective_max_w)
            }
            TextStyle::Fixed { text, fg, char_h } => {
                let ch = char_h_within_height(*char_h, max_h, 1);
                font::render_text_height(text, *fg, self.bg, ch)
            }
            TextStyle::Wrapped { text, fg, char_h } => {
                let ch = char_h_within_height(*char_h, max_h, 1);
                font::render_text_wrapped(text, *fg, self.bg, ch, max_w)
            }
        };
        let max_dim = safety::limits().max_dim;
        if result.1 > max_dim || result.2 > max_dim {
            return (vec![], 0, 0);
        }
        result
    }

    /// Cheap measure — no rasterization. Returns the natural `(w, h)`
    /// the rasterized buffer would have given `(max_w, max_h)`.
    pub(super) fn natural(&self, max_w: u32, max_h: u32) -> Size {
        let max_w = safety::clamp_dim(max_w);
        let max_h = safety::clamp_dim(max_h);
        if max_w == 0 {
            return Size::ZERO;
        }
        let (w, h) = match &self.style {
            TextStyle::Lines(lines) => {
                let refs: Vec<(&str, Color)> =
                    lines.iter().map(|(s, c)| (s.as_str(), *c)).collect();
                let effective_max_w = lines_height_bounded_max_w(&refs, max_w, max_h);
                font::measure_lines_fitted(&refs, effective_max_w)
            }
            TextStyle::Fixed { text, char_h, .. } => {
                let ch = char_h_within_height(*char_h, max_h, 1);
                font::measure_text_height(text, ch)
            }
            TextStyle::Wrapped { text, char_h, .. } => {
                let ch = char_h_within_height(*char_h, max_h, 1);
                font::measure_text_wrapped(text, ch, max_w)
            }
        };
        safety::clamp_size(Size::new(w, h))
    }
}

/// Given a `Lines` text and a `(max_w, max_h)` rect, return the
/// *effective* `max_w` to feed [`font::measure_lines_fitted`] /
/// [`font::render_lines_fitted`] so that the resulting char height
/// also fits within `max_h`.
///
/// `font::*_lines_fitted` derives char height from `max_w / longest *
/// (BASE_H / BASE_W)`, then clamps to `[BASE_H/4, BASE_H]`. Inverting
/// the formula gives a smaller `max_w` that yields the desired char
/// height when height is the binding constraint.
fn lines_height_bounded_max_w(refs: &[(&str, Color)], max_w: u32, max_h: u32) -> u32 {
    if max_h == 0 || refs.is_empty() {
        return max_w;
    }
    // Width-fit char height (after the lower clamp).
    let longest = refs.iter().map(|(s, _)| s.len()).max().unwrap_or(1).max(1) as u32;
    let n_lines = refs.len().max(1) as u32;
    let h_at_full_w = (max_w as u64 * 54 / (longest as u64 * 26)) as u32;
    let total_h_at_full_w = h_at_full_w * n_lines;
    if total_h_at_full_w <= max_h {
        return max_w;
    }
    let target_char_h = max_h / n_lines;
    if target_char_h == 0 {
        return 0;
    }
    let derived_max_w = ((target_char_h as u64) * (longest as u64) * 26 / 54) as u32;
    derived_max_w.min(max_w).max(1)
}

/// Cap a fixed `char_h` to fit within `max_h` for `n_lines` lines,
/// preserving the safety upper bound.
fn char_h_within_height(char_h: u32, max_h: u32, n_lines: u32) -> u32 {
    let base = safety::clamp_char_h(char_h);
    if max_h == 0 || n_lines == 0 {
        return base;
    }
    base.min(max_h / n_lines.max(1))
}

#[cfg(test)]
mod tests {
    use super::super::color::{BLACK, WHITE};
    use super::*;

    #[test]
    fn text_lines_measures_match_render() {
        let spec = TextSpec::lines(
            vec![("HELLO".to_string(), WHITE), ("WORLD!".to_string(), WHITE)],
            BLACK,
        );
        let measured = spec.natural(200, 0);
        let (_, w, h) = spec.rasterize(200, 0);
        assert_eq!(measured, Size::new(w, h));
    }

    #[test]
    fn text_lines_shrinks_to_fit_height() {
        // 60×30 box, 1 char "a": width-fit alone → char_h=54 (overflow).
        // Height-aware should produce char_h ≤ 30.
        let spec = TextSpec::lines(vec![("a".to_string(), WHITE)], BLACK);
        let s = spec.natural(60, 30);
        assert!(s.h <= 30, "expected text to fit in 30-tall box, got {s:?}");
    }
}
