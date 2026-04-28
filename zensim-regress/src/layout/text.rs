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
#[non_exhaustive]
pub struct TextSpec {
    pub style: TextStyle,
    /// Background color the glyph rasterizer alpha-blends against — set
    /// to match the strip behind the text for clean edges.
    pub bg: Color,
    /// Line-height multiplier (CSS `line-height: <ratio>`). Applies to
    /// every line stride in multi-line text — `Lines` and the multi-line
    /// output of `Wrapped`. The first line still occupies `char_h`
    /// pixels; each subsequent line advances by `round(char_h *
    /// line_height)` (clamped to ≥ 1px). Default `1.0` matches the
    /// pre-existing tight stacking. `1.2` gives a typographic gap.
    pub line_height: f32,
}

/// Default line-height multiplier — `1.0` reproduces the legacy
/// zero-leading stacking. CSS browsers default to ~1.2.
pub const DEFAULT_LINE_HEIGHT: f32 = 1.0;

impl TextSpec {
    pub fn lines(lines: Vec<(impl Into<String>, Color)>, bg: Color) -> Self {
        Self {
            style: TextStyle::Lines(lines.into_iter().map(|(s, c)| (s.into(), c)).collect()),
            bg,
            line_height: DEFAULT_LINE_HEIGHT,
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
            line_height: DEFAULT_LINE_HEIGHT,
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
            line_height: DEFAULT_LINE_HEIGHT,
        }
    }

    /// Set the line-height multiplier. `1.0` = tight stack (legacy), `1.2` =
    /// CSS-default typographic gap, `1.4` = airier. Values ≤ 0 or non-finite
    /// are clamped to `DEFAULT_LINE_HEIGHT`.
    pub fn line_height(mut self, ratio: f32) -> Self {
        self.line_height = if ratio.is_finite() && ratio > 0.0 {
            ratio
        } else {
            DEFAULT_LINE_HEIGHT
        };
        self
    }

    /// Multiply explicit `char_h` (in `Fixed` / `Wrapped` variants) by
    /// `scale`. `Lines` is auto-fit and unchanged. `line_height` is a
    /// ratio so it is preserved unchanged.
    pub(super) fn scaled(self, scale: f32) -> Self {
        let scale_h = |h: u32| ((h as f32) * scale).round() as u32;
        Self {
            bg: self.bg,
            line_height: self.line_height,
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
        let lh = self.line_height;
        let result = match &self.style {
            TextStyle::Lines(lines) => {
                let refs: Vec<(&str, Color)> =
                    lines.iter().map(|(s, c)| (s.as_str(), *c)).collect();
                let effective_max_w = lines_height_bounded_max_w(&refs, max_w, max_h, lh);
                font::render_lines_fitted_lh(&refs, self.bg, effective_max_w, lh)
            }
            TextStyle::Fixed { text, fg, char_h } => {
                let ch = char_h_within_height(*char_h, max_h, 1, lh);
                font::render_text_height_lh(text, *fg, self.bg, ch, lh)
            }
            TextStyle::Wrapped { text, fg, char_h } => {
                let n_lines = wrapped_line_count(text, *char_h, max_w);
                let ch = char_h_within_height(*char_h, max_h, n_lines, lh);
                font::render_text_wrapped_lh(text, *fg, self.bg, ch, max_w, lh)
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
        let lh = self.line_height;
        let (w, h) = match &self.style {
            TextStyle::Lines(lines) => {
                let refs: Vec<(&str, Color)> =
                    lines.iter().map(|(s, c)| (s.as_str(), *c)).collect();
                let effective_max_w = lines_height_bounded_max_w(&refs, max_w, max_h, lh);
                font::measure_lines_fitted_lh(&refs, effective_max_w, lh)
            }
            TextStyle::Fixed { text, char_h, .. } => {
                let ch = char_h_within_height(*char_h, max_h, 1, lh);
                font::measure_text_height_lh(text, ch, lh)
            }
            TextStyle::Wrapped { text, char_h, .. } => {
                let n_lines = wrapped_line_count(text, *char_h, max_w);
                let ch = char_h_within_height(*char_h, max_h, n_lines, lh);
                font::measure_text_wrapped_lh(text, ch, max_w, lh)
            }
        };
        safety::clamp_size(Size::new(w, h))
    }
}

/// Estimate how many lines a `Wrapped` text will produce at `(char_h, max_w)`.
/// Used by [`char_h_within_height`] to bound the requested char height when
/// `max_h` is tight.
fn wrapped_line_count(text: &str, char_h: u32, max_w: u32) -> u32 {
    if char_h == 0 || max_w == 0 || text.is_empty() {
        return 1;
    }
    // measure_text_wrapped tells us the natural total height at the
    // requested char_h; divide by char_h to get the line count.
    let (_, total_h) = font::measure_text_wrapped_lh(text, char_h, max_w, 1.0);
    if total_h == 0 {
        1
    } else {
        total_h.div_ceil(char_h).max(1)
    }
}

/// Given a `Lines` text and a `(max_w, max_h)` rect, return the
/// *effective* `max_w` to feed [`font::measure_lines_fitted_lh`] /
/// [`font::render_lines_fitted_lh`] so that the resulting char height
/// — stacked at `line_height` ratio — also fits within `max_h`.
///
/// `font::*_lines_fitted` derives char height from `max_w / longest *
/// (BASE_H / BASE_W)`, then clamps to `[BASE_H/4, BASE_H]`. Total
/// stacked height is `(n-1) * char_h * line_height + char_h`. Inverting
/// the formula gives a smaller `max_w` that yields the desired char
/// height when height is the binding constraint.
fn lines_height_bounded_max_w(
    refs: &[(&str, Color)],
    max_w: u32,
    max_h: u32,
    line_height: f32,
) -> u32 {
    if max_h == 0 || refs.is_empty() {
        return max_w;
    }
    let lh = if line_height.is_finite() && line_height > 0.0 {
        line_height.max(1.0) as f64
    } else {
        1.0
    };
    let longest = refs.iter().map(|(s, _)| s.len()).max().unwrap_or(1).max(1) as u32;
    let n_lines = refs.len().max(1) as u32;
    // char_h_at_full_w = max_w * 54 / (longest * 26)
    let h_at_full_w = (max_w as u64 * 54 / (longest as u64 * 26)) as u32;
    // total_h = (n-1) * h * lh + h = h * ((n-1) * lh + 1)
    let stacked_factor = ((n_lines as f64 - 1.0) * lh + 1.0).max(1.0);
    let total_h_at_full_w = ((h_at_full_w as f64) * stacked_factor) as u32;
    if total_h_at_full_w <= max_h {
        return max_w;
    }
    // Solve for the largest char_h such that h * stacked_factor <= max_h.
    let target_char_h = ((max_h as f64) / stacked_factor).floor() as u32;
    if target_char_h == 0 {
        return 0;
    }
    let derived_max_w = ((target_char_h as u64) * (longest as u64) * 26 / 54) as u32;
    derived_max_w.min(max_w).max(1)
}

/// Cap a fixed `char_h` to fit within `max_h` for `n_lines` lines stacked
/// at `line_height` ratio, preserving the safety upper bound.
///
/// Total stacked height = `(n - 1) * char_h * line_height + char_h`
/// = `char_h * ((n - 1) * line_height + 1)`. Solving for char_h with
/// the constraint `total ≤ max_h` gives the cap.
fn char_h_within_height(char_h: u32, max_h: u32, n_lines: u32, line_height: f32) -> u32 {
    let base = safety::clamp_char_h(char_h);
    if max_h == 0 || n_lines == 0 {
        return base;
    }
    let lh = if line_height.is_finite() && line_height > 0.0 {
        line_height.max(1.0)
    } else {
        1.0
    };
    let n = n_lines as f32;
    // total / char_h = (n - 1) * lh + 1
    let stacked_factor = ((n - 1.0) * lh + 1.0).max(1.0);
    let cap = ((max_h as f32) / stacked_factor).floor() as u32;
    base.min(cap.max(1))
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

    #[test]
    fn line_height_default_is_tight_legacy_stacking() {
        let spec = TextSpec::wrapped(
            "this is enough text to require multiple wrapped lines",
            WHITE,
            BLACK,
            8,
        );
        assert_eq!(spec.line_height, DEFAULT_LINE_HEIGHT);
        let s = spec.natural(80, 1000);
        // n lines × 8px (legacy: zero leading).
        assert_eq!(
            s.h % 8,
            0,
            "default line_height=1.0 stacks at exactly char_h: {s:?}"
        );
    }

    #[test]
    fn line_height_increases_wrapped_text_height() {
        let mk = |lh: f32| {
            TextSpec::wrapped(
                "this is enough text to require multiple wrapped lines",
                WHITE,
                BLACK,
                8,
            )
            .line_height(lh)
        };
        let h_tight = mk(1.0).natural(80, 1000).h;
        let h_loose = mk(1.5).natural(80, 1000).h;
        assert!(
            h_loose > h_tight,
            "line_height=1.5 should be taller than 1.0 (got loose={h_loose}, tight={h_tight})",
        );
    }

    #[test]
    fn line_height_increases_lines_height() {
        let mk = |lh: f32| {
            TextSpec::lines(
                vec![
                    ("LINE 1".to_string(), WHITE),
                    ("LINE 2".to_string(), WHITE),
                    ("LINE 3".to_string(), WHITE),
                ],
                BLACK,
            )
            .line_height(lh)
        };
        let h_tight = mk(1.0).natural(200, 1000).h;
        let h_loose = mk(1.4).natural(200, 1000).h;
        assert!(
            h_loose > h_tight,
            "line_height=1.4 should be taller than 1.0 (got loose={h_loose}, tight={h_tight})",
        );
    }

    #[test]
    fn line_height_below_one_clamps_to_one() {
        // line_advance_px floors at char_h; values < 1.0 don't cause line overlap.
        let lh_low = TextSpec::wrapped("wrapped text wrapped text wrapped text", WHITE, BLACK, 8)
            .line_height(0.5);
        let lh_one = TextSpec::wrapped("wrapped text wrapped text wrapped text", WHITE, BLACK, 8)
            .line_height(1.0);
        assert_eq!(lh_low.natural(60, 1000), lh_one.natural(60, 1000));
    }

    #[test]
    fn line_height_invalid_falls_back_to_default() {
        let spec = TextSpec::fixed("hi", WHITE, BLACK, 8).line_height(f32::NAN);
        assert_eq!(spec.line_height, DEFAULT_LINE_HEIGHT);
        let spec = TextSpec::fixed("hi", WHITE, BLACK, 8).line_height(-2.0);
        assert_eq!(spec.line_height, DEFAULT_LINE_HEIGHT);
    }

    #[test]
    fn line_height_natural_matches_rasterize() {
        let spec = TextSpec::wrapped(
            "wrap me at narrow width to trigger multiple lines",
            WHITE,
            BLACK,
            6,
        )
        .line_height(1.4);
        let measured = spec.natural(60, 0);
        let (_, w, h) = spec.rasterize(60, 0);
        assert_eq!(
            measured,
            Size::new(w, h),
            "natural() and rasterize() must agree under non-default line_height",
        );
    }

    #[test]
    fn line_height_height_constrained_clamps_char_h() {
        // Wide-and-short box: width is plentiful (so the natural char_h
        // derived from width is the BASE_CHAR_H ceiling), but height is
        // binding. With line_height=1.5 and 2 lines, the largest char_h
        // that fits in 60px is 60 / 2.5 = 24. Pre-fix this test would
        // have stacked at exactly 2 × char_h ignoring the leading.
        let spec = TextSpec::lines(
            vec![("LINE A".to_string(), WHITE), ("LINE B".to_string(), WHITE)],
            BLACK,
        )
        .line_height(1.5);
        let s = spec.natural(2000, 60);
        assert!(
            s.h <= 60,
            "stacked height {s:?} must fit max_h=60 under line_height=1.5",
        );
        // And it must actually use the available height — i.e., not
        // collapse to the legacy 2 × char_h ignoring leading.
        let tight = TextSpec::lines(
            vec![("LINE A".to_string(), WHITE), ("LINE B".to_string(), WHITE)],
            BLACK,
        )
        .natural(2000, 60);
        assert!(
            s.h <= tight.h,
            "lh=1.5 must shrink char_h further than lh=1.0 under tight max_h \
             (got lh1.5_h={}, lh1.0_h={})",
            s.h,
            tight.h,
        );
    }
}
