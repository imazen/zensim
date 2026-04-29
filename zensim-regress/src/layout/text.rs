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
#[non_exhaustive]
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
    /// Word-wrapped text whose char height auto-fits the constraint
    /// rect. Binary-searches the largest `char_h ∈ [min_char_h,
    /// max_char_h]` such that the wrapped result fits in
    /// `(max_w, max_h)` honoring [`TextSpec::line_height`]. The
    /// "fit a paragraph into a box" mode.
    AutoFit {
        /// The text to render. Newlines are honored as hard breaks.
        text: String,
        /// Foreground color.
        fg: Color,
        /// Lower clamp on char height. Default 4.
        min_char_h: u32,
        /// Upper clamp on char height. Default [`crate::font::GLYPH_H`] (54).
        max_char_h: u32,
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
    /// Horizontal text padding as a fraction of the rendered `char_h`,
    /// applied to both left and right edges of the rasterized buffer.
    /// Default [`DEFAULT_PAD_H_FRAC`].
    pub pad_h_frac: f32,
    /// Vertical text padding as a fraction of `char_h`, top and bottom.
    /// Default [`DEFAULT_PAD_V_FRAC`].
    pub pad_v_frac: f32,
}

/// Default line-height multiplier — `1.0` reproduces the legacy
/// zero-leading stacking. CSS browsers default to ~1.2.
pub const DEFAULT_LINE_HEIGHT: f32 = 1.0;

/// Lower clamp on the auto-fit char-height search. Below ~4 px the
/// font's anti-aliasing produces unreadable smudges.
pub const DEFAULT_AUTOFIT_MIN_CHAR_H: u32 = 4;

/// Default per-axis text padding as a fraction of `char_h`. The padding
/// is rendered as `bg`-colored margin around the glyph block — purely
/// typographic breathing room. Override via
/// [`TextSpec::with_padding`].
///
/// `(0.10, 0.05)` mirrors typical CSS text padding inside a button —
/// small enough that it doesn't visually push other content, large
/// enough that text never butts directly against a colored container's
/// edge.
pub const DEFAULT_PAD_H_FRAC: f32 = 0.10;
/// See [`DEFAULT_PAD_H_FRAC`].
pub const DEFAULT_PAD_V_FRAC: f32 = 0.05;

impl TextSpec {
    /// Build a multi-color line strip. Each input string is split on `\n`;
    /// the segments share the input's color. Empty strings produce a blank
    /// line (rendered as one bg-colored row).
    pub fn lines(lines: Vec<(impl Into<String>, Color)>, bg: Color) -> Self {
        let mut expanded: Vec<(String, Color)> = Vec::with_capacity(lines.len());
        for (s, c) in lines {
            let s: String = s.into();
            if s.is_empty() {
                expanded.push((String::new(), c));
                continue;
            }
            let mut first = true;
            for part in s.split('\n') {
                if !first || !part.is_empty() {
                    expanded.push((part.to_string(), c));
                }
                first = false;
            }
            // Trailing `\n` produces an empty final segment; preserve it.
            if s.ends_with('\n') {
                expanded.push((String::new(), c));
            }
        }
        Self {
            style: TextStyle::Lines(expanded),
            bg,
            line_height: DEFAULT_LINE_HEIGHT,
            pad_h_frac: DEFAULT_PAD_H_FRAC,
            pad_v_frac: DEFAULT_PAD_V_FRAC,
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
            pad_h_frac: DEFAULT_PAD_H_FRAC,
            pad_v_frac: DEFAULT_PAD_V_FRAC,
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
            pad_h_frac: DEFAULT_PAD_H_FRAC,
            pad_v_frac: DEFAULT_PAD_V_FRAC,
        }
    }

    /// Word-wrapped text whose char height auto-fits the constraint rect.
    /// Defaults: `min_char_h = 4`, `max_char_h = font::GLYPH_H` (54).
    /// Adjust via [`TextSpec::with_min_em`] / [`TextSpec::with_max_em`].
    pub fn auto_fit(text: impl Into<String>, fg: Color, bg: Color) -> Self {
        Self {
            style: TextStyle::AutoFit {
                text: text.into(),
                fg,
                min_char_h: DEFAULT_AUTOFIT_MIN_CHAR_H,
                max_char_h: font::GLYPH_H,
            },
            bg,
            line_height: DEFAULT_LINE_HEIGHT,
            pad_h_frac: DEFAULT_PAD_H_FRAC,
            pad_v_frac: DEFAULT_PAD_V_FRAC,
        }
    }

    /// Override the lower char-height clamp on an [`TextStyle::AutoFit`]
    /// spec. No effect on other styles.
    pub fn with_min_char_h(mut self, n: u32) -> Self {
        if let TextStyle::AutoFit { min_char_h, .. } = &mut self.style {
            *min_char_h = n.max(1);
        }
        self
    }

    /// Override the upper char-height clamp on an [`TextStyle::AutoFit`]
    /// spec. No effect on other styles.
    pub fn with_max_char_h(mut self, n: u32) -> Self {
        if let TextStyle::AutoFit { max_char_h, .. } = &mut self.style {
            *max_char_h = n.max(1);
        }
        self
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

    /// Override the per-axis text padding fractions. Both default to
    /// [`DEFAULT_PAD_H_FRAC`] / [`DEFAULT_PAD_V_FRAC`] respectively;
    /// pass `(0.0, 0.0)` to disable padding entirely. Values are clamped
    /// to non-negative finite numbers.
    pub fn with_padding(mut self, pad_h_frac: f32, pad_v_frac: f32) -> Self {
        let clean = |v: f32| if v.is_finite() && v >= 0.0 { v } else { 0.0 };
        self.pad_h_frac = clean(pad_h_frac);
        self.pad_v_frac = clean(pad_v_frac);
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
                TextStyle::AutoFit {
                    text,
                    fg,
                    min_char_h,
                    max_char_h,
                } => TextStyle::AutoFit {
                    text,
                    fg,
                    min_char_h: scale_h(min_char_h),
                    max_char_h: scale_h(max_char_h),
                },
            },
            pad_h_frac: self.pad_h_frac,
            pad_v_frac: self.pad_v_frac,
        }
    }

    /// Rasterize against `(max_w, max_h)` — produces the pixel buffer
    /// used at paint. Char height shrinks to fit *either* axis (so a
    /// short string in a tall-narrow rect doesn't overflow downward).
    /// The output buffer includes the configured per-axis padding,
    /// `bg`-filled. `max_h == 0` means unbounded height; `max_w == 0`
    /// returns empty.
    pub(super) fn rasterize(&self, max_w: u32, max_h: u32) -> (Vec<u8>, u32, u32) {
        let max_w = safety::clamp_dim(max_w);
        let max_h = safety::clamp_dim(max_h);
        if max_w == 0 {
            return (vec![], 0, 0);
        }
        let (inner_max_w, inner_max_h, est_pad_h, est_pad_v) =
            self.estimate_pad_and_inner(max_w, max_h);
        let lh = self.line_height;
        let (inner_buf, inner_w, inner_h, char_h_used) = match &self.style {
            TextStyle::Lines(lines) => {
                let refs: Vec<(&str, Color)> =
                    lines.iter().map(|(s, c)| (s.as_str(), *c)).collect();
                let effective_max_w =
                    lines_height_bounded_max_w(&refs, inner_max_w, inner_max_h, lh);
                let (buf, w, h) = font::render_lines_fitted_lh(&refs, self.bg, effective_max_w, lh);
                let lines_n = refs.len().max(1) as u32;
                let ch_used = if h == 0 || lines_n == 0 {
                    0
                } else {
                    // total = (n-1)*ch*lh + ch ⇒ ch = total / ((n-1)*lh + 1)
                    let factor = ((lines_n as f32 - 1.0) * lh + 1.0).max(1.0);
                    ((h as f32) / factor).round() as u32
                };
                (buf, w, h, ch_used)
            }
            TextStyle::Fixed { text, fg, char_h } => {
                let ch = char_h_within_height(*char_h, inner_max_h, 1, lh);
                let (buf, w, h) = font::render_text_height_lh(text, *fg, self.bg, ch, lh);
                (buf, w, h, ch)
            }
            TextStyle::Wrapped { text, fg, char_h } => {
                let n_lines = wrapped_line_count(text, *char_h, inner_max_w);
                let ch = char_h_within_height(*char_h, inner_max_h, n_lines, lh);
                let (buf, w, h) =
                    font::render_text_wrapped_lh(text, *fg, self.bg, ch, inner_max_w, lh);
                (buf, w, h, ch)
            }
            TextStyle::AutoFit {
                text,
                fg,
                min_char_h,
                max_char_h,
            } => {
                let ch =
                    autofit_char_h(text, inner_max_w, inner_max_h, lh, *min_char_h, *max_char_h);
                let (buf, w, h) =
                    font::render_text_wrapped_lh(text, *fg, self.bg, ch, inner_max_w, lh);
                (buf, w, h, ch)
            }
        };
        // Recompute padding from the actually-used char_h so the
        // estimate-based inner_max accounts for itself.
        let (pad_h, pad_v) = self.padding_px(char_h_used.max(1));
        let (pad_h, pad_v) = (
            pad_h.max(est_pad_h.min(pad_h)),
            pad_v.max(est_pad_v.min(pad_v)),
        );
        let outer_w = inner_w.saturating_add(pad_h.saturating_mul(2));
        let outer_h = inner_h.saturating_add(pad_v.saturating_mul(2));
        let max_dim = safety::limits().max_dim;
        if outer_w > max_dim || outer_h > max_dim {
            return (vec![], 0, 0);
        }
        if inner_w == 0 || inner_h == 0 {
            return (vec![], 0, 0);
        }
        if pad_h == 0 && pad_v == 0 {
            return (inner_buf, inner_w, inner_h);
        }
        // Wrap the inner buffer with bg-colored padding on all four sides.
        let mut out = vec![0u8; (outer_w as usize) * (outer_h as usize) * 4];
        for px in out.chunks_exact_mut(4) {
            px.copy_from_slice(&self.bg);
        }
        for row in 0..inner_h {
            let dst_off = (((row + pad_v) as usize) * (outer_w as usize) + pad_h as usize) * 4;
            let src_off = (row as usize) * (inner_w as usize) * 4;
            let n = (inner_w as usize) * 4;
            out[dst_off..dst_off + n].copy_from_slice(&inner_buf[src_off..src_off + n]);
        }
        (out, outer_w, outer_h)
    }

    /// Cheap measure — no rasterization. Returns the natural `(w, h)`
    /// the rasterized buffer would have given `(max_w, max_h)`,
    /// including the configured per-axis padding.
    pub(super) fn natural(&self, max_w: u32, max_h: u32) -> Size {
        let max_w = safety::clamp_dim(max_w);
        let max_h = safety::clamp_dim(max_h);
        if max_w == 0 {
            return Size::ZERO;
        }
        let (inner_max_w, inner_max_h, est_pad_h, est_pad_v) =
            self.estimate_pad_and_inner(max_w, max_h);
        let lh = self.line_height;
        let ((w, h), char_h_used) = match &self.style {
            TextStyle::Lines(lines) => {
                let refs: Vec<(&str, Color)> =
                    lines.iter().map(|(s, c)| (s.as_str(), *c)).collect();
                let effective_max_w =
                    lines_height_bounded_max_w(&refs, inner_max_w, inner_max_h, lh);
                let (w, h) = font::measure_lines_fitted_lh(&refs, effective_max_w, lh);
                let lines_n = refs.len().max(1) as u32;
                let ch = if h == 0 {
                    0
                } else {
                    let factor = ((lines_n as f32 - 1.0) * lh + 1.0).max(1.0);
                    ((h as f32) / factor).round() as u32
                };
                ((w, h), ch)
            }
            TextStyle::Fixed { text, char_h, .. } => {
                let ch = char_h_within_height(*char_h, inner_max_h, 1, lh);
                (font::measure_text_height_lh(text, ch, lh), ch)
            }
            TextStyle::Wrapped { text, char_h, .. } => {
                let n_lines = wrapped_line_count(text, *char_h, inner_max_w);
                let ch = char_h_within_height(*char_h, inner_max_h, n_lines, lh);
                (font::measure_text_wrapped_lh(text, ch, inner_max_w, lh), ch)
            }
            TextStyle::AutoFit {
                text,
                min_char_h,
                max_char_h,
                ..
            } => {
                let ch =
                    autofit_char_h(text, inner_max_w, inner_max_h, lh, *min_char_h, *max_char_h);
                (font::measure_text_wrapped_lh(text, ch, inner_max_w, lh), ch)
            }
        };
        let (pad_h, pad_v) = self.padding_px(char_h_used.max(1));
        let (pad_h, pad_v) = (
            pad_h.max(est_pad_h.min(pad_h)),
            pad_v.max(est_pad_v.min(pad_v)),
        );
        safety::clamp_size(Size::new(
            w.saturating_add(pad_h.saturating_mul(2)),
            h.saturating_add(pad_v.saturating_mul(2)),
        ))
    }

    /// Compute the per-axis padding in pixels for a given used char_h.
    fn padding_px(&self, char_h: u32) -> (u32, u32) {
        let h = ((char_h as f32) * self.pad_h_frac).round() as u32;
        let v = ((char_h as f32) * self.pad_v_frac).round() as u32;
        (h, v)
    }

    /// Estimate padding from the spec's *requested* char_h and shrink
    /// the inner constraint by that amount. Returns
    /// `(inner_max_w, inner_max_h, est_pad_h, est_pad_v)`. The real
    /// padding is recomputed from the used char_h after measure; for
    /// most cases the estimate is exact (pad_h_frac is small enough
    /// that one-pass approximation is < 1px off in the final outer
    /// size).
    fn estimate_pad_and_inner(&self, max_w: u32, max_h: u32) -> (u32, u32, u32, u32) {
        let nominal_ch = match &self.style {
            TextStyle::Lines(_) => font::GLYPH_H,
            TextStyle::Fixed { char_h, .. } | TextStyle::Wrapped { char_h, .. } => *char_h,
            TextStyle::AutoFit { max_char_h, .. } => *max_char_h,
        };
        let (h, v) = self.padding_px(nominal_ch.max(1));
        let inner_w = max_w.saturating_sub(h.saturating_mul(2));
        let inner_h = if max_h == 0 {
            0
        } else {
            max_h.saturating_sub(v.saturating_mul(2))
        };
        (inner_w, inner_h, h, v)
    }
}

/// Binary-search the largest `char_h ∈ [min_char_h, max_char_h]` such
/// that wrapping `text` at that char_h fits within `(max_w, max_h)`
/// honoring `line_height`. Returns `min_char_h` if even the minimum
/// overflows (rasterizer will then clip / overflow on its own).
fn autofit_char_h(
    text: &str,
    max_w: u32,
    max_h: u32,
    line_height: f32,
    min_char_h: u32,
    max_char_h: u32,
) -> u32 {
    if max_w == 0 || text.is_empty() {
        return min_char_h.max(1);
    }
    let lo = min_char_h.max(1);
    let hi = max_char_h.max(lo);
    let fits = |ch: u32| {
        let (w, h) = font::measure_text_wrapped_lh(text, ch, max_w, line_height);
        let w_ok = w <= max_w;
        let h_ok = max_h == 0 || h <= max_h;
        w_ok && h_ok
    };
    if fits(hi) {
        return hi;
    }
    if !fits(lo) {
        return lo;
    }
    // Invariant: fits(lo) == true, fits(hi) == false.
    let mut lo = lo;
    let mut hi = hi;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
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

    // ── AutoFit ────────────────────────────────────────────────────────

    #[test]
    fn autofit_picks_largest_char_h_within_box() {
        let spec = TextSpec::auto_fit("hello world", WHITE, BLACK);
        // Rect tall enough for char_h=20 single-line at width 240 (no wrap).
        let s = spec.natural(240, 30);
        // Should fit within the rect.
        assert!(
            s.w <= 240 && s.h <= 30,
            "AutoFit at (240, 30) should fit, got {s:?}",
        );
        // But shouldn't be tiny — the result should have used a char_h
        // close to the available height (minus padding).
        assert!(
            s.h >= 16,
            "expected char_h close to max_h=30, got h={}",
            s.h
        );
    }

    #[test]
    fn autofit_shrinks_for_narrow_box() {
        let spec = TextSpec::auto_fit("a long paragraph that wraps several times", WHITE, BLACK);
        let s_wide = spec.clone().natural(1000, 200);
        let s_narrow = spec.natural(60, 200);
        assert!(
            s_narrow.h > 0,
            "narrow box should still produce some output"
        );
        assert!(
            s_narrow.w <= 60,
            "AutoFit must respect max_w=60, got w={}",
            s_narrow.w,
        );
        assert!(
            s_wide.w <= 1000,
            "AutoFit must respect max_w=1000, got w={}",
            s_wide.w,
        );
        // The narrow box is forced to use a smaller char_h (= more
        // wraps but smaller glyphs); the wide box keeps glyphs large
        // and wraps less. The narrow box's individual line height
        // should therefore be smaller, even though both fit their
        // bounds. Sanity check: wide box width usage is much larger
        // than narrow.
        assert!(s_wide.w > s_narrow.w * 4);
    }

    #[test]
    fn autofit_clamps_min_char_h_when_box_is_too_tight() {
        let spec = TextSpec::auto_fit("very wide text", WHITE, BLACK)
            .with_min_char_h(8)
            .with_max_char_h(54);
        // 1-pixel tall box — even min_char_h=8 doesn't fit.
        let s = spec.natural(200, 1);
        // Output exists but doesn't fit (caller's job to clip).
        assert!(s.h > 0);
    }

    #[test]
    fn autofit_natural_matches_rasterize() {
        let spec = TextSpec::auto_fit("example paragraph text here", WHITE, BLACK);
        let measured = spec.natural(180, 60);
        let (_, w, h) = spec.rasterize(180, 60);
        assert_eq!(measured, Size::new(w, h));
    }

    // ── Default text padding ───────────────────────────────────────────

    #[test]
    fn default_padding_adds_breathing_room() {
        let s_with = TextSpec::fixed("X", WHITE, BLACK, 20).natural(2000, 2000);
        let s_no_pad = TextSpec::fixed("X", WHITE, BLACK, 20)
            .with_padding(0.0, 0.0)
            .natural(2000, 2000);
        assert!(
            s_with.w > s_no_pad.w,
            "default padding should make natural width larger \
             (with={s_with:?}, no_pad={s_no_pad:?})",
        );
    }

    #[test]
    fn padding_override_zero_matches_legacy_size() {
        // Two leaves at the same char_h — one with padding off — should
        // produce identical natural sizes for the no-pad one against
        // measure_text_height (the pre-padding behavior).
        let no_pad = TextSpec::fixed("X", WHITE, BLACK, 20)
            .with_padding(0.0, 0.0)
            .natural(2000, 2000);
        let raw = font::measure_text_height("X", 20);
        assert_eq!(no_pad, Size::new(raw.0, raw.1));
    }

    #[test]
    fn padding_invalid_clamps_to_zero() {
        let spec = TextSpec::fixed("hi", WHITE, BLACK, 20).with_padding(f32::NAN, -1.0);
        assert_eq!(spec.pad_h_frac, 0.0);
        assert_eq!(spec.pad_v_frac, 0.0);
    }

    // ── Lines auto-splits \n ───────────────────────────────────────────

    #[test]
    fn lines_splits_newlines_within_strings() {
        let spec = TextSpec::lines(vec![("line one\nline two".to_string(), WHITE)], BLACK);
        if let TextStyle::Lines(lines) = &spec.style {
            assert_eq!(lines.len(), 2);
            assert_eq!(lines[0].0, "line one");
            assert_eq!(lines[1].0, "line two");
        } else {
            panic!("expected Lines");
        }
    }

    #[test]
    fn lines_preserves_color_across_split_segments() {
        let red = [255, 0, 0, 255];
        let blue = [0, 0, 255, 255];
        let spec = TextSpec::lines(
            vec![
                ("red\nstill red".to_string(), red),
                ("blue".to_string(), blue),
            ],
            BLACK,
        );
        if let TextStyle::Lines(lines) = &spec.style {
            assert_eq!(lines.len(), 3);
            assert_eq!(lines[0], ("red".to_string(), red));
            assert_eq!(lines[1], ("still red".to_string(), red));
            assert_eq!(lines[2], ("blue".to_string(), blue));
        } else {
            panic!("expected Lines");
        }
    }
}
