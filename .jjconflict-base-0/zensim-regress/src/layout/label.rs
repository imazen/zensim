//! Label strips — the pervasive "background-filled bar with centered
//! text" pattern used in diff montages. Two flavors:
//!
//! - [`LabelStyle::strip`] — single text, horizontally aligned per
//!   `LabelStyle::align`, vertically centered.
//! - [`LabelStyle::segmented_strip`] — multiple [`LabelSegment`]s,
//!   each aligned independently within the same strip via [`Layers`].

use crate::pixel_ops::Bitmap;

use super::color::{Color, rgb};
use super::geom::{HAlign, Insets, Rect, Size, VAlign};
use super::modifiers::{LayoutMod, wrap_padded};
use super::node::{Node, fill, text};
use super::paint::{fill_rect, render_image};
use super::sizing::Fit;
use super::text::TextSpec;
use crate::font;

#[derive(Clone, Debug)]
pub struct LabelStyle {
    pub fg: Color,
    pub bg: Color,
    pub align: HAlign,
    pub padding: Insets,
    /// `None` → auto-fit width via `font::measure_lines_fitted`.
    pub char_h: Option<u32>,
}

impl Default for LabelStyle {
    fn default() -> Self {
        Self {
            fg: rgb(220, 220, 220),
            bg: rgb(40, 40, 40),
            align: HAlign::Center,
            padding: Insets::xy(8, 2),
            char_h: None,
        }
    }
}

impl LabelStyle {
    pub fn with_fg(mut self, fg: Color) -> Self {
        self.fg = fg;
        self
    }
    pub fn with_bg(mut self, bg: Color) -> Self {
        self.bg = bg;
        self
    }
    pub fn with_align(mut self, align: HAlign) -> Self {
        self.align = align;
        self
    }
    pub fn with_padding(mut self, padding: Insets) -> Self {
        self.padding = padding;
        self
    }
    pub fn with_char_h(mut self, char_h: u32) -> Self {
        self.char_h = Some(char_h);
        self
    }

    /// Build a single-text label strip — `fill_width` background-filled
    /// box with vertically-centered, horizontally-aligned-per-style text.
    pub fn strip(&self, s: impl Into<String>) -> Node {
        let inner = match self.char_h {
            Some(h) => text(TextSpec::fixed(s, self.fg, self.bg, h)),
            None => text(TextSpec::lines(vec![(s.into(), self.fg)], self.bg)),
        };
        let mut inner: Node = inner.align(self.align, VAlign::Center);
        if self.padding != Insets::default() {
            inner = wrap_padded(inner, self.padding);
        }
        inner.background(self.bg).fill_width()
    }

    /// Build a width-aware segmented label strip — left/center/right
    /// groups sized so they always fit side-by-side at any reasonable
    /// strip width, with a guaranteed minimum gap between groups.
    ///
    /// At paint time, the variant computes a single `char_h` from the
    /// actual rect width such that:
    ///
    /// 1. Total content width + min-gaps ≤ inner strip width.
    /// 2. `char_h` is as large as possible (preferring readability).
    /// 3. `char_h` ≤ `style.char_h` if explicitly set (caller's cap).
    /// 4. `char_h` ≤ inner strip height (so glyphs never extend
    ///    above/below the strip).
    /// 5. `char_h` ≥ `font::GLYPH_H / 4` (legibility floor).
    ///
    /// This avoids both "labels too small at narrow strips" (the old
    /// fixed `GLYPH_H/2` default) and "labels overflow strip" (when
    /// caller forces too large a `char_h`).
    pub fn segmented_strip(&self, segments: Vec<LabelSegment>) -> Node {
        if segments.is_empty() {
            return fill(self.bg);
        }
        // Wrap in fill_width so the strip always spans its container's
        // cross axis (matching the old grid-based implementation).
        // Without this, a hug-cross stack would size the strip to its
        // natural content width and leave canvas_bg showing through
        // around it.
        Node::SegmentedStrip {
            segments,
            style: Box::new(self.clone()),
        }
        .fill_width()
    }
}

// ── SegmentedStrip measure / paint ──────────────────────────────────────

/// Pixel gap between adjacent groups, used both to space left+right
/// when both are present and as a guard inside groups.
pub(super) const SEGMENT_INTER_GROUP_GAP: u32 = 6;
/// Pixel gap between segments inside the same group.
pub(super) const SEGMENT_INTRA_GROUP_GAP: u32 = 2;

/// Total character count across all segments — used to invert the
/// `char_w(h) = round(BASE_W × h / BASE_H)` formula and pick the
/// largest `h` that fits.
fn total_chars(segments: &[LabelSegment]) -> u32 {
    segments.iter().map(|s| s.text.chars().count() as u32).sum()
}

/// Number of non-empty (left, center, right) groups in `segments`.
fn group_count(segments: &[LabelSegment]) -> u32 {
    let mut l = false;
    let mut c = false;
    let mut r = false;
    for s in segments {
        match s.align {
            HAlign::Left => l = true,
            HAlign::Center => c = true,
            HAlign::Right => r = true,
        }
    }
    [l, c, r].iter().filter(|x| **x).count() as u32
}

/// Pick the largest `char_h` such that all segments fit side-by-side
/// in `inner_w` × `inner_h`. The result is bounded by
/// `style.char_h` (if set), `inner_h`, [`font::GLYPH_H`] (no upscaling
/// past the embedded strip's native height), and a legibility floor
/// of `font::GLYPH_H / 4`.
pub(super) fn fit_segmented_char_h(
    segments: &[LabelSegment],
    style: &LabelStyle,
    inner_w: u32,
    inner_h: u32,
) -> u32 {
    let total = total_chars(segments).max(1);
    let groups = group_count(segments);
    let intra: u32 = segments
        .iter()
        .fold(([0u32; 3], 0u32), |(mut counts, _), s| {
            let i = match s.align {
                HAlign::Left => 0,
                HAlign::Center => 1,
                HAlign::Right => 2,
            };
            counts[i] += 1;
            (counts, 0)
        })
        .0
        .iter()
        .map(|n| SEGMENT_INTRA_GROUP_GAP * n.saturating_sub(1))
        .sum();
    let inter = SEGMENT_INTER_GROUP_GAP * groups.saturating_sub(1);
    let glyph_budget = inner_w.saturating_sub(intra + inter);
    // char_w(h) ≈ BASE_W × h / BASE_H.
    // total × char_w(h) ≤ glyph_budget
    // h ≤ glyph_budget × BASE_H / (total × BASE_W)
    let max_h_by_width = ((glyph_budget as u64) * (font::GLYPH_H as u64)
        / ((total as u64) * (font::GLYPH_W as u64))) as u32;
    // When the caller explicitly sets `style.char_h`, respect their
    // request (capped to inner_h and max_h_by_width to prevent
    // overflow but NOT to font::GLYPH_H — the rasterizer will upscale
    // if a larger char_h was requested).
    //
    // When auto-fitting (`char_h is None`), cap at font::GLYPH_H to
    // avoid blurry upscale and at max_h_by_width so content fits.
    let max_h = match style.char_h {
        Some(h) => h.min(inner_h).min(max_h_by_width),
        None => font::GLYPH_H.min(inner_h).min(max_h_by_width),
    };
    max_h.max(font::GLYPH_H / 4)
}

/// Approximate scaled-character width at `char_h` — mirrors
/// [`font::char_width_for_height`] (which is private). Off by ≤ 1 px.
fn approx_char_w(char_h: u32) -> u32 {
    (font::GLYPH_W * char_h + font::GLYPH_H / 2) / font::GLYPH_H
}

/// Measure pass — width hugs `max.w` if [`SizeRule::Fill`] is wrapping
/// us, else the natural content width at `style.char_h.unwrap_or(GLYPH_H/2)`.
/// Height = char_h + padding (so a SegmentedStrip placed inside a
/// hug-content stack still gets a reasonable height).
pub(super) fn measure_segmented_strip(
    segments: &[LabelSegment],
    style: &LabelStyle,
    max: Size,
) -> Size {
    let inner_w = max.w.saturating_sub(style.padding.horizontal());
    let inner_h = max.h.saturating_sub(style.padding.vertical());
    let char_h = fit_segmented_char_h(segments, style, inner_w, inner_h);
    let total_w = approx_char_w(char_h).saturating_mul(total_chars(segments).max(1));
    let groups = group_count(segments);
    let intra: u32 = segments
        .iter()
        .fold([0u32; 3], |mut counts, s| {
            let i = match s.align {
                HAlign::Left => 0,
                HAlign::Center => 1,
                HAlign::Right => 2,
            };
            counts[i] += 1;
            counts
        })
        .iter()
        .map(|n| SEGMENT_INTRA_GROUP_GAP * n.saturating_sub(1))
        .sum();
    let inter = SEGMENT_INTER_GROUP_GAP * groups.saturating_sub(1);
    let natural_w = total_w + intra + inter + style.padding.horizontal();
    let h = char_h + style.padding.vertical();
    Size::new(natural_w.min(max.w), h.min(max.h))
}

/// Paint pass — uses the actual rect to compute a width-fit `char_h`,
/// then rasterizes each segment and lays them out left/center/right
/// with the inter- and intra-group gaps.
pub(super) fn paint_segmented_strip(
    segments: &[LabelSegment],
    style: &LabelStyle,
    rect: Rect,
    canvas: &mut Bitmap,
) {
    fill_rect(canvas, rect, style.bg);

    if segments.is_empty() {
        return;
    }
    let inner = Rect::new(
        rect.x.saturating_add(style.padding.left),
        rect.y.saturating_add(style.padding.top),
        rect.w.saturating_sub(style.padding.horizontal()),
        rect.h.saturating_sub(style.padding.vertical()),
    );
    if inner.w == 0 || inner.h == 0 {
        return;
    }

    let char_h = fit_segmented_char_h(segments, style, inner.w, inner.h);
    if char_h == 0 {
        return;
    }

    // Rasterize each segment once so we have actual widths.
    let mut rasters: Vec<(LabelSegment, Bitmap)> = Vec::with_capacity(segments.len());
    for seg in segments {
        // Per-segment char_h overrides take precedence; cap at the
        // shared char_h derived from the strip's geometry so all
        // segments share a baseline. Don't cap at GLYPH_H — caller's
        // explicit char_h is honored even if it requires upscale.
        let h = seg.char_h.unwrap_or(char_h).min(inner.h);
        let (buf, w, glyph_h) = font::render_text_height(&seg.text, seg.fg, style.bg, h);
        if w == 0 || glyph_h == 0 {
            continue;
        }
        let Some(img) = Bitmap::from_raw(w, glyph_h, buf) else {
            continue;
        };
        rasters.push((seg.clone(), img));
    }

    // Group widths.
    let mut widths = [0u32; 3];
    let mut group_segs: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (i, (seg, img)) in rasters.iter().enumerate() {
        let g = match seg.align {
            HAlign::Left => 0,
            HAlign::Center => 1,
            HAlign::Right => 2,
        };
        group_segs[g].push(i);
        if !group_segs[g].is_empty() && widths[g] > 0 {
            widths[g] += SEGMENT_INTRA_GROUP_GAP;
        }
        widths[g] += img.width();
    }

    // Place each group within `inner`.
    let typo_shift = ((char_h as f32) * font::TYPO_CAP_MID_OFFSET).round() as u32;
    let baseline_y = inner.y + (inner.h.saturating_sub(char_h)) / 2 + typo_shift;

    let place_group = |g: usize, x_start: u32, canvas: &mut Bitmap| {
        let mut x = x_start;
        let mut first = true;
        for &i in &group_segs[g] {
            let (_seg, img) = &rasters[i];
            if !first {
                x += SEGMENT_INTRA_GROUP_GAP;
            }
            first = false;
            render_image(
                img,
                Fit::None,
                Rect::new(x, baseline_y, img.width(), img.height()),
                canvas,
            );
            x += img.width();
        }
    };

    // Left group hugs the left edge.
    if !group_segs[0].is_empty() {
        place_group(0, inner.x, canvas);
    }
    // Right group hugs the right edge.
    if !group_segs[2].is_empty() {
        let x = inner.x.saturating_add(inner.w).saturating_sub(widths[2]);
        place_group(2, x, canvas);
    }
    // Center group centers in remaining space (between left's end and
    // right's start, with min inter-group gaps preserved).
    if !group_segs[1].is_empty() {
        let left_end = if group_segs[0].is_empty() {
            inner.x
        } else {
            inner.x + widths[0] + SEGMENT_INTER_GROUP_GAP
        };
        let right_start = if group_segs[2].is_empty() {
            inner.x + inner.w
        } else {
            inner.x + inner.w - widths[2] - SEGMENT_INTER_GROUP_GAP
        };
        let avail = right_start.saturating_sub(left_end);
        let x_start = left_end + (avail.saturating_sub(widths[1])) / 2;
        place_group(1, x_start, canvas);
    }
}

#[derive(Clone, Debug)]
pub struct LabelSegment {
    pub text: String,
    pub fg: Color,
    pub align: HAlign,
    pub char_h: Option<u32>,
}

impl LabelSegment {
    pub fn left(text: impl Into<String>, fg: Color) -> Self {
        Self {
            text: text.into(),
            fg,
            align: HAlign::Left,
            char_h: None,
        }
    }
    pub fn right(text: impl Into<String>, fg: Color) -> Self {
        Self {
            text: text.into(),
            fg,
            align: HAlign::Right,
            char_h: None,
        }
    }
    pub fn center(text: impl Into<String>, fg: Color) -> Self {
        Self {
            text: text.into(),
            fg,
            align: HAlign::Center,
            char_h: None,
        }
    }
    pub fn with_char_h(mut self, h: u32) -> Self {
        self.char_h = Some(h);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::super::geom::Size;
    use super::super::node::image as image_node;
    use super::super::sizing::SizeRule;
    use super::*;
    use crate::pixel_ops::Bitmap;

    fn solid(w: u32, h: u32, c: Color) -> Bitmap {
        Bitmap::from_pixel(w, h, c)
    }

    /// Helpers for SegmentedStrip pixel-level structural tests. Render
    /// a strip + return the canvas for inspection.
    fn render_seg_strip(segments: Vec<LabelSegment>, style: &LabelStyle, w: u32, h: u32) -> Bitmap {
        let n: Node = style
            .segmented_strip(segments)
            .width(SizeRule::Fixed(w))
            .height(SizeRule::Fixed(h));
        super::super::render(&n, w)
    }

    /// Find rightmost x at row `y` matching `is_target`.
    fn rightmost_x(img: &Bitmap, y: u32, is_target: impl Fn(&[u8; 4]) -> bool) -> Option<u32> {
        (0..img.width())
            .rev()
            .find(|&x| is_target(&img.get_pixel(x, y)))
    }
    fn leftmost_x(img: &Bitmap, y: u32, is_target: impl Fn(&[u8; 4]) -> bool) -> Option<u32> {
        (0..img.width()).find(|&x| is_target(&img.get_pixel(x, y)))
    }
    fn is_orange(p: &[u8; 4]) -> bool {
        p[0] > 200 && (140..200).contains(&p[1]) && p[2] < 130
    }
    fn is_cyan(p: &[u8; 4]) -> bool {
        p[0] < 130 && p[1] > 180 && p[2] > 180
    }

    #[test]
    fn label_default_strip_above_image() {
        let img = image_node(solid(40, 40, [255, 0, 0, 255])).label("EXPECTED");
        let s = img.measure(Size::new(400, 400));
        assert!(s.h > 40);
    }

    #[test]
    fn label_styled_uses_provided_bg() {
        let style = LabelStyle::default()
            .with_bg(rgb(100, 0, 0))
            .with_fg(rgb(255, 255, 0));
        let canvas = image_node(solid(40, 40, [255, 0, 0, 255]))
            .label_styled("HELLO", &style)
            .width(SizeRule::Fixed(40))
            .render(40);
        assert_eq!(canvas.get_pixel(0, 0), [100, 0, 0, 255]);
    }

    #[test]
    fn segmented_strip_renders_both_groups() {
        let canvas = render_seg_strip(
            vec![
                LabelSegment::left("ADD", rgb(255, 180, 80)),
                LabelSegment::right("REMOVE", rgb(80, 220, 220)),
            ],
            &LabelStyle::default(),
            240,
            58,
        );
        // Both colors must appear somewhere in the strip.
        let mut saw_orange = false;
        let mut saw_cyan = false;
        for y in 0..canvas.height() {
            for x in 0..canvas.width() {
                let p = canvas.get_pixel(x, y);
                if is_orange(&p) {
                    saw_orange = true;
                }
                if is_cyan(&p) {
                    saw_cyan = true;
                }
            }
        }
        assert!(saw_orange, "ADD (orange) not rendered");
        assert!(saw_cyan, "REMOVE (cyan) not rendered");
    }

    #[test]
    fn segmented_strip_no_overflow_at_typical_width() {
        // 240×58 strip. ADD/REMOVE must fit inside.
        let canvas = render_seg_strip(
            vec![
                LabelSegment::left("ADD", rgb(255, 180, 80)),
                LabelSegment::right("REMOVE", rgb(80, 220, 220)),
            ],
            &LabelStyle::default(),
            240,
            58,
        );
        // Find rightmost cyan in the canvas. Must be < 240 (within strip).
        let mid_y = canvas.height() / 2;
        let right = (0..canvas.height())
            .filter_map(|y| rightmost_x(&canvas, y, is_cyan))
            .max();
        if let Some(x) = right {
            assert!(
                x < 240,
                "REMOVE rightmost pixel at x={} overflows 240-wide strip",
                x
            );
            // Strip padding xy(8,2) → inner right at 240-8=232; allow small slop.
            assert!(
                x <= 232 + 1,
                "REMOVE at x={} encroaches on right padding (mid_y={})",
                x,
                mid_y
            );
        } else {
            panic!("no cyan pixels found at all");
        }
        // Same for ADD on the left.
        let left = (0..canvas.height())
            .filter_map(|y| leftmost_x(&canvas, y, is_orange))
            .min();
        if let Some(x) = left {
            assert!(
                x >= 8 - 1,
                "ADD leftmost pixel at x={} overlaps left padding",
                x
            );
        } else {
            panic!("no orange pixels found");
        }
    }

    #[test]
    fn segmented_strip_no_overflow_at_narrow_width() {
        // 80×30 strip — very tight. char_h must shrink so both segments fit.
        let canvas = render_seg_strip(
            vec![
                LabelSegment::left("ADD", rgb(255, 180, 80)),
                LabelSegment::right("REMOVE", rgb(80, 220, 220)),
            ],
            &LabelStyle::default(),
            80,
            30,
        );
        let right = (0..canvas.height())
            .filter_map(|y| rightmost_x(&canvas, y, is_cyan))
            .max();
        assert!(right.is_some(), "REMOVE not rendered at 80-wide strip");
        let x = right.unwrap();
        assert!(
            x < 80,
            "REMOVE rightmost pixel at x={} overflows 80-wide strip",
            x
        );
    }

    #[test]
    fn segmented_strip_no_overflow_at_oversized_explicit_char_h() {
        // Caller forces char_h=54 on 80-wide strip — segmented_strip
        // must clamp char_h down to actually fit. Old implementation
        // overflowed past the strip's right edge.
        let style = LabelStyle::default().with_char_h(54);
        let canvas = render_seg_strip(
            vec![
                LabelSegment::left("ADD", rgb(255, 180, 80)),
                LabelSegment::right("REMOVE", rgb(80, 220, 220)),
            ],
            &style,
            80,
            30,
        );
        let right = (0..canvas.height())
            .filter_map(|y| rightmost_x(&canvas, y, is_cyan))
            .max();
        assert!(right.is_some(), "REMOVE not rendered");
        assert!(
            right.unwrap() < 80,
            "REMOVE at x={} overflows 80-wide strip with explicit char_h=54",
            right.unwrap()
        );
    }

    #[test]
    fn segmented_strip_uses_large_char_h_when_room() {
        // 400×60 strip with short ADD/REMOVE — char_h should approach
        // GLYPH_H since there's plenty of room.
        let canvas = render_seg_strip(
            vec![
                LabelSegment::left("AB", rgb(255, 180, 80)),
                LabelSegment::right("CD", rgb(80, 220, 220)),
            ],
            &LabelStyle::default(),
            400,
            60,
        );
        // Measure orange glyph height — should be > 30 (more than half).
        let orange_ys: Vec<u32> = (0..canvas.height())
            .filter(|&y| (0..canvas.width()).any(|x| is_orange(&canvas.get_pixel(x, y))))
            .collect();
        assert!(!orange_ys.is_empty(), "no orange found");
        let glyph_h = orange_ys.last().unwrap() - orange_ys.first().unwrap() + 1;
        assert!(
            glyph_h >= 24,
            "char_h too small: glyph spans only {} rows in 60-tall strip with plenty of room",
            glyph_h
        );
    }

    #[test]
    fn segmented_strip_min_gap_preserved() {
        // 120×30 strip where ADD+REMOVE fit but barely. Inter-group
        // gap should still be ≥ SEGMENT_INTER_GROUP_GAP.
        let canvas = render_seg_strip(
            vec![
                LabelSegment::left("ADD", rgb(255, 180, 80)),
                LabelSegment::right("REMOVE", rgb(80, 220, 220)),
            ],
            &LabelStyle::default(),
            120,
            30,
        );
        // ADD's rightmost x and REMOVE's leftmost x at the row containing both.
        for y in 0..canvas.height() {
            let add_right = rightmost_x(&canvas, y, is_orange);
            let rem_left = leftmost_x(&canvas, y, is_cyan);
            if let (Some(ar), Some(rl)) = (add_right, rem_left) {
                let gap = rl as i32 - ar as i32 - 1;
                assert!(
                    gap >= SEGMENT_INTER_GROUP_GAP as i32 - 2,
                    "gap between ADD(x={}) and REMOVE(x={}) at y={} is {} < min {}",
                    ar,
                    rl,
                    y,
                    gap,
                    SEGMENT_INTER_GROUP_GAP
                );
                return;
            }
        }
        panic!("never found a row with both ADD and REMOVE");
    }

    #[test]
    fn segmented_strip_left_at_left_edge() {
        let canvas = render_seg_strip(
            vec![
                LabelSegment::left("ADD", rgb(255, 180, 80)),
                LabelSegment::right("REMOVE", rgb(80, 220, 220)),
            ],
            &LabelStyle::default(),
            240,
            58,
        );
        let left = (0..canvas.height())
            .filter_map(|y| leftmost_x(&canvas, y, is_orange))
            .min();
        // ADD should start within ~2 px of the strip's left padding (8 px).
        assert!(
            (8..=12).contains(&left.unwrap()),
            "ADD's leftmost pixel at x={} not near strip left padding",
            left.unwrap()
        );
    }

    #[test]
    fn segmented_strip_right_at_right_edge() {
        let canvas = render_seg_strip(
            vec![
                LabelSegment::left("ADD", rgb(255, 180, 80)),
                LabelSegment::right("REMOVE", rgb(80, 220, 220)),
            ],
            &LabelStyle::default(),
            240,
            58,
        );
        let right = (0..canvas.height())
            .filter_map(|y| rightmost_x(&canvas, y, is_cyan))
            .max();
        // REMOVE's rightmost glyph pixel should be near (but ≤) the
        // strip's right inner edge. char_w(h) is integer-rounded so the
        // total content rarely fills the inner exactly — allow up to
        // ~10 px of slack but never overflow.
        let r = right.unwrap();
        assert!(r < 240, "REMOVE at x={} overflows 240-wide strip", r);
        assert!(
            r >= 220,
            "REMOVE at x={} sits too far from right edge (expected ≥ 220)",
            r
        );
    }

    #[test]
    fn segmented_strip_vertical_centering() {
        // Glyph mid-row should land near strip mid-row.
        let canvas = render_seg_strip(
            vec![
                LabelSegment::left("ADD", rgb(255, 180, 80)),
                LabelSegment::right("REMOVE", rgb(80, 220, 220)),
            ],
            &LabelStyle::default(),
            240,
            60,
        );
        let orange_ys: Vec<u32> = (0..canvas.height())
            .filter(|&y| (0..canvas.width()).any(|x| is_orange(&canvas.get_pixel(x, y))))
            .collect();
        let glyph_top = *orange_ys.first().unwrap();
        let glyph_bot = *orange_ys.last().unwrap();
        let glyph_mid = (glyph_top + glyph_bot) / 2;
        let strip_mid = canvas.height() / 2;
        let drift = (glyph_mid as i32 - strip_mid as i32).abs();
        // Glyph mid should be within ~1/8 of strip height of strip mid
        // (typographic shift accounts for glyph cell whitespace).
        assert!(
            drift <= (canvas.height() / 8) as i32,
            "glyph mid y={} drifts {} px from strip mid {}",
            glyph_mid,
            drift,
            strip_mid
        );
    }
}
