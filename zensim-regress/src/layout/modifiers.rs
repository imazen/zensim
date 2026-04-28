//! Modifier nodes — `Padded` / `Sized` / `Constrain` / `Aspect` /
//! `Align` / `Fit` / `Background` / `Border` — together with the
//! [`LayoutMod`] trait that gives every node and builder fluent
//! modifier methods (`.padding(8)`, `.center()`, `.background(c)`, ...).

use crate::pixel_ops::Bitmap;

use super::color::Color;
use super::geom::{HAlign, Insets, Rect, Size, VAlign};
use super::label::{LabelSegment, LabelStyle};
use super::node::Node;
use super::paint;
use super::safety;
use super::sizing::{Fit, SizeRule};

// ── Wrap helpers ───────────────────────────────────────────────────────

pub(super) fn wrap_padded(child: Node, insets: Insets) -> Node {
    Node::Padded {
        insets,
        child: Box::new(child),
    }
}
pub(super) fn wrap_sized(child: Node, w: SizeRule, h: SizeRule) -> Node {
    Node::Sized {
        w,
        h,
        child: Box::new(child),
    }
}
pub(super) fn wrap_constrain(
    child: Node,
    min_w: Option<u32>,
    max_w: Option<u32>,
    min_h: Option<u32>,
    max_h: Option<u32>,
) -> Node {
    Node::Constrain {
        min_w,
        max_w,
        min_h,
        max_h,
        child: Box::new(child),
    }
}

// ── Per-modifier measure helpers ──────────────────────────────────────

pub(super) fn measure_padded(insets: Insets, child: &Node, max: Size) -> Size {
    let inner_max = Size::new(
        max.w.saturating_sub(insets.horizontal()),
        max.h.saturating_sub(insets.vertical()),
    );
    let inner = child.measure(inner_max);
    Size::new(inner.w + insets.horizontal(), inner.h + insets.vertical())
}

pub(super) fn measure_sized(w: SizeRule, h: SizeRule, child: &Node, max: Size) -> Size {
    // For non-Hug rules, the rule resolves to a concrete axis size that
    // *is* the constraint we should pass to the child — otherwise a
    // wrapped-text leaf inside e.g. `.width(Fixed(50))` would measure
    // itself against the parent's wide max and report a single-line
    // height. For Hug we pass the parent's max unchanged (no extra
    // constraint added by us).
    let probe = |rule: SizeRule, axis_max: u32| -> u32 {
        match rule {
            SizeRule::Hug => axis_max,
            // Fill / Grow inside an unbounded constraint would otherwise
            // happily stretch to u32::MAX; clamp to MAX_DIM and to the
            // available axis_max.
            SizeRule::Fill | SizeRule::Grow(_) => safety::clamp_dim(axis_max),
            SizeRule::Fixed(v) => safety::clamp_dim(v).min(axis_max),
            SizeRule::Percent(p) => {
                if axis_max == 0 || p.is_nan() {
                    0
                } else {
                    let v = ((axis_max as f32) * p.clamp(0.0, 1.0)).round() as u32;
                    safety::clamp_dim(v).min(axis_max)
                }
            }
        }
    };
    let child_max = Size::new(probe(w, max.w), probe(h, max.h));
    let child_size = child.measure(child_max);
    let resolve = |rule: SizeRule, axis_max: u32, child_dim: u32| -> u32 {
        match rule {
            SizeRule::Hug => child_dim,
            SizeRule::Fill | SizeRule::Grow(_) => safety::clamp_dim(axis_max),
            SizeRule::Fixed(v) => safety::clamp_dim(v),
            SizeRule::Percent(p) => {
                if axis_max == 0 || p.is_nan() {
                    0
                } else {
                    let v = ((axis_max as f32) * p.clamp(0.0, 1.0)).round() as u32;
                    safety::clamp_dim(v)
                }
            }
        }
    };
    Size::new(
        resolve(w, max.w, child_size.w),
        resolve(h, max.h, child_size.h),
    )
}

pub(super) fn measure_constrain(
    min_w: Option<u32>,
    max_w: Option<u32>,
    min_h: Option<u32>,
    max_h: Option<u32>,
    child: &Node,
    max: Size,
) -> Size {
    let cap_max_w = max_w.map(safety::clamp_dim);
    let cap_max_h = max_h.map(safety::clamp_dim);
    let cap_min_w = min_w.map(safety::clamp_dim);
    let cap_min_h = min_h.map(safety::clamp_dim);
    let child_max = Size::new(
        cap_max_w.map_or(max.w, |v| v.min(max.w)),
        cap_max_h.map_or(max.h, |v| v.min(max.h)),
    );
    let s = child.measure(child_max);
    safety::clamp_size(Size::new(
        s.w.max(cap_min_w.unwrap_or(0))
            .min(cap_max_w.unwrap_or(u32::MAX))
            .min(max.w),
        s.h.max(cap_min_h.unwrap_or(0))
            .min(cap_max_h.unwrap_or(u32::MAX))
            .min(max.h),
    ))
}

pub(super) fn measure_aspect(num: u32, den: u32, child: &Node, max: Size) -> Size {
    let s = child.measure(max);
    let cw = s.w.max(1);
    let ch = s.h.max(1);
    let h_from_w = cw * den / num;
    let w_from_h = ch * num / den;
    if h_from_w <= max.h && cw <= max.w && h_from_w >= ch {
        Size::new(cw, h_from_w)
    } else if w_from_h <= max.w && ch <= max.h {
        Size::new(w_from_h, ch)
    } else {
        // Both candidates exceed; scale to fit max while preserving ratio.
        let h_from_max_w = max.w * den / num;
        if h_from_max_w <= max.h {
            Size::new(max.w, h_from_max_w)
        } else {
            Size::new(max.h * num / den, max.h)
        }
    }
}

// ── Per-modifier paint helpers ────────────────────────────────────────

pub(super) fn paint_padded(insets: Insets, child: &Node, rect: Rect, canvas: &mut Bitmap) {
    let inner = Rect::new(
        rect.x.saturating_add(insets.left),
        rect.y.saturating_add(insets.top),
        rect.w.saturating_sub(insets.horizontal()),
        rect.h.saturating_sub(insets.vertical()),
    );
    child.paint(inner, canvas);
}

pub(super) fn paint_align(h: HAlign, v: VAlign, child: &Node, rect: Rect, canvas: &mut Bitmap) {
    let child_size = child.measure(rect.size());
    let x_off = match h {
        HAlign::Left => 0,
        HAlign::Center => rect.w.saturating_sub(child_size.w) / 2,
        HAlign::Right => rect.w.saturating_sub(child_size.w),
    };
    let mut y_off = match v {
        VAlign::Top => 0,
        VAlign::Center => rect.h.saturating_sub(child_size.h) / 2,
        VAlign::Bottom => rect.h.saturating_sub(child_size.h),
    };
    // Typographic centering: when vertically centering a text-bearing
    // child, shift it down by `font::TYPO_CAP_MID_OFFSET` of its
    // height so the inked cap-mid (~42.6% of cell) lands at the box
    // geometric mid. Without this, capitals visibly skew above center
    // because the embedded glyph cell has whitespace above cap-top.
    if matches!(v, VAlign::Center) && contains_text(child) {
        let typo_shift = ((child_size.h as f32) * crate::font::TYPO_CAP_MID_OFFSET).round() as u32;
        y_off = y_off.saturating_add(typo_shift);
    }
    let inner = Rect::new(
        rect.x.saturating_add(x_off),
        rect.y.saturating_add(y_off),
        child_size.w.min(rect.w),
        child_size.h.min(rect.h),
    );
    child.paint(inner, canvas);
}

/// `true` if `node` reaches a [`Node::Text`] leaf via pass-through
/// modifier wrappers (Padded, Sized, Constrain, Aspect, Background,
/// Border, Fit). Stops at `Align` (don't double-apply) and at
/// containers (Stack/Grid/Layers — those manage their own children).
fn contains_text(node: &Node) -> bool {
    match node {
        Node::Text(_) => true,
        Node::Padded { child, .. }
        | Node::Sized { child, .. }
        | Node::Constrain { child, .. }
        | Node::Aspect { child, .. }
        | Node::Background { child, .. }
        | Node::Border { child, .. }
        | Node::Fit { child, .. } => contains_text(child),
        _ => false,
    }
}

pub(super) fn paint_fit(mode: Fit, child: &Node, rect: Rect, canvas: &mut Bitmap) {
    if let Node::Image(img) = child {
        paint::render_image(img, mode, rect, canvas);
    } else {
        child.paint(rect, canvas);
    }
}

pub(super) fn paint_background(color: Color, child: &Node, rect: Rect, canvas: &mut Bitmap) {
    paint::fill_rect(canvas, rect, color);
    child.paint(rect, canvas);
}

pub(super) fn paint_border(color: Color, child: &Node, rect: Rect, canvas: &mut Bitmap) {
    child.paint(rect, canvas);
    paint::draw_rect_border(canvas, rect, color);
}

// ── LayoutMod trait — fluent modifier methods ─────────────────────────

/// Modifier methods on any layout node or builder. Auto-imported via
/// `use zensim_regress::layout::*;`.
pub trait LayoutMod: Sized {
    fn into_node(self) -> Node;

    // ── Padding (CSS `padding`) ────────────────────────────────────────
    fn padding(self, v: u32) -> Node {
        wrap_padded(self.into_node(), Insets::all(v))
    }
    fn padding_xy(self, x: u32, y: u32) -> Node {
        wrap_padded(self.into_node(), Insets::xy(x, y))
    }
    /// CSS shorthand order: top, right, bottom, left.
    fn padding_each(self, t: u32, r: u32, b: u32, l: u32) -> Node {
        wrap_padded(self.into_node(), Insets::each(t, r, b, l))
    }

    // ── Sizing (CSS `width`, `height`) ────────────────────────────────
    fn width(self, r: SizeRule) -> Node {
        wrap_sized(self.into_node(), r, SizeRule::Hug)
    }
    fn height(self, r: SizeRule) -> Node {
        wrap_sized(self.into_node(), SizeRule::Hug, r)
    }
    fn size(self, w: u32, h: u32) -> Node {
        wrap_sized(self.into_node(), SizeRule::Fixed(w), SizeRule::Fixed(h))
    }
    fn fill_width(self) -> Node {
        wrap_sized(self.into_node(), SizeRule::Fill, SizeRule::Hug)
    }
    fn fill_height(self) -> Node {
        wrap_sized(self.into_node(), SizeRule::Hug, SizeRule::Fill)
    }
    fn fill(self) -> Node {
        wrap_sized(self.into_node(), SizeRule::Fill, SizeRule::Fill)
    }
    /// Weighted main-axis grow — CSS `flex-grow`.
    fn grow(self, weight: u32) -> Node {
        wrap_sized(self.into_node(), SizeRule::Grow(weight), SizeRule::Hug)
    }

    /// Width as a fraction of the parent constraint, `[0.0, 1.0]`
    /// (CSS `width: 50%` = `width_percent(0.5)`).
    fn width_percent(self, p: f32) -> Node {
        wrap_sized(self.into_node(), SizeRule::Percent(p), SizeRule::Hug)
    }
    /// Height as a fraction of the parent constraint.
    fn height_percent(self, p: f32) -> Node {
        wrap_sized(self.into_node(), SizeRule::Hug, SizeRule::Percent(p))
    }
    /// `(width%, height%)` of the parent constraint.
    fn size_percent(self, w: f32, h: f32) -> Node {
        wrap_sized(self.into_node(), SizeRule::Percent(w), SizeRule::Percent(h))
    }

    // ── Min/Max constraints (CSS `min-width` etc.) ────────────────────
    fn min_width(self, n: u32) -> Node {
        wrap_constrain(self.into_node(), Some(n), None, None, None)
    }
    fn max_width(self, n: u32) -> Node {
        wrap_constrain(self.into_node(), None, Some(n), None, None)
    }
    fn min_height(self, n: u32) -> Node {
        wrap_constrain(self.into_node(), None, None, Some(n), None)
    }
    fn max_height(self, n: u32) -> Node {
        wrap_constrain(self.into_node(), None, None, None, Some(n))
    }

    /// CSS `aspect-ratio: <num> / <den>`.
    fn aspect_ratio(self, num: u32, den: u32) -> Node {
        Node::Aspect {
            num: num.max(1),
            den: den.max(1),
            child: Box::new(self.into_node()),
        }
    }

    // ── Alignment (CSS `place-self`) ──────────────────────────────────
    fn align(self, h: HAlign, v: VAlign) -> Node {
        Node::Align {
            h,
            v,
            child: Box::new(self.into_node()),
        }
    }
    fn center(self) -> Node {
        self.align(HAlign::Center, VAlign::Center)
    }
    fn align_h(self, h: HAlign) -> Node {
        self.align(h, VAlign::Top)
    }
    fn align_v(self, v: VAlign) -> Node {
        self.align(HAlign::Left, v)
    }

    // ── Image fit (CSS `object-fit`) ──────────────────────────────────
    fn fit(self, mode: Fit) -> Node {
        Node::Fit {
            mode,
            child: Box::new(self.into_node()),
        }
    }
    fn fit_contain(self) -> Node {
        self.fit(Fit::Contain)
    }
    fn fit_cover(self) -> Node {
        self.fit(Fit::Cover)
    }
    fn fit_stretch(self) -> Node {
        self.fit(Fit::Stretch)
    }

    // ── Painting ──────────────────────────────────────────────────────
    fn background(self, c: Color) -> Node {
        Node::Background {
            color: c,
            child: Box::new(self.into_node()),
        }
    }
    fn border(self, c: Color) -> Node {
        Node::Border {
            color: c,
            child: Box::new(self.into_node()),
        }
    }

    // ── Label shortcuts ───────────────────────────────────────────────
    fn label(self, s: impl Into<String>) -> Node {
        self.label_styled(s, &LabelStyle::default())
    }
    fn label_styled(self, s: impl Into<String>, style: &LabelStyle) -> Node {
        super::stack::column()
            .gap(0)
            .child(style.strip(s))
            .child(self.into_node())
            .into()
    }
    fn label_segments(self, segments: Vec<LabelSegment>, style: &LabelStyle) -> Node {
        super::stack::column()
            .gap(0)
            .child(style.segmented_strip(segments))
            .child(self.into_node())
            .into()
    }

    /// Render this tree into an [`Bitmap`] of width `max_w`.
    /// Convenience for [`super::render`] when chaining off a builder.
    fn render(self, max_w: u32) -> Bitmap {
        super::render(&self.into_node(), max_w)
    }
}

impl LayoutMod for Node {
    fn into_node(self) -> Node {
        self
    }
}
impl LayoutMod for super::stack::Stack {
    fn into_node(self) -> Node {
        self.into()
    }
}
impl LayoutMod for super::grid::Grid {
    fn into_node(self) -> Node {
        self.into()
    }
}
impl LayoutMod for super::layers::Layers {
    fn into_node(self) -> Node {
        self.into()
    }
}

#[cfg(test)]
mod tests {
    use super::super::color::{BLACK, WHITE};
    use super::super::node::{empty, image as image_node};
    use super::*;
    use crate::pixel_ops::Bitmap;

    fn solid(w: u32, h: u32, c: Color) -> Bitmap {
        Bitmap::from_pixel(w, h, c)
    }

    #[test]
    fn padded_grows_by_insets() {
        let n = image_node(solid(50, 30, WHITE)).padding(10);
        assert_eq!(n.measure(Size::new(200, 200)), Size::new(70, 50));
    }
    #[test]
    fn size_overrides() {
        let n = image_node(solid(50, 30, WHITE)).size(100, 80);
        assert_eq!(n.measure(Size::new(200, 200)), Size::new(100, 80));
    }
    #[test]
    fn fill_width_consumes_max() {
        let n = image_node(solid(50, 30, WHITE)).fill_width();
        assert_eq!(n.measure(Size::new(200, 200)), Size::new(200, 30));
    }
    #[test]
    fn min_width_grows_to_minimum() {
        let n = image_node(solid(10, 10, WHITE)).min_width(50);
        assert_eq!(n.measure(Size::new(200, 200)), Size::new(50, 10));
    }
    #[test]
    fn max_width_caps_natural() {
        let n = image_node(solid(80, 10, WHITE)).max_width(40);
        assert_eq!(n.measure(Size::new(200, 200)), Size::new(40, 10));
    }
    #[test]
    fn aspect_ratio_16_9_from_width() {
        let n = image_node(solid(160, 50, WHITE)).aspect_ratio(16, 9);
        assert_eq!(n.measure(Size::new(200, 200)), Size::new(160, 90));
    }

    /// Regression: `Sized<Fixed(narrow), Hug>` containing wrapped text used
    /// to measure the child against the parent's wide max instead of the
    /// resolved 50px width, so `Hug` reported a single-line height. The
    /// width constraint must be propagated to the child measure.
    #[test]
    fn sized_fixed_width_propagates_to_wrapped_text_height() {
        use super::super::node::text;
        use super::super::text::TextSpec;
        let spec = TextSpec::wrapped(
            "this sentence is long enough to require multiple lines",
            WHITE,
            BLACK,
            8,
        );
        let unconstrained_h = text(spec.clone()).measure(Size::new(2000, 2000)).h;
        let narrowed_h = text(spec)
            .width(SizeRule::Fixed(50))
            .measure(Size::new(2000, 2000))
            .h;
        assert!(
            narrowed_h > unconstrained_h,
            "wrapping at 50px should be taller than wrapping at 2000px \
             (got narrowed_h={narrowed_h}, unconstrained_h={unconstrained_h})",
        );
    }

    /// Sister regression: same shape but for the height axis — `Sized<Hug,
    /// Fixed(short)>` should clamp wrapped text's measured height even
    /// when the child would naturally be taller. (Pre-fix this happened
    /// to work because `resolve(Fixed)` returns the rule unconditionally,
    /// but verifies the probe doesn't regress that path.)
    #[test]
    fn sized_fixed_height_overrides_child_intrinsic() {
        use super::super::node::text;
        use super::super::text::TextSpec;
        let spec = TextSpec::wrapped(
            "this sentence is long enough to require multiple lines",
            WHITE,
            BLACK,
            8,
        );
        let n = text(spec).height(SizeRule::Fixed(20));
        assert_eq!(n.measure(Size::new(2000, 2000)).h, 20);
    }

    /// `Sized<Percent(0.25), Hug>` on wrapped text inside a 400-wide
    /// parent should wrap at 100px and report the corresponding height,
    /// not the height-at-400.
    #[test]
    fn sized_percent_width_propagates_to_wrapped_text_height() {
        use super::super::node::text;
        use super::super::text::TextSpec;
        let spec = TextSpec::wrapped(
            "this sentence is long enough to require multiple lines",
            WHITE,
            BLACK,
            8,
        );
        let h_full = text(spec.clone()).measure(Size::new(400, 1000)).h;
        let h_quarter = text(spec)
            .width(SizeRule::Percent(0.25))
            .measure(Size::new(400, 1000))
            .h;
        assert!(
            h_quarter > h_full,
            "Percent(0.25) of 400 → 100px wrap should be taller than wrap at 400px \
             (got h_quarter={h_quarter}, h_full={h_full})",
        );
    }
    #[test]
    fn align_centers_in_oversized_rect() {
        let img = image_node(solid(20, 10, [255, 255, 0, 255]))
            .center()
            .size(100, 60)
            .render(100);
        assert_eq!(img.get_pixel(50, 30), [255, 255, 0, 255]);
        assert_eq!(img.get_pixel(0, 0), BLACK);
    }
    #[test]
    fn background_paints_first() {
        let img = empty()
            .background([10, 20, 30, 255])
            .size(20, 20)
            .render(20);
        assert_eq!(img.get_pixel(5, 5), [10, 20, 30, 255]);
    }
    #[test]
    fn border_paints_outline() {
        let img = empty()
            .background(BLACK)
            .border(WHITE)
            .size(10, 10)
            .render(10);
        assert_eq!(img.get_pixel(0, 0), WHITE);
        assert_eq!(img.get_pixel(9, 0), WHITE);
        assert_eq!(img.get_pixel(5, 5), BLACK);
    }
    #[test]
    fn fit_contain_letterboxes() {
        let img = image_node(solid(20, 10, [255, 0, 0, 255]))
            .fit_contain()
            .size(40, 40)
            .render(40);
        assert_eq!(img.get_pixel(20, 0), BLACK);
        assert_eq!(img.get_pixel(20, 20), [255, 0, 0, 255]);
    }
}
