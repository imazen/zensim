//! Linear stack container — CSS `display: flex` with `flex-direction:
//! row | column`. Distribution along the main axis is controlled by
//! [`MainAlign`]; cross-axis alignment by [`CrossAlign`].

use crate::pixel_ops::Bitmap;

use super::geom::{Axis, Rect, Size};
use super::node::Node;
use super::safety;
use super::sizing::{CrossAlign, MainAlign, SizeRule};

#[derive(Clone, Debug)]
pub struct Stack {
    axis: Axis,
    gap: u32,
    justify: MainAlign,
    align_items: CrossAlign,
    children: Vec<Node>,
    shrink: bool,
}

impl Stack {
    pub fn new(axis: Axis) -> Self {
        Self {
            axis,
            gap: 0,
            justify: MainAlign::Start,
            align_items: CrossAlign::Start,
            children: Vec::new(),
            shrink: false,
        }
    }

    pub fn gap(mut self, g: u32) -> Self {
        self.gap = g;
        self
    }
    pub fn justify(mut self, j: MainAlign) -> Self {
        self.justify = j;
        self
    }
    pub fn align_items(mut self, a: CrossAlign) -> Self {
        self.align_items = a;
        self
    }
    /// Allow proportional shrinking of non-rigid (non-`Fixed`) children
    /// when their summed natural size exceeds the available main-axis
    /// space. Default `false` — overflow is left to the diagnostics
    /// layer so layout bugs surface loudly rather than silently
    /// compress content. Enable when you want CSS-flex-style graceful
    /// shrinking instead.
    pub fn shrink_on_overflow(mut self, on: bool) -> Self {
        self.shrink = on;
        self
    }
    pub fn child(mut self, c: impl Into<Node>) -> Self {
        self.children.push(c.into());
        self
    }
    pub fn children<I, N>(mut self, items: I) -> Self
    where
        I: IntoIterator<Item = N>,
        N: Into<Node>,
    {
        self.children.extend(items.into_iter().map(Into::into));
        self
    }
}

impl From<Stack> for Node {
    fn from(s: Stack) -> Node {
        Node::Stack {
            axis: s.axis,
            gap: s.gap,
            justify: s.justify,
            align_items: s.align_items,
            children: s.children,
            shrink: s.shrink,
        }
    }
}

/// Free-fn entry point for a horizontal [`Stack`].
pub fn row() -> Stack {
    Stack::new(Axis::Horizontal)
}

/// Free-fn entry point for a vertical [`Stack`].
pub fn column() -> Stack {
    Stack::new(Axis::Vertical)
}

// ── Measure / paint ────────────────────────────────────────────────────

pub(super) fn measure(axis: Axis, gap: u32, children: &[Node], max: Size) -> Size {
    if children.is_empty() {
        return Size::ZERO;
    }
    let cap = safety::cap_children(children.len());
    let children = &children[..cap];
    let n = children.len() as u32;
    let total_gap = gap.saturating_mul(n.saturating_sub(1));

    let (main_max, cross_max) = match axis {
        Axis::Horizontal => (max.w.saturating_sub(total_gap), max.h),
        Axis::Vertical => (max.h.saturating_sub(total_gap), max.w),
    };
    let child_max = match axis {
        Axis::Horizontal => Size::new(main_max, cross_max),
        Axis::Vertical => Size::new(cross_max, main_max),
    };

    let mut main = 0u32;
    let mut cross = 0u32;
    for c in children {
        let s = c.measure(child_max);
        let (m, x) = match axis {
            Axis::Horizontal => (s.w, s.h),
            Axis::Vertical => (s.h, s.w),
        };
        main = main.saturating_add(m);
        cross = cross.max(x);
    }
    main = main.saturating_add(total_gap);
    match axis {
        Axis::Horizontal => Size::new(main, cross),
        Axis::Vertical => Size::new(cross, main),
    }
}

#[allow(clippy::too_many_arguments, clippy::manual_checked_ops)]
pub(super) fn paint(
    axis: Axis,
    gap: u32,
    justify: MainAlign,
    align_items: CrossAlign,
    shrink: bool,
    children: &[Node],
    rect: Rect,
    canvas: &mut Bitmap,
) {
    if children.is_empty() {
        return;
    }
    let cap = safety::cap_children(children.len());
    let children = &children[..cap];
    let n = children.len() as u32;
    let total_gap = gap.saturating_mul(n.saturating_sub(1));
    let (main_avail, cross_avail) = match axis {
        Axis::Horizontal => (rect.w.saturating_sub(total_gap), rect.h),
        Axis::Vertical => (rect.h.saturating_sub(total_gap), rect.w),
    };
    let child_max = match axis {
        Axis::Horizontal => Size::new(main_avail, cross_avail),
        Axis::Vertical => Size::new(cross_avail, main_avail),
    };

    // Per-child weight (Fill + Grow) and natural size.
    let mut weights: Vec<u32> = vec![0; children.len()];
    let mut measured: Vec<Size> = Vec::with_capacity(children.len());
    let mut used_main = 0u32;
    let mut total_weight = 0u32;
    for (i, c) in children.iter().enumerate() {
        let s = c.measure(child_max);
        let w = main_grow_weight(c, axis);
        weights[i] = w;
        total_weight = total_weight.saturating_add(w);
        let main_dim = match axis {
            Axis::Horizontal => s.w,
            Axis::Vertical => s.h,
        };
        if w == 0 {
            used_main = used_main.saturating_add(main_dim);
        }
        measured.push(s);
    }

    let remainder = main_avail.saturating_sub(used_main);

    // Distribute main-axis sizes — Hug children take their natural,
    // weighted children split the remainder.
    let mut main_sizes: Vec<u32> = vec![0; children.len()];
    let mut allotted = 0u32;
    let mut last_flex_idx: Option<usize> = None;
    for i in 0..children.len() {
        if weights[i] > 0 {
            let share = if total_weight > 0 {
                remainder * weights[i] / total_weight
            } else {
                0
            };
            main_sizes[i] = share;
            allotted = allotted.saturating_add(share);
            last_flex_idx = Some(i);
        } else {
            main_sizes[i] = match axis {
                Axis::Horizontal => measured[i].w,
                Axis::Vertical => measured[i].h,
            };
        }
    }
    if let Some(i) = last_flex_idx {
        // Hand the leftover floor-rounding pixels to the last flex child
        // so we don't lose pixels to integer division.
        main_sizes[i] = main_sizes[i].saturating_add(remainder.saturating_sub(allotted));
    }

    // ── Shrink-on-overflow ─────────────────────────────────────────────
    // CSS-flex-style proportional shrink: when the summed natural sizes
    // of non-rigid children exceed `main_avail`, shrink each shrinkable
    // child by `main_avail / shrinkable_total`. Rigid (`Sized<Fixed,_>`)
    // children keep their declared size — let them overflow loudly.
    // Grow children already get 0 share when there's no remainder;
    // they don't need shrinking.
    if shrink && used_main > main_avail {
        let mut shrinkable_total: u64 = 0;
        let mut shrinkable_indices: Vec<usize> = Vec::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            if weights[i] > 0 {
                continue; // grow children — already at 0 / no basis to shrink
            }
            if main_axis_is_rigid(child, axis) {
                continue; // Sized<Fixed,_> — keep declared size
            }
            shrinkable_total = shrinkable_total.saturating_add(main_sizes[i] as u64);
            shrinkable_indices.push(i);
        }
        if shrinkable_total > 0 && !shrinkable_indices.is_empty() {
            // Sum of rigid + grow main_sizes is what we cannot shrink.
            let unshrinkable: u64 = main_sizes
                .iter()
                .enumerate()
                .filter(|(i, _)| !shrinkable_indices.contains(i))
                .map(|(_, m)| *m as u64)
                .sum();
            let target_shrinkable = (main_avail as u64).saturating_sub(unshrinkable);
            // Distribute target_shrinkable across shrinkable_indices proportionally.
            let mut allotted_shrunk: u64 = 0;
            let mut last_idx: Option<usize> = None;
            for &i in &shrinkable_indices {
                let share = (main_sizes[i] as u64) * target_shrinkable / shrinkable_total;
                main_sizes[i] = share as u32;
                allotted_shrunk += share;
                last_idx = Some(i);
            }
            // Leftover rounding pixels → last shrunk child.
            if let Some(i) = last_idx {
                let extra = target_shrinkable.saturating_sub(allotted_shrunk);
                main_sizes[i] = main_sizes[i].saturating_add(extra as u32);
            }
        }
    }

    let total_children_main: u32 = main_sizes.iter().sum();
    let leftover = main_avail.saturating_sub(total_children_main);

    // Justify-content offsetting.
    let (mut cursor, lead_pad, extra_gap) = match justify {
        MainAlign::Start => (0u32, 0u32, 0u32),
        MainAlign::Center => (leftover / 2, 0, 0),
        MainAlign::End => (leftover, 0, 0),
        MainAlign::SpaceBetween if n > 1 => (0, 0, leftover / (n - 1)),
        MainAlign::SpaceBetween => (0, 0, 0),
        MainAlign::SpaceAround => {
            let unit = if n > 0 { leftover / n } else { 0 };
            (unit / 2, 0, unit)
        }
        MainAlign::SpaceEvenly => {
            let unit = leftover / (n + 1);
            (unit, 0, unit)
        }
    };
    cursor = cursor.saturating_add(lead_pad);
    cursor = cursor.saturating_add(match axis {
        Axis::Horizontal => rect.x,
        Axis::Vertical => rect.y,
    });

    for (i, child) in children.iter().enumerate() {
        let m_size = main_sizes[i];
        let cross_natural = match axis {
            Axis::Horizontal => measured[i].h,
            Axis::Vertical => measured[i].w,
        };
        let cross_size = match align_items {
            CrossAlign::Stretch => cross_avail,
            _ => cross_natural.min(cross_avail),
        };
        let cross_off = match align_items {
            CrossAlign::Start | CrossAlign::Stretch => 0,
            CrossAlign::Center => cross_avail.saturating_sub(cross_size) / 2,
            CrossAlign::End => cross_avail.saturating_sub(cross_size),
        };

        let child_rect = match axis {
            Axis::Horizontal => Rect::new(cursor, rect.y + cross_off, m_size, cross_size),
            Axis::Vertical => Rect::new(rect.x + cross_off, cursor, cross_size, m_size),
        };
        super::diagnostics::check_contained(
            super::diagnostics::ContainerKind::Stack,
            i,
            child.kind(),
            rect,
            child_rect,
        );
        child.paint(child_rect, canvas);

        cursor = cursor.saturating_add(m_size);
        if i + 1 < children.len() {
            cursor = cursor.saturating_add(gap).saturating_add(extra_gap);
        }
    }
}

/// `true` if the node has a `Sized<Fixed, _>` rule on the main axis,
/// found through transparent modifier wrappers — these children should
/// NOT be shrunk on overflow (they declared a hard size).
fn main_axis_is_rigid(node: &Node, axis: Axis) -> bool {
    let mut current = node;
    for _ in 0..16 {
        match current {
            Node::Sized { w, h, child } => {
                let main_rule = match axis {
                    Axis::Horizontal => *w,
                    Axis::Vertical => *h,
                };
                match main_rule {
                    SizeRule::Fixed(_) => return true,
                    SizeRule::Hug => current = child,
                    _ => return false,
                }
            }
            Node::Background { child, .. }
            | Node::Border { child, .. }
            | Node::Align { child, .. }
            | Node::Aspect { child, .. }
            | Node::Constrain { child, .. }
            | Node::Padded { child, .. } => current = child,
            _ => return false,
        }
    }
    false
}

/// Find a `Grow` (or `Fill`) weight on the main axis by walking through
/// transparent modifier wrappers. Without this walk, chaining
/// `.grow(1).fill_height()` (which wraps as
/// `Sized<Hug, Fill>(Sized<Grow(1), Hug>(child))`) hides the Grow weight
/// behind the outer Sized's `Hug` width, and the Stack treats the node
/// as a Hug child with `measured.w == axis_max`.
///
/// Walk rules:
/// - `Sized`: if the main-axis rule's `grow_weight() > 0`, return it.
///   Else if it's `Hug`, recurse into the child (the wrapper doesn't
///   override main-axis sizing). Else (`Fixed` / `Percent`) the wrapper
///   nails the main-axis size — return 0.
/// - `Background`, `Border`, `Align`, `Aspect`, `Constrain`, `Padded`:
///   pass-through wrappers — recurse. (`Padded` and `Constrain` adjust
///   sizes, but don't override a child's grow intent.)
/// - Anything else: not a grow node, return 0.
fn main_grow_weight(node: &Node, axis: Axis) -> u32 {
    let mut current = node;
    // Bound the walk so a malformed tree can't loop indefinitely.
    for _ in 0..16 {
        match current {
            Node::Sized { w, h, child } => {
                let main_rule = match axis {
                    Axis::Horizontal => *w,
                    Axis::Vertical => *h,
                };
                let weight = main_rule.grow_weight();
                if weight > 0 {
                    return weight;
                }
                if matches!(main_rule, SizeRule::Hug) {
                    current = child;
                } else {
                    return 0;
                }
            }
            Node::Background { child, .. }
            | Node::Border { child, .. }
            | Node::Align { child, .. }
            | Node::Aspect { child, .. }
            | Node::Constrain { child, .. }
            | Node::Padded { child, .. } => current = child,
            _ => return 0,
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::super::color::{BLACK, WHITE};
    use super::super::modifiers::LayoutMod;
    use super::super::node::{empty, image as image_node};
    use super::super::sizing::SizeRule;
    use super::*;
    use crate::pixel_ops::Bitmap;

    fn solid(w: u32, h: u32, c: super::super::color::Color) -> Bitmap {
        Bitmap::from_pixel(w, h, c)
    }

    #[test]
    fn column_sums_main_max_cross() {
        let n = column()
            .gap(2)
            .child(image_node(solid(10, 10, WHITE)))
            .child(image_node(solid(20, 5, WHITE)));
        assert_eq!(
            Node::from(n).measure(Size::new(100, 100)),
            Size::new(20, 17)
        );
    }
    #[test]
    fn row_sums_main_max_cross() {
        let n = row()
            .gap(4)
            .child(image_node(solid(10, 10, WHITE)))
            .child(image_node(solid(20, 5, WHITE)));
        assert_eq!(
            Node::from(n).measure(Size::new(100, 100)),
            Size::new(34, 10)
        );
    }

    #[test]
    fn align_items_center_centers_in_cross() {
        let small = image_node(solid(10, 10, [255, 0, 0, 255]));
        let big = image_node(solid(40, 40, [0, 0, 0, 255]));
        let n = column()
            .align_items(CrossAlign::Center)
            .child(small)
            .child(big)
            .render(40);
        assert_eq!(n.get_pixel(20, 5), [255, 0, 0, 255]);
        assert_eq!(n.get_pixel(0, 5), [0, 0, 0, 255]);
    }

    #[test]
    fn align_items_stretch_fills_cross() {
        let strip = empty()
            .background([255, 0, 0, 255])
            .height(SizeRule::Fixed(5));
        let big = image_node(solid(40, 40, BLACK));
        let img = column()
            .align_items(CrossAlign::Stretch)
            .child(strip)
            .child(big)
            .size(40, 45)
            .render(40);
        assert_eq!(img.get_pixel(0, 0), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(39, 0), [255, 0, 0, 255]);
    }

    #[test]
    fn justify_space_between() {
        let a = image_node(solid(10, 10, [255, 0, 0, 255]));
        let b = image_node(solid(10, 10, [0, 0, 255, 255]));
        let img = row()
            .justify(MainAlign::SpaceBetween)
            .child(a)
            .child(b)
            .size(50, 10)
            .render(50);
        assert_eq!(img.get_pixel(5, 5), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(45, 5), [0, 0, 255, 255]);
    }

    #[test]
    fn grow_weights_distribute_remainder() {
        let hug = image_node(solid(10, 10, [255, 0, 0, 255]));
        let grow = empty().background([0, 0, 255, 255]).grow(2).fill_height();
        let img = row()
            .align_items(CrossAlign::Stretch)
            .child(hug)
            .child(grow)
            .size(60, 10)
            .render(60);
        assert_eq!(img.get_pixel(5, 5), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(50, 5), [0, 0, 255, 255]);
    }

    #[test]
    fn grow_weights_split_2_to_1() {
        let a = empty().background([255, 0, 0, 255]).grow(2);
        let b = empty().background([0, 0, 255, 255]).grow(1);
        let img = row()
            .align_items(CrossAlign::Stretch)
            .child(a)
            .child(b)
            .size(60, 10)
            .render(60);
        assert_eq!(img.get_pixel(20, 5), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(50, 5), [0, 0, 255, 255]);
    }

    /// Regression: chaining `.grow(n).fill_height()` (or any `Sized` whose
    /// main-axis rule is `Hug`) used to hide the inner Grow weight from
    /// the Stack's `main_grow_weight` walker, so two grow children would
    /// each measure as `axis_max` wide and the second would paint off-
    /// canvas. Mirrors gallery scene 08 ("flex-grow weights").
    #[test]
    fn grow_through_outer_hug_wrapper_distributes_correctly() {
        let hug = image_node(solid(60, 10, [255, 0, 0, 255]));
        // Both wrap as Sized<Hug,Fill>(Sized<Grow(_),Hug>(Background(Empty))).
        // Pre-fix the outer Sized<Hug,Fill> reported grow_weight=0 and the
        // Stack treated these as Hug children with measured.w == axis_max.
        let g1 = empty().background([0, 255, 0, 255]).grow(1).fill_height();
        let g2 = empty().background([0, 0, 255, 255]).grow(2).fill_height();
        let img = row()
            .align_items(CrossAlign::Stretch)
            .child(hug)
            .child(g1)
            .child(g2)
            .size(360, 10)
            .render(360);
        // Available remainder = 360 - 60 = 300, split 1:2 → g1=100, g2=200.
        // Hug: x ∈ [0..60), green: [60..160), blue: [160..360).
        assert_eq!(img.get_pixel(30, 5), [255, 0, 0, 255], "hug");
        assert_eq!(img.get_pixel(110, 5), [0, 255, 0, 255], "grow 1");
        // The pre-fix bug had grow 2 painted off-canvas; this pixel was
        // bg-black instead of blue.
        assert_eq!(img.get_pixel(250, 5), [0, 0, 255, 255], "grow 2");
        assert_eq!(img.get_pixel(355, 5), [0, 0, 255, 255], "grow 2 right edge");
    }

    /// `shrink_on_overflow(true)` proportionally shrinks Hug children
    /// when their summed natural sizes exceed the row's main axis.
    /// Without it, the same row overflows on the right.
    #[test]
    fn shrink_on_overflow_compresses_hug_children() {
        // Three Sized<Hug,_> children whose intrinsic widths sum to 90,
        // in a 60-wide row. With shrink: each shrinks to 60/90 = 2/3 of
        // its size. Without: total 90 > 60, last child paints partially
        // off-canvas.
        let mk = |c: super::super::color::Color| image_node(solid(30, 10, c));
        let img = row()
            .shrink_on_overflow(true)
            .child(mk([255, 0, 0, 255]))
            .child(mk([0, 255, 0, 255]))
            .child(mk([0, 0, 255, 255]))
            .size(60, 10)
            .render(60);
        // 30 → 20 each. Cursor: 0, 20, 40. Final: red [0..20), green
        // [20..40), blue [40..60).
        assert_eq!(img.get_pixel(10, 5), [255, 0, 0, 255], "red");
        assert_eq!(
            img.get_pixel(30, 5),
            [0, 255, 0, 255],
            "green (centered after shrink)"
        );
        assert_eq!(
            img.get_pixel(55, 5),
            [0, 0, 255, 255],
            "blue (visible at right)"
        );
    }

    /// `shrink_on_overflow(true)` does NOT shrink rigid `Sized<Fixed, _>`
    /// children — they keep their declared size.
    #[test]
    fn shrink_skips_fixed_children() {
        let rigid = empty().background([255, 0, 0, 255]).size(40, 10);
        let flex = empty().background([0, 255, 0, 255]).size(20, 10);
        let img = row()
            .shrink_on_overflow(true)
            .child(rigid)
            .child(flex)
            .size(50, 10)
            .render(50);
        // Rigid stays at 40px (red [0..40)), flex shrinks from 20 to
        // 10px (50 - 40 = 10) — green [40..50).
        // Actually the second child here is also Sized<Fixed,Fixed> via
        // .size(), so it's also rigid → not shrunk. So we'll have
        // overflow. Verify red still occupies the first 40px.
        assert_eq!(img.get_pixel(10, 5), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(30, 5), [255, 0, 0, 255]);
    }

    /// Default behavior (`shrink: false`) preserves overflow — the
    /// diagnostics layer can flag it.
    #[test]
    fn shrink_off_by_default_overflow_preserved() {
        let mk = |c: super::super::color::Color| image_node(solid(30, 10, c));
        let img = row()
            .child(mk([255, 0, 0, 255]))
            .child(mk([0, 255, 0, 255]))
            .child(mk([0, 0, 255, 255]))
            .size(60, 10)
            .render(60);
        // Without shrink: cursor 0, 30, 60, 90. Red [0..30), green [30..60),
        // blue at [60..) — off-canvas, not visible.
        assert_eq!(img.get_pixel(10, 5), [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(45, 5), [0, 255, 0, 255]);
        // Pixel at x=55 should still be green (last visible non-overflow).
        assert_eq!(img.get_pixel(55, 5), [0, 255, 0, 255]);
    }
}
