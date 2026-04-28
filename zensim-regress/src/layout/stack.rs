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
}

impl Stack {
    pub fn new(axis: Axis) -> Self {
        Self {
            axis,
            gap: 0,
            justify: MainAlign::Start,
            align_items: CrossAlign::Start,
            children: Vec::new(),
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
}
