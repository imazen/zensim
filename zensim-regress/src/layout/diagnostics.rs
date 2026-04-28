//! Runtime paint-time containment diagnostics.
//!
//! Catches the bug class where a layout container assigns a child rect
//! that extends past its own paint rect — e.g., a Stack distributing
//! `cursor + m_size > rect.x + rect.w` because a Grow weight was
//! hidden behind an outer `Sized<Hug, _>` wrapper. Pre-fix, those
//! pixels were silently clipped by [`crate::pixel_ops::Bitmap::put_pixel`]'s
//! out-of-bounds early-return, so the bug only surfaced as missing
//! glyphs in rendered output.
//!
//! The check is opt-in: paint always succeeds and produces a bitmap;
//! the errors observed during paint are surfaced via the
//! [`super::render_checked`] entry point. Production callers can keep
//! using the infallible [`super::render`] / [`super::render_with_config`].
//! Tests, debug builds, and CI should prefer the checked variant so
//! containment violations fail loudly rather than leaving phantom pixels.

use std::cell::RefCell;

use super::geom::Rect;

/// Which container reported the violation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContainerKind {
    /// Row or column.
    Stack,
    /// Grid.
    Grid,
    /// Layered overlap.
    Layers,
}

/// Coarse classification of the offending child node — useful in error
/// messages without exposing the full `Node` enum (which would be a
/// public-API commitment).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NodeKind {
    Empty,
    Fill,
    Image,
    Text,
    Stack,
    Grid,
    Layers,
    Padded,
    Sized,
    Constrain,
    Aspect,
    Align,
    Fit,
    Background,
    Border,
    SegmentedStrip,
}

/// Per-edge overflow amounts in pixels. `0` on an edge means the child
/// did not exceed the parent on that edge.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Overflow {
    /// Pixels by which the child extends past the parent's left edge.
    pub left: u32,
    /// Pixels by which the child extends past the parent's top edge.
    pub top: u32,
    /// Pixels by which the child extends past the parent's right edge.
    pub right: u32,
    /// Pixels by which the child extends past the parent's bottom edge.
    pub bottom: u32,
}

impl Overflow {
    /// Compute the per-edge overflow of `child` relative to `parent`.
    /// Returns all-zero if `child` is fully contained.
    pub fn of(parent: Rect, child: Rect) -> Self {
        let parent_x_end = parent.x.saturating_add(parent.w);
        let parent_y_end = parent.y.saturating_add(parent.h);
        let child_x_end = child.x.saturating_add(child.w);
        let child_y_end = child.y.saturating_add(child.h);
        Self {
            left: parent.x.saturating_sub(child.x),
            top: parent.y.saturating_sub(child.y),
            right: child_x_end.saturating_sub(parent_x_end),
            bottom: child_y_end.saturating_sub(parent_y_end),
        }
    }

    /// `true` if any edge has nonzero overflow.
    pub fn is_any(&self) -> bool {
        self.left | self.top | self.right | self.bottom != 0
    }
}

/// One paint-time containment violation.
///
/// Emitted when a layout container hands a child a paint rect that
/// extends past the container's own rect. The paint still proceeds
/// (off-canvas pixels are clipped by the bitmap's bounds checks); this
/// record exists so callers can detect the bug rather than chase
/// phantom missing glyphs.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LayoutError {
    /// Which container kind reported the violation.
    pub container: ContainerKind,
    /// Index of the offending child within its container.
    pub child_index: usize,
    /// Coarse kind of the offending child node.
    pub child_kind: NodeKind,
    /// The container's paint rect.
    pub parent_rect: Rect,
    /// The child's intended paint rect.
    pub child_rect: Rect,
    /// Per-edge overflow in pixels.
    pub overflow: Overflow,
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = self.parent_rect;
        let c = self.child_rect;
        let o = self.overflow;
        write!(
            f,
            "off-canvas paint in {:?}: child #{} ({:?}) at {}+{}×{}+{} \
             overflows parent {}+{}×{}+{} by L{} T{} R{} B{}",
            self.container,
            self.child_index,
            self.child_kind,
            c.x,
            c.w,
            c.y,
            c.h,
            p.x,
            p.w,
            p.y,
            p.h,
            o.left,
            o.top,
            o.right,
            o.bottom,
        )
    }
}

impl std::error::Error for LayoutError {}

// ── Thread-local collector ───────────────────────────────────────────

thread_local! {
    /// `Some` when a collector is active for this thread; paint reports
    /// containment errors into the buffer. `None` (the default) means
    /// no collector — `report` is a cheap no-op.
    static COLLECTOR: RefCell<Option<Vec<LayoutError>>> = const { RefCell::new(None) };
}

/// Run `f` while collecting paint-time containment errors. Returns the
/// function result and the errors observed during the call.
///
/// Reentrant calls share the outer collector — the inner call sees
/// errors from anything painted underneath it. This matches how layout
/// recursion works.
pub fn collect_paint_errors<R>(f: impl FnOnce() -> R) -> (R, Vec<LayoutError>) {
    let active_at_entry =
        COLLECTOR.with(|c| c.borrow().is_some() || c.replace(Some(Vec::new())).is_some());
    let result = f();
    if active_at_entry {
        // Outer collector owns the buffer; just return the function result
        // with an empty local view. The outer collect_paint_errors drains.
        (result, Vec::new())
    } else {
        let errs = COLLECTOR.with(|c| c.replace(None)).unwrap_or_default();
        (result, errs)
    }
}

/// Report a containment violation. No-op if no collector is active.
pub(super) fn report(err: LayoutError) {
    COLLECTOR.with(|c| {
        if let Some(buf) = c.borrow_mut().as_mut() {
            buf.push(err);
        }
    });
}

/// Helper for paint sites: if `child_rect` is not contained in
/// `parent_rect`, emit a [`LayoutError`] (when a collector is active).
/// Always cheap — the bounds check is a few u32 compares; the
/// allocation only happens on violation AND when collecting.
pub(super) fn check_contained(
    container: ContainerKind,
    child_index: usize,
    child_kind: NodeKind,
    parent_rect: Rect,
    child_rect: Rect,
) {
    let overflow = Overflow::of(parent_rect, child_rect);
    if !overflow.is_any() {
        return;
    }
    // Allocate / report only if a collector exists.
    COLLECTOR.with(|c| {
        if c.borrow().is_some() {
            report(LayoutError {
                container,
                child_index,
                child_kind,
                parent_rect,
                child_rect,
                overflow,
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_of_contained_is_zero() {
        let p = Rect::new(0, 0, 100, 100);
        let c = Rect::new(10, 10, 80, 80);
        let o = Overflow::of(p, c);
        assert!(!o.is_any());
        assert_eq!(o, Overflow::default());
    }

    #[test]
    fn overflow_right_and_bottom() {
        let p = Rect::new(0, 0, 100, 100);
        let c = Rect::new(10, 20, 200, 90);
        let o = Overflow::of(p, c);
        // child x-end = 210, parent x-end = 100 → right = 110.
        // child y-end = 110, parent y-end = 100 → bottom = 10.
        assert_eq!(o.right, 110);
        assert_eq!(o.bottom, 10);
        assert_eq!(o.left, 0);
        assert_eq!(o.top, 0);
        assert!(o.is_any());
    }

    #[test]
    fn overflow_left_and_top() {
        let p = Rect::new(50, 50, 100, 100);
        let c = Rect::new(10, 20, 50, 50);
        let o = Overflow::of(p, c);
        // child x = 10, parent x = 50 → left = 40.
        // child y = 20, parent y = 50 → top = 30.
        assert_eq!(o.left, 40);
        assert_eq!(o.top, 30);
        assert!(o.is_any());
    }

    #[test]
    fn collect_returns_no_errors_for_clean_paint() {
        let (val, errs) = collect_paint_errors(|| 42);
        assert_eq!(val, 42);
        assert!(errs.is_empty());
    }

    #[test]
    fn collect_captures_reported_errors() {
        let (_, errs) = collect_paint_errors(|| {
            check_contained(
                ContainerKind::Stack,
                0,
                NodeKind::Image,
                Rect::new(0, 0, 100, 100),
                Rect::new(0, 0, 200, 100),
            );
        });
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].overflow.right, 100);
        assert_eq!(errs[0].container, ContainerKind::Stack);
        assert_eq!(errs[0].child_kind, NodeKind::Image);
    }

    #[test]
    fn report_outside_collector_is_noop() {
        // No panic, no error — just silently drops.
        check_contained(
            ContainerKind::Stack,
            0,
            NodeKind::Image,
            Rect::new(0, 0, 100, 100),
            Rect::new(0, 0, 200, 100),
        );
    }

    #[test]
    fn nested_collectors_use_outer_buffer() {
        let (_, outer_errs) = collect_paint_errors(|| {
            check_contained(
                ContainerKind::Stack,
                0,
                NodeKind::Image,
                Rect::new(0, 0, 10, 10),
                Rect::new(0, 0, 20, 10),
            );
            let (_, inner_errs) = collect_paint_errors(|| {
                check_contained(
                    ContainerKind::Grid,
                    1,
                    NodeKind::Text,
                    Rect::new(0, 0, 10, 10),
                    Rect::new(0, 0, 30, 10),
                );
            });
            // Inner collect yields nothing — the outer owns the buffer.
            assert!(inner_errs.is_empty());
        });
        // Outer collected both.
        assert_eq!(outer_errs.len(), 2);
    }
}
