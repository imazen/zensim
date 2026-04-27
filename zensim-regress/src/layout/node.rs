//! The retained-tree IR — one [`Node`] enum covering leaves, containers,
//! and modifiers, plus the top-level [`Node::measure`] / [`Node::paint`]
//! dispatch tables.
//!
//! The dispatch arms delegate to per-variant helpers in:
//! - [`super::paint`] — image/text leaves and primitives,
//! - [`super::stack`] / [`super::grid`] / [`super::layers`] — containers,
//! - [`super::modifiers`] — modifier nodes.

use image::RgbaImage;

use super::color::Color;
use super::geom::Size;
use super::safety;
use super::sizing::{Fit, SizeRule};
use super::text::TextSpec;
use super::{geom, grid, layers, modifiers, paint, stack};

/// The retained layout tree.
///
/// Construct via the free functions ([`image`], [`text`], [`line`],
/// [`fill`], [`empty`]) and the builders in
/// [`super::stack`] / [`super::grid`] / [`super::layers`]; chain
/// modifiers from the [`super::LayoutMod`] trait.
#[derive(Clone, Debug)]
pub enum Node {
    // ── Leaves ────────────────────────────────────────────────────────
    Empty,
    Fill(Color),
    Image(RgbaImage),
    Text(TextSpec),

    // ── Containers ────────────────────────────────────────────────────
    Stack {
        axis: geom::Axis,
        gap: u32,
        justify: super::sizing::MainAlign,
        align_items: super::sizing::CrossAlign,
        children: Vec<Node>,
    },
    Grid {
        cols: Vec<super::sizing::Track>,
        rows: Vec<super::sizing::Track>,
        gap: (u32, u32),
        pad: u32,
        cells: Vec<(grid::GridSpan, Node)>,
    },
    Layers(Vec<Node>),

    // ── Modifiers ─────────────────────────────────────────────────────
    Padded {
        insets: geom::Insets,
        child: Box<Node>,
    },
    Sized {
        w: SizeRule,
        h: SizeRule,
        child: Box<Node>,
    },
    Constrain {
        min_w: Option<u32>,
        max_w: Option<u32>,
        min_h: Option<u32>,
        max_h: Option<u32>,
        child: Box<Node>,
    },
    Aspect {
        num: u32,
        den: u32,
        child: Box<Node>,
    },
    Align {
        h: geom::HAlign,
        v: geom::VAlign,
        child: Box<Node>,
    },
    Fit {
        mode: Fit,
        child: Box<Node>,
    },
    Background {
        color: Color,
        child: Box<Node>,
    },
    Border {
        color: Color,
        child: Box<Node>,
    },

    // ── Specialised composite nodes ───────────────────────────────────
    /// A label strip with multiple aligned segments. Width-aware: at
    /// paint time, computes a single `char_h` that lets all segments
    /// fit side-by-side within the actual rect, so it works at any
    /// strip width without caller char_h tuning. See
    /// [`super::label::LabelStyle::segmented_strip`].
    SegmentedStrip {
        segments: Vec<super::label::LabelSegment>,
        style: Box<super::label::LabelStyle>,
    },
}

// ── Implicit conversions ───────────────────────────────────────────────

impl From<&str> for Node {
    /// Default: white text on black (pre-blended). For non-default
    /// background, use [`text`] / [`line`] explicitly.
    fn from(s: &str) -> Node {
        Node::Text(TextSpec::lines(
            vec![(s.to_string(), super::color::WHITE)],
            super::color::BLACK,
        ))
    }
}

impl From<String> for Node {
    fn from(s: String) -> Node {
        Node::Text(TextSpec::lines(
            vec![(s, super::color::WHITE)],
            super::color::BLACK,
        ))
    }
}

impl From<RgbaImage> for Node {
    fn from(img: RgbaImage) -> Node {
        Node::Image(img)
    }
}

// ── Free-function leaf constructors ────────────────────────────────────

pub fn empty() -> Node {
    Node::Empty
}
pub fn fill(c: Color) -> Node {
    Node::Fill(c)
}
pub fn image(img: RgbaImage) -> Node {
    Node::Image(img)
}
pub fn text(spec: TextSpec) -> Node {
    Node::Text(spec)
}

/// Default minimum padding (in pixels) wrapped around [`line`]'s text
/// so it doesn't kiss the edges of a tight container. Use [`text`]
/// directly for raw, no-padding text.
pub const DEFAULT_LINE_PADDING: u32 = 4;

/// Single-line text on a transparent background, sized to fit the
/// constraint width, with [`DEFAULT_LINE_PADDING`] of breathing room
/// applied uniformly. For text without padding, use [`text`] directly.
pub fn line(s: impl Into<String>, fg: Color) -> Node {
    use super::modifiers::LayoutMod;
    Node::Text(TextSpec::lines(
        vec![(s.into(), fg)],
        super::color::TRANSPARENT,
    ))
    .padding(DEFAULT_LINE_PADDING)
}

// ── Dispatch tables ────────────────────────────────────────────────────

impl Node {
    /// Measure this node given the maximum available `(w, h)`. The
    /// returned size is clamped to `max` and to [`safety::MAX_DIM`].
    /// If recursion exceeds [`safety::MAX_DEPTH`] (e.g., a malicious
    /// or deserialized tree with absurd nesting), [`Size::ZERO`] is
    /// returned.
    pub fn measure(&self, max: Size) -> Size {
        safety::with_depth(|| {
            let raw = self.measure_raw(max);
            safety::clamp_size(Size::new(raw.w.min(max.w), raw.h.min(max.h)))
        })
        .unwrap_or(Size::ZERO)
    }

    fn measure_raw(&self, max: Size) -> Size {
        match self {
            Node::Empty | Node::Fill(_) => Size::ZERO,
            Node::Image(img) => Size::new(img.width(), img.height()),
            Node::Text(spec) => spec.natural(max.w, max.h),
            Node::Stack {
                axis,
                gap,
                children,
                ..
            } => stack::measure(*axis, *gap, children, max),
            Node::Grid {
                cols,
                rows,
                gap,
                pad,
                cells,
            } => grid::measure(cols, rows, *gap, *pad, cells, max),
            Node::Layers(children) => layers::measure(children, max),
            Node::Padded { insets, child } => modifiers::measure_padded(*insets, child, max),
            Node::Sized { w, h, child } => modifiers::measure_sized(*w, *h, child, max),
            Node::Constrain {
                min_w,
                max_w,
                min_h,
                max_h,
                child,
            } => modifiers::measure_constrain(*min_w, *max_w, *min_h, *max_h, child, max),
            Node::Aspect { num, den, child } => modifiers::measure_aspect(*num, *den, child, max),
            Node::Align { child, .. }
            | Node::Fit { child, .. }
            | Node::Background { child, .. }
            | Node::Border { child, .. } => child.measure(max),

            Node::SegmentedStrip { segments, style } => {
                super::label::measure_segmented_strip(segments, style, max)
            }
        }
    }

    /// Walk the tree and multiply every fixed-pixel quantity by
    /// `scale` (Sized::Fixed, Insets, Track::Px, gap, pad, char_h).
    /// Relative quantities (Hug / Fill / Grow / Percent / Fr / Auto)
    /// are unchanged. `scale == 1.0`, `<= 0.0`, or non-finite returns
    /// `self` unchanged (no allocation).
    ///
    /// Used by [`super::render_with_config`] to apply a CSS-`dppx`-style
    /// uniform scale at render time.
    pub fn scaled(self, scale: f32) -> Self {
        if scale == 1.0 || scale <= 0.0 || !scale.is_finite() {
            return self;
        }
        let s = |v: u32| ((v as f32) * scale).round() as u32;
        match self {
            Node::Empty | Node::Fill(_) | Node::Image(_) => self,
            Node::Text(spec) => Node::Text(spec.scaled(scale)),
            Node::Stack {
                axis,
                gap,
                justify,
                align_items,
                children,
            } => Node::Stack {
                axis,
                justify,
                align_items,
                gap: s(gap),
                children: children.into_iter().map(|c| c.scaled(scale)).collect(),
            },
            Node::Grid {
                cols,
                rows,
                gap,
                pad,
                cells,
            } => Node::Grid {
                cols: cols.into_iter().map(|t| t.scaled(scale)).collect(),
                rows: rows.into_iter().map(|t| t.scaled(scale)).collect(),
                gap: (s(gap.0), s(gap.1)),
                pad: s(pad),
                cells: cells
                    .into_iter()
                    .map(|(span, n)| (span, n.scaled(scale)))
                    .collect(),
            },
            Node::Layers(children) => {
                Node::Layers(children.into_iter().map(|c| c.scaled(scale)).collect())
            }
            Node::Padded { insets, child } => Node::Padded {
                insets: insets.scaled(scale),
                child: Box::new(child.scaled(scale)),
            },
            Node::Sized { w, h, child } => Node::Sized {
                w: w.scaled(scale),
                h: h.scaled(scale),
                child: Box::new(child.scaled(scale)),
            },
            Node::Constrain {
                min_w,
                max_w,
                min_h,
                max_h,
                child,
            } => Node::Constrain {
                min_w: min_w.map(s),
                max_w: max_w.map(s),
                min_h: min_h.map(s),
                max_h: max_h.map(s),
                child: Box::new(child.scaled(scale)),
            },
            Node::Aspect { num, den, child } => Node::Aspect {
                num,
                den,
                child: Box::new(child.scaled(scale)),
            },
            Node::Align { h, v, child } => Node::Align {
                h,
                v,
                child: Box::new(child.scaled(scale)),
            },
            Node::Fit { mode, child } => Node::Fit {
                mode,
                child: Box::new(child.scaled(scale)),
            },
            Node::Background { color, child } => Node::Background {
                color,
                child: Box::new(child.scaled(scale)),
            },
            Node::Border { color, child } => Node::Border {
                color,
                child: Box::new(child.scaled(scale)),
            },
            Node::SegmentedStrip {
                segments,
                mut style,
            } => {
                // Scale only the explicit `char_h` cap and `padding`.
                // The auto-fit char_h derived at paint time already
                // adapts to the (scaled) rect width.
                if let Some(h) = style.char_h {
                    style.char_h = Some(((h as f32) * scale).round() as u32);
                }
                style.padding = style.padding.scaled(scale);
                let segments = segments
                    .into_iter()
                    .map(|mut s| {
                        if let Some(h) = s.char_h {
                            s.char_h = Some(((h as f32) * scale).round() as u32);
                        }
                        s
                    })
                    .collect();
                Node::SegmentedStrip { segments, style }
            }
        }
    }

    pub(super) fn paint(&self, rect: super::geom::Rect, canvas: &mut RgbaImage) {
        if rect.w == 0 || rect.h == 0 {
            return;
        }
        // Bounce out at recursion-depth cap — return without painting
        // rather than overflowing the stack on hostile trees.
        let Some(()) = safety::with_depth(|| self.paint_inner(rect, canvas)) else {
            return;
        };
    }

    fn paint_inner(&self, rect: super::geom::Rect, canvas: &mut RgbaImage) {
        match self {
            Node::Empty => {}
            Node::Fill(c) => paint::fill_rect(canvas, rect, *c),
            Node::Image(img) => paint::render_image(img, Fit::None, rect, canvas),
            Node::Text(spec) => paint::render_text(spec, rect, canvas),
            Node::Stack {
                axis,
                gap,
                justify,
                align_items,
                children,
            } => stack::paint(*axis, *gap, *justify, *align_items, children, rect, canvas),
            Node::Grid {
                cols,
                rows,
                gap,
                pad,
                cells,
            } => grid::paint(cols, rows, *gap, *pad, cells, rect, canvas),
            Node::Layers(children) => layers::paint(children, rect, canvas),
            Node::Padded { insets, child } => modifiers::paint_padded(*insets, child, rect, canvas),
            Node::Sized { child, .. }
            | Node::Constrain { child, .. }
            | Node::Aspect { child, .. } => child.paint(rect, canvas),
            Node::Align { h, v, child } => modifiers::paint_align(*h, *v, child, rect, canvas),
            Node::Fit { mode, child } => modifiers::paint_fit(*mode, child, rect, canvas),
            Node::Background { color, child } => {
                modifiers::paint_background(*color, child, rect, canvas)
            }
            Node::Border { color, child } => modifiers::paint_border(*color, child, rect, canvas),

            Node::SegmentedStrip { segments, style } => {
                super::label::paint_segmented_strip(segments, style, rect, canvas);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::color::WHITE;
    use super::*;
    use image::Rgba;

    fn solid(w: u32, h: u32, c: Color) -> RgbaImage {
        RgbaImage::from_pixel(w, h, Rgba(c))
    }

    #[test]
    fn empty_measures_zero() {
        assert_eq!(empty().measure(Size::new(100, 100)), Size::ZERO);
    }
    #[test]
    fn image_natural_size() {
        let n: Node = solid(50, 30, [255, 0, 0, 255]).into();
        assert_eq!(n.measure(Size::new(200, 200)), Size::new(50, 30));
    }
    #[test]
    fn from_str_makes_text_node() {
        let n: Node = "hello".into();
        assert!(matches!(n, Node::Text(_)));
    }
    #[test]
    fn line_helper_uses_provided_fg() {
        let _ = line("hi", WHITE);
    }
}
