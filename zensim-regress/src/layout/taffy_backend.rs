//! The **default** layout backend: route the retained
//! [`Node`](super::Node) tree through the [`taffy`] CSS flex/grid solver,
//! then paint the taffy-computed rects with our *existing* paint
//! primitives. Selected by [`super::Backend::Taffy`] (the default);
//! [`super::Backend::Native`] keeps the hand-written solver available for
//! A/B and for the native-only overflow diagnostics.
//!
//! Because painting is identical across backends (same [`super::paint`]
//! calls, same font, same compositing) and only the rect assignment
//! changes, the two backends differ only in geometry — verified in the
//! parity harness (`examples/taffy_parity.rs`): the core flex/grid math
//! matches to sub-pixel rounding.
//!
//! Mapping is 1 [`Node`] → 1 taffy node so the paint walk is a plain
//! parallel recursion. Modifiers become taffy `Style` on a single-child
//! node; decorations (background/border/fit) are applied during the walk.
//!
//! Constructs taffy cannot express faithfully are recorded in
//! [`Coverage`] rather than silently mis-rendered.
//!
//! Safety: `build` honors the active [`super::safety`] caps
//! (`max_children` / `max_cells` / `max_tracks` / `max_depth`); the
//! canvas is clamped to `max_dim` + the pixel budget in
//! [`render_into_canvas`]. So hostile/deserialized trees are bounded on
//! the taffy path exactly as on the native path.

use taffy::prelude::*;
use taffy::style_helpers::{auto, fr, length, minmax, percent};
use taffy::{Line as TaffyLine, Rect as TaffyRect, Size as TaffySize, TaffyTree};

use crate::pixel_ops::Bitmap;

use super::color::Color;
use super::geom::{HAlign, Rect, VAlign};
use super::label::LabelSegment;
use super::node::Node;
use super::sizing::{CrossAlign, Fit, MainAlign, SizeRule, Track};
use super::text::TextSpec;
use super::{RenderConfig, paint};

/// Per-leaf measurement context handed to taffy's measure callback.
#[derive(Clone)]
enum Ctx {
    /// Intrinsic fixed size (Image → its pixels; Empty/Fill → zero).
    Fixed(f32, f32),
    /// Text leaf — measured via [`TextSpec::natural`] against the
    /// available width/height taffy offers.
    Text(TextSpec),
    /// Segmented label strip — width-aware; measured via the label module.
    Strip(Vec<LabelSegment>, Box<super::label::LabelStyle>),
}

/// What the taffy bridge could and could not express for a given tree.
#[derive(Debug, Default, Clone)]
pub struct Coverage {
    /// Node kinds that mapped to a taffy construct with matching intent.
    pub mapped: Vec<&'static str>,
    /// Node kinds that required an approximation (recorded so the report
    /// is honest about where taffy semantics diverge from ours).
    pub approximated: Vec<(&'static str, &'static str)>,
}

impl Coverage {
    fn map(&mut self, k: &'static str) {
        if !self.mapped.contains(&k) {
            self.mapped.push(k);
        }
    }
    fn approx(&mut self, k: &'static str, why: &'static str) {
        if !self.approximated.iter().any(|(a, _)| *a == k) {
            self.approximated.push((k, why));
        }
    }
}

// ── Style construction from our modifier/sizing vocabulary ─────────────

fn dim_from_rule(rule: SizeRule) -> Dimension {
    match rule {
        SizeRule::Hug => Dimension::auto(),
        SizeRule::Fixed(v) => Dimension::length(v as f32),
        // Our `Fill` = 100% of the parent constraint.
        SizeRule::Fill => Dimension::percent(1.0_f32),
        // Grow is expressed via flex_grow; the base size stays auto.
        SizeRule::Grow(_) => Dimension::auto(),
        SizeRule::Percent(p) => Dimension::percent(p.clamp(0.0, 1.0)),
    }
}

fn track_component(t: Track) -> GridTemplateComponent<String> {
    match t {
        Track::Px(v) => GridTemplateComponent::Single(length(v as f32)),
        Track::Fr(w) => GridTemplateComponent::Single(fr(w as f32)),
        Track::Auto => GridTemplateComponent::Single(auto()),
        Track::Percent(p) => GridTemplateComponent::Single(percent(p.clamp(0.0, 1.0))),
        // minmax(min_px, weight·fr) — the direct CSS analogue of FrMin.
        Track::FrMin { weight, min_px } => {
            GridTemplateComponent::Single(minmax(length(min_px as f32), fr(weight as f32)))
        }
    }
}

// ── Tree construction ──────────────────────────────────────────────────

/// Build a taffy subtree for `node`, returning its node id. Records
/// coverage as it goes and honors the active [`super::safety`] depth /
/// child / cell / track caps (a tree past `max_depth` becomes an empty
/// leaf, matching the native solver's bounce-out).
fn build(tree: &mut TaffyTree<Ctx>, node: &Node, cov: &mut Coverage) -> NodeId {
    match super::safety::with_depth(|| build_inner(tree, node, cov)) {
        Some(id) => id,
        None => tree.new_leaf(Style::default()).unwrap(),
    }
}

fn build_inner(tree: &mut TaffyTree<Ctx>, node: &Node, cov: &mut Coverage) -> NodeId {
    match node {
        Node::Empty => {
            cov.map("Empty");
            tree.new_leaf_with_context(Style::default(), Ctx::Fixed(0.0, 0.0))
                .unwrap()
        }
        Node::Fill(_) => {
            cov.map("Fill");
            // Our `Fill` paints its whole given rect — semantically "fill
            // the parent." In taffy's box model a zero-intrinsic leaf must
            // say so explicitly (100% of the parent's inner box), else it
            // collapses to 0×0 and paints nothing.
            let style = Style {
                size: TaffySize {
                    width: percent(1.0_f32),
                    height: percent(1.0_f32),
                },
                ..Default::default()
            };
            tree.new_leaf_with_context(style, Ctx::Fixed(0.0, 0.0))
                .unwrap()
        }
        Node::Image(img) => {
            cov.map("Image");
            tree.new_leaf_with_context(
                Style::default(),
                Ctx::Fixed(img.width() as f32, img.height() as f32),
            )
            .unwrap()
        }
        Node::Text(spec) => {
            cov.map("Text");
            if matches!(spec.style, super::text::TextStyle::AutoFit { .. }) {
                cov.approx(
                    "Text",
                    "AutoFit binary-search fit-to-box vs taffy content measure",
                );
            }
            tree.new_leaf_with_context(Style::default(), Ctx::Text(spec.clone()))
                .unwrap()
        }
        Node::SegmentedStrip { segments, style } => {
            cov.map("SegmentedStrip");
            tree.new_leaf_with_context(
                Style::default(),
                Ctx::Strip(segments.clone(), style.clone()),
            )
            .unwrap()
        }

        // ── Containers ─────────────────────────────────────────────────
        Node::Stack {
            axis,
            gap,
            justify,
            align_items,
            children,
            ..
        } => {
            cov.map("Stack");
            let cap = super::safety::cap_children(children.len());
            let kids: Vec<NodeId> = children
                .iter()
                .take(cap)
                .map(|c| build(tree, c, cov))
                .collect();
            let (fd, gap_size) = match axis {
                super::geom::Axis::Horizontal => (
                    FlexDirection::Row,
                    TaffySize {
                        width: length(*gap as f32),
                        height: length(0.0_f32),
                    },
                ),
                super::geom::Axis::Vertical => (
                    FlexDirection::Column,
                    TaffySize {
                        width: length(0.0_f32),
                        height: length(*gap as f32),
                    },
                ),
            };
            let style = Style {
                display: Display::Flex,
                flex_direction: fd,
                gap: gap_size,
                justify_content: Some(map_justify(*justify)),
                align_items: Some(map_align(*align_items)),
                ..Default::default()
            };
            tree.new_with_children(style, &kids).unwrap()
        }
        Node::Grid {
            cols,
            rows,
            gap,
            pad,
            cells,
        } => {
            cov.map("Grid");
            let cell_cap = super::safety::cap_cells(cells.len());
            let kids: Vec<NodeId> = cells
                .iter()
                .take(cell_cap)
                .map(|(span, child)| {
                    let id = build(tree, child, cov);
                    // Our GridSpan is 0-indexed cell coords; taffy grid
                    // lines are 1-indexed. colspan/rowspan → span().
                    let mut s = tree.style(id).unwrap().clone();
                    s.grid_column = TaffyLine {
                        start: taffy::style_helpers::line(span.col as i16 + 1),
                        end: taffy::style_helpers::span(span.colspan.max(1) as u16),
                    };
                    s.grid_row = TaffyLine {
                        start: taffy::style_helpers::line(span.row as i16 + 1),
                        end: taffy::style_helpers::span(span.rowspan.max(1) as u16),
                    };
                    tree.set_style(id, s).unwrap();
                    id
                })
                .collect();
            let col_cap = super::safety::cap_tracks(cols.len());
            let row_cap = super::safety::cap_tracks(rows.len());
            let style = Style {
                display: Display::Grid,
                grid_template_columns: cols
                    .iter()
                    .take(col_cap)
                    .map(|t| track_component(*t))
                    .collect(),
                grid_template_rows: rows
                    .iter()
                    .take(row_cap)
                    .map(|t| track_component(*t))
                    .collect(),
                gap: TaffySize {
                    width: length(gap.0 as f32),
                    height: length(gap.1 as f32),
                },
                padding: TaffyRect {
                    left: length(*pad as f32),
                    right: length(*pad as f32),
                    top: length(*pad as f32),
                    bottom: length(*pad as f32),
                },
                ..Default::default()
            };
            tree.new_with_children(style, &kids).unwrap()
        }
        Node::Layers(children) => {
            cov.map("Layers");
            cov.approx(
                "Layers",
                "z-stack via position:absolute inset:0 (no native z-order)",
            );
            let cap = super::safety::cap_children(children.len());
            let kids: Vec<NodeId> = children
                .iter()
                .take(cap)
                .map(|c| {
                    let id = build(tree, c, cov);
                    let mut s = tree.style(id).unwrap().clone();
                    s.position = Position::Absolute;
                    s.inset = TaffyRect {
                        left: LengthPercentageAuto::length(0.0_f32),
                        right: LengthPercentageAuto::length(0.0_f32),
                        top: LengthPercentageAuto::length(0.0_f32),
                        bottom: LengthPercentageAuto::length(0.0_f32),
                    };
                    tree.set_style(id, s).unwrap();
                    id
                })
                .collect();
            // The container holds absolute children that each fill it.
            // (block_layout isn't enabled; Flex is fine — absolute
            // children are out of flow regardless of the container's
            // display mode.)
            let style = Style {
                display: Display::Flex,
                ..Default::default()
            };
            tree.new_with_children(style, &kids).unwrap()
        }

        // ── Modifiers (1 child) ────────────────────────────────────────
        Node::Padded { insets, child } => {
            cov.map("Padded");
            let inner = build(tree, child, cov);
            set_fill(tree, inner);
            let style = Style {
                padding: TaffyRect {
                    left: length(insets.left as f32),
                    right: length(insets.right as f32),
                    top: length(insets.top as f32),
                    bottom: length(insets.bottom as f32),
                },
                ..Default::default()
            };
            tree.new_with_children(style, &[inner]).unwrap()
        }
        Node::Sized { w, h, child } => {
            cov.map("Sized");
            let inner = build(tree, child, cov);
            set_fill(tree, inner);
            let style = Style {
                size: TaffySize {
                    width: dim_from_rule(*w),
                    height: dim_from_rule(*h),
                },
                flex_grow: match (*w, *h) {
                    (SizeRule::Grow(n), _) | (_, SizeRule::Grow(n)) => n as f32,
                    (SizeRule::Fill, _) | (_, SizeRule::Fill) => 0.0,
                    _ => 0.0,
                },
                ..Default::default()
            };
            tree.new_with_children(style, &[inner]).unwrap()
        }
        Node::Constrain {
            min_w,
            max_w,
            min_h,
            max_h,
            child,
        } => {
            cov.map("Constrain");
            let inner = build(tree, child, cov);
            set_fill(tree, inner);
            let d = |o: Option<u32>| {
                o.map(|v| Dimension::length(v as f32))
                    .unwrap_or(Dimension::auto())
            };
            let style = Style {
                min_size: TaffySize {
                    width: d(*min_w),
                    height: d(*min_h),
                },
                max_size: TaffySize {
                    width: d(*max_w),
                    height: d(*max_h),
                },
                ..Default::default()
            };
            tree.new_with_children(style, &[inner]).unwrap()
        }
        Node::Aspect { num, den, child } => {
            cov.map("Aspect");
            let inner = build(tree, child, cov);
            set_fill(tree, inner);
            let style = Style {
                aspect_ratio: Some(*num as f32 / (*den as f32).max(1.0)),
                ..Default::default()
            };
            tree.new_with_children(style, &[inner]).unwrap()
        }
        Node::Align { h, v, child } => {
            cov.map("Align");
            let inner = build(tree, child, cov);
            // Position the single child within this box via a flex
            // container: main axis = row, justify = horizontal, align =
            // vertical.
            let style = Style {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                justify_content: Some(match h {
                    HAlign::Left => JustifyContent::START,
                    HAlign::Center => JustifyContent::CENTER,
                    HAlign::Right => JustifyContent::END,
                }),
                align_items: Some(match v {
                    VAlign::Top => AlignItems::START,
                    VAlign::Center => AlignItems::CENTER,
                    VAlign::Bottom => AlignItems::END,
                }),
                ..Default::default()
            };
            tree.new_with_children(style, &[inner]).unwrap()
        }
        Node::Fit { child, .. } => {
            cov.map("Fit");
            // Fit is object-fit (paint-time); geometry is passthrough.
            let inner = build(tree, child, cov);
            set_fill(tree, inner);
            tree.new_with_children(Style::default(), &[inner]).unwrap()
        }
        Node::Background { child, .. } => {
            cov.map("Background");
            let inner = build(tree, child, cov);
            set_fill(tree, inner);
            tree.new_with_children(Style::default(), &[inner]).unwrap()
        }
        Node::Border { child, .. } => {
            cov.map("Border");
            let inner = build(tree, child, cov);
            set_fill(tree, inner);
            tree.new_with_children(Style::default(), &[inner]).unwrap()
        }
    }
}

/// Make a just-built child fill its transparent-modifier wrapper on both
/// axes — our modifier model hands the child the wrapper's whole rect,
/// whereas taffy would otherwise auto-size the child to hug and starve any
/// flex/grid distribution inside it. Not applied to `Align` (which
/// positions a natural-size child) or `Layers` (absolute children).
fn set_fill(tree: &mut TaffyTree<Ctx>, id: NodeId) {
    let mut s = tree.style(id).unwrap().clone();
    s.flex_grow = 1.0;
    s.flex_shrink = 1.0;
    s.align_self = Some(AlignItems::STRETCH);
    // Only widen `auto` axes — never override an explicit Fixed/Percent
    // size the child already carries (e.g. a Sized leaf).
    if s.size.width == Dimension::auto() {
        s.size.width = percent(1.0_f32);
    }
    if s.size.height == Dimension::auto() {
        s.size.height = percent(1.0_f32);
    }
    tree.set_style(id, s).unwrap();
}

fn map_justify(j: MainAlign) -> JustifyContent {
    match j {
        MainAlign::Start => JustifyContent::START,
        MainAlign::Center => JustifyContent::CENTER,
        MainAlign::End => JustifyContent::END,
        MainAlign::SpaceBetween => JustifyContent::SPACE_BETWEEN,
        MainAlign::SpaceAround => JustifyContent::SPACE_AROUND,
        MainAlign::SpaceEvenly => JustifyContent::SPACE_EVENLY,
    }
}

fn map_align(a: CrossAlign) -> AlignItems {
    match a {
        CrossAlign::Start => AlignItems::START,
        CrossAlign::Center => AlignItems::CENTER,
        CrossAlign::End => AlignItems::END,
        CrossAlign::Stretch => AlignItems::STRETCH,
    }
}

// ── Measure callback ───────────────────────────────────────────────────

fn measure(
    known: TaffySize<Option<f32>>,
    avail: TaffySize<AvailableSpace>,
    ctx: Option<&mut Ctx>,
) -> TaffySize<f32> {
    let Some(ctx) = ctx else {
        return TaffySize {
            width: known.width.unwrap_or(0.0),
            height: known.height.unwrap_or(0.0),
        };
    };
    let avail_dim = |a: AvailableSpace, fallback: u32| -> u32 {
        match a {
            AvailableSpace::Definite(v) => v.max(0.0) as u32,
            AvailableSpace::MaxContent | AvailableSpace::MinContent => fallback,
        }
    };
    match ctx {
        Ctx::Fixed(w, h) => TaffySize {
            width: known.width.unwrap_or(*w),
            height: known.height.unwrap_or(*h),
        },
        Ctx::Text(spec) => {
            let mw = known
                .width
                .map(|v| v as u32)
                .unwrap_or_else(|| avail_dim(avail.width, 100_000));
            let mh = known
                .height
                .map(|v| v as u32)
                .unwrap_or_else(|| avail_dim(avail.height, 100_000));
            let nat = spec.natural(mw, mh);
            TaffySize {
                width: known.width.unwrap_or(nat.w as f32),
                height: known.height.unwrap_or(nat.h as f32),
            }
        }
        Ctx::Strip(segments, style) => {
            let mw = known
                .width
                .map(|v| v as u32)
                .unwrap_or_else(|| avail_dim(avail.width, 100_000));
            let mh = known
                .height
                .map(|v| v as u32)
                .unwrap_or_else(|| avail_dim(avail.height, 100_000));
            let s = super::label::measure_segmented_strip(
                segments,
                style,
                super::geom::Size::new(mw, mh),
            );
            TaffySize {
                width: known.width.unwrap_or(s.w as f32),
                height: known.height.unwrap_or(s.h as f32),
            }
        }
    }
}

// ── Paint walk (taffy geometry → our paint primitives) ─────────────────

fn abs_rect(tree: &TaffyTree<Ctx>, id: NodeId, origin: (f32, f32)) -> (Rect, (f32, f32)) {
    let l = tree.layout(id).unwrap();
    let x = origin.0 + l.location.x;
    let y = origin.1 + l.location.y;
    (
        Rect::new(
            x.round().max(0.0) as u32,
            y.round().max(0.0) as u32,
            l.size.width.round().max(0.0) as u32,
            l.size.height.round().max(0.0) as u32,
        ),
        (x, y),
    )
}

fn paint_node(
    tree: &TaffyTree<Ctx>,
    node: &Node,
    id: NodeId,
    origin: (f32, f32),
    fit: Fit,
    canvas: &mut Bitmap,
) {
    let (rect, abs_origin) = abs_rect(tree, id, origin);
    if rect.w == 0 || rect.h == 0 {
        return;
    }
    let kids: Vec<NodeId> = tree.children(id).unwrap_or_default();
    match node {
        Node::Empty => {}
        Node::Fill(c) => paint::fill_rect(canvas, rect, *c),
        Node::Image(img) => paint::render_image(img, fit, rect, canvas),
        Node::Text(spec) => paint::render_text(spec, rect, canvas),
        Node::SegmentedStrip { segments, style } => {
            super::label::paint_segmented_strip(segments, style, rect, canvas)
        }
        Node::Stack { children, .. } => {
            for (c, kid) in children.iter().zip(kids.iter()) {
                paint_node(tree, c, *kid, abs_origin, Fit::None, canvas);
            }
        }
        Node::Grid { cells, .. } => {
            for ((_, c), kid) in cells.iter().zip(kids.iter()) {
                paint_node(tree, c, *kid, abs_origin, Fit::None, canvas);
            }
        }
        Node::Layers(children) => {
            for (c, kid) in children.iter().zip(kids.iter()) {
                paint_node(tree, c, *kid, abs_origin, Fit::None, canvas);
            }
        }
        Node::Background { color, child } => {
            paint::fill_rect(canvas, rect, *color);
            if let Some(kid) = kids.first() {
                paint_node(tree, child, *kid, abs_origin, fit, canvas);
            }
        }
        Node::Border { color, child } => {
            if let Some(kid) = kids.first() {
                paint_node(tree, child, *kid, abs_origin, fit, canvas);
            }
            paint::draw_rect_border(canvas, rect, *color);
        }
        Node::Fit { mode, child } => {
            if let Some(kid) = kids.first() {
                paint_node(tree, child, *kid, abs_origin, *mode, canvas);
            }
        }
        Node::Padded { child, .. }
        | Node::Sized { child, .. }
        | Node::Constrain { child, .. }
        | Node::Aspect { child, .. }
        | Node::Align { child, .. } => {
            if let Some(kid) = kids.first() {
                paint_node(tree, child, *kid, abs_origin, fit, canvas);
            }
        }
    }
}

/// Core: solve `tree` with taffy against `max_w` (height hugged), clamp
/// the canvas to the active [`super::safety`] limits, and paint. MUST be
/// called inside a [`super::safety::with_limits`] + `with_base_em` scope
/// (the leaf-measure callback and the clamps read the active limits).
/// Returns the bitmap and the coverage report.
fn render_into_canvas(tree: &Node, max_w: u32, bg: Color) -> (Bitmap, Coverage) {
    let mut cov = Coverage::default();
    let mut taffy: TaffyTree<Ctx> = TaffyTree::new();
    let root = build(&mut taffy, tree, &mut cov);

    let avail = TaffySize {
        width: AvailableSpace::Definite(max_w as f32),
        height: AvailableSpace::MaxContent,
    };
    taffy
        .compute_layout_with_measure(root, avail, |known, av, _id, ctx, _style| {
            measure(known, av, ctx)
        })
        .unwrap();

    // Clamp the canvas exactly like the native root: per-axis max_dim,
    // then aspect-preserving down-scale to the pixel budget.
    let size = taffy.layout(root).unwrap().size;
    let measured = super::safety::clamp_size(super::geom::Size::new(
        size.width.round().max(0.0) as u32,
        size.height.round().max(0.0) as u32,
    ));
    let s = super::safety::clamp_to_pixel_budget(measured);
    let mut canvas = Bitmap::from_pixel(s.w.max(1), s.h.max(1), bg);
    paint_node(&taffy, tree, root, (0.0, 0.0), Fit::None, &mut canvas);
    (canvas, cov)
}

/// Backend entry point for [`super::render_with_config`]. Assumes the
/// caller already established the limits / base_em scope and applied
/// `scale` to `tree`.
pub fn render_scaled(tree: &Node, max_w: u32, bg: Color) -> Bitmap {
    render_into_canvas(tree, max_w, bg).0
}

/// Standalone render (sets up its own limits/base_em scope) returning the
/// bitmap plus the taffy coverage report — for the parity harness.
pub fn render_via_taffy(tree: &Node, cfg: &RenderConfig) -> (Bitmap, Coverage) {
    super::safety::with_limits(cfg.limits, || {
        super::safety::with_base_em(cfg.base_em, || {
            let scaled = tree.clone().scaled(cfg.scale);
            let max_w = super::safety::clamp_dim(((cfg.max_w as f32) * cfg.scale).round() as u32);
            render_into_canvas(&scaled, max_w, cfg.bg)
        })
    })
}

/// Convenience: render with a default config and given background.
pub fn render_via_taffy_bg(tree: &Node, max_w: u32, bg: Color) -> (Bitmap, Coverage) {
    render_via_taffy(tree, &RenderConfig::new(max_w).with_bg(bg))
}
