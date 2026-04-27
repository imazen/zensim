//! Retained-tree layout for composing labeled image grids.
//!
//! v0 — CSS-flavored fluent API, two-pass measure → render. Designed for
//! diff montages, heatmap cell grids, and any "labeled boxes of pixels and
//! text" composition.
//!
//! # The model
//!
//! - **[`Node`]** is the IR — an enum of leaves (image, text, fill),
//!   containers (row/column stacks, grid, layers), and modifiers (padding,
//!   sizing, alignment, fit, background, border).
//! - **Builders** ([`row`], [`column`], [`grid`], [`layers`]) wrap the IR
//!   in fluent types that accept children and structural settings;
//!   [`LayoutMod`] adds modifier methods to anything that converts to
//!   [`Node`].
//! - **Two passes:** [`Node::measure`] returns the size a node wants given
//!   maximum constraints; [`render`] lays out and paints into a fresh
//!   canvas.
//!
//! # Example
//!
//! ```ignore
//! use zensim_regress::layout::*;
//!
//! let canvas = grid()
//!     .areas(&[
//!         "title title",
//!         "exp   act ",
//!         "pdiff sdiff",
//!     ])
//!     .row_heights([Track::Auto, Track::Px(panel_h), Track::Px(panel_h)])
//!     .col_widths([Track::Fr(1), Track::Fr(1)])
//!     .gap(8)
//!     .padding(8)
//!     .place("title", "FAILED — zdsim 0.13 > 0.01")
//!     .place("exp",   image(expected).label("EXPECTED"))
//!     .place("act",   image(actual).label("ACTUAL"))
//!     .place("pdiff", image(pdiff).label("PIXEL DIFF"))
//!     .place("sdiff", image(sdiff).label_segments(vec![
//!         LabelSegment::left("ADD",    rgb(255, 180, 80)),
//!         LabelSegment::right("REMOVE", rgb(80, 220, 220)),
//!     ], &LabelStyle::default()))
//!     .background(hex("#121212"))
//!     .render(800);
//! ```
//!
//! # File layout
//!
//! | Concern | File |
//! |---|---|
//! | Color type, named constants, `rgb`/`rgba`/`hex` | [`color`] |
//! | Sizes, rects, insets, axis/halign/valign | [`geom`] |
//! | `SizeRule`/`MainAlign`/`CrossAlign`/`Fit`/`Track` | [`sizing`] |
//! | `TextStyle` / `TextSpec` (font integration) | [`text`] |
//! | Drawing primitives + image/text leaf paint | [`paint`] |
//! | `Node` IR + measure/paint dispatch | [`node`] |
//! | `Stack` / `row` / `column` builders + measure/paint | [`stack`] |
//! | `Grid` / `Track` solver + areas + measure/paint | [`grid`] |
//! | `Layers` builder + measure/paint | [`layers`] |
//! | Modifier nodes + `LayoutMod` trait | [`modifiers`] |
//! | `LabelStyle` / `LabelSegment` strip builders | [`label`] |

#![allow(clippy::too_many_arguments)]
// Many `pub` items in this module exist for external callers gated by
// the `_internal_api` feature (the layout_gallery example). When the
// feature is off, those items appear unused — silence the noise.
#![cfg_attr(not(feature = "_internal_api"), allow(dead_code, unused_imports))]

use image::{Rgba, RgbaImage};

pub mod color;
pub mod geom;
pub mod grid;
pub mod label;
pub mod layers;
pub mod modifiers;
pub mod node;
pub mod paint;
pub mod safety;
pub mod sizing;
pub mod stack;
pub mod text;

// ── Public API re-exports ─────────────────────────────────────────────

pub use color::{BLACK, Color, TRANSPARENT, WHITE, hex, rgb, rgba};
pub use geom::{Axis, HAlign, Insets, Rect, Size, VAlign};
pub use grid::{Grid, GridSpan, grid};
pub use label::{LabelSegment, LabelStyle};
pub use layers::{Layers, layers};
pub use modifiers::LayoutMod;
pub use node::{DEFAULT_LINE_PADDING, Node, empty, fill, image, line, text};
pub use paint::{draw_rect_border, fill_rect};
pub use safety::{DEFAULT_BASE_EM, LayoutLimits};
pub use sizing::{CrossAlign, Fit, MainAlign, SizeRule, Track};
pub use stack::{Stack, column, row};
pub use text::{TextSpec, TextStyle};

// ── Top-level render entry points ──────────────────────────────────────

// ── RenderConfig ──────────────────────────────────────────────────────

/// Per-render configuration — separate from [`LayoutLimits`] (which is
/// strictly safety) and from the tree itself.
///
/// `scale` is the CSS-`dppx` analogue: a uniform multiplier applied to
/// every fixed-pixel quantity (Sized::Fixed, Insets, Track::Px, gap,
/// pad, char_h) by walking the tree once via [`Node::scaled`]. `Fr`,
/// `Percent`, and `Hug` are unchanged. `1.0` is no-op.
///
/// `base_em` is the `em(1.0)` baseline in pixels — used by the
/// [`em`] free function to express padding/sizes relative to a
/// design-baseline text height.
///
/// Default: `scale = 1.0`, `base_em = 16`, `bg = BLACK`,
/// `limits = LayoutLimits::default()`.
#[derive(Clone, Debug)]
pub struct RenderConfig {
    pub max_w: u32,
    pub bg: Color,
    pub scale: f32,
    pub base_em: u32,
    pub limits: LayoutLimits,
}

impl RenderConfig {
    pub fn new(max_w: u32) -> Self {
        Self {
            max_w,
            bg: BLACK,
            scale: 1.0,
            base_em: DEFAULT_BASE_EM,
            limits: LayoutLimits::default(),
        }
    }
    pub fn with_bg(mut self, bg: Color) -> Self {
        self.bg = bg;
        self
    }
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
    pub fn with_base_em(mut self, base_em: u32) -> Self {
        self.base_em = base_em;
        self
    }
    pub fn with_limits(mut self, limits: LayoutLimits) -> Self {
        self.limits = limits;
        self
    }
}

/// `em(1.0)` returns `base_em` pixels. `em(0.5)` is half. Reads the
/// active baseline (set via [`RenderConfig::with_base_em`]; defaults
/// to [`DEFAULT_BASE_EM`]).
///
/// Resolves at call site, so use it inside the same scope where you
/// build the tree if you want it to track the active baseline.
pub fn em(units: f32) -> u32 {
    let base = safety::base_em() as f32;
    (base * units).round().max(0.0) as u32
}

// ── Render entry points ───────────────────────────────────────────────

/// Render `tree` against `max_w` using default limits, scale 1, and
/// black background.
pub fn render(tree: &Node, max_w: u32) -> RgbaImage {
    render_with_config(tree, &RenderConfig::new(max_w))
}

/// As [`render`], with a custom default canvas background.
pub fn render_with(tree: &Node, max_w: u32, bg: Color) -> RgbaImage {
    render_with_config(tree, &RenderConfig::new(max_w).with_bg(bg))
}

/// As [`render_with`], with explicit safety limits.
pub fn render_with_limits(tree: &Node, max_w: u32, bg: Color, limits: LayoutLimits) -> RgbaImage {
    render_with_config(
        tree,
        &RenderConfig::new(max_w).with_bg(bg).with_limits(limits),
    )
}

/// Render `tree` per `cfg` — the full-control entry point.
///
/// Applies (in order):
/// 1. `cfg.base_em` is set as the active em-baseline.
/// 2. `cfg.limits` is set as the active safety limits.
/// 3. The tree is walked once via [`Node::scaled`] with `cfg.scale`,
///    multiplying every fixed-pixel quantity (no-op when `scale == 1.0`).
/// 4. The constraint width is `cfg.max_w * cfg.scale`.
/// 5. The canvas is sized to the measured tree, clamped per axis to
///    `limits.max_dim` and rescaled (preserving aspect) to fit
///    `limits.max_pixels`.
pub fn render_with_config(tree: &Node, cfg: &RenderConfig) -> RgbaImage {
    safety::with_limits(cfg.limits, || {
        safety::with_base_em(cfg.base_em, || {
            // Apply scale by walking the tree once. Cheap clone for
            // scale == 1.0 (returns self without allocation).
            let scaled = tree.clone().scaled(cfg.scale);
            let scaled_max_w = ((cfg.max_w as f32) * cfg.scale).round() as u32;
            let max_w = safety::clamp_dim(scaled_max_w);
            let root_max = Size::new(max_w, u32::MAX / 2);
            let measured = safety::clamp_size(scaled.measure(root_max));
            let s = safety::clamp_to_pixel_budget(measured);
            let mut canvas = RgbaImage::from_pixel(s.w.max(1), s.h.max(1), Rgba(cfg.bg));
            scaled.paint(Rect::new(0, 0, s.w, s.h), &mut canvas);
            canvas
        })
    })
}

/// Render the tree directly into an existing canvas at `rect`. The
/// caller is responsible for calling this within a
/// [`safety::with_limits`] scope if non-default caps are wanted.
pub fn render_into(tree: &Node, rect: Rect, canvas: &mut RgbaImage) {
    tree.paint(rect, canvas);
}

#[cfg(test)]
mod tests {
    //! Cross-module integration tests — single-module tests live in
    //! their respective submodules.

    use super::*;
    use image::Rgba;

    #[test]
    fn builder_chain_produces_node() {
        let _: Node = column()
            .gap(8)
            .align_items(CrossAlign::Center)
            .child("hi")
            .child(image(image::RgbaImage::from_pixel(
                10,
                10,
                Rgba([255, 255, 255, 255]),
            )))
            .padding(4)
            .background(hex("#123456"));
    }

    #[test]
    fn implicit_str_to_text_in_builder() {
        let n: Node = row().child("a").child("b").into();
        if let Node::Stack { children, .. } = n {
            assert_eq!(children.len(), 2);
            assert!(matches!(children[0], Node::Text(_)));
        } else {
            panic!("expected Stack");
        }
    }

    #[test]
    fn text_in_vstack_hugs_then_takes_full_width_cross() {
        let n = column()
            .gap(0)
            .child(line("EXPECTED", WHITE).background(BLACK));
        let s = Node::from(n).measure(Size::new(400, 400));
        assert!(s.w > 0 && s.w <= 400);
        assert!(s.h > 0);
    }

    // ── Safety / hostile-tree tests ────────────────────────────────

    fn default_max_dim() -> u32 {
        LayoutLimits::default().max_dim
    }

    #[test]
    fn safety_fixed_size_is_capped() {
        let n = empty().size(u32::MAX, u32::MAX);
        let s = n.measure(Size::new(4096, 4096));
        assert!(s.w <= default_max_dim());
        assert!(s.h <= default_max_dim());
    }

    #[test]
    fn safety_fr_falls_back_to_auto_when_unbounded() {
        // Without the fallback this would measure to u32::MAX/2 height.
        let n = grid()
            .cols(2)
            .equal_rows(2)
            .cell(
                0,
                0,
                image(image::RgbaImage::from_pixel(20, 20, image::Rgba(WHITE))),
            )
            .cell(
                1,
                0,
                image(image::RgbaImage::from_pixel(20, 20, image::Rgba(WHITE))),
            )
            .cell(
                0,
                1,
                image(image::RgbaImage::from_pixel(20, 20, image::Rgba(WHITE))),
            )
            .cell(
                1,
                1,
                image(image::RgbaImage::from_pixel(20, 20, image::Rgba(WHITE))),
            );
        let canvas = render(&Node::from(n), 200);
        assert!(canvas.height() < default_max_dim());
    }

    #[test]
    fn safety_canvas_alloc_clamped() {
        let n = empty().size(u32::MAX, u32::MAX).background(BLACK);
        let canvas = render(&n, u32::MAX);
        // Per-axis cap.
        assert!(canvas.width() <= default_max_dim());
        assert!(canvas.height() <= default_max_dim());
        // Total-pixel cap.
        let total = canvas.width() as u64 * canvas.height() as u64;
        assert!(total <= LayoutLimits::default().max_pixels as u64);
    }

    #[test]
    fn safety_pixel_budget_scales_huge_canvas_down() {
        // 8192 × 8192 = 67 MP, but DEFAULT.max_pixels = 20 MP.
        let n = empty().size(8192, 8192).background(BLACK);
        let canvas = render(&n, 8192);
        let total = canvas.width() as u64 * canvas.height() as u64;
        assert!(total <= LayoutLimits::default().max_pixels as u64);
    }

    #[test]
    fn safety_permissive_allows_more_pixels() {
        let n = empty().size(8192, 8192).background(BLACK);
        let canvas = render_with_limits(&n, 8192, BLACK, LayoutLimits::PERMISSIVE);
        // Permissive caps at 200 MP — the full 67 MP canvas fits.
        let total = canvas.width() as u64 * canvas.height() as u64;
        assert!(total > 60_000_000);
    }

    #[test]
    fn safety_strict_clamps_more_aggressively() {
        let n = empty().size(8192, 8192).background(BLACK);
        let canvas = render_with_limits(&n, 8192, BLACK, LayoutLimits::STRICT);
        // Strict: 4 MP budget.
        let total = canvas.width() as u64 * canvas.height() as u64;
        assert!(total <= LayoutLimits::STRICT.max_pixels as u64);
    }

    #[test]
    fn safety_deep_nesting_does_not_overflow_stack() {
        let mut n: Node = empty().size(10, 10);
        for _ in 0..200 {
            n = n.padding(0);
        }
        let _ = n.measure(Size::new(100, 100));
        let _ = render(&n, 100);
    }

    #[test]
    fn safety_huge_child_count_is_truncated() {
        let mut s = row();
        for _ in 0..10_000 {
            s = s.child(empty().size(1, 1));
        }
        let measured = Node::from(s).measure(Size::new(2000, 100));
        assert!(measured.w <= default_max_dim());
    }

    // ── Percent / scale / em ───────────────────────────────────────

    #[test]
    fn percent_width_takes_half_of_constraint() {
        let n = empty().width_percent(0.5);
        let s = n.measure(Size::new(400, 200));
        assert_eq!(s.w, 200);
    }
    #[test]
    fn percent_size_takes_quarter() {
        let n = empty().size_percent(0.25, 0.5);
        let s = n.measure(Size::new(400, 200));
        assert_eq!(s, Size::new(100, 100));
    }
    #[test]
    fn track_percent_consumes_proportional() {
        let img = grid()
            .columns([Track::Percent(0.25), Track::Percent(0.75)])
            .equal_rows(1)
            .gap(0)
            .cell(0, 0, empty().background([255, 0, 0, 255]).fill())
            .cell(1, 0, empty().background([0, 0, 255, 255]).fill())
            .size(100, 10)
            .render(100);
        // 25% / 75% split → red x=0..24, blue x=25..99.
        assert_eq!(img.get_pixel(10, 5), &Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(50, 5), &Rgba([0, 0, 255, 255]));
        assert_eq!(img.get_pixel(90, 5), &Rgba([0, 0, 255, 255]));
    }

    #[test]
    fn scale_factor_doubles_fixed_quantities() {
        // A 100×40 fixed-size node, rendered at scale=2 → 200×80 canvas.
        let n = empty().background(WHITE).size(100, 40);
        let canvas = render_with_config(&n, &RenderConfig::new(100).with_scale(2.0));
        assert_eq!(canvas.width(), 200);
        assert_eq!(canvas.height(), 80);
    }
    #[test]
    fn scale_factor_leaves_percent_alone() {
        // Percent is already relative; max_w doubles, so percent of
        // max_w doubles too — but the percent value itself is unchanged.
        let n = empty().background(WHITE).width_percent(0.5);
        let canvas = render_with_config(&n, &RenderConfig::new(200).with_scale(2.0).with_bg(BLACK));
        // max_w = 200 * 2 = 400 px, width_percent(0.5) = 200 px wide.
        assert_eq!(canvas.width(), 200);
    }

    #[test]
    fn em_uses_base_em() {
        // Default base_em = 16, so em(1.0) = 16, em(0.5) = 8.
        let n = empty().background(WHITE).size(em(2.0), em(1.0));
        let canvas = render(&n, 100);
        assert_eq!(canvas.width(), 32);
        assert_eq!(canvas.height(), 16);
    }
    #[test]
    fn em_with_custom_base() {
        // em() resolves at call time, so we need to build inside the
        // RenderConfig scope. We do that by computing em() within a
        // closure passed to render_with_config; direct render() uses
        // the default base. Demonstrated via with_base_em scope.
        super::safety::with_base_em(20, || {
            assert_eq!(em(1.0), 20);
            assert_eq!(em(0.5), 10);
            assert_eq!(em(2.5), 50);
        });
        // Reset.
        assert_eq!(em(1.0), 16);
    }

    #[test]
    fn safety_huge_char_h_is_clamped() {
        let n = text(TextSpec::fixed("x", WHITE, BLACK, u32::MAX)).background(BLACK);
        let s = n.measure(Size::new(1000, 1000));
        assert!(s.h <= default_max_dim());
    }
}
