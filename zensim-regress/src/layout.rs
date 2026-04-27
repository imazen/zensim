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
//!   maximum constraints; [`Node::render`] paints into an existing canvas
//!   at a given rect. [`render`] is the convenience entry point.
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
//! # CSS correspondence
//!
//! | Layout API | CSS analogue |
//! |---|---|
//! | [`row()`] / [`column()`] | `display: flex; flex-direction: row | column` |
//! | [`Stack::gap`] | `gap` |
//! | [`Stack::justify`] | `justify-content` |
//! | [`Stack::align_items`] | `align-items` |
//! | [`grid()`] + [`Track`] | `display: grid; grid-template-columns/rows` |
//! | [`Grid::areas`] | `grid-template-areas` |
//! | [`Grid::span`] | `grid-column: span N` |
//! | [`layers()`] | `position: absolute` siblings (z-stack) |
//! | [`LayoutMod::padding`] | `padding` |
//! | [`LayoutMod::width`]/[`LayoutMod::height`] | `width`/`height` |
//! | [`LayoutMod::min_width`]/[`max_width`](LayoutMod::max_width) | `min-width`/`max-width` |
//! | [`LayoutMod::aspect_ratio`] | `aspect-ratio` |
//! | [`LayoutMod::align`] | `place-self` |
//! | [`LayoutMod::fit`] | `object-fit` |
//! | [`LayoutMod::background`] | `background-color` |
//! | [`LayoutMod::border`] | `outline` (1-px) |

use std::collections::HashMap;

use image::{Rgba, RgbaImage, imageops};

use crate::font;

// ════════════════════════════════════════════════════════════════════════
// Color
// ════════════════════════════════════════════════════════════════════════

/// RGBA color (straight alpha, the `image` crate's storage convention).
pub type Color = [u8; 4];

pub const WHITE: Color = [255, 255, 255, 255];
pub const BLACK: Color = [0, 0, 0, 255];
pub const TRANSPARENT: Color = [0, 0, 0, 0];

/// Build an opaque RGB color.
pub const fn rgb(r: u8, g: u8, b: u8) -> Color {
    [r, g, b, 255]
}

/// Build an RGBA color.
pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Color {
    [r, g, b, a]
}

/// Parse a CSS-style hex color: `#RGB`, `#RRGGBB`, or `#RRGGBBAA`. The
/// leading `#` is optional. Panics at compile time (or runtime, if not
/// in const context) on malformed input.
pub const fn hex(s: &str) -> Color {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let start = if len > 0 && bytes[0] == b'#' { 1 } else { 0 };
    let n = len - start;

    if n == 3 {
        let r = hex1(bytes[start]) * 17;
        let g = hex1(bytes[start + 1]) * 17;
        let b = hex1(bytes[start + 2]) * 17;
        [r, g, b, 255]
    } else if n == 6 {
        [
            hex2(bytes[start], bytes[start + 1]),
            hex2(bytes[start + 2], bytes[start + 3]),
            hex2(bytes[start + 4], bytes[start + 5]),
            255,
        ]
    } else if n == 8 {
        [
            hex2(bytes[start], bytes[start + 1]),
            hex2(bytes[start + 2], bytes[start + 3]),
            hex2(bytes[start + 4], bytes[start + 5]),
            hex2(bytes[start + 6], bytes[start + 7]),
        ]
    } else {
        panic!("invalid hex color literal — expected #RGB, #RRGGBB, or #RRGGBBAA")
    }
}

const fn hex2(hi: u8, lo: u8) -> u8 {
    hex1(hi) * 16 + hex1(lo)
}

const fn hex1(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => panic!("invalid hex digit"),
    }
}

// ════════════════════════════════════════════════════════════════════════
// Geometry
// ════════════════════════════════════════════════════════════════════════

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Size {
    pub w: u32,
    pub h: u32,
}

impl Size {
    pub const fn new(w: u32, h: u32) -> Self {
        Self { w, h }
    }
    pub const ZERO: Self = Self { w: 0, h: 0 };
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub const fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }
    pub const fn size(&self) -> Size {
        Size::new(self.w, self.h)
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Insets {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

impl Insets {
    pub const fn all(v: u32) -> Self {
        Self {
            left: v,
            top: v,
            right: v,
            bottom: v,
        }
    }
    /// `Insets::xy(x, y)` — symmetric horizontal/vertical.
    pub const fn xy(x: u32, y: u32) -> Self {
        Self {
            left: x,
            top: y,
            right: x,
            bottom: y,
        }
    }
    /// CSS shorthand order: top, right, bottom, left.
    pub const fn each(top: u32, right: u32, bottom: u32, left: u32) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
        }
    }
    pub const fn horizontal(&self) -> u32 {
        self.left + self.right
    }
    pub const fn vertical(&self) -> u32 {
        self.top + self.bottom
    }
}

// ════════════════════════════════════════════════════════════════════════
// Sizing & alignment enums
// ════════════════════════════════════════════════════════════════════════

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Axis {
    Horizontal,
    Vertical,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum HAlign {
    Left,
    #[default]
    Center,
    Right,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum VAlign {
    Top,
    #[default]
    Center,
    Bottom,
}

/// Main-axis distribution within a [`Stack`]. Mirrors CSS `justify-content`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum MainAlign {
    #[default]
    Start,
    Center,
    End,
    /// Equal gaps between children, none at edges.
    SpaceBetween,
    /// Half-gaps at edges, full gaps between children.
    SpaceAround,
    /// Equal gaps everywhere (between, before, after).
    SpaceEvenly,
}

/// Cross-axis alignment of children within a [`Stack`]. Mirrors CSS
/// `align-items`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CrossAlign {
    #[default]
    Start,
    Center,
    End,
    /// Stretch each child to fill the cross axis.
    Stretch,
}

/// How an [`image`] leaf scales into its allotted rect — CSS `object-fit`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Fit {
    /// Place at original size, clipped if larger than rect (CSS `none`).
    #[default]
    None,
    /// Scale preserving aspect, fit entirely (letterbox).
    Contain,
    /// Scale preserving aspect, fill rect (crop overflow).
    Cover,
    /// Stretch to rect ignoring aspect.
    Stretch,
}

/// Per-axis sizing rule for a node — analogous to CSS `width: auto/100%/<n>px`.
///
/// [`SizeRule::Grow`] is CSS `flex-grow: n` — among Grow children of a
/// stack, remaining main-axis space is split proportionally to weight.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum SizeRule {
    /// Hug content (CSS `auto`).
    #[default]
    Hug,
    /// Fixed pixel size.
    Fixed(u32),
    /// Fill the parent constraint (CSS `100%`). Equivalent to `Grow(1)`
    /// when used as the only flex rule, but inside a stack with mixed
    /// `Grow`/`Fill` it consumes a single unit of weight.
    Fill,
    /// Weighted fill — CSS `flex-grow: n`.
    Grow(u32),
}

impl SizeRule {
    fn grow_weight(self) -> u32 {
        match self {
            SizeRule::Fill => 1,
            SizeRule::Grow(w) => w,
            _ => 0,
        }
    }
}

/// Grid track sizing — CSS `grid-template-columns` / `-rows`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Track {
    /// Fixed pixel size.
    Px(u32),
    /// Fraction of remaining space (CSS `<n>fr`).
    Fr(u32),
    /// Hug the maximum content size in this track (CSS `auto` /
    /// `min-content`).
    Auto,
}

// ════════════════════════════════════════════════════════════════════════
// Text
// ════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub enum TextStyle {
    /// One or more lines, each with its own color, sized so the longest
    /// fits within the constraint width. [`font::render_lines_fitted`].
    Lines(Vec<(String, Color)>),
    /// Single line at a fixed character pixel-height. [`font::render_text_height`].
    Fixed { text: String, fg: Color, char_h: u32 },
    /// Word-wrapped text at a fixed character pixel-height.
    /// [`font::render_text_wrapped`].
    Wrapped {
        text: String,
        fg: Color,
        char_h: u32,
    },
}

#[derive(Clone, Debug)]
pub struct TextSpec {
    pub style: TextStyle,
    /// Background color the glyph rasterizer alpha-blends against — set
    /// to match the strip behind the text for clean edges.
    pub bg: Color,
}

impl TextSpec {
    pub fn lines(lines: Vec<(impl Into<String>, Color)>, bg: Color) -> Self {
        Self {
            style: TextStyle::Lines(lines.into_iter().map(|(s, c)| (s.into(), c)).collect()),
            bg,
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
        }
    }

    fn rasterize(&self, max_w: u32) -> (Vec<u8>, u32, u32) {
        if max_w == 0 {
            return (vec![], 0, 0);
        }
        match &self.style {
            TextStyle::Lines(lines) => {
                let refs: Vec<(&str, Color)> =
                    lines.iter().map(|(s, c)| (s.as_str(), *c)).collect();
                font::render_lines_fitted(&refs, self.bg, max_w)
            }
            TextStyle::Fixed { text, fg, char_h } => {
                font::render_text_height(text, *fg, self.bg, *char_h)
            }
            TextStyle::Wrapped { text, fg, char_h } => {
                font::render_text_wrapped(text, *fg, self.bg, *char_h, max_w)
            }
        }
    }

    fn natural(&self, max_w: u32) -> Size {
        if max_w == 0 {
            return Size::ZERO;
        }
        let (w, h) = match &self.style {
            TextStyle::Lines(lines) => {
                let refs: Vec<(&str, Color)> =
                    lines.iter().map(|(s, c)| (s.as_str(), *c)).collect();
                font::measure_lines_fitted(&refs, max_w)
            }
            TextStyle::Fixed { text, char_h, .. } => font::measure_text_height(text, *char_h),
            TextStyle::Wrapped { text, char_h, .. } => {
                font::measure_text_wrapped(text, *char_h, max_w)
            }
        };
        Size::new(w, h)
    }
}

// ════════════════════════════════════════════════════════════════════════
// Node IR
// ════════════════════════════════════════════════════════════════════════

/// The retained layout tree.
///
/// Construct via the free functions ([`image`], [`text`], [`line`],
/// [`fill`], [`empty`]) and builders ([`row`], [`column`], [`grid`],
/// [`layers`]); chain modifiers from the [`LayoutMod`] trait.
#[derive(Clone, Debug)]
pub enum Node {
    // ── Leaves ────────────────────────────────────────────────────────
    Empty,
    Fill(Color),
    Image(RgbaImage),
    Text(TextSpec),

    // ── Containers ────────────────────────────────────────────────────
    Stack {
        axis: Axis,
        gap: u32,
        justify: MainAlign,
        align_items: CrossAlign,
        children: Vec<Node>,
    },
    Grid {
        cols: Vec<Track>,
        rows: Vec<Track>,
        gap: (u32, u32),
        pad: u32,
        cells: Vec<(GridSpan, Node)>,
    },
    Layers(Vec<Node>),

    // ── Modifiers ─────────────────────────────────────────────────────
    Padded {
        insets: Insets,
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
        h: HAlign,
        v: VAlign,
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
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GridSpan {
    pub col: u32,
    pub row: u32,
    pub colspan: u32,
    pub rowspan: u32,
}

impl GridSpan {
    pub const fn cell(col: u32, row: u32) -> Self {
        Self {
            col,
            row,
            colspan: 1,
            rowspan: 1,
        }
    }
    pub const fn span(col: u32, row: u32, colspan: u32, rowspan: u32) -> Self {
        Self {
            col,
            row,
            colspan,
            rowspan,
        }
    }
    fn cs(&self) -> u32 {
        self.colspan.max(1)
    }
    fn rs(&self) -> u32 {
        self.rowspan.max(1)
    }
}

// ── Implicit conversions ───────────────────────────────────────────────

impl From<&str> for Node {
    /// Default: white text on black (pre-blended). For non-default
    /// background, use [`text`] / [`line`] explicitly.
    fn from(s: &str) -> Node {
        Node::Text(TextSpec::lines(vec![(s.to_string(), WHITE)], BLACK))
    }
}
impl From<String> for Node {
    fn from(s: String) -> Node {
        Node::Text(TextSpec::lines(vec![(s, WHITE)], BLACK))
    }
}
impl From<RgbaImage> for Node {
    fn from(img: RgbaImage) -> Node {
        Node::Image(img)
    }
}

// ════════════════════════════════════════════════════════════════════════
// Free-function constructors
// ════════════════════════════════════════════════════════════════════════

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
/// Single-line text on a transparent background, sized to fit the
/// constraint width.
pub fn line(s: impl Into<String>, fg: Color) -> Node {
    Node::Text(TextSpec::lines(vec![(s.into(), fg)], TRANSPARENT))
}

pub fn row() -> Stack {
    Stack::new(Axis::Horizontal)
}
pub fn column() -> Stack {
    Stack::new(Axis::Vertical)
}
pub fn grid() -> Grid {
    Grid::new()
}
pub fn layers() -> Layers {
    Layers::new()
}

// ════════════════════════════════════════════════════════════════════════
// Stack builder
// ════════════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════════════
// Grid builder
// ════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, Default)]
pub struct Grid {
    cols: Vec<Track>,
    rows: Vec<Track>,
    gap: (u32, u32),
    pad: u32,
    cells: Vec<(GridSpan, Node)>,
    areas: HashMap<String, GridSpan>,
}

impl Grid {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn columns(mut self, tracks: impl IntoIterator<Item = Track>) -> Self {
        self.cols = tracks.into_iter().collect();
        self
    }
    pub fn rows(mut self, tracks: impl IntoIterator<Item = Track>) -> Self {
        self.rows = tracks.into_iter().collect();
        self
    }
    /// Alias for [`Grid::columns`] — reads better paired with [`row_heights`].
    pub fn col_widths(self, tracks: impl IntoIterator<Item = Track>) -> Self {
        self.columns(tracks)
    }
    /// Alias for [`Grid::rows`].
    pub fn row_heights(self, tracks: impl IntoIterator<Item = Track>) -> Self {
        self.rows(tracks)
    }
    /// Equal-weight fr columns: shorthand for `vec![Track::Fr(1); n]`.
    pub fn cols(mut self, n: u32) -> Self {
        self.cols = (0..n).map(|_| Track::Fr(1)).collect();
        self
    }
    /// Equal-weight fr rows.
    pub fn equal_rows(mut self, n: u32) -> Self {
        self.rows = (0..n).map(|_| Track::Fr(1)).collect();
        self
    }

    pub fn gap(mut self, g: u32) -> Self {
        self.gap = (g, g);
        self
    }
    pub fn gap_xy(mut self, x: u32, y: u32) -> Self {
        self.gap = (x, y);
        self
    }
    pub fn padding(mut self, p: u32) -> Self {
        self.pad = p;
        self
    }

    pub fn cell(mut self, col: u32, row: u32, n: impl Into<Node>) -> Self {
        self.cells.push((GridSpan::cell(col, row), n.into()));
        self
    }
    pub fn span(
        mut self,
        col: u32,
        row: u32,
        colspan: u32,
        rowspan: u32,
        n: impl Into<Node>,
    ) -> Self {
        self.cells
            .push((GridSpan::span(col, row, colspan, rowspan), n.into()));
        self
    }

    /// Define named areas by an ASCII-art template — like CSS
    /// `grid-template-areas`. Each row is a string of whitespace-separated
    /// tokens; `.` is empty. All rows must have the same column count.
    /// Repeated tokens form a single area whose bounding box is computed.
    ///
    /// If `cols`/`rows` haven't been set explicitly, this also infers
    /// equal-weight Fr tracks from the template's dimensions.
    pub fn areas(mut self, rows: &[&str]) -> Self {
        if rows.is_empty() {
            return self;
        }
        let row_tokens: Vec<Vec<&str>> = rows
            .iter()
            .map(|r| r.split_whitespace().collect())
            .collect();
        let cols_count = row_tokens[0].len() as u32;
        for (i, r) in row_tokens.iter().enumerate() {
            assert_eq!(
                r.len() as u32,
                cols_count,
                "Grid::areas row {i} has {} cols, expected {cols_count}",
                r.len()
            );
        }
        let rows_count = rows.len() as u32;

        // name → (col_min, row_min, col_max, row_max)
        let mut bbox: HashMap<String, (u32, u32, u32, u32)> = HashMap::new();
        for (r, line) in row_tokens.iter().enumerate() {
            for (c, tok) in line.iter().enumerate() {
                if *tok == "." {
                    continue;
                }
                let key = (*tok).to_string();
                let entry = bbox
                    .entry(key)
                    .or_insert((c as u32, r as u32, c as u32, r as u32));
                entry.0 = entry.0.min(c as u32);
                entry.1 = entry.1.min(r as u32);
                entry.2 = entry.2.max(c as u32);
                entry.3 = entry.3.max(r as u32);
            }
        }
        self.areas = bbox
            .into_iter()
            .map(|(name, (c0, r0, c1, r1))| {
                (
                    name,
                    GridSpan::span(c0, r0, c1 - c0 + 1, r1 - r0 + 1),
                )
            })
            .collect();

        if self.cols.is_empty() {
            self.cols = (0..cols_count).map(|_| Track::Fr(1)).collect();
        }
        if self.rows.is_empty() {
            self.rows = (0..rows_count).map(|_| Track::Fr(1)).collect();
        }
        self
    }

    /// Place a child in a named area defined by [`Grid::areas`].
    /// Panics if `name` is unknown.
    pub fn place(mut self, name: &str, n: impl Into<Node>) -> Self {
        let span = *self
            .areas
            .get(name)
            .unwrap_or_else(|| panic!("Grid::place: no area named {name:?}"));
        self.cells.push((span, n.into()));
        self
    }
}

impl From<Grid> for Node {
    fn from(g: Grid) -> Node {
        Node::Grid {
            cols: g.cols,
            rows: g.rows,
            gap: g.gap,
            pad: g.pad,
            cells: g.cells,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Layers builder
// ════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, Default)]
pub struct Layers {
    children: Vec<Node>,
}

impl Layers {
    pub fn new() -> Self {
        Self::default()
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

impl From<Layers> for Node {
    fn from(l: Layers) -> Node {
        Node::Layers(l.children)
    }
}

// ════════════════════════════════════════════════════════════════════════
// LayoutMod — fluent modifiers on anything that converts to Node
// ════════════════════════════════════════════════════════════════════════

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
    /// Fixed `(w, h)`.
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
    /// Weighted main-axis grow (CSS `flex-grow`). Only meaningful inside
    /// a stack with other Grow/Fill children.
    fn grow(self, weight: u32) -> Node {
        wrap_sized(self.into_node(), SizeRule::Grow(weight), SizeRule::Hug)
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
        column().gap(0).child(style.strip(s)).child(self.into_node()).into()
    }
    fn label_segments(self, segments: Vec<LabelSegment>, style: &LabelStyle) -> Node {
        column()
            .gap(0)
            .child(style.segmented_strip(segments))
            .child(self.into_node())
            .into()
    }

    /// Render this tree into an [`RgbaImage`] of width `max_w`. Convenience
    /// for `render(&node, max_w)` when you have a builder in hand.
    fn render(self, max_w: u32) -> RgbaImage {
        render(&self.into_node(), max_w)
    }
}

fn wrap_padded(child: Node, insets: Insets) -> Node {
    Node::Padded {
        insets,
        child: Box::new(child),
    }
}
fn wrap_sized(child: Node, w: SizeRule, h: SizeRule) -> Node {
    Node::Sized {
        w,
        h,
        child: Box::new(child),
    }
}
fn wrap_constrain(
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

impl LayoutMod for Node {
    fn into_node(self) -> Node {
        self
    }
}
impl LayoutMod for Stack {
    fn into_node(self) -> Node {
        self.into()
    }
}
impl LayoutMod for Grid {
    fn into_node(self) -> Node {
        self.into()
    }
}
impl LayoutMod for Layers {
    fn into_node(self) -> Node {
        self.into()
    }
}

// ════════════════════════════════════════════════════════════════════════
// Measurement
// ════════════════════════════════════════════════════════════════════════

impl Node {
    /// Measure this node given the maximum available `(w, h)`. Returns
    /// the size the node wants, always clamped to `max`.
    pub fn measure(&self, max: Size) -> Size {
        let raw = self.measure_raw(max);
        Size::new(raw.w.min(max.w), raw.h.min(max.h))
    }

    fn measure_raw(&self, max: Size) -> Size {
        match self {
            Node::Empty => Size::ZERO,
            Node::Fill(_) => Size::ZERO,
            Node::Image(img) => Size::new(img.width(), img.height()),
            Node::Text(spec) => spec.natural(max.w),

            Node::Stack {
                axis,
                gap,
                children,
                ..
            } => measure_stack(*axis, *gap, children, max),

            Node::Grid {
                cols,
                rows,
                gap,
                pad,
                cells,
            } => measure_grid(cols, rows, *gap, *pad, cells, max),

            Node::Layers(children) => {
                let mut out = Size::ZERO;
                for c in children {
                    let s = c.measure(max);
                    out.w = out.w.max(s.w);
                    out.h = out.h.max(s.h);
                }
                out
            }

            Node::Padded { insets, child } => {
                let inner_max = Size::new(
                    max.w.saturating_sub(insets.horizontal()),
                    max.h.saturating_sub(insets.vertical()),
                );
                let inner = child.measure(inner_max);
                Size::new(inner.w + insets.horizontal(), inner.h + insets.vertical())
            }

            Node::Sized { w, h, child } => {
                let child_size = child.measure(max);
                let out_w = match w {
                    SizeRule::Hug => child_size.w,
                    SizeRule::Fill | SizeRule::Grow(_) => max.w,
                    SizeRule::Fixed(v) => *v,
                };
                let out_h = match h {
                    SizeRule::Hug => child_size.h,
                    SizeRule::Fill | SizeRule::Grow(_) => max.h,
                    SizeRule::Fixed(v) => *v,
                };
                Size::new(out_w, out_h)
            }

            Node::Constrain {
                min_w,
                max_w,
                min_h,
                max_h,
                child,
            } => {
                let child_max = Size::new(
                    max_w.map_or(max.w, |v| v.min(max.w)),
                    max_h.map_or(max.h, |v| v.min(max.h)),
                );
                let s = child.measure(child_max);
                Size::new(
                    s.w.max(min_w.unwrap_or(0))
                        .min(max_w.unwrap_or(u32::MAX))
                        .min(max.w),
                    s.h.max(min_h.unwrap_or(0))
                        .min(max_h.unwrap_or(u32::MAX))
                        .min(max.h),
                )
            }

            Node::Aspect { num, den, child } => measure_aspect(*num, *den, child, max),

            Node::Align { child, .. } => child.measure(max),
            Node::Fit { child, .. } => child.measure(max),
            Node::Background { child, .. } => child.measure(max),
            Node::Border { child, .. } => child.measure(max),
        }
    }
}

fn measure_stack(axis: Axis, gap: u32, children: &[Node], max: Size) -> Size {
    if children.is_empty() {
        return Size::ZERO;
    }
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

fn measure_grid(
    cols: &[Track],
    rows: &[Track],
    gap: (u32, u32),
    pad: u32,
    cells: &[(GridSpan, Node)],
    max: Size,
) -> Size {
    if cols.is_empty() || rows.is_empty() {
        return Size::ZERO;
    }
    let inner_w = max
        .w
        .saturating_sub(pad * 2)
        .saturating_sub(gap.0.saturating_mul(cols.len().saturating_sub(1) as u32));
    let inner_h = max
        .h
        .saturating_sub(pad * 2)
        .saturating_sub(gap.1.saturating_mul(rows.len().saturating_sub(1) as u32));
    let col_widths = solve_tracks(cols, inner_w, |c| {
        max_natural_in_track(cells, c, true, gap, max)
    });
    let row_heights = solve_tracks(rows, inner_h, |r| {
        max_natural_in_track(cells, r, false, gap, max)
    });
    let total_w =
        pad * 2 + col_widths.iter().sum::<u32>() + gap.0.saturating_mul(cols.len() as u32 - 1);
    let total_h =
        pad * 2 + row_heights.iter().sum::<u32>() + gap.1.saturating_mul(rows.len() as u32 - 1);
    Size::new(total_w.min(max.w), total_h.min(max.h))
}

/// For Auto-track sizing: find the max natural size of any cell whose
/// span includes track index `idx` along the given axis, divided by the
/// span length so a 2-col-spanning child contributes half its width to
/// each spanned column.
fn max_natural_in_track(
    cells: &[(GridSpan, Node)],
    idx: u32,
    is_col: bool,
    gap: (u32, u32),
    max: Size,
) -> u32 {
    let mut out = 0u32;
    for (span, child) in cells {
        let (start, len) = if is_col {
            (span.col, span.cs())
        } else {
            (span.row, span.rs())
        };
        if idx < start || idx >= start + len {
            continue;
        }
        // Measure with a generous constraint — auto tracks should hug.
        let s = child.measure(max);
        let dim = if is_col { s.w } else { s.h };
        // Subtract internal gaps when distributing across span.
        let internal_gap = if is_col { gap.0 } else { gap.1 };
        let spread = dim
            .saturating_sub(internal_gap.saturating_mul(len.saturating_sub(1)))
            .div_ceil(len.max(1));
        out = out.max(spread);
    }
    out
}

/// Solve track sizes given total available space along that axis.
///
/// 1. Px tracks consume their fixed size.
/// 2. Auto tracks consume their content size (via `auto_size` callback).
/// 3. Remaining space is distributed among Fr tracks proportionally to
///    their fr weight.
#[allow(clippy::manual_checked_ops)]
fn solve_tracks(tracks: &[Track], available: u32, auto_size: impl Fn(u32) -> u32) -> Vec<u32> {
    let mut sizes = vec![0u32; tracks.len()];
    let mut consumed = 0u32;
    let mut total_fr = 0u32;

    for (i, t) in tracks.iter().enumerate() {
        match t {
            Track::Px(v) => {
                sizes[i] = *v;
                consumed = consumed.saturating_add(*v);
            }
            Track::Auto => {
                let v = auto_size(i as u32);
                sizes[i] = v;
                consumed = consumed.saturating_add(v);
            }
            Track::Fr(w) => {
                total_fr = total_fr.saturating_add(*w);
            }
        }
    }
    if total_fr > 0 {
        let remainder = available.saturating_sub(consumed);
        // First (n-1) Fr tracks get floor share; last Fr track gets the
        // remainder so we don't lose pixels to integer division.
        let mut allotted = 0u32;
        let mut last_fr_idx: Option<usize> = None;
        for (i, t) in tracks.iter().enumerate() {
            if let Track::Fr(w) = t {
                let share = remainder * *w / total_fr;
                sizes[i] = share;
                allotted = allotted.saturating_add(share);
                last_fr_idx = Some(i);
            }
        }
        if let Some(i) = last_fr_idx {
            sizes[i] = sizes[i].saturating_add(remainder.saturating_sub(allotted));
        }
    }
    sizes
}

fn measure_aspect(num: u32, den: u32, child: &Node, max: Size) -> Size {
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

// ════════════════════════════════════════════════════════════════════════
// Rendering
// ════════════════════════════════════════════════════════════════════════

/// Render `tree` against `max_w`. The canvas height is derived from the
/// tree's measured height. The default canvas background is opaque black.
pub fn render(tree: &Node, max_w: u32) -> RgbaImage {
    render_with(tree, max_w, BLACK)
}

/// As [`render`], with a custom default canvas background.
pub fn render_with(tree: &Node, max_w: u32, bg: Color) -> RgbaImage {
    let root_max = Size::new(max_w, u32::MAX / 2);
    let s = tree.measure(root_max);
    let mut canvas = RgbaImage::from_pixel(s.w.max(1), s.h.max(1), Rgba(bg));
    tree.paint(Rect::new(0, 0, s.w, s.h), &mut canvas);
    canvas
}

/// Render the tree into an existing canvas at `rect`.
pub fn render_into(tree: &Node, rect: Rect, canvas: &mut RgbaImage) {
    tree.paint(rect, canvas);
}

impl Node {
    fn paint(&self, rect: Rect, canvas: &mut RgbaImage) {
        if rect.w == 0 || rect.h == 0 {
            return;
        }
        match self {
            Node::Empty => {}
            Node::Fill(c) => fill_rect(canvas, rect, *c),
            Node::Image(img) => render_image(img, Fit::None, rect, canvas),
            Node::Text(spec) => render_text(spec, rect, canvas),

            Node::Stack {
                axis,
                gap,
                justify,
                align_items,
                children,
            } => render_stack(*axis, *gap, *justify, *align_items, children, rect, canvas),

            Node::Grid {
                cols,
                rows,
                gap,
                pad,
                cells,
            } => render_grid(cols, rows, *gap, *pad, cells, rect, canvas),

            Node::Layers(children) => {
                for c in children {
                    c.paint(rect, canvas);
                }
            }

            Node::Padded { insets, child } => {
                let inner = Rect::new(
                    rect.x.saturating_add(insets.left),
                    rect.y.saturating_add(insets.top),
                    rect.w.saturating_sub(insets.horizontal()),
                    rect.h.saturating_sub(insets.vertical()),
                );
                child.paint(inner, canvas);
            }
            Node::Sized { child, .. } => child.paint(rect, canvas),
            Node::Constrain { child, .. } => child.paint(rect, canvas),
            Node::Aspect { child, .. } => child.paint(rect, canvas),
            Node::Align { h, v, child } => {
                let child_size = child.measure(rect.size());
                let x_off = match h {
                    HAlign::Left => 0,
                    HAlign::Center => rect.w.saturating_sub(child_size.w) / 2,
                    HAlign::Right => rect.w.saturating_sub(child_size.w),
                };
                let y_off = match v {
                    VAlign::Top => 0,
                    VAlign::Center => rect.h.saturating_sub(child_size.h) / 2,
                    VAlign::Bottom => rect.h.saturating_sub(child_size.h),
                };
                let inner = Rect::new(
                    rect.x.saturating_add(x_off),
                    rect.y.saturating_add(y_off),
                    child_size.w.min(rect.w),
                    child_size.h.min(rect.h),
                );
                child.paint(inner, canvas);
            }
            Node::Fit { mode, child } => {
                if let Node::Image(img) = child.as_ref() {
                    render_image(img, *mode, rect, canvas);
                } else {
                    child.paint(rect, canvas);
                }
            }
            Node::Background { color, child } => {
                fill_rect(canvas, rect, *color);
                child.paint(rect, canvas);
            }
            Node::Border { color, child } => {
                child.paint(rect, canvas);
                draw_rect_border(canvas, rect, *color);
            }
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::manual_checked_ops)]
fn render_stack(
    axis: Axis,
    gap: u32,
    justify: MainAlign,
    align_items: CrossAlign,
    children: &[Node],
    rect: Rect,
    canvas: &mut RgbaImage,
) {
    if children.is_empty() {
        return;
    }
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

    // Gather grow weights (Fill + Grow) and natural main sizes.
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

    // Compute per-child main size.
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
        main_sizes[i] = main_sizes[i].saturating_add(remainder.saturating_sub(allotted));
    }

    let total_children_main: u32 = main_sizes.iter().sum();
    let leftover = main_avail.saturating_sub(total_children_main);

    // Compute starting offset and inter-child spacing per justify.
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
        child.paint(child_rect, canvas);

        cursor = cursor.saturating_add(m_size);
        if i + 1 < children.len() {
            cursor = cursor.saturating_add(gap).saturating_add(extra_gap);
        }
    }
}

fn main_grow_weight(node: &Node, axis: Axis) -> u32 {
    match node {
        Node::Sized { w, h, .. } => {
            let r = match axis {
                Axis::Horizontal => *w,
                Axis::Vertical => *h,
            };
            r.grow_weight()
        }
        _ => 0,
    }
}

fn render_grid(
    cols: &[Track],
    rows: &[Track],
    gap: (u32, u32),
    pad: u32,
    cells: &[(GridSpan, Node)],
    rect: Rect,
    canvas: &mut RgbaImage,
) {
    if cols.is_empty() || rows.is_empty() {
        return;
    }
    let inner_w = rect
        .w
        .saturating_sub(pad * 2)
        .saturating_sub(gap.0.saturating_mul(cols.len().saturating_sub(1) as u32));
    let inner_h = rect
        .h
        .saturating_sub(pad * 2)
        .saturating_sub(gap.1.saturating_mul(rows.len().saturating_sub(1) as u32));
    let col_widths = solve_tracks(cols, inner_w, |c| {
        max_natural_in_track(cells, c, true, gap, rect.size())
    });
    let row_heights = solve_tracks(rows, inner_h, |r| {
        max_natural_in_track(cells, r, false, gap, rect.size())
    });

    // Cumulative offsets.
    let mut col_off = vec![0u32; cols.len() + 1];
    for i in 0..cols.len() {
        col_off[i + 1] = col_off[i] + col_widths[i] + if i + 1 < cols.len() { gap.0 } else { 0 };
    }
    let mut row_off = vec![0u32; rows.len() + 1];
    for i in 0..rows.len() {
        row_off[i + 1] = row_off[i] + row_heights[i] + if i + 1 < rows.len() { gap.1 } else { 0 };
    }

    for (span, child) in cells {
        if span.col >= cols.len() as u32 || span.row >= rows.len() as u32 {
            continue;
        }
        let cs = span.cs().min(cols.len() as u32 - span.col);
        let rs = span.rs().min(rows.len() as u32 - span.row);
        let x0 = rect.x + pad + col_off[span.col as usize];
        let y0 = rect.y + pad + row_off[span.row as usize];
        let x1 = rect.x + pad + col_off[(span.col + cs) as usize].saturating_sub(gap.0);
        let y1 = rect.y + pad + row_off[(span.row + rs) as usize].saturating_sub(gap.1);
        let w = x1.saturating_sub(x0);
        let h = y1.saturating_sub(y0);
        child.paint(Rect::new(x0, y0, w, h), canvas);
    }
}

// ── Leaf renderers ─────────────────────────────────────────────────────

fn render_image(img: &RgbaImage, mode: Fit, rect: Rect, canvas: &mut RgbaImage) {
    if rect.w == 0 || rect.h == 0 {
        return;
    }
    let (iw, ih) = (img.width(), img.height());
    if iw == 0 || ih == 0 {
        return;
    }
    let scaled: Option<RgbaImage> = match mode {
        Fit::None => None,
        Fit::Stretch if (iw, ih) != (rect.w, rect.h) => Some(imageops::resize(
            img,
            rect.w,
            rect.h,
            imageops::FilterType::Triangle,
        )),
        Fit::Stretch => None,
        Fit::Contain | Fit::Cover => {
            let sx = rect.w as f32 / iw as f32;
            let sy = rect.h as f32 / ih as f32;
            let s = if matches!(mode, Fit::Contain) {
                sx.min(sy)
            } else {
                sx.max(sy)
            };
            let nw = (iw as f32 * s).round().max(1.0) as u32;
            let nh = (ih as f32 * s).round().max(1.0) as u32;
            Some(imageops::resize(
                img,
                nw,
                nh,
                imageops::FilterType::Triangle,
            ))
        }
    };

    let (src, sw, sh) = match &scaled {
        Some(s) => (s, s.width(), s.height()),
        None => (img, iw, ih),
    };

    let x_off = (rect.w.saturating_sub(sw)) / 2;
    let y_off = (rect.h.saturating_sub(sh)) / 2;

    if sw <= rect.w && sh <= rect.h {
        imageops::overlay(
            canvas,
            src,
            rect.x.saturating_add(x_off) as i64,
            rect.y.saturating_add(y_off) as i64,
        );
    } else {
        let crop_x = (sw.saturating_sub(rect.w)) / 2;
        let crop_y = (sh.saturating_sub(rect.h)) / 2;
        let cw = rect.w.min(sw);
        let ch = rect.h.min(sh);
        let cropped = imageops::crop_imm(src, crop_x, crop_y, cw, ch).to_image();
        imageops::overlay(canvas, &cropped, rect.x as i64, rect.y as i64);
    }
}

fn render_text(spec: &TextSpec, rect: Rect, canvas: &mut RgbaImage) {
    let (buf, w, h) = spec.rasterize(rect.w);
    if w == 0 || h == 0 {
        return;
    }
    let Some(img) = RgbaImage::from_raw(w, h, buf) else {
        return;
    };
    imageops::overlay(canvas, &img, rect.x as i64, rect.y as i64);
}

// ════════════════════════════════════════════════════════════════════════
// Drawing primitives
// ════════════════════════════════════════════════════════════════════════

pub fn fill_rect(img: &mut RgbaImage, rect: Rect, color: Color) {
    let px = Rgba(color);
    let img_w = img.width();
    let img_h = img.height();
    let x_end = rect.x.saturating_add(rect.w).min(img_w);
    let y_end = rect.y.saturating_add(rect.h).min(img_h);
    for y in rect.y..y_end {
        for x in rect.x..x_end {
            img.put_pixel(x, y, px);
        }
    }
}

pub fn draw_rect_border(img: &mut RgbaImage, rect: Rect, color: Color) {
    if rect.w == 0 || rect.h == 0 {
        return;
    }
    let px = Rgba(color);
    let img_w = img.width();
    let img_h = img.height();
    let x_end = rect.x.saturating_add(rect.w).min(img_w);
    let y_end = rect.y.saturating_add(rect.h).min(img_h);
    for x in rect.x..x_end {
        if rect.y < img_h {
            img.put_pixel(x, rect.y, px);
        }
        let bot = y_end.saturating_sub(1);
        if bot < img_h && bot > rect.y {
            img.put_pixel(x, bot, px);
        }
    }
    for y in rect.y..y_end {
        if rect.x < img_w {
            img.put_pixel(rect.x, y, px);
        }
        let right = x_end.saturating_sub(1);
        if right < img_w && right > rect.x {
            img.put_pixel(right, y, px);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Label styling
// ════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct LabelStyle {
    pub fg: Color,
    pub bg: Color,
    pub align: HAlign,
    pub padding: Insets,
    /// `None` → auto-fit width via [`font::measure_lines_fitted`].
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

    fn strip(&self, s: impl Into<String>) -> Node {
        let inner = match self.char_h {
            Some(h) => text(TextSpec::fixed(s, self.fg, self.bg, h)),
            None => text(TextSpec::lines(vec![(s.into(), self.fg)], self.bg)),
        };
        let mut inner: Node = inner.align_h(self.align);
        if self.padding != Insets::default() {
            inner = wrap_padded(inner, self.padding);
        }
        inner.background(self.bg).fill_width()
    }

    fn segmented_strip(&self, segments: Vec<LabelSegment>) -> Node {
        let mut l = layers().child(fill(self.bg));
        for seg in segments {
            let txt = match self.char_h.or(seg.char_h) {
                Some(h) => text(TextSpec::fixed(seg.text, seg.fg, self.bg, h)),
                None => text(TextSpec::lines(vec![(seg.text, seg.fg)], self.bg)),
            };
            l = l.child(txt.align(seg.align, VAlign::Center).padding_xy(
                self.padding.left.max(self.padding.right),
                self.padding.top.max(self.padding.bottom),
            ));
        }
        l.fill_width()
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

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: Color) -> RgbaImage {
        RgbaImage::from_pixel(w, h, Rgba(c))
    }

    // ── Color ──────────────────────────────────────────────────────────

    #[test]
    fn hex_3_digit() {
        assert_eq!(hex("#abc"), [0xaa, 0xbb, 0xcc, 255]);
    }
    #[test]
    fn hex_6_digit() {
        assert_eq!(hex("#1a2b3c"), [0x1a, 0x2b, 0x3c, 255]);
    }
    #[test]
    fn hex_8_digit() {
        assert_eq!(hex("#11223344"), [0x11, 0x22, 0x33, 0x44]);
    }
    #[test]
    fn hex_no_hash() {
        assert_eq!(hex("ff0080"), [0xff, 0x00, 0x80, 255]);
    }
    #[test]
    fn rgb_helper_is_opaque() {
        assert_eq!(rgb(10, 20, 30), [10, 20, 30, 255]);
    }

    // ── Geometry / measurement ─────────────────────────────────────────

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
    fn padded_grows_by_insets() {
        let n = image(solid(50, 30, WHITE)).padding(10);
        assert_eq!(n.measure(Size::new(200, 200)), Size::new(70, 50));
    }
    #[test]
    fn size_overrides() {
        let n = image(solid(50, 30, WHITE)).size(100, 80);
        assert_eq!(n.measure(Size::new(200, 200)), Size::new(100, 80));
    }
    #[test]
    fn fill_width_consumes_max() {
        let n = image(solid(50, 30, WHITE)).fill_width();
        assert_eq!(n.measure(Size::new(200, 200)), Size::new(200, 30));
    }

    // ── Stack ──────────────────────────────────────────────────────────

    #[test]
    fn column_sums_main_max_cross() {
        let n = column()
            .gap(2)
            .child(image(solid(10, 10, WHITE)))
            .child(image(solid(20, 5, WHITE)));
        assert_eq!(n.into_node().measure(Size::new(100, 100)), Size::new(20, 17));
    }
    #[test]
    fn row_sums_main_max_cross() {
        let n = row()
            .gap(4)
            .child(image(solid(10, 10, WHITE)))
            .child(image(solid(20, 5, WHITE)));
        assert_eq!(n.into_node().measure(Size::new(100, 100)), Size::new(34, 10));
    }

    #[test]
    fn align_items_center_centers_in_cross() {
        let small = image(solid(10, 10, [255, 0, 0, 255]));
        let big = image(solid(40, 40, [0, 0, 0, 255]));
        let n = column()
            .align_items(CrossAlign::Center)
            .child(small)
            .child(big)
            .render(40);
        // Small centered within 40 cross-width: red pixel at x=15..25, y=0..9.
        assert_eq!(n.get_pixel(20, 5), &Rgba([255, 0, 0, 255]));
        assert_eq!(n.get_pixel(0, 5), &Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn align_items_stretch_fills_cross() {
        // Empty has 0 natural height — give the strip explicit height so
        // it has a main-axis size to paint into.
        let strip = empty().background([255, 0, 0, 255]).height(SizeRule::Fixed(5));
        let big = image(solid(40, 40, BLACK));
        let img = column()
            .align_items(CrossAlign::Stretch)
            .child(strip)
            .child(big)
            .size(40, 45)
            .render(40);
        // Strip stretched to width 40; first row red.
        assert_eq!(img.get_pixel(0, 0), &Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(39, 0), &Rgba([255, 0, 0, 255]));
    }

    #[test]
    fn justify_space_between() {
        let a = image(solid(10, 10, [255, 0, 0, 255]));
        let b = image(solid(10, 10, [0, 0, 255, 255]));
        let img = row()
            .justify(MainAlign::SpaceBetween)
            .child(a)
            .child(b)
            .size(50, 10)
            .render(50);
        // a at x=0..10 (red), b at x=40..50 (blue).
        assert_eq!(img.get_pixel(5, 5), &Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(45, 5), &Rgba([0, 0, 255, 255]));
    }

    #[test]
    fn grow_weights_distribute_remainder() {
        // Two children: a hug (10px), and a Grow(2). Total stack 60px main.
        // Hug consumes 10; remainder 50; grow gets all 50.
        let hug = image(solid(10, 10, [255, 0, 0, 255]));
        let grow = empty().background([0, 0, 255, 255]).grow(2).fill_height();
        let img = row()
            .align_items(CrossAlign::Stretch)
            .child(hug)
            .child(grow)
            .size(60, 10)
            .render(60);
        assert_eq!(img.get_pixel(5, 5), &Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(50, 5), &Rgba([0, 0, 255, 255]));
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
        // a gets ~40, b gets ~20.
        assert_eq!(img.get_pixel(20, 5), &Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(50, 5), &Rgba([0, 0, 255, 255]));
    }

    // ── Grid ───────────────────────────────────────────────────────────

    #[test]
    fn grid_uniform_fr_fills_constraint() {
        let img = grid()
            .cols(2)
            .equal_rows(2)
            .gap(0)
            .cell(0, 0, empty().background([255, 0, 0, 255]).fill())
            .cell(1, 0, empty().background([0, 255, 0, 255]).fill())
            .cell(0, 1, empty().background([0, 0, 255, 255]).fill())
            .cell(1, 1, empty().background([255, 255, 0, 255]).fill())
            .size(80, 80)
            .render(80);
        assert_eq!(img.get_pixel(20, 20), &Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(60, 20), &Rgba([0, 255, 0, 255]));
        assert_eq!(img.get_pixel(20, 60), &Rgba([0, 0, 255, 255]));
        assert_eq!(img.get_pixel(60, 60), &Rgba([255, 255, 0, 255]));
    }

    #[test]
    fn grid_track_px_takes_fixed_size() {
        let img = grid()
            .columns([Track::Px(20), Track::Fr(1)])
            .equal_rows(1)
            .gap(0)
            .cell(0, 0, empty().background([255, 0, 0, 255]).fill())
            .cell(1, 0, empty().background([0, 0, 255, 255]).fill())
            .size(100, 10)
            .render(100);
        // Px(20) = red on x<20, Fr fills x>=20 with blue.
        assert_eq!(img.get_pixel(10, 5), &Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(50, 5), &Rgba([0, 0, 255, 255]));
    }

    #[test]
    fn grid_track_auto_hugs_content() {
        let img = grid()
            .columns([Track::Auto, Track::Fr(1)])
            .equal_rows(1)
            .gap(0)
            .cell(0, 0, image(solid(15, 10, [255, 0, 0, 255])))
            .cell(1, 0, empty().background([0, 0, 255, 255]).fill())
            .size(100, 10)
            .render(100);
        // Auto col = 15 wide (red), Fr fills rest (blue).
        assert_eq!(img.get_pixel(7, 5), &Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(50, 5), &Rgba([0, 0, 255, 255]));
    }

    #[test]
    fn grid_areas_parses_template() {
        let img = grid()
            .areas(&["title title", "exp   act"])
            .row_heights([Track::Px(10), Track::Px(20)])
            .gap(0)
            .place("title", empty().background([255, 0, 0, 255]).fill())
            .place("exp", empty().background([0, 255, 0, 255]).fill())
            .place("act", empty().background([0, 0, 255, 255]).fill())
            .size(40, 30)
            .render(40);
        // Title row spans 0..40, height 10 → red at top.
        assert_eq!(img.get_pixel(20, 5), &Rgba([255, 0, 0, 255]));
        // exp/act split at x=20, y=10..30.
        assert_eq!(img.get_pixel(10, 20), &Rgba([0, 255, 0, 255]));
        assert_eq!(img.get_pixel(30, 20), &Rgba([0, 0, 255, 255]));
    }

    #[test]
    fn grid_span_via_areas() {
        let img = grid()
            .areas(&["banner banner", "a a"])
            .equal_rows(2)
            .gap(0)
            .place("banner", empty().background([0, 255, 0, 255]).fill())
            .place("a", empty().background([255, 0, 0, 255]).fill())
            .size(40, 40)
            .render(40);
        // Banner spans both columns of row 0.
        assert_eq!(img.get_pixel(10, 10), &Rgba([0, 255, 0, 255]));
        assert_eq!(img.get_pixel(30, 10), &Rgba([0, 255, 0, 255]));
        assert_eq!(img.get_pixel(20, 30), &Rgba([255, 0, 0, 255]));
    }

    // ── Layers / modifiers ─────────────────────────────────────────────

    #[test]
    fn layers_painters_order() {
        let bg = empty().background([255, 0, 0, 255]).fill();
        let dot = image(solid(4, 4, [0, 255, 0, 255])).center();
        let img = layers().child(bg).child(dot).size(20, 20).render(20);
        assert_eq!(img.get_pixel(0, 0), &Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(10, 10), &Rgba([0, 255, 0, 255]));
    }

    #[test]
    fn align_centers_in_oversized_rect() {
        let img = image(solid(20, 10, [255, 255, 0, 255]))
            .center()
            .size(100, 60)
            .render(100);
        assert_eq!(img.get_pixel(50, 30), &Rgba([255, 255, 0, 255]));
        assert_eq!(img.get_pixel(0, 0), &Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn background_paints_first() {
        let img = empty().background([10, 20, 30, 255]).size(20, 20).render(20);
        assert_eq!(img.get_pixel(5, 5), &Rgba([10, 20, 30, 255]));
    }

    #[test]
    fn border_paints_outline() {
        let img = empty()
            .background(BLACK)
            .border(WHITE)
            .size(10, 10)
            .render(10);
        assert_eq!(img.get_pixel(0, 0), &Rgba([255, 255, 255, 255]));
        assert_eq!(img.get_pixel(9, 0), &Rgba([255, 255, 255, 255]));
        assert_eq!(img.get_pixel(5, 5), &Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn fit_contain_letterboxes() {
        let img = image(solid(20, 10, [255, 0, 0, 255]))
            .fit_contain()
            .size(40, 40)
            .render(40);
        assert_eq!(img.get_pixel(20, 0), &Rgba([0, 0, 0, 255]));
        assert_eq!(img.get_pixel(20, 20), &Rgba([255, 0, 0, 255]));
    }

    // ── min/max/aspect ─────────────────────────────────────────────────

    #[test]
    fn min_width_grows_to_minimum() {
        let n = image(solid(10, 10, WHITE)).min_width(50);
        assert_eq!(n.measure(Size::new(200, 200)), Size::new(50, 10));
    }
    #[test]
    fn max_width_caps_natural() {
        let n = image(solid(80, 10, WHITE)).max_width(40);
        assert_eq!(n.measure(Size::new(200, 200)), Size::new(40, 10));
    }
    #[test]
    fn aspect_ratio_16_9_from_width() {
        let n = image(solid(160, 50, WHITE)).aspect_ratio(16, 9);
        // cw=160, h_from_w = 160*9/16 = 90 → (160, 90)
        assert_eq!(n.measure(Size::new(200, 200)), Size::new(160, 90));
    }

    // ── Text ───────────────────────────────────────────────────────────

    #[test]
    fn text_lines_measures_match_render() {
        let spec = TextSpec::lines(
            vec![("HELLO".to_string(), WHITE), ("WORLD!".to_string(), WHITE)],
            BLACK,
        );
        let measured = spec.natural(200);
        let (_, w, h) = spec.rasterize(200);
        assert_eq!(measured, Size::new(w, h));
    }
    #[test]
    fn from_str_makes_text_node() {
        let n: Node = "hello".into();
        assert!(matches!(n, Node::Text(_)));
    }

    // ── Label ──────────────────────────────────────────────────────────

    #[test]
    fn label_default_strip_above_image() {
        let img = image(solid(40, 40, [255, 0, 0, 255])).label("EXPECTED");
        let s = img.measure(Size::new(400, 400));
        assert!(s.h > 40);
    }
    #[test]
    fn label_styled_uses_provided_bg() {
        let style = LabelStyle::default()
            .with_bg(rgb(100, 0, 0))
            .with_fg(rgb(255, 255, 0));
        let canvas = image(solid(40, 40, [255, 0, 0, 255]))
            .label_styled("HELLO", &style)
            .width(SizeRule::Fixed(40))
            .render(40);
        assert_eq!(canvas.get_pixel(0, 0), &Rgba([100, 0, 0, 255]));
    }

    // ── Builder fluency smoke tests ────────────────────────────────────

    #[test]
    fn builder_chain_produces_node() {
        let _: Node = column()
            .gap(8)
            .align_items(CrossAlign::Center)
            .child("hi")
            .child(image(solid(10, 10, WHITE)))
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
}
