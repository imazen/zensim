//! Self-documenting gallery for the layout module.
//!
//! Generates an HTML index page where each "scene" shows its source code
//! beside its rendered PNG. The macro [`scene!`] captures the body's
//! source via `stringify!` and runs the same code, so the displayed
//! source is exactly what produced the image.
//!
//! ```text
//! cargo run -p zensim-regress --example layout_gallery -- /tmp/gallery
//! # then open /tmp/gallery/index.html
//! ```

use std::fs;
use std::path::PathBuf;

use zensim_regress::Bitmap;
use zensim_regress::layout::compose::*;
use zensim_regress::layout::*;
use zensim_regress::{column, row};

// ── Helper "templates" used by the template scenes ────────────────────

/// A small status card — title + role + accent color, padded over a
/// dark background. Hugs content; no `min_width` so a row of these
/// fits whatever canvas width the caller chose.
fn status_card(title: &str, role: &str, accent: Color) -> Node {
    column()
        .gap(4)
        .align_items(CrossAlign::Start)
        .child(line(title, WHITE))
        .child(line(role, accent))
        .padding(12)
        .background(hex("#202531"))
        .border(hex("#3a3f4d"))
}

/// Header bar — title left, status right, padded over a colored strip.
fn header_bar(title: &str, status: &str, status_color: Color) -> Node {
    layers()
        .child(fill(hex("#1f2733")))
        .child(
            line(title, WHITE)
                .align(HAlign::Left, VAlign::Center)
                .padding(12),
        )
        .child(
            line(status, status_color)
                .align(HAlign::Right, VAlign::Center)
                .padding(12),
        )
        .fill_width()
        .height(SizeRule::Fixed(40))
}

/// Solid colored swatch with a centered label — useful for showing
/// container layout.
fn swatch(label: &str, color: Color) -> Node {
    layers()
        .child(fill(color))
        .child(line(label, WHITE).center())
        .into()
}

/// Colorful gradient square — used as a stand-in for an image.
fn gradient(w: u32, h: u32) -> Bitmap {
    Bitmap::from_fn(w, h, |x, y| {
        [
            ((x * 255) / w.max(1)) as u8,
            ((y * 255) / h.max(1)) as u8,
            (((x + y) * 200) / (w + h).max(1) + 30) as u8,
            255,
        ]
    })
}

// ── Scene plumbing ────────────────────────────────────────────────────

struct Scene {
    title: String,
    description: String,
    source: String,
    width: u32,
    tree: Node,
    /// Optional scale override — when `Some(s)`, the scene renders via
    /// `render_with_config` at scale `s` instead of the plain `render`.
    scale: Option<f32>,
}

/// Capture the body's source via `stringify!` and build the scene.
/// `body` must yield something convertible to [`Node`] — a builder, a
/// modifier chain, a string literal, etc.
macro_rules! scene {
    ($title:literal, $desc:literal, $width:expr, $body:expr $(,)?) => {{
        Scene {
            title: $title.to_string(),
            description: $desc.to_string(),
            source: stringify!($body).to_string(),
            width: $width,
            tree: Node::from($body),
            scale: None,
        }
    }};
}

/// Variant that renders at an explicit scale via `render_with_config`.
macro_rules! scene_at_scale {
    ($title:literal, $desc:literal, $width:expr, $scale:expr, $body:expr $(,)?) => {{
        Scene {
            title: $title.to_string(),
            description: $desc.to_string(),
            source: stringify!($body).to_string(),
            width: $width,
            tree: Node::from($body),
            scale: Some($scale),
        }
    }};
}

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/zenlayout_gallery".to_string())
        .into();
    fs::create_dir_all(&dir).expect("create output dir");

    let scenes: Vec<Scene> = vec![
        // ── Leaves ────────────────────────────────────────────────────
        scene!(
            "Solid fill",
            "`fill(c)` paints `c` over its rect. Wrap in `.size(w, h)` so it has dimensions to paint.",
            240,
            fill(hex("#ff6f3c")).size(240, 80)
        ),
        scene!(
            "Plain text",
            "`line(s, fg)` is a single-line label, sized to fit the constraint width.",
            400,
            line("Hello, world!", WHITE).background(hex("#1a1a1a"))
        ),
        scene!(
            "Padded text",
            "`.padding(n)` wraps a node in [`Node::Padded`]; `.background(c)` paints `c` behind.",
            300,
            line("padded", WHITE).padding(20).background(hex("#243949"))
        ),
        scene!(
            "Centered text in oversized box",
            "`.center()` aligns the child within the rect its parent gives it. Wrap in `.size(w, h)` to give it room.",
            300,
            line("centered", WHITE)
                .center()
                .size(300, 100)
                .background(hex("#243949"))
        ),
        // ── Row / column basics ───────────────────────────────────────
        scene!(
            "Row of swatches",
            "`row()` is HStack. `gap(n)` separates children. Each `swatch` here is a helper template (see top of file).",
            600,
            row()
                .gap(8)
                .child(swatch("A", hex("#ef4444")).size(120, 60))
                .child(swatch("B", hex("#10b981")).size(120, 60))
                .child(swatch("C", hex("#3b82f6")).size(120, 60))
                .child(swatch("D", hex("#f59e0b")).size(120, 60))
                .padding(8)
                .background(hex("#0e0e10"))
        ),
        scene!(
            "Column with align_items: Center",
            "Cross-axis alignment applied once on the parent rather than per-child.",
            300,
            column()
                .gap(8)
                .align_items(CrossAlign::Center)
                .child(line("title", WHITE))
                .child(line("subtitle", hex("#888")))
                .child(swatch("body", hex("#3b82f6")).size(200, 60))
                .padding(12)
                .background(hex("#0e0e10"))
        ),
        scene!(
            "justify-content variants",
            "Same children, different `justify(...)`. Top to bottom: Start, Center, End, SpaceBetween, SpaceAround, SpaceEvenly.",
            500,
            column()
                .gap(8)
                .child(
                    row()
                        .justify(MainAlign::Start)
                        .child(swatch("a", hex("#ef4444")).size(60, 30))
                        .child(swatch("b", hex("#10b981")).size(60, 30))
                        .child(swatch("c", hex("#3b82f6")).size(60, 30))
                        .fill_width()
                        .background(hex("#1c1c1c"))
                )
                .child(
                    row()
                        .justify(MainAlign::Center)
                        .child(swatch("a", hex("#ef4444")).size(60, 30))
                        .child(swatch("b", hex("#10b981")).size(60, 30))
                        .child(swatch("c", hex("#3b82f6")).size(60, 30))
                        .fill_width()
                        .background(hex("#1c1c1c"))
                )
                .child(
                    row()
                        .justify(MainAlign::End)
                        .child(swatch("a", hex("#ef4444")).size(60, 30))
                        .child(swatch("b", hex("#10b981")).size(60, 30))
                        .child(swatch("c", hex("#3b82f6")).size(60, 30))
                        .fill_width()
                        .background(hex("#1c1c1c"))
                )
                .child(
                    row()
                        .justify(MainAlign::SpaceBetween)
                        .child(swatch("a", hex("#ef4444")).size(60, 30))
                        .child(swatch("b", hex("#10b981")).size(60, 30))
                        .child(swatch("c", hex("#3b82f6")).size(60, 30))
                        .fill_width()
                        .background(hex("#1c1c1c"))
                )
                .child(
                    row()
                        .justify(MainAlign::SpaceAround)
                        .child(swatch("a", hex("#ef4444")).size(60, 30))
                        .child(swatch("b", hex("#10b981")).size(60, 30))
                        .child(swatch("c", hex("#3b82f6")).size(60, 30))
                        .fill_width()
                        .background(hex("#1c1c1c"))
                )
                .child(
                    row()
                        .justify(MainAlign::SpaceEvenly)
                        .child(swatch("a", hex("#ef4444")).size(60, 30))
                        .child(swatch("b", hex("#10b981")).size(60, 30))
                        .child(swatch("c", hex("#3b82f6")).size(60, 30))
                        .fill_width()
                        .background(hex("#1c1c1c"))
                )
                .padding(8)
                .background(hex("#0e0e10"))
        ),
        scene!(
            "flex-grow weights",
            "`SizeRule::Grow(n)` (or `.grow(n)`) shares the remainder among Grow children proportionally. Hug children take their natural size first.",
            500,
            row()
                .child(swatch("hug", hex("#ef4444")).size(80, 40))
                .child(swatch("grow 1", hex("#10b981")).grow(1).fill_height())
                .child(swatch("grow 2", hex("#3b82f6")).grow(2).fill_height())
                .padding(8)
                .background(hex("#0e0e10"))
                .size(500, 56)
        ),
        // ── Grid ──────────────────────────────────────────────────────
        scene!(
            "Uniform Fr grid",
            "`.cols(n)` is shorthand for n equal-Fr columns. Cells fill the constraint evenly.",
            400,
            grid()
                .cols(3)
                .equal_rows(2)
                .gap(4)
                .cell(0, 0, swatch("0,0", hex("#ef4444")))
                .cell(1, 0, swatch("1,0", hex("#10b981")))
                .cell(2, 0, swatch("2,0", hex("#3b82f6")))
                .cell(0, 1, swatch("0,1", hex("#f59e0b")))
                .cell(1, 1, swatch("1,1", hex("#a855f7")))
                .cell(2, 1, swatch("2,1", hex("#ec4899")))
                .padding(8)
                .background(hex("#0e0e10"))
                .size(400, 200)
        ),
        scene!(
            "Mixed Track sizes (Px / Fr / Auto)",
            "Like CSS `grid-template-columns: 80px 1fr 2fr`. Px is fixed; Fr splits the remainder weighted; Auto hugs content.",
            500,
            grid()
                .columns([Track::Px(80), Track::Fr(1), Track::Fr(2)])
                .equal_rows(1)
                .gap(4)
                .cell(0, 0, swatch("80px", hex("#ef4444")))
                .cell(1, 0, swatch("1fr", hex("#10b981")))
                .cell(2, 0, swatch("2fr", hex("#3b82f6")))
                .padding(8)
                .background(hex("#0e0e10"))
                .size(500, 60)
        ),
        scene!(
            "ASCII-art areas (`grid-template-areas`)",
            "Define a layout as text — repeated tokens form a single area whose bounding box is computed. `.place(name, child)` looks up by name.",
            500,
            grid()
                .areas(&[
                    "header header header",
                    "side   main   main",
                    "side   main   main",
                    "footer footer footer",
                ])
                .row_heights([Track::Px(40), Track::Fr(1), Track::Fr(1), Track::Px(30)])
                .col_widths([Track::Px(100), Track::Fr(1), Track::Fr(1)])
                .gap(4)
                .place("header", swatch("HEADER", hex("#3b82f6")))
                .place("side", swatch("SIDE", hex("#10b981")))
                .place("main", swatch("MAIN", hex("#f59e0b")))
                .place("footer", swatch("FOOTER", hex("#a855f7")))
                .padding(8)
                .background(hex("#0e0e10"))
                .size(500, 280)
        ),
        scene!(
            "Grid with span",
            "`.span(col, row, colspan, rowspan, child)` extends a cell across multiple tracks.",
            400,
            grid()
                .cols(3)
                .equal_rows(3)
                .gap(4)
                .span(0, 0, 3, 1, swatch("banner (3 cols)", hex("#3b82f6")))
                .cell(0, 1, swatch("a", hex("#ef4444")))
                .cell(1, 1, swatch("b", hex("#10b981")))
                .span(2, 1, 1, 2, swatch("tall (2 rows)", hex("#f59e0b")))
                .cell(0, 2, swatch("c", hex("#a855f7")))
                .cell(1, 2, swatch("d", hex("#ec4899")))
                .padding(8)
                .background(hex("#0e0e10"))
                .size(400, 280)
        ),
        // ── Layers / modifiers ────────────────────────────────────────
        scene!(
            "Layers (z-stack)",
            "Children render in painter's order — first child is the bottom of the stack.",
            300,
            layers()
                .child(fill(hex("#1f2937")))
                .child(swatch("dot", hex("#ef4444")).size(40, 40).center())
                .child(
                    line("on top", WHITE)
                        .align(HAlign::Right, VAlign::Bottom)
                        .padding(8)
                )
                .size(300, 200)
        ),
        scene!(
            "Aspect ratio",
            "`.aspect_ratio(num, den)` constrains the box to a given ratio.",
            400,
            row()
                .gap(8)
                .child(
                    swatch("16:9", hex("#3b82f6"))
                        .aspect_ratio(16, 9)
                        .fill_width()
                )
                .child(swatch("1:1", hex("#10b981")).aspect_ratio(1, 1))
                .padding(8)
                .background(hex("#0e0e10"))
                .size(400, 120)
        ),
        scene!(
            "Image fit modes",
            "`Fit::None` (top-left, no scale), `Fit::Contain` (scale to fit, letterbox), `Fit::Cover` (scale to fill, crop), `Fit::Stretch` (ignore aspect).",
            520,
            row()
                .gap(4)
                .child(
                    column().gap(4).child(line("None", WHITE)).child(
                        image(gradient(60, 40))
                            .fit(Fit::None)
                            .size(120, 80)
                            .background(hex("#1c1c1c"))
                    )
                )
                .child(
                    column().gap(4).child(line("Contain", WHITE)).child(
                        image(gradient(60, 40))
                            .fit(Fit::Contain)
                            .size(120, 80)
                            .background(hex("#1c1c1c"))
                    )
                )
                .child(
                    column().gap(4).child(line("Cover", WHITE)).child(
                        image(gradient(60, 40))
                            .fit(Fit::Cover)
                            .size(120, 80)
                            .background(hex("#1c1c1c"))
                    )
                )
                .child(
                    column().gap(4).child(line("Stretch", WHITE)).child(
                        image(gradient(60, 40))
                            .fit(Fit::Stretch)
                            .size(120, 80)
                            .background(hex("#1c1c1c"))
                    )
                )
                .padding(8)
                .background(hex("#0e0e10"))
        ),
        // ── Labels ────────────────────────────────────────────────────
        scene!(
            "Image with default label",
            "`.label(s)` is the pervasive image-with-label-bar pattern — a `column(label_strip, self)`.",
            300,
            image(gradient(200, 120)).label("EXPECTED")
        ),
        scene!(
            "Image with styled label",
            "`LabelStyle` configures fg/bg/align/padding. `.with_*` builder methods.",
            300,
            image(gradient(200, 120)).label_styled(
                "ACTUAL",
                &LabelStyle::default()
                    .with_fg(hex("#ffd166"))
                    .with_bg(hex("#1f1f3a"))
                    .with_align(HAlign::Left)
            )
        ),
        scene!(
            "Image with segmented label (ADD / REMOVE)",
            "`label_segments` renders multiple aligned segments inside a single strip via Layers.",
            300,
            image(gradient(200, 120)).label_segments(
                vec![
                    LabelSegment::left("ADD", hex("#ffb454")),
                    LabelSegment::right("REMOVE", hex("#50e3c2")),
                ],
                &LabelStyle::default()
            )
        ),
        // ── Template / data injection ─────────────────────────────────
        scene!(
            "Template: status_card called with data",
            "`status_card(title, role, color)` is defined once at the top of this file. `.grow(1)` on each card makes them share the row width evenly — like CSS `flex: 1`.",
            560,
            row()
                .gap(8)
                .child(status_card("Alice", "Engineer", hex("#88f")).grow(1))
                .child(status_card("Bob", "Designer", hex("#f88")).grow(1))
                .child(status_card("Carol", "PM", hex("#8f8")).grow(1))
                .child(status_card("Dan", "Researcher", hex("#fc8")).grow(1))
                .padding(8)
                .background(hex("#0e0e10"))
        ),
        scene!(
            "Template: header_bar over content",
            "Same template idea — a `header_bar` helper composed with body content. `.grow(1)` lets the cards share the row width.",
            500,
            column()
                .gap(0)
                .child(header_bar("zensim-regress", "OK", hex("#10b981")))
                .child(
                    row()
                        .gap(8)
                        .child(status_card("CI", "passing", hex("#10b981")).grow(1))
                        .child(status_card("Tests", "276 / 276", hex("#88f")).grow(1))
                        .child(status_card("Clippy", "clean", hex("#ffd166")).grow(1))
                        .padding(16)
                        .background(hex("#0e0e10"))
                )
                .background(hex("#0a0a0a"))
        ),
        // ── Percent / scale / em ──────────────────────────────────────
        scene!(
            "Percent sizing",
            "`SizeRule::Percent(0.5)` and `Track::Percent(p)` give CSS-`width: 50%`-style relative sizing.",
            500,
            row()
                .gap(4)
                .child(
                    swatch("25%", hex("#ef4444"))
                        .width_percent(0.25)
                        .fill_height()
                )
                .child(
                    swatch("50%", hex("#10b981"))
                        .width_percent(0.5)
                        .fill_height()
                )
                .child(swatch("rest", hex("#3b82f6")).grow(1).fill_height())
                .padding(8)
                .background(hex("#0e0e10"))
                .size(500, 80)
        ),
        scene!(
            "Track::Percent in a grid",
            "Same idea on grid columns — proportional to total available track-axis space.",
            500,
            grid()
                .columns([Track::Percent(0.2), Track::Percent(0.3), Track::Fr(1)])
                .equal_rows(1)
                .gap(4)
                .cell(0, 0, swatch("20%", hex("#ef4444")))
                .cell(1, 0, swatch("30%", hex("#10b981")))
                .cell(2, 0, swatch("rest (Fr)", hex("#3b82f6")))
                .padding(8)
                .background(hex("#0e0e10"))
                .size(500, 60)
        ),
        scene_at_scale!(
            "Same tree, scale = 1.5×",
            "`render_with_config(..., RenderConfig::new(w).with_scale(1.5))` walks the tree once via `Node::scaled` and multiplies every fixed-pixel quantity (Sized::Fixed, Insets, Px tracks, gap, char_h). Fr / Percent / Grow / Hug are unchanged. `.grow(1)` cards distribute the larger canvas width evenly.",
            500,
            1.5,
            row()
                .gap(8)
                .child(status_card("Alice", "Engineer", hex("#88f")).grow(1))
                .child(status_card("Bob", "Designer", hex("#f88")).grow(1))
                .child(status_card("Carol", "PM", hex("#8f8")).grow(1))
                .padding(8)
                .background(hex("#0e0e10"))
        ),
        scene!(
            "em-relative card",
            "`em(units)` returns `base_em * units` pixels (default base_em = 16). Used inline at construction time, so spacing scales with the design baseline.",
            400,
            column()
                .gap(em(0.5))
                .child(line("em-relative", WHITE))
                .child(line("padding em(1.0)", hex("#888")))
                .child(swatch("body", hex("#3b82f6")).size(em(15.0), em(3.0)))
                .padding(em(1.0))
                .background(hex("#0e0e10"))
        ),
        // ── Mini diff montage ─────────────────────────────────────────
        scene!(
            "Mini diff montage",
            "A 2×2 grid of labeled images plus a primary strip and an extra strip — same shape as `compose_montage`. Each cell forces its label to a uniform `label_h` via `.height(SizeRule::Fixed(label_h))` so all 4 headers align horizontally and the row tracks have matching heights.",
            520,
            // label_h chosen to fit both auto-fit "EXPECTED"/"ACTUAL"/etc.
            // (~58 incl. padding) and the multi-segment ADD/REMOVE strip,
            // matching compose_montage's max-of-strip-heights derivation.
            // image_h = 160 below.
            column()
                .gap(0)
                .align_items(CrossAlign::Stretch)
                .child(
                    grid()
                        .columns([Track::Px(240), Track::Px(240)])
                        .rows([Track::Px(58 + 160), Track::Px(58 + 160)])
                        .gap(8)
                        .cell(
                            0,
                            0,
                            column()
                                .gap(0)
                                .child(
                                    LabelStyle::default()
                                        .strip("EXPECTED")
                                        .height(SizeRule::Fixed(58))
                                )
                                .child(image(gradient(240, 160)))
                        )
                        .cell(
                            1,
                            0,
                            column()
                                .gap(0)
                                .child(
                                    LabelStyle::default()
                                        .strip("ACTUAL")
                                        .height(SizeRule::Fixed(58))
                                )
                                .child(image(gradient(240, 160)))
                        )
                        .cell(
                            0,
                            1,
                            column()
                                .gap(0)
                                .child(
                                    LabelStyle::default()
                                        .strip("PIXEL DIFF")
                                        .height(SizeRule::Fixed(58))
                                )
                                .child(image(gradient(240, 160)))
                        )
                        .cell(
                            1,
                            1,
                            column()
                                .gap(0)
                                .child(
                                    LabelStyle::default()
                                        // Multi-segment strips fall back to a
                                        // conservative default char_h so the
                                        // L/R groups don't overflow the strip
                                        // even at narrow widths. Override
                                        // explicitly only when you've sized
                                        // your strip to fit.
                                        .segmented_strip(vec![
                                            LabelSegment::left("ADD", hex("#ffb454")),
                                            LabelSegment::right("REMOVE", hex("#50e3c2")),
                                        ])
                                        .height(SizeRule::Fixed(58))
                                )
                                .child(image(gradient(240, 160)))
                        )
                        .padding(8)
                )
                .child(
                    line("FAIL — zdsim 0.18 > 0.01", hex("#ff5050"))
                        .center()
                        .padding(8)
                        .fill_width()
                        .background(hex("#1e1e1e"))
                )
                .child(
                    line(
                        "alpha: max delta 0  •  pixels_differing 34.2%",
                        hex("#aaaaaa")
                    )
                    .padding(8)
                    .fill_width()
                    .background(hex("#191919"))
                )
                .background(hex("#121212"))
        ),
        // ── Syntax styles ─────────────────────────────────────────────
        scene!(
            "Style: fluent builder (status quo)",
            "The original chained-method syntax. Modifiers attach via `.gap`, `.padding`, etc. Children added via `.child(...)` calls.",
            500,
            row()
                .gap(4)
                .child(swatch("A", hex("#ef4444")))
                .child(swatch("B", hex("#10b981")))
                .child(swatch("C", hex("#3b82f6")))
                .padding(8)
                .background(hex("#0e0e10"))
                .size(500, 80)
        ),
        scene!(
            "Style: array-children macros",
            "`column![...]` / `row![...]` / `layers![...]` macros expand to existing builder chains. Children are an array literal — tighter tree shape, modifiers chain after the macro. Internal-only API (gated behind the `_internal_api` feature).",
            500,
            row![
                swatch("A", hex("#ef4444")),
                swatch("B", hex("#10b981")),
                swatch("C", hex("#3b82f6")),
            ]
            .gap(4)
            .padding(8)
            .background(hex("#0e0e10"))
            .size(500, 80)
        ),
        scene!(
            "Style: closure-builder (compose)",
            "`column_block(|c| {...})` / `row_block(|r| {...})` / `layers_block(|l| {...})` — lexical nesting matches the box tree. Each closure receives a `&mut ColumnCtx` / `RowCtx` / `LayersCtx`. Refactoring a sub-tree into `fn add_pair(r: &mut RowCtx, ...)` is the killer feature.",
            500,
            row_block(|r| {
                r.gap(4);
                r.push(swatch("A", hex("#ef4444")));
                r.push(swatch("B", hex("#10b981")));
                r.push(swatch("C", hex("#3b82f6")));
            })
            .padding(8)
            .background(hex("#0e0e10"))
            .size(500, 80)
        ),
        scene!(
            "Style: factored fragments (compose's killer feature)",
            "A function `fn add_swatches(r: &mut RowCtx, colors: &[Color])` is a real, type-checked, container-affine layout fragment — it can only be called inside a row. Fluent and macros can't do this without macros-on-top-of-macros or runtime branching.",
            500,
            row_block(|r| {
                r.gap(4);
                fn add_swatches(r: &mut RowCtx, colors: &[Color]) {
                    for (i, c) in colors.iter().enumerate() {
                        r.push(swatch(["X", "Y", "Z", "W", "V", "U"][i.min(5)], *c));
                    }
                }
                add_swatches(
                    r,
                    &[
                        hex("#ef4444"),
                        hex("#10b981"),
                        hex("#3b82f6"),
                        hex("#f59e0b"),
                        hex("#a855f7"),
                    ],
                );
            })
            .padding(8)
            .background(hex("#0e0e10"))
            .size(500, 80)
        ),
        scene!(
            "Style: nested macros for grid-of-grids feel",
            "`column![ row![...], row![...] ]` reads almost like HTML's `<col><row>...</row></col>`. Modifier-as-suffix is the only thing keeping it from being literal HTML.",
            520,
            column![
                row![
                    swatch("EXPECTED", hex("#ef4444")),
                    swatch("ACTUAL", hex("#10b981")),
                ]
                .gap(4),
                row![
                    swatch("PIXEL DIFF", hex("#3b82f6")),
                    swatch("HEATMAP", hex("#f59e0b")),
                ]
                .gap(4),
            ]
            .gap(4)
            .padding(8)
            .background(hex("#0e0e10"))
            .size(520, 180)
        ),
        scene!(
            "Style: closure-builder layered",
            "Nested `layers_block` inside `column_block`. Each `&mut Ctx` is its own type; a function expecting `&mut LayersCtx` can't accidentally be called inside a column.",
            520,
            column_block(|c| {
                c.gap(4);
                c.layers(|l| {
                    l.fill(hex("#1f2733"));
                    l.push(line("EXPECTED — closure builder", WHITE).padding(8));
                });
                c.row(|r| {
                    r.gap(4);
                    r.push(swatch("A", hex("#ef4444")));
                    r.push(swatch("B", hex("#10b981")));
                    r.push(swatch("C", hex("#3b82f6")));
                });
            })
            .padding(8)
            .background(hex("#0e0e10"))
            .size(520, 160)
        ),
    ];

    // ── Render & save ─────────────────────────────────────────────────
    let mut html_scenes = String::new();
    let mut toc = String::new();
    for (i, sc) in scenes.iter().enumerate() {
        let canvas = match sc.scale {
            None => render(&sc.tree, sc.width),
            Some(scale) => {
                render_with_config(&sc.tree, &RenderConfig::new(sc.width).with_scale(scale))
            }
        };
        let img_path = dir.join(format!("scene_{:02}.png", i + 1));
        canvas.save(&img_path).expect("save png");
        let img_name = img_path.file_name().unwrap().to_string_lossy().into_owned();
        let anchor = format!("scene-{:02}", i + 1);
        toc.push_str(&format!(
            "  <a href=\"#{anchor}\">{:02}. {}</a>\n",
            i + 1,
            html_escape(&sc.title)
        ));
        html_scenes.push_str(&render_scene_html(
            i + 1,
            &anchor,
            sc,
            &img_name,
            canvas.width(),
            canvas.height(),
        ));
    }

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>zensim-regress layout gallery</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>zensim-regress layout gallery</h1>
  <p>Each scene below shows the source code that produced the rendered PNG beside it.
     Generated by <code>cargo run -p zensim-regress --example layout_gallery</code>.</p>
  <nav class="toc">
{toc}  </nav>
</header>
<main>
{html_scenes}</main>
<footer>
  <p>{n} scenes — rendered with <code>zensim_regress::layout</code>.</p>
</footer>
</body>
</html>
"#,
        css = CSS,
        n = scenes.len()
    );

    let html_path = dir.join("index.html");
    fs::write(&html_path, html).expect("write index.html");

    println!("Wrote {} scenes to {}", scenes.len(), dir.display());
    println!("Open: file://{}", html_path.display());
}

fn render_scene_html(
    n: usize,
    anchor: &str,
    sc: &Scene,
    img_name: &str,
    img_w: u32,
    img_h: u32,
) -> String {
    let pretty = pretty_source(&sc.source);
    format!(
        r#"<section class="scene" id="{anchor}">
  <h2>{n:02}. {title}</h2>
  <p class="desc">{desc}</p>
  <div class="scene-grid">
    <pre><code>{code}</code></pre>
    <div class="scene-image">
      <img src="{img_name}" alt="{title}">
      <div class="scene-image-meta">{img_w}×{img_h} • constraint width = {cw}</div>
    </div>
  </div>
</section>
"#,
        title = html_escape(&sc.title),
        desc = html_escape(&sc.description),
        code = html_escape(&pretty),
        cw = sc.width,
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// `stringify!` runs the tokens through `proc_macro::TokenStream`'s
/// formatter, which spaces tokens out but does not insert line breaks.
/// Add line breaks after `;` and after method-call dots when the line
/// is long, plus indentation for `{` / `}` and `[` / `]` blocks.
fn pretty_source(s: &str) -> String {
    let mut out = String::new();
    let mut indent: i32 = 0;
    let mut chars = s.chars().peekable();
    let mut in_string = false;
    let mut last_char = ' ';

    while let Some(ch) = chars.next() {
        if in_string {
            out.push(ch);
            if ch == '"' && last_char != '\\' {
                in_string = false;
            }
            last_char = ch;
            continue;
        }

        match ch {
            '"' => {
                out.push(ch);
                in_string = true;
            }
            '{' | '[' => {
                out.push(ch);
                indent += 1;
                push_newline(&mut out, indent);
            }
            '}' | ']' => {
                indent -= 1;
                trim_trailing_ws(&mut out);
                push_newline(&mut out, indent);
                out.push(ch);
            }
            ',' => {
                out.push(ch);
                push_newline(&mut out, indent);
                // Skip space directly after a comma (we just added a newline).
                while let Some(&next) = chars.peek() {
                    if next == ' ' {
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            '.' if last_char.is_alphanumeric() || last_char == ')' || last_char == ']' => {
                // Method-chain call. Break before . when line is already long.
                let line_len = out.lines().last().map_or(0, |l| l.len());
                if line_len > 50 {
                    push_newline(&mut out, indent + 1);
                    out.push(ch);
                } else {
                    out.push(ch);
                }
            }
            _ => out.push(ch),
        }
        last_char = ch;
    }
    out.trim().to_string()
}

fn push_newline(out: &mut String, indent: i32) {
    out.push('\n');
    for _ in 0..indent.max(0) {
        out.push_str("    ");
    }
}

fn trim_trailing_ws(out: &mut String) {
    while let Some(c) = out.chars().last() {
        if c == ' ' || c == '\n' {
            out.pop();
        } else {
            break;
        }
    }
}

const CSS: &str = r#"
:root {
    --bg: #0a0a0a;
    --panel: #161616;
    --code-bg: #1c1c1c;
    --fg: #e4e4e7;
    --muted: #71717a;
    --accent: #88c0ff;
    --border: #27272a;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    padding: 32px;
    background: var(--bg);
    color: var(--fg);
    font: 15px/1.5 -apple-system, "Segoe UI", "Helvetica Neue", system-ui, sans-serif;
}
header { max-width: 1100px; margin: 0 auto 32px; }
main { max-width: 1100px; margin: 0 auto; }
footer { max-width: 1100px; margin: 32px auto 0; color: var(--muted); }
h1 { color: var(--accent); margin: 0 0 8px; font-weight: 600; font-size: 28px; }
h2 { color: var(--accent); margin: 0 0 4px; font-size: 18px; font-weight: 600; }
p { margin: 4px 0 0; }
code { font-family: "SF Mono", Menlo, Consolas, ui-monospace, monospace; font-size: 13px; }
.toc {
    margin-top: 16px;
    padding: 12px 16px;
    background: var(--panel);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    columns: 2;
    column-gap: 24px;
}
.toc a {
    display: block;
    color: var(--fg);
    text-decoration: none;
    padding: 2px 0;
    font-size: 13px;
}
.toc a:hover { color: var(--accent); }
.scene {
    margin: 24px 0;
    padding: 20px 24px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
}
.desc { color: var(--muted); margin-top: 4px; max-width: 70ch; }
.scene-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-top: 16px;
    align-items: start;
}
@media (max-width: 900px) { .scene-grid { grid-template-columns: 1fr; } }
.scene-grid pre {
    margin: 0;
    padding: 14px 16px;
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: auto;
    max-height: 600px;
    font: 12.5px/1.5 "SF Mono", Menlo, Consolas, ui-monospace, monospace;
    color: #d4d4d4;
    white-space: pre;
    tab-size: 4;
}
.scene-image {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.scene-image img {
    /* Native pixel size only. `max-width: 100%` would force the browser
       to downscale wide PNGs, and `image-rendering: pixelated` does NOT
       prevent the smoothing browsers apply on bilinear downscale (only
       on upscale). Wrapping container handles overflow with horizontal
       scroll. Result: every pixel rendered is a real pixel from the
       PNG — no phantom-stripe artifacts from interpolation. */
    width: auto;
    height: auto;
    max-width: none;
    image-rendering: pixelated;
    image-rendering: crisp-edges;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: #000;
    display: block;
}
.scene-image {
    overflow-x: auto;
}
.scene-image-meta {
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
    font-family: "SF Mono", Menlo, Consolas, ui-monospace, monospace;
}
"#;
