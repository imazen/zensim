# zengrid — extraction analysis

Worktree: `~/work/zen/zensim--zengrid-analysis/` (jj workspace `zengrid-analysis`).
No code edits — investigation only.

## What "layout/montage" code currently lives in zensim-regress

All in two files (~2.65k lines combined, ~18% of the crate):

| File | Lines | Concern |
|------|------:|---------|
| `src/diff_image.rs` | 2232 | montage composition + diff rendering (mixed) |
| `src/font.rs` | 419 | embedded Consolas glyph strip + text rendering |

`diff_image.rs` is doing two distinct jobs glued together:

1. **Diff math** — `generate_diff_image`, `generate_structural_diff`, `spatial_analysis`, `RegionStats`, `SpatialAnalysis`, `pixelate_upscale`. Fundamentally about *pixels*.
2. **Visual layout** — `MontageOptions`, `render_montage_impl`, `render_mismatched_montage`, `render_heatmap_grid`, `create_montage`, `fill_rect`, `draw_rect_border`, deprecated `create_*_montage*` family, `AnnotationText`, label bars, title strip, primary strip, vertical stacking. Fundamentally about *boxes*.

The split is clean: ~1200 lines are layout, ~1000 lines are diff. Nothing in the layout code reaches into diff internals beyond reading `RgbaImage` panels and a `SpatialAnalysis { regions, cols, rows }` struct.

## The recurring shapes

Reading the two big render functions, the same primitives appear over and over:

| Primitive | Where it shows up |
|-----------|-------------------|
| **2×2 grid of cells**, each cell = label-bar + image | `render_montage_impl` lines 810–914, `render_mismatched_montage` 1196–1302 |
| **N×M cell grid** with text inside each cell, severity-colored | `render_heatmap_grid` 1381–1491 |
| **Horizontal strip** of N panels with gap | `create_montage` 116–134 |
| **Vertical stack of variable-height strips** below the grid (title, primary, heatmap, extra) | both render functions, ~80 lines each, near-identical |
| **Center text in box** | repeated ~6× as `(box_w.saturating_sub(text_w)) / 2` |
| **Left-/right-aligned text with inset** | ADD / REMOVE label bar, lines 783–798 and 1170–1185 |
| **Best-fit char height** to fit longest label in width | `label_char_h = …clamp(GLYPH_H, GLYPH_H*3)` repeated verbatim in both functions |
| **Word-wrap text within max width** | `render_text_wrapped` |
| **Fill rect** / **draw rect border** | private helpers, ~50 lines |

`render_montage_impl` and `render_mismatched_montage` are **near-duplicates** — same 2×2 grid math, same label sizing, same text-strip stacking, with a different panel-preparation prologue. That's the strongest internal signal that an abstraction is missing.

## Consumers

- **Internal:** `display.rs` (sixels + PNG save), `examples/visual_diff.rs`.
- **External, today:** only `zenjpeg` (5 files, all using deprecated `create_comparison_montage_raw`).
- **Plausible future consumers** in the zen workspace that already produce regression-style or A/B image output: `codec-eval`, `fast-ssim2`, `butteraugli`, `resamplescope-rs`, `zensally` viz, `zenfilters` parity tests, `zenresize` filter visualizations, `zenpipe` integration tests, `zenbench` for image-shaped benchmarks.

So: one real cross-crate consumer today. Several plausible ones — but none that have asked.

## Dependencies the layout code actually needs

- `image::{RgbaImage, imageops::{overlay, resize, crop_imm, FilterType}}`
- `std::sync::OnceLock` (font cache)
- the embedded font PNG (`font_strip.png`, ~few KB)

That's it. No `image-tiff`, no archmage, no zensim-specific types. A clean extraction is genuinely lightweight — no dependency graph pain.

## Why a `zengrid` crate would help

1. **Kills the duplicate render functions.** The same-dim and mismatched-dim paths collapse to one `Grid::build(...).into_image()` plus a 30-line panel-prep prologue each. ~300 → ~50 lines per function.
2. **Reusable shape.** "Labeled image grid + text strips" is what every codec/quality/saliency tool produces; today each one either reinvents it (zenjpeg has its own ad-hoc montages) or reaches into zensim-regress (which drags `zensim` + `seahash` + `fs2` + classification features along).
3. **Frees zensim-regress to depend on zengrid without dragging weight.** Currently any crate that wants `MontageOptions` has to take the entire regression-testing harness as a dependency.
4. **Self-contained font is a real product.** `render_text_height` + `render_lines_fitted` over an embedded Consolas strip is genuinely useful and not specific to grids — half the workspace could use it for badge/label rendering. Crates that today reach for `ab_glyph` or `fontdue` (TTF parsing + rasterization) get a much smaller, faster, deterministic alternative for monospace labels.
5. **Sets up future polish** (legend strips, badges, axis ticks, scrollbar hints) without bloating zensim-regress.

## Why NOT to extract right now

1. **Only one external consumer**, and it uses a *deprecated* API. Three similar usage sites is the usual threshold for an abstraction; we have ~1.
2. **Premature-generalization risk.** The current API is heavily tuned for "diff montage": dark theme, COLOR_FAIL/OK palette, 2×2 panel layout, automatic spatial heatmap from `SpatialAnalysis`. A first extraction that bakes those defaults in just relocates the problem; one that strips them out has to be rebuilt the moment the second consumer's needs differ.
3. **The 2×2 layout is mostly straight-line code**, not algorithmic. A clever generic `Cell/Row/Column` builder is not dramatically shorter than the current explicit math — most of the savings come from collapsing the duplicate function, which can be done *inside* zensim-regress with no new crate.
4. **Cross-crate breakage cost.** zenjpeg is mid-publish-train (per active session marker). Breaking `create_comparison_montage_raw` to route through a new dep is friction we don't need.
5. **Two crates, not one, if done right.** The font code is independent enough that bundling it under "grid" is awkward — it really wants `zenfont-mono` or similar. Splitting out two crates for one consumer is a lot of plumbing.

## Recommendation

**Phase 1 (do now, no new crate):** refactor *inside* zensim-regress.

- Add `src/layout.rs` with `Cell`, `Grid::build`, `VStack`, centering/alignment helpers, and `fit_char_height_for_label`.
- Move `fill_rect`, `draw_rect_border`, the 2×2 grid math, the title/primary/heatmap/extra strip stacking into it.
- Rewrite `render_montage_impl` and `render_mismatched_montage` to share the same `Grid` + `VStack` plumbing. The diff is now only the panel-prep prologue.
- Keep `font.rs` as-is for now.

This gets you ~80% of the duplication kill at zero blast-radius cost, and **prototypes the API before promoting it to a public crate**.

**Phase 2 (when the second non-zensim consumer asks):** promote `layout.rs` + `font.rs` into a `zengrid` crate.

- Strip diff-specific defaults (palette, heatmap auto-compute) from the public API; pass them in as caller-supplied content.
- Embedded font becomes either part of `zengrid` or a sibling `zenfont-mono`. My read: keep them in one crate (`zengrid`) until proven wrong — fewer Cargo.toml entries, identical release cadence, no real reason to split.
- zensim-regress, zenjpeg, codec-eval, fast-ssim2, butteraugli adopt it.

**Sketched Phase 2 API** (for reference, not commitment):

```rust
// zengrid::layout
let title  = Strip::title(annotation.title.as_deref(), grid_w);
let panels = Grid::new(2, 2).pad(8).gap(2)
    .cell_size(panel_w, label_h + panel_h)
    .cell(0, 0, Cell::image(expected).top_label("EXPECTED"))
    .cell(1, 0, Cell::image(actual)  .top_label("ACTUAL"))
    .cell(0, 1, Cell::image(pdiff)   .top_label("PIXEL DIFF"))
    .cell(1, 1, Cell::image(sdiff)   .top_label_segments(&[
        Segment::left("ADD",    [255,180,80,255]),
        Segment::right("REMOVE",[80,220,220,255]),
    ]));
let heatmap = Strip::cells(spatial.cols, spatial.rows, |r, c| { ... });
let extra   = Strip::wrapped_text(&annotation.extra, COLOR_DETAIL, fit);

VStack::new().bg([18,18,18,255])
    .push(title).push(panels).push(primary).push(heatmap).push(extra)
    .render()
```

```rust
// zengrid::font
use zengrid::font::{render_text_height, render_lines_fitted, GLYPH_W, GLYPH_H};
```

## Bottom line

A `zengrid` crate **is** a good idea — the abstractions are real, the dependencies are clean, the duplication exists. But the cheapest first step is an internal `mod layout` refactor inside zensim-regress that proves the API on the existing two render functions. Promote to a crate **after** the next consumer (most likely codec-eval or zenjpeg's parity tooling) is ready to adopt it; that's the moment when generalizing the palette/heatmap defaults pays for itself instead of just moving them.

Phase 1 is ~1 day of work and immediately useful. Phase 2 is ~1 day of repackaging plus consumer migrations, deferrable until needed.
