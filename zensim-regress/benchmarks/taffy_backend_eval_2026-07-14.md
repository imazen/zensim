# taffy as an alternate layout backend for zensim-regress — evaluation

**Date:** 2026-07-14 · **Workspace:** `zensim--taffy-eval` · **taffy:** 0.12.1
(features `flexbox,grid,std,taffy_tree`; +arrayvec/grid/slotmap = 5 crates,
~1.3 s cold build).

## What was built

- `zensim-regress/src/layout/taffy_backend.rs` (feature `taffy-backend`): a
  `Node` → taffy bridge + a paint walk that sources **only** taffy geometry
  through our existing `paint::*` primitives (same font, same compositing),
  so every pixel delta vs `layout::render` is a pure layout-solver
  difference.
- `zensim-regress/examples/taffy_parity.rs`: renders a 23-scene corpus with
  both backends, reports dims-match / % pixels differing / mean|Δ| / max|Δ|,
  and emits per-scene + montage PNGs. Output:
  `/mnt/v/output/zensim/taffy-eval/{REPORT.md,montage.png,*_native.png,*_taffy.png}`.

## Coverage — taffy expresses the whole vocabulary

All 11 node kinds we use mapped: Stack (flex), Grid (px/fr/auto/percent
tracks + **named areas → 1-indexed line placement** + spans), Sized, Padded,
Constrain (min/max), Aspect, Align, Background, Border, Fit, Image, Text,
Layers. Only **Layers** needed an approximation — taffy has no z-order, so
overlay children become `position:absolute; inset:0`. It rendered correctly.
`Fit` is object-fit (paint-time), so it's a passthrough in layout — correct.

## Parity — the core solver matches; our text convention is the gap

Isolating the solver from our text/label convention with **text-free scenes
(pure fill boxes)** is decisive:

| scene | dims | % pixels differ | note |
|---|:--:|--:|---|
| `pure_justify_between` | match | **0.00%** | space-between exact |
| `pure_justify_center` | match | **0.00%** | exact |
| `pure_justify_evenly` | match | 2.22% | 1-px even-split rounding |
| `pure_grow_1_2` | match | **0.00%** | flex-grow exact |
| `pure_grid_px_fr` | match | 0.56% | 1-px fr rounding |
| `pure_percent_split` | match | 0.50% | 1-px rounding |
| `fixed_leaf` / `padding` | match | **0.00%** | exact |
| `aspect_16_9` | match | 5.9% | ratio exact; label content differs |
| `grid_named_areas` | match | 21% | **placement exact**; only cell labels differ |

**Taffy's flex/grid solver reproduces our hand-written math to sub-pixel
rounding.** Residual on fr / space-evenly is integer-rounding divergence
(taffy rounds fractional tracks differently than our "leftover pixels → last
flex child" rule).

The larger divergences (labeled-swatch scenes 40–75%) trace to **one root
cause**, not a taffy limitation: our `.label()` stacks a **fitted text
strip** that greedily measures to fill the available width, and our modifier
model hands a child its parent's whole rect. taffy uses CSS **intrinsic**
(min/max-content) text measure — labels size to their glyph width, not the
container. So every labeled element diverges, and it cascades up through hug
sizing. This is a **measure-protocol mismatch**, reconcilable only by
teaching the taffy leaf-measure to replicate our fill-to-width fitted-text
behavior (doable — the measure callback is where it'd live).

## Two bridge lessons (a naive port fails, quietly)

1. **1:1 Node→taffy starves flex/grid.** `.size(w,h)` wraps a container in a
   `Sized` node; taffy then auto-sizes the flex/grid *child* to hug and
   leaves no free space — justify/grow/fr silently collapse to start-packed.
   Fix: transparent modifiers (`Sized/Constrain/Aspect/Padded/Background/
   Border/Fit`) must make their child fill the wrapper (`flex_grow:1` +
   `align_self:stretch` + widen `auto` axes to `100%`). A real bridge would
   *flatten* modifier chains onto the styled node instead.
2. **Zero-intrinsic leaves collapse.** `Fill(color)` measures 0×0; taffy
   paints nothing unless told `size:100%`.

Both are invisible without a pixel-diff harness — they look like "taffy can't
do X" when they're bridge modeling bugs.

## Verdict (eval)

- **Capability:** taffy can back these layouts. Nothing in the vocabulary is
  unexpressible; placement/sizing math matches to rounding.
- **Cost:** not pixel-identical — fr/even-split rounding differs, and a
  correct bridge must flatten modifiers + reuse our fitted-text measure. But
  our text measure (`spec.natural`) is *already* what the taffy leaf-measure
  callback calls, so the "text port" is a non-issue, and the eval's concern
  that "every golden re-baselines" turned out empirically false (below).
- **When it pays off:** real CSS flex-wrap / grid auto-fill / min/max-content
  intrinsic sizing — the features the native solver lacks — come for free and
  correct. That is the direction the imageflow-JSON layout surface points.

## Adopted — 2026-07-14

Owner directive: adopt. Taffy is now the **default** backend behind the
existing `Node` API / paint layer; the native solver stays selectable via
`Backend::Native`.

- `RenderConfig::backend` (`Backend::{Taffy,Native}`, default `Taffy`);
  `render_with_config` dispatches. taffy is a normal (non-optional) dep.
- Safety limits (`max_dim` / pixel budget / `max_depth` / `max_children` /
  `max_cells` / `max_tracks`) integrated into the taffy path — hostile trees
  bounded exactly as on the native path.
- **Empirical break-set: 3 of 393 lib tests**, all testing *native-only*
  features — `shrink_on_overflow`'s exact distribution and `render_checked`'s
  overflow diagnostics. Both are moot under taffy (CSS flex shrinks to fit
  rather than overflowing), so they were pinned to `Backend::Native` (same
  assertions, correct backend) — not relaxed. The feared 75-assertion
  re-baseline did not happen: most of those are direct `measure()` unit tests
  on the native solver (untouched), and the render-path pixel tests match
  taffy.
- **Shipping montages verified** through taffy: the real
  `create_structural_montage` / `MontageOptions` output (2×2 image panels,
  segmented ADD/REMOVE strips, 3×3 heatmap grids, mismatched-dimension
  shared-canvas path) renders correctly — see
  `/mnt/v/output/zensim/taffy-eval/montages-taffy/`.
- Full suite: 393 lib + 10 integration + 8 doctests pass; clippy + fmt clean.

Behavior notes: `render_checked` overflow diagnostics and
`shrink_on_overflow` are native-backend features; under taffy content shrinks
to fit (CSS flex) rather than overflowing. `Backend::Native` remains for A/B
and pre-taffy geometry. Bridge lives in `src/layout/taffy_backend.rs`; the
parity harness behind the `taffy-backend` feature (`examples/taffy_parity.rs`).
