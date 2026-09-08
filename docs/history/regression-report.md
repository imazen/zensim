# Layout-module port — regression report

## Summary

`render_montage_impl`, `render_mismatched_montage`, and `render_heatmap_grid`
in `zensim-regress/src/diff_image.rs` rewritten on top of the new
`layout` module (`src/layout.rs`).

**6 of 6 regression scenes are byte-identical to the pre-port baseline.**
**272 of 272 lib tests pass. Clippy clean.**

## What changed

| Function | Before | After | Notes |
|---|---:|---:|---|
| `render_montage_impl` (same-dim) | 290 lines | 30 lines + shared composer | composes via `compose_montage` |
| `render_mismatched_montage` | 386 lines | ~120 lines (panel prep only) + shared composer | same composer |
| `render_heatmap_grid` | 110 lines | 90 lines | layout `Grid` + `Layers` per cell |
| `compose_montage` (new) | — | 165 lines | shared 2×2 grid + strip-stack |
| `fill_rect` (private helper) | 10 lines | 0 (deleted) | layout module owns this now |

`diff_image.rs` net **2232 → 1892 lines (−340)**, plus 2269 lines added in the
new layout module. The two near-duplicate render functions are gone — they
now share `compose_montage`. The mismatched function only owns its
panel-prep prologue.

## Regression-test scenes

`zensim-regress/examples/montage_regression.rs` exercises six representative
scenes covering both code paths and the optional annotation strips:

| Scene | Dim | Annotations | Heatmap | Path | Pixels |
|---|---|---|---|---|---|
| 01 same-dim plain | 96×96 same | none | yes | same-dim | identical |
| 02 same-dim annotated | 96×96 same | title + 3 primary lines + extra | yes | same-dim | identical |
| 03 mismatched plain | 96×64 vs 72×96 | none | (no diff in plain) | mismatched | identical |
| 04 mismatched annotated | 96×64 vs 72×96 | title + primary + extra | yes | mismatched | identical |
| 05 tiny pixelate | 16×16 | none | yes | same-dim, pixelate-upscale | identical |
| 06 custom labels | 96×96 | "REFERENCE-LONG-NAME" / "CANDIDATE", heatmap off | no | same-dim | identical |

Compare via:

```sh
cargo run -p zensim-regress --example montage_regression -- /tmp/montage_baseline
# (port the changes, then)
cargo run -p zensim-regress --example montage_regression -- /tmp/montage_new
cargo run -p zensim-regress --example montage_diff
```

## Bugs found and fixed during the port

The diff-driven workflow surfaced three real bugs in the new layout
module that the unit tests had missed (all had clean tests before — they
were missing coverage, not broken):

1. **`render_grid` clipped the last column by `gap` pixels**
   `col_off[i+1] - gap` was applied unconditionally, but the last
   col_off has no trailing gap. Right-most cells got `cell_w - pad`
   instead of `cell_w`. Fixed by subtracting gap only for non-trailing
   spans. Added a `grid_with_gap_last_col_full_width` test.

2. **`Image` rendered with `Fit::None` was centered, not top-left**
   When the rect was bigger than the image (e.g. heatmap 598 wide
   inside a 600-wide column with `align_items: Stretch`), the image
   got `(rect.w - img.w) / 2` left-offset. Original code expected
   top-left placement. Fix: only center when `Fit::Contain`/`Cover`
   are explicitly set; otherwise paint at rect's origin. Centering is
   recoverable via `.center()`.

3. **Auto-fit text char_h shifted between measure and paint**
   `font::render_lines_fitted` derives char height from `max_w` via
   integer arithmetic. When `Align` reduces a rect to the measured
   natural width, paint sees a smaller `max_w` than measure did and
   may compute a different char_h (e.g., 40 vs 39). Fix in this
   port: pre-rasterize annotation text into `Image` leaves rather
   than relying on `Node::Text`'s auto-fit at paint time. Plain
   labels and ADD/REMOVE happen to hit a clamp boundary so they're
   stable through the round-trip.

These bugs would have caused subtle 1–2-px visual drift in production
with no test signal — caught only by the byte-level regression
comparison.

## Layout-module additions/changes during the port

- `LabelStyle::strip` and `LabelStyle::segmented_strip` made `pub`
  (needed by the port to build label nodes with explicit height).
- `LabelStyle::strip` switched from `align_h(self.align)` to
  `align(self.align, VAlign::Center)` so labels vertically center
  inside an over-sized strip.
- New unit test for `Grid` with non-zero gap at the last column.

## Next step (separate change)

The shared `compose_montage` is right where it is for now — clean and
self-contained. The layout module is internal to `zensim-regress`; the
text auto-fit caveat (workaround: pre-rasterize) is the strongest
argument for caching `char_h` inside `Node::Text` in a future iteration,
which would make the API friendlier for non-pre-rasterized annotation
strips. Not blocking.
