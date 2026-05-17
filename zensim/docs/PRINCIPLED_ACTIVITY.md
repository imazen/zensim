# Principled per-channel H-blur activity (masked / IW features)

**Status**: shipped 2026-05-17 on branch `feat/principled-activity`.

## Background — accidental cascade

The masked-block (features 228..300) and IW-block (300..372) extended
features both compute an "activity map" per channel:

```
activity[c] = box_blur(|src[c] - mu1[c]|)
```

then derive per-pixel weights:

```
mask_weight[c]    = 1 / (1 + k_mask * activity[c])
iw_weight[c]      = 1 + k_iw    * activity[c]
```

For the 1-pass blur path (`config.blur_passes == 1`, the
production code path used at every scale by all profiles), the
input to the activity computation is `bufs.mu1`. The pre-existing
implementation reused `bufs.mu1` across the three channels (X → Y → B)
via `std::mem::swap(&mut bufs.mu1, &mut bufs.mask)` after the fused
V-blur. The fused V-blur **only writes inner rows** of `mu1`, leaving
strip-overlap rows untouched. The resulting overlap-row contents
turn out to be:

| Strip | Channel | Source of overlap-row `bufs.mu1` |
| --- | --- | --- |
| 0 | X | `0.0` (initial `ScaleBuffers::new` zero) |
| 0 | Y | `src_X(gy, x)` (X's swap moved raw stale state) |
| 0 | B | `\|src_Y(gy, x) - src_X(gy, x)\|` |
| K≥1 | X | inherited from prev-strip B's `bufs.mask` |
| K≥1 | Y/B | similar cross-strip cascade |

This was an *accidental* artifact of buffer reuse, not a designed
algorithm. The diagnostic agent (CPU TSV dumps at
`/tmp/zensim_diag_cpu_scale0_strip{0,1}_chan{0,1,2}.tsv`) confirmed
the cross-channel state leaks empirically.

The GPU side had been trying to reproduce this cascade exactly via a
precomputed `carryover` plane (per-(strip, channel, row, x) host-side
simulation of the CPU's per-strip state). That work is intricate,
fragile, and an unhealthy amount of code for what is supposed to be
a perceptual masking signal.

## Replacement algorithm

For each strip K, for each channel c in {X, Y, B}:

1. `bufs.mu1[c]` at ALL strip rows (inner + overlap) = `H_blur(src[c])`
   at those rows. This is the **horizontal pass** of the source's
   box blur, computed by `box_blur_h` (which already runs across the
   full strip including overlap rows).
2. `activity[c]` = `box_blur(|src[c] - mu1[c]|)` (strip-local mirror
   for the V-blur, unchanged).

Channels are decoupled. No cross-channel reuse of mu1 for activity
computation. mu1 is purely the H-blur of the same channel's source,
strip-local.

## Why this is right

- **mu1 at overlap rows had no defined meaning before.** The fused
  V-blur skipped them; whatever was there came from whatever happened
  to be in the buffer (zero or prior-channel garbage). Activity
  computed against undefined data is undefined.
- **H-blur of src at the same row** is a principled "local mean"
  signal. It is the *first half* of what mu1 means at inner rows
  (mu1 inner = V-blur of H-blur of src). At overlap rows the V-blur
  has nowhere to go (no inner-row neighbors above/below it for the
  strip), so the H-blur alone is the natural per-row local mean.
- **Activity values become strip-continuous.** At inner rows the
  activity uses V-blurred mu1; at overlap rows it uses H-blurred mu1.
  The difference is small (mu1's V-blur of an H-blur ≈ H-blur for
  smooth-in-y content) and bounded. The V-blur of activity then
  averages across the strip rows with valid mu1 throughout —
  contrasted with the prior behavior, where the V-blur reached into
  rows whose mu1 was effectively zero (or a stale cross-channel
  artifact), inflating activity at strip boundaries.
- **GPU parity becomes trivial.** Per channel: compute H-blur of src
  at overlap rows from the source plane directly. No carryover plane.
  No cross-channel simulation. No host-side replay.

## Precise CPU change

In `zensim/src/streaming.rs::process_strip_channel`, at the start of
the `need_activity` block in the `blur_passes == 1` fast path (around
line 1261), **before** the existing `abs_diff_into` call, replace
the in-place `bufs.mu1` contents with the H-blur of the current
channel's source plane at all strip rows. Use `box_blur_h`
(`zensim::blur::box_blur_h`, already `pub(crate)`) writing into
`bufs.temp_blur` first and then swap, or write directly into the
strip rows of `bufs.mu1`.

Pseudo-Rust:

```rust
// Before computing activity: overwrite bufs.mu1 at ALL strip rows
// with H_blur(src_c). Inner rows lose the V-blurred value used by
// the ssim_d/edge accumulators above (those are already accumulated
// at this point), so this is safe. Overlap rows go from
// stale-cross-channel garbage to a well-defined per-row local mean.
box_blur_h(
    &src_c[..strip_n],
    &mut bufs.temp_blur[..strip_n],
    width,
    strip_h,
    config.blur_radius,
);
std::mem::swap(&mut bufs.mu1, &mut bufs.temp_blur);
```

The masked-edge block at the end of the function (lines 1417..1446)
still uses `bufs.mu1` as the inner-row "mu1 reference plane" for
`edge_diff_channel_masked*` — but it only reads `bufs.mu1` at
**inner rows**, where H-blur and V-blur-of-H-blur give close-enough
values (smooth-in-y) for the masked-edge metric. (If the regression
testing shows otherwise, we can stash a copy of the V-blurred inner-
row mu1 before overwriting.) **Verified by running the test suite**;
if `masked_art4` / `masked_det4` diverge by more than the existing
tolerance, the implementation switches to "stash V-blurred mu1 at
inner rows in a small scratch, restore after activity computation."

## Why the basic 228 features are unaffected

The basic features (slots 0..228) are accumulated **before** the
`need_activity` block runs. They use `bufs.mu1` containing the
V-blurred values produced by `fused_vblur_features_ssim` /
`fused_vblur_features_edge`, written before activity is computed.
After the activity block, the function returns. The change only
mutates `bufs.mu1` after all basic-feature accumulation is done —
the basic-block contribution is bit-identical.

## What the GPU change mirrors

Drop the `carry` plane and the entire `populate_carryover` host-side
simulator. In `masked_iw_strip_kernel`, the overlap-row branch becomes
a plain on-the-fly H-blur of `src` (the same channel's plane):

```text
mu1_row[load_x] = H_blur_of_src_at(channel, gy, gx)
```

For inner rows, continue reading from the persist mu1 plane (which is
V-blur of H-blur, identical to before).

Cross-channel cascade (X → Y → B) is **deleted**. The strip 0 vs
strip K≥1 distinction is **deleted**. Per channel, mu1 is the H-blur
of that channel's source, period.

## Test plan

CPU:
- `cargo test --release` (workspace) — basic-block features must be
  bit-identical (any pre-existing test asserting their values stays
  green).
- Masked/IW block outputs change. Run any zensim CPU bench / test
  that asserts on those slots; either delete the stale fixture or
  regenerate to the new values.

GPU:
- `cargo test --release -p zensim-gpu --features cuda
  --no-default-features --lib --tests` — extended_parity should pass
  at ≤0.5e-2 rel for X, Y, B at all scales (tighten from current
  1.5e-2 rel multi-strip widening).

12 MP WithIw bench: target ≤ 1.2× of prior 26.92 ms baseline; expected
faster because the carry-plane host simulator and the per-pixel
cross-channel branch in the kernel both vanish.

## Caveats for downstream

- Parquet sweep data that includes masked/IW features (300-col
  sidecars from zenmetrics' v15 sweep) was scored against the old
  cascade semantics. New feature values will differ in the slots
  affected by overlap-row mu1 (mostly low-amplitude — the activity
  signal is bounded by the H-blur of src difference). Retraining
  any picker that consumed those features at face value is
  recommended; the magnitude of shift is bounded by the 1.5-4 % rel
  GPU residual that this fix eliminates.
- Re-bake any V_X model whose training corpus included masked/IW
  features from the contaminated semantics (V0_4 onward consumed
  these as the extended-IW feature block). Until then, the CPU and
  GPU runtimes agree on the new semantics, so production scoring is
  consistent — but the bakes were trained on slightly different
  numbers.
