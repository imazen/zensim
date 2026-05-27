# Structural-Corruption Distortion Corpus — spec (2026-05-27)

## Purpose

zensim's score has an intentional **negative tail** (≤ 100 above; unbounded
below). Positive scores mean "honestly degraded" (uniform lossy compression
— globally softer/blockier but structurally faithful). **Below 0 means
"worse than a simple lq encode" — structurally BROKEN**: a localized,
catastrophic defect no honest encoder would produce (a decoder/render bug,
a channel swap, an off-by-one edge).

This corpus exists to **validate and calibrate that negative tail**. It is
the ground truth for the property the regression-test use case needs: a
broken decode must score WORSE than a legitimately-lossy one, so the test
catches the bug instead of passing it.

## The gate (what every corpus entry asserts)

Each entry is a triple `(reference, corruption, honest_lq_anchor)` where
`honest_lq_anchor` = a uniform low-quality encode of the same reference
(JPEG q20 and q10 — "honestly degraded"). The metric MUST satisfy:

    score(ref, corruption) < score(ref, honest_lq_anchor)

i.e. the structural corruption ranks below an honest lossy encode. For the
most egregious corruptions (large region, opaque channel swap), the
calibrated score should be **< 0**. Subtle ones (8×8 block, 1px 20%-opacity
border) need only rank below the lq anchor — they're the hard cases.

The corpus is NOT for training the metric's positive band (that's the
human-MOS corpora). It is a held-out **falsification set for the negative
tail + the structural-corruption ordering**.

## Distortion families

Each family is parameterized by **region size** (sweep: whole-image → 1/4 →
1/16 → 64×64 → 16×16 → 8×8 → single-pixel where meaningful) and **severity**
(opacity / magnitude). Generate on a content-varied reference set (photo,
screen/UI, line-art, text, gradient — reuse codec-corpus references).

1. **Channel corruption** — within a rectangle: invert (255−v) per channel;
   channel swap (RGB→BGR, R↔G, R↔B, G↔B); single-channel zero/max. Models
   a decoder writing the wrong channel order / a corrupted plane.
2. **Block corruption** — within an N×N block: zero-fill, mid-gray fill,
   garbage (random), copy from a wrong location (MCU mispredict), repeat the
   neighboring block. Models dropped/corrupted MCUs, Huffman desync.
3. **Edge / border artifacts** — a k-px border (k ∈ {1,2,4}) at p% opacity
   (p ∈ {20,50,100}) on one edge / all edges; a 1px row/column shift of the
   interior (off-by-one crop/pad); duplicated or dropped edge row. Models
   partial-MCU edge handling, padding/cropping off-by-one.
4. **Salt-and-pepper / bit errors** — k random pixels (k from 1 to 0.1% of
   pixels) set to black/white/random; single-bit flips in a channel. Models
   transmission/bit corruption.
5. **Local tone / gamma** — within a block: wrong transfer function
   (apply/undo sRGB gamma), local contrast boost (×1.5 around the block
   mean), brightness offset. Models a decoder color-management bug confined
   to a region.
6. **Low-opacity overlay** — draw a shape (rect, line, glyph) at low opacity
   (5–30%) somewhere. Models a render leak / watermark bleed / compositing
   bug.
7. **Chroma-boundary mismatch** — upsample chroma with the wrong phase /
   a different kernel than luma at block boundaries (the CLAUDE.md
   "chroma boundary errors from mismatched upsampling"). Confined to block
   edges. Models the buffered-vs-streaming chroma divergence.
8. **Aliasing / moiré** — nearest-neighbor downscale→upscale (vs a clean
   reference), or a high-frequency pattern downsampled without prefilter.
   Models a resampler bug.
9. **Geometric** — 1px translation / sub-pixel shift / small rotation of a
   region; horizontal/vertical flip of a block; 1px skew. Models a transform
   off-by-one.
10. **Wrong-background compositing** — treat premultiplied alpha as straight
    (or vice-versa); composite onto the wrong background color (black vs
    white vs gray). Models the alpha-handling bugs in `tests/common/
    distortions.rs` (`premul_as_straight`, `wrong_bg_black`).
11. **Global off-by-one / bit-depth rounding** — image-wide, max_diff = 1 but
    systematic: u16→u8 truncation (`>>8`) vs round, 10-bit→8-bit roundtrip
    LSB drift, unpremultiply truncation, missing U/V chroma rounding, a
    uniform ±1 added to every pixel. This is the **saturating-metric worst
    case** — a tiny per-pixel error spread over the whole image is exactly
    what a metric with a saturated near-lossless tail will wrongly rank as
    "perfect," yet it's a real shipped-bug class (zenpng `d88325c`/`838cad7`,
    zenavif `4509713`/`42d06a7`, zenwebp `11465dd`). Sweep the rounding
    magnitude (±1, ±2) and the conversion (truncate vs round vs round-half-
    even). Unlike families 1–10 this is NOT region-localized — the region
    axis is fixed at whole-image; severity is the per-pixel error magnitude
    and the fraction of pixels affected (truncation hits a deterministic
    subset, e.g. odd LSBs). The gate is the hard one: this must still rank
    below an honest q20 encode even though its max per-pixel delta is far
    smaller — because the *honest* encode is structurally faithful while
    this is a correctness bug. (Mined gap, surfaced by the historical-bug
    audit → codec-corpus#7.)

## Real-bug reproductions (mined, mandatory)

The corpus MUST include reproductions of **historical decoder/renderer bugs**
from imageflow + the zensim-regress codec users (zenjpeg, zenwebp, zengif,
zenjxl, zenavif, zenpng, heic). For each documented bug that produced wrong
pixels, add an entry: the reference + the buggy output (if recoverable from
the bug report / fixed test) OR a synthetic repro of the bug's pixel
pattern. Known starting points (verify + expand by mining issues/CHANGELOGs/
git history):
- zenjpeg Huffman-table bug on non-mod-8 dimensions → garbled trailing
  blocks.
- chroma-boundary errors from mismatched upsampling (buffered vs streaming).
- partial-MCU edge artifacts from wrong padding/cropping.
- zenwebp systematic quality cliffs at q75→80 / q87→90 (mode switches).
- rounding divergence between buffered and streaming decode.
- premultiplied-vs-straight alpha confusion; wrong-bg compositing.
- GIF/zengif palette/transparency edge cases.

These are the gold-standard members: they are the exact failures the metric
must rank as "broken," and they connect the corpus to real shipped bugs.

## Corpus structure (in codec-corpus)

- New module/dir (e.g. `corruptions/`) with deterministic Rust generators
  (seeded), one fn per family, parameterized by region + severity.
- A `_MANIFEST.json` (or the codec-corpus convention) per entry:
  `{ ref_id, family, params, expected_below_lq: true, expected_negative:
  bool, source: "synthetic" | "real-bug:<repo>#<issue>" }`.
- References: reuse codec-corpus's curated set across content classes; do
  NOT commit large generated images to git — follow codec-corpus's existing
  storage convention (block storage + pointer, or on-demand generation from
  seeds). Prefer **on-demand generation from seeds + the reference id** so
  the corpus is reproducible without committing bytes.
- Provide a small driver that, given a reference, emits (ref, corruption,
  q20-anchor, q10-anchor) so the zensim gate can be run directly.

## Severity / coverage discipline

Per the CLAUDE.md sweep rule: cover the region-size axis densely from
whole-image down to 8×8 (and 1px for edge/geometric), and severity from
subtle (20% opacity, 8×8) to obvious (opaque, whole-image). The subtle end
is the hard, important case — that's where a metric saturates and lets a
real bug pass. ≥ 5 content classes, ≥ 10 references each.

## Acceptance

The corpus ships with a zensim eval that reports, per family × region ×
severity: `score(corruption)`, `score(q20)`, `score(q10)`, and the gate
pass/fail `score(corruption) < score(q20)`. A faithful metric passes the
gate on the obvious end and is measured (not asserted) on the subtle end —
the subtle failures are the research signal for the negative-tail
calibration + the partial-monotone-residual head (option 3, task #32).
