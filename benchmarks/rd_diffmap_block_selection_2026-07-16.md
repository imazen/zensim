# Can a zensim diffmap drive codec RD block-selection? (2026-07-16)

**Question (user).** Codecs allocating a fixed bit budget must choose *which*
regions to spend bits on. The classic RD loop ranks blocks by squared error
(SSE / PSNR-optimal); jxl-encoder's adaptive quantizer ranks by a **butteraugli**
diffmap. Does zensim's per-pixel **diffmap** pick better blocks — and how does it
compare to butteraugli under multiple judges (butter max-norm, butter 3-norm,
SSIM2, zensim-B)?

**Method** (`zensim/examples/rd_block_selection.rs` + `scripts/v_next/rd_diffmap_eval.py`).
Ref `R`, distorted encode `D`. Tile into blocks; for budget fraction f=0.25,
"refine" the top-f blocks by each selector — copy `R`'s pixels back (the
unlimited-bits limit). Every selector refines the **same block count** → same
rate. An **independent** judge scores the refined image. Selectors: `sse`,
`zensim`-diffmap, `butteraugli`-diffmap (crate 0.9.3, jxl-encoder's deployed
signal), `random`. Judges: `butter_max`, `butter_3norm`, `ssim2`, `zensim_B`
(shipped Profile B). n=50 pairs, mozjpeg q15/q30. Home-turf cells (selector under
its own metric) are marked `*` and excluded from fair reads.

## Results — mean perceptual improvement over the unrefined encode

### block = 32

| selector | butter_max | butter_3norm | ssim2 | zensim_B |
|---|---|---|---|---|
| sse | +0.754 | +0.303 | +5.925 | +7.195 |
| **zensim** | +0.036 | +0.161 | **+11.815** | +5.361* |
| butteraugli | +0.922* | +0.375* | +8.933 | +5.977 |
| random | +0.054 | +0.167 | +5.835 | +3.715 |

### block = 16 (consistent)

| selector | butter_max | butter_3norm | ssim2 | zensim_B |
|---|---|---|---|---|
| sse | +1.429 | +0.390 | +6.916 | +8.997 |
| **zensim** | +0.070 | +0.181 | **+12.399** | +5.706* |
| butteraugli | +1.641* | +0.532* | +10.608 | +6.902 |
| random | +0.133 | +0.192 | +6.056 | +3.817 |

**Block overlap** (how differently selectors choose): sse∩zensim 4–6%,
zensim∩butter 33–35%, sse∩butter 44–47%. zensim makes genuinely different
choices from both; sse and butteraugli agree most (both track pixel-error
concentration).

## The honest verdict — it depends entirely on the judge

1. **Under SSIM2 (the neutral third judge), zensim's diffmap is the BEST RD
   driver** — beats SSE by ~+5.9 and butteraugli by ~+2.9, consistently across
   both block sizes. If your quality target is SSIM2-like structural quality,
   ranking blocks by the zensim diffmap spends bits better than either the
   codec's SSE default or butteraugli's deployed diffmap.
2. **Under butteraugli (max & 3-norm), zensim's diffmap is the WORST of the
   real selectors** — SSE and butteraugli win. butteraugli rewards fixing the
   single peak-error region; the zensim diffmap distributes attention by
   structural error and largely ignores that peak.
3. **The two perceptual selectors are near-orthogonal** (zensim∩butter ~34%).
   zensim's diffmap ≈ SSIM2's spatial sensitivity (both structural metrics);
   butteraugli's ≈ SSE's (both peak/pixel-error). "Which diffmap is the better
   RD driver" has no answer independent of "which metric are you targeting."

## The surprising internal finding — zensim diffmap ↔ scalar disagree

**The zensim-selector LOSES on its own zensim-B judge** (+5.36 vs sse +7.20 at
block 32; +5.71 vs +9.00 at block 16). Refining where zensim's *diffmap* says
error is highest improves the zensim *scalar score* LESS than refining SSE-picked
blocks does. So the spatial diffmap (`DiffmapWeighting::default()`, Σ per block)
and the pooled scalar disagree about which regions dominate the score — plausibly
the scalar is peak-weighted while the diffmap sum is total-weighted. **If we want
a zensim-diffmap-driven RD loop that optimizes the zensim score, the diffmap
weighting must be aligned to the scalar pooling** — an open, concrete follow-up.

## Caveats (do not over-read)

- "Refine = copy reference" is a clean upper-bound proxy for "spend unlimited
  bits here," not a real re-encode. Directionally sound for *which blocks*, not a
  bitrate claim.
- n=50, two q levels (15/30), one codec (mozjpeg). Not a size/quality/content
  sweep — this is a mechanism probe, not a calibration.
- zensim↔SSIM2 kinship: both are structural multi-scale metrics, so their spatial
  agreement is partly expected. The result is "structural-diffmap RD helps a
  structural judge," which is honest but not a free lunch versus butteraugli's
  own target.

Data: `/mnt/v/output/zensim/rd-eval/matrix_b{32,16}.txt`. Harness is Rust
(diffmaps + blocks) + zenmetrics (judges); Python only fans out.
