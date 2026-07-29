# Guided mass-conserving redistribution of coarse-scale diffmap contributions (E-JBU) — 2026-07-30

**Verdict: TBD (protocol pre-registered before any implementation run; results appended below).**

Sibling-workspace experiment (`zensim--diffmap-jbu`, based on C1 `12774c42`), run in
parallel with the C-series attribution-map build by explicit user sanction. Zero changes
to scoring, features, or the attrmap module; the render option is opt-in and default OFF.

## Idea under test

E-M6 (`342b7b1e`): 924-era training puts ~89% of basic gradient mass on coarse-scale MSE
(s2+s3), so the mass-blended signal fold renders at effectively 1/8 resolution — M3
collapses while M2 stays 0.99, and E-M8b showed the coarse preference is a data
equilibrium training can't move. Render-time fix candidate: the coarse scales carry mass
but not localization, while the fold already computes a full-res localization signal (the
scale-0 per-pixel contribution plane). For each coarse-scale contribution cell (s1=2×,
s2=4×, s3=8× footprints), instead of NN/box upsampling, redistribute the cell's mass
within its footprint proportional to the fine guide:

```
out(x,y) += cell_mass · g(x,y) / Σ_cell g        g = |s0 contribution plane| + ε
```

Properties required by construction: (a) per-cell mass conservation → the scalar score
and footprint-aggregated attribution unchanged (verified numerically; f32 drift
reported); (b) ε-fallback → flat guide reproduces current NN behavior; (c) O(N), off the
scoring hot path (diffmap render only).

## PROTOCOL (pre-registered 2026-07-30, before any result was produced)

### Models

| bake | regime | role |
|---|---|---|
| `/mnt/v/output/zensim/bakes/coherent-089/em2/EM2_fold924_s99.bin` | 924 | pathological coarse-MSE bake (E-M9 row 1) |
| `/mnt/v/output/zensim/bakes/coherent-089/em2/EM4_mask2_kw0.15_s42.bin` | 924 | selected candidate, pathological (E-M9 row 2) |
| `/mnt/v/output/zensim/bakes/p1kadis/foldmlp_bigcodec_kadis_720.bin` | 720 | healthy baseline (C1's no-regression bake) |

Pair: `city.png` vs `city_576_q50.jpg` (576², `/mnt/v/output/zensim/diffmap-coherence-2026-07-18/`)
— the E-M9/C1 anchor pair. Harness: `diffmap_block_coherence --bake` (q50 spatial-SROCC
protocol per `diffmap_coherence_2026-07-16.md`; its ~0.66 pooling ceiling caveat applies:
the scalar is non-additive, so no per-pixel map reaches 1.0).

### Primary endpoint

M3 = SROCC(model-sensitivity map block sums, ΔS_bake) before vs after redistribution, at
blocks {16, 32, 64, 128} (the E-M9 grid), on all 3 bakes, computed in ONE harness process
per (bake, block) so ΔS/s_k/M2 are shared exactly between arms (`ZENSIM_JBU_AB=1`).

- **Success bar**: registered ≥ +0.05 at the failing granularities (EM2@128 = −0.36,
  candidate@128 = −0.08, K720@64 = +0.003 are the collapse/inversion cells).
- **M2 must be bit-unchanged** (shared computation; reported as exact).
- **Scalar drift** must be ≤ 1e-6 relative or exactly 0 (the diffmap option must not
  perturb scoring).
- **Mass-conservation gate**: total-map relative drift and per-block drift reported
  (f32 summation-order drift expected ~1e-6 or below).

### Pre-registered structural prediction (stated before running)

The current fold's coarse upsample is NN-replicate (`upsample_pow2x_add`), and the
pyramid halves by aligned 2×2 box (`downscale_2x_inplace`, floor dims), so a scale-s
cell's footprint is exactly `[sx·2^s, (sx+1)·2^s) × [sy·2^s, (sy+1)·2^s)` — grid-aligned,
max 8×8. Every E-M9 evaluation block (16-128px, grid-aligned multiples of 8; 576 is
divisible by 8 so no cell straddles the trim) therefore contains WHOLE footprints only.
**Any per-cell mass-conserving redistribution is invisible to aligned block sums ≥ the
footprint size: the primary endpoint is predicted to be EXACTLY null (f32 noise only),
by construction.** If the measurement contradicts this, the implementation (or the
alignment analysis) is wrong — that is what the drift gates detect.

The informative measurements are therefore pre-registered at the granularities where
redistribution CAN move mass across evaluation-cell boundaries:

### Sub-footprint + per-pixel endpoints (where the mechanism can bite)

1. **M3 at block=4** (s3 footprints straddle 4 aligned 4px blocks → s3 mass is mobile
   across blocks): EM2_fold924_s99 only, budget-capped — per-refine cost measured at
   block=16 first, projected count-linearly to 20 736 refines; run only if ≤ ~3 h,
   else report the measured estimate and skip honestly. (block=8 is still invariant by
   the same theorem — covered by a unit test, not a paid run.)
2. **Per-pixel A/B stats** (every bake × block run): fraction of pixels changed,
   max |Δpixel|, per-pixel SROCC(map_off, map_on) — demonstrates the map DOES change
   sub-footprint while block sums don't.
3. **Unit-level exactness tests** (committed): per-cell conservation vs NN on synthetic
   pairs (aligned 8px block sums equal within f32 tolerance, per-pixel maps differ);
   uniform-guide ⇒ NN-equivalence (ε-fallback); non-multiple-of-8 dims edge clipping.
4. **Visual A/B**: diffmap PNGs (shared normalization) to
   `/mnt/v/output/zensim/diffmap-jbu-2026-07-30/` (block storage, not git).

### Secondary (only if the primary pays, which the prediction says it cannot)

Guide-variant A/B (s0-only vs s0+s1 blend); hierarchical cascade (s3→s2→s1). Note: any
cascade that conserves mass per cell at each level is equally block-invariant at ≥8px,
so these fire only if the primary shows movement (i.e. only if the prediction is wrong).

### Perf

ms/MP of the redistribution pass at 1 MP (1024²) and 12 MP (4000×3000): median of 4
runs of `compute_with_diffmap` with the option OFF vs ON on a synthesized noise+gradient
pair, `nice -n19 ionice -c3`, default release build (runtime SIMD dispatch, no
`-C target-cpu=native`). Redistribution cost = ON − OFF medians; reported per-MP.

### Honest-null clause

If M3 doesn't move at the primary grid (as predicted), the finding is the diagnosis
itself: **coarse mass is mislocated at the CELL level (which 8×8 cells carry mass, and
with what sign), not merely coarsely spread** — no render-time redistribution that
conserves per-cell mass can repair block-level steering at codec-aligned partition sizes
≥ 8px. That sharpens E-M6's "effectively 1/8-res" mechanism: the fold's block-level
failure at 64/128px is cell-value placement (coarse-plane MSE ranks blocks wrongly),
and the render-time win from redistribution is confined to sub-8px localization
(per-pixel visualization sharpness, 4px partitions).

## Implementation (this workspace)

- `DiffmapOptions.guided_coarse_redistribution: bool` (gated `custom-profiles`, default
  `false`; OFF path byte-identical to the pre-change fold).
- `redistribute_pow2x_guided_add` in `streaming.rs`: per-cell two-pass (guide-sum, then
  deposit `mass·g/Σg`), f64 per-cell accumulation, guide = |scale-0 plane| + ε with
  ε = 1e-6·mean|s0| + 1e-20 (global-scale-invariant per cell; all-zero guide ⇒ uniform ⇒
  NN behavior).
- Harness: `ZENSIM_JBU_AB=1` computes both arms in one process; `ZENSIM_JBU_DUMP=prefix`
  writes the PNG A/B.

## Results

TBD — appended after the runs, never edited into the protocol above.
