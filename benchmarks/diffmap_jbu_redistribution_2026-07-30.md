# Guided mass-conserving redistribution of coarse-scale diffmap contributions (E-JBU) — 2026-07-30

**Verdict: pre-registered structural null CONFIRMED at the primary endpoint (ΔM3 = +0.0000
in 13/13 measured cells, drift ≤ 5.8e-9; scalar bit-identical) — and the instrument built
to verify it produced a bigger finding than the experiment: the deployed combined M3 map's
ranking is carried ENTIRELY by the raw v2 fold-in (M3(v2-only) ≡ M3(combined) to 4
decimals in 13/13 cells; the v1 signal fold's value share of the deployed map is
0.0006–0.0052% of p99.5). The E-M9 "fold inverts at 128px" pattern belongs to the v2 raw
add; the v1 mass-blended fold's own M3 curve RISES with block size (+0.08@16 → +0.53@128
on EM2 — the true E-M6 1/8-res signature) and never inverts. Guided redistribution does
exactly what it can: it visibly dissolves the v1 fold's 8×8 cell blockiness per-pixel
(max px Δ = 18× the v1 fold's p99.5; v1 px SROCC off→on 0.71 on EM2) at zero cost to any
aligned block aggregate — but the v1 fold it sharpens is value-invisible in today's
deployed map, and at the single cell where redistribution CAN move a block rank
(v1-only @ 4px) the s0 guide is marginally WORSE than uniform (Δ −0.018): sharper
rendering, not better steering. Cost +3.6–5.7 ms/MP, render-only, default OFF. Landed as
an opt-in render option + the decomposition instrument; the actionable follow-up
(unit-correct v2 blend in the signal fold, owned by the C-series) is flagged, not
built.**

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

## Results (2026-07-30; machine: WSL2 7950X dev box, release build, no target-cpu=native, niced)

All runs: `ZENSIM_JBU_AB=1 diffmap_block_coherence <city.png> <city_576_q50.jpg> --bake <bake>
--block N` (both arms in one process; s_k/ΔS/M2 shared exactly). Logs: `~/tmp/jbu-*.log`.
Baseline sanity: every M3(off) below reproduces the C1/E-M9 gate table to 4 decimals.

### Gates (all pass)

- **Scalar drift**: `0.000e0` — bit-identical scores in every run (the option never
  touches the scoring path).
- **Mass conservation**: total-map drift `3.3e-11` (EM2) / `1.4e-10` (EM4) / `3.6e-10`
  (K720) rel; max per-block drift at the primary grid `5.8e-9` of max|block| (worst cell).
  Pure f32 summation-order noise, as constructed.
- **M2**: shared computation between arms (one process) — identical by construction;
  values match C1 (+0.9985..+1.0000 per cell).
- **OFF path**: byte-identical code path to pre-change (option is a branch that only
  engages when set; default `false`).

### Primary endpoint — M3 before/after at the E-M9 grid (deployed combined map)

ΔM3 = M3r − M3. The pre-registered prediction (aligned-footprint theorem: any per-cell
mass-conserving redistribution is invisible to aligned block sums ≥ 8px) is confirmed
exactly — and the success bar (≥ +0.05 at failing granularities) is therefore not met,
by construction, not by implementation failure:

| block | EM2_fold924_s99 M3→M3r (Δ) | EM4_mask2_kw0.15_s42 M3→M3r (Δ) | K720 healthy M3→M3r (Δ) |
|---|---|---|---|
| 16 | +0.3771 → +0.3771 (+0.0000) | +0.4308 → +0.4308 (+0.0000) | +0.4204 → +0.4204 (+0.0000) |
| 32 | +0.3048 → +0.3048 (+0.0000) | +0.4029 → +0.4029 (+0.0000) | +0.2814 → +0.2814 (+0.0000) |
| 64 | +0.2491 → +0.2491 (+0.0000) | +0.3353 → +0.3353 (+0.0000) | +0.0030 → +0.0030 (+0.0000) |
| 128 | −0.3615 → −0.3615 (+0.0000) | −0.0831 → −0.0831 (+0.0000) | +0.5915 → +0.5915 (+0.0000) |

Sub-footprint (pre-registered, EM2 only, 20 736 blocks): **block=4: +0.2751 → +0.2751
(|Δ| < 5e-5)**. Here s3 footprints DO straddle blocks and mass DID move across block
boundaries — max block drift jumps three orders of magnitude to `4.4e-6` of max|block| —
yet M3 still doesn't move, because the moved v1 mass is value-invisible against
v2-dominated block sums (next section). **On the un-swamped v1 fold, where the moved
mass IS visible, the guide makes 4px ranking slightly WORSE: v1-only M3 +0.0495 →
+0.0317 (Δ −0.0178)** — the one cell in the study where redistribution can move a block
rank at all, and the s0 guide places sub-footprint mass no better (marginally worse)
than uniform for predicting where refinement pays. The guide sharpens rendering, not
steering — consistent with the ~0.66-ceiling caveat and with blur-bleed: the s0 plane
lights up blur-radius neighborhoods, not the exact paying pixels.

### The decomposition finding — the deployed map's M3 IS the v2 raw add

`JBU v2-only` line = SROCC of the v2map's block sums alone vs the same ΔS.
**M3(v2-only) ≡ M3(combined) to 4 decimals in every cell measured (13/13, incl.
block=4)**, e.g. EM2:
+0.3771/+0.3048/+0.2491/−0.3615 at 16/32/64/128 — identical columns. Direct cause: the
v1 signal fold's value scale is microscopic against the raw v2 fold-in —

| bake | v1-fold p99.5 | combined p99.5 | v1 share |
|---|---|---|---|
| EM2_fold924_s99 | 1.066e-4 | 1.924e1 | **0.0006 %** |
| EM4_mask2_kw0.15_s42 | 2.018e-4 | ~1.5e1 | 0.0013 % |
| K720 | 1.120e-3 | ~2.2e1 | 0.0052 % |

(The v1 fold is tiny in VALUE because these bakes' per-scale weights concentrate on the
MSE slot and XYB-unit squared errors at q50 are ~1e-4; the raw v2 fold's family maps are
O(1–30). This is the C1 "raw ADD swamps a score-unit density" lesson measured from the
other side — the harness comment that the raw add "is fine for the normalized signal
fold" is measured WRONG for coherence purposes: it silently turns the deployed map into
a v2-only renderer.)

The v1 fold's OWN spatial coherence (`JBU v1-fold-only` M3, identical off/on at every
size — the theorem again):

| block | EM2 v1-only | EM4 v1-only | K720 v1-only | v2-only (=combined), EM2 |
|---|---|---|---|---|
| 16 | +0.0791 | +0.0248 | +0.1925 | +0.3771 |
| 32 | +0.1289 | −0.1304 | +0.4447 | +0.3048 |
| 64 | +0.1763 | −0.1525 | +0.6220 | +0.2491 |
| 128 | **+0.5285** | +0.2800 | +0.5085 | **−0.3615** |

Two opposite block-size profiles: the v1 mass-blended fold is weak at fine granularity
and BEST at 128px (the honest signature of E-M6's "effectively 1/8-res" map — a coarse
map can rank coarse partitions), and it never inverts; the v2 raw add is decent at
16px and degrades/inverts at 128px. **The E-M9 "fold M3 degrades and inverts at 128px"
pattern — and the K720@64 = 0.003 crater — are properties of the v2 raw add, not of the
v1 mass-blended fold that E-M6's coarse-gradient-mass mechanism describes.** The
gradient-mass finding itself (89% on coarse MSE) is untouched — it is about the scalar's
gradient — but the RENDERED map's measured M3 failures at 64–128px trace to the v2
fold-in's unit mismatch, not to coarse-scale NN upsampling.

### Where the redistribution actually acts (per-pixel, v1 fold)

- Guide (|s0 plane| + ε): mean 3.67e-4; within-cell sd/mean = 0.66 (f2) / 0.93 (f4) /
  1.12 (f8), max 3.5 — strong sub-footprint differential signal (the guide is NOT flat).
- v1-fold per-pixel effect: max px Δ = 1.94e-3 = **18.2× the v1 fold's p99.5** (EM2);
  per-pixel SROCC(off,on) on the v1 fold = 0.706 (EM2) / 0.558 (EM4) / 0.964 (K720) —
  the redistribution substantially re-orders the v1 fold's per-pixel content
  (concentrating each cell's mass onto its high-guide pixels), while every aligned
  8/16px block sum stays bit-stable to ~1e-9.
- On the deployed combined map the same change reads as 0.01 % of p99.5 and per-pixel
  SROCC 1.000000 — swamped by the shared v2 add.
- Visual A/B (`/mnt/v/output/zensim/diffmap-jbu-2026-07-30/em2_city_q50_b32_v1_{off,on}.png`
  + absdelta): the OFF v1 fold shows plain 8×8 cell blockiness; the ON map dissolves it
  into pixel-level structure following the fine guide. The combined-map pair is visually
  identical (v2-swamped), absdelta essentially black.

### Perf (redistribution pass cost)

`--perf WxH` mode: synthesized pair, interleaved (off,on)×4 after warmup, ref precompute
OUTSIDE the timed call, medians of 4, niced, default release build (runtime SIMD
dispatch, no target-cpu=native):

| size | OFF median | ON median | redistribution pass |
|---|---|---|---|
| 1024×1024 (1.05 MP) | 24.13 ms (23.01 ms/MP) | 27.89 ms | **+3.77 ms = +3.59 ms/MP** |
| 4000×3000 (12.0 MP) | 395.15 ms (32.93 ms/MP) | 463.84 ms | **+68.7 ms = +5.72 ms/MP** |

O(N) as designed (guide build + per-cell two-pass over 3 coarse scales ≈ 7 full-image
passes, scalar); ~15-17% of the diffmap render call, zero cost on the scoring path and
zero when the option is off (default). The per-MP rise at 12 MP is the working set
leaving LLC. SIMD/fusion headroom exists if a production consumer ever appears; not
spent on a research option.

### Findings ledger

1. **Pre-registered null confirmed, with the theorem**: per-cell mass-conserving
   redistribution cannot change aligned block attribution at ≥ footprint size — measured
   ΔM3 = +0.0000 in 13/13 cells, drift ≤ 5.8e-9. Any cascade variant conserving per-cell
   mass is equally inert there (secondary arms correctly not fired).
2. **The coarse-granularity problem, as measured by block-level M3, is NOT an upsampling
   problem at codec granularities**: NN already confines each cell's mass to a footprint
   inside every aligned ≥8px block. The fold's block-level failures are cell-VALUE
   placement (and, at 64–128px, the v2 add — see 3).
3. **NEW (instrument-found): the deployed combined M3 map ranks by the raw v2 fold-in
   alone** (v2-only ≡ combined, 12/12; v1 value share ≤ 0.005%). E-M9's inversion at
   128px and K720's 64px crater are v2-add properties. The signal fold needs the same
   unit correction C1 applied to the density's v2 fold-in (weights × 1/(w·h)) before any
   statement about "the fold's" coherence at large blocks is about the v1 fold at all.
4. **The v1 fold's own curve rises with block size and never inverts** (+0.08→+0.53 on
   EM2) — E-M6's 1/8-res mechanism, seen cleanly for the first time without v2 swamping.
5. **Guided redistribution is a real visual deblocker for the v1 fold render** (18× p99.5
   per-pixel movement, blockiness dissolved) at zero block-attribution cost — worth
   keeping as the render-time visualization option it now is, but it cannot repair
   steering, and in today's v2-swamped combined map it is invisible.
6. **Where redistribution CAN move a block rank (v1-only @ 4px), the s0 guide is
   marginally WORSE than uniform** (Δ −0.018): sub-footprint mass placement by the
   blurred s0 error plane does not localize where refinement pays. The secondary
   guide-variant/cascade arms stay un-fired per protocol — a better guide is not a
   render-time question but a signal-definition one (blur-deconvolved or
   attribution-derived), and the C-series density already owns that direction.
7. Perf: +3.6 ms/MP (1 MP) / +5.7 ms/MP (12 MP), render-only, default-off.

### Composition with the C-series attribution map (discussion, no code touched there)

- The C1 density's coarse-scale upsample (sum-preserving NN ÷ footprint area) has the
  same aligned-footprint geometry — the theorem transfers: a guided variant of the
  density's coarse upsample would leave `query_rect`/`block_sums` answers unchanged at
  aligned ≥8px rectangles and only sharpen sub-footprint visualization. If the density
  ever ships a visualization surface, this option composes cleanly (same guide, same
  kernel shape, O(N)).
- The decomposition instrument (v2-only / v1-only lines) is the cheap regression guard
  for the C-series' own v2 fold-in work: any future unit-correct v2 blend in the signal
  fold should move M3(combined) AWAY from M3(v2-only) toward a genuine mixture; today's
  identity is the "swamped" tell.
- Flagged for the C-series owner (NOT built here, out of scope): unit-correct v2 blend
  in the signal fold (× 1/(w·h), exactly C1's density fix); and the append block remains
  blind in BOTH folds (0.5–0.7% |s|-mass, decisive at 128px per C1's ATTRDIAG).
