# V39 compression-quality monotonicity characterization (2026-05-27)

**Read-only analysis. No retrain, no weight swap, no test change.** This
doc characterizes the shipped zensim bake's quality-monotonicity failure
*across encode quality on compression distortions*, using the fresh
334,080-row dense zenjpeg picker sweep — the first substrate we've had that
pairs `metric_zensim_gpu` and `metric_ssim2_gpu` per cell across dense q,
16 sizes, 6 content classes, and 36 codec cells.

- **Shipped bake:** `zensim/weights/v39_v32plus_spline_seed17_2026-05-25.bin`
  (`ZensimProfile::A` / `PreviewV0_3`).
- **Data:** `/mnt/v/zen/picker-dense-full-2026-05-27/parquet/picker_dense_full_zenjpeg.parquet`
  (334,080 rows; 20 sources × 16 sizes × 36 cells × 29 q-points; codec = zenjpeg).
- **Content classes:** derived from source-name prefix — `gen-chart`,
  `gen-doc`, `gen-line`, `gen-mixed`, `gen-screen`; hex-named sources = `photo`.
  (5 sources each for chart/line/mixed, 1 each for doc/screen, 3 photo;
  16 size variants each.)
- **Script:** `zensim/scripts/analyze_compression_monotonicity.py`.
- **Full output:** `/mnt/v/zen/picker-dense-full-2026-05-27/monotonicity_analysis/`
  (`per_curve_monotonicity.tsv` 11,520 curves, `summary.txt`).

A *curve* = one (source, size, cell) triple, scores ordered by q ascending.
A *q-step reversal* = an adjacent-q step where the score **decreases as q
increases** — a monotone quality metric should have ~0. ssim2 is the monotone
reference. (Note: the GPU sweep's `metric_ssim2_gpu` is mis-scaled to large
negatives on some synthetic-content sources — likely a ref-mismatch in the
sweep harness, unrelated to zensim — but each ssim2 curve is still internally
monotone and used only as the within-curve reference; the photo ssim2 values
are clean and behave correctly.)

## Headline

| Metric | q-step reversal rate | mean drop | median drop | max drop | curves that collapse (q-high < q-low) |
|---|--:|--:|--:|--:|--:|
| **zensim (V39)** | **69.11%** | 9.97 | 7.08 | 119.10 | **95.26%** (10,974/11,520) |
| ssim2 (reference) | 20.02% | 1.74 | 0.56 | 89.71 | 18.12% (2,088/11,520) |

**The shipped dial runs backwards on compression quality.** On 69% of
adjacent-q steps zensim *decreases* as encode quality *increases*, with a mean
drop of ~10 score points. Across full curves, **95% score the best-quality
encode lower than the worst-quality encode.** ssim2's 20% step-reversal is
mostly sub-point noise (median 0.56 pt) at the saturated top of curves; its
full-curve collapse is 18%.

### The decisive q90-vs-q30 check (the picker's framing)

Comparing q=90 vs q=30 directly on each curve (where both q-points exist):

| | q90 scored LOWER than q30 | median(score₉₀ − score₃₀) |
|---|--:|--:|
| **zensim (V39)** | **86.4%** (9,948/11,520) | **−143.51** |
| ssim2 | 16.2% (1,868/11,520) | (positive — correct) |

zensim ranks a q90 encode **143 score-points worse** than a q30 encode on the
median curve. This is not a wiggle — it is a near-total inversion of the dial.

## Where it concentrates

### By content class (worst → best reversal rate)

| class | zensim rate | zensim mean-drop | zensim max-drop | ssim2 rate | q90<q30 inversion |
|---|--:|--:|--:|--:|--:|
| mixed | 73.2% | 8.88 | 119.10 | 17.1% | 89.0% |
| photo | 71.5% | 9.44 | 81.42 | 4.4% | 82.2% |
| chart | 71.0% | 8.99 | 99.51 | 26.6% | 82.4% |
| screen | 70.7% | 11.36 | 91.98 | 20.6% | 85.4% |
| line | 63.6% | 11.96 | 91.20 | 22.9% | 87.8% |
| doc | 58.0% | 12.10 | 100.86 | 33.6% | 99.3% |

The defect is **near-uniform across content class** (58–73%), with *photo*
right in the middle at 71.5% — so this is NOT a synthetic-content-only
problem. Photo's ssim2 reference rate is the cleanest (4.4%), which makes the
zensim/ssim2 gap *largest* on photo. doc has the lowest step-rate but the
highest full-collapse (99.3%) — its curve declines smoothly-but-monotonically
backwards rather than zig-zagging.

### By q-band (lower-q endpoint of each step)

| q-band | zensim reversal rate | mean drop | max drop |
|---|--:|--:|--:|
| low (q<30) | 34.6% | 14.00 | 119.10 |
| mid (30≤q<70) | 70.7% | 14.19 | 101.40 |
| high (q≥70) | 79.8% | 7.39 | 99.51 |

**Reversals intensify as quality rises.** The high-q band (q≥70 — the
"visually-lossless" product regime where users live) reverses on 80% of steps.
Low-q reverses less often but with larger individual drops.

### By size class

| size group | zensim reversal rate |
|---|--:|
| tiny (sz32–48) | ~15% |
| small (sz64–160) | ~80% |
| medium (sz192–512) | ~83% |
| large (sz640–1024) | ~80% |

**A sharp cliff at sz48→sz64.** Tiny images (≤48px) stay near-monotone (15%);
everything ≥64px is catastrophically non-monotone (~80%). The q-band × size
cross-tab makes the interaction explicit:

| reversal rate | tiny(≤48) | small(64–160) | med(192–512) | large(>512) |
|---|--:|--:|--:|--:|
| low q<30 | 0.0% | 43.1% | 43.6% | 39.9% |
| mid 30–70 | 2.2% | 82.1% | 90.4% | 87.3% |
| high q≥70 | 27.1% | 91.5% | 92.5% | 91.7% |

For any non-tiny image at mid-or-high q — **the entire production regime** —
the reversal rate is 82–93%.

### By codec cell (subsampling / progressive / sharp_yuv / effort)

The codec knobs are **not** a meaningful factor. All 36 cells sit in a narrow
67–70% band; axis marginals are flat:

- `progressive` true vs false: **identical** (69.11% both).
- `sharp_yuv` true vs false: 69.07% vs 69.16%.
- `effort` 0/1/2: 68.4% / 69.5% / 69.5%.
- `subsampling` 420/422/444: 69.6% / 68.3% / 69.5%.

**The defect is a property of the metric × (content, q, size), independent of
the encode cell.** No cell choice rescues monotonicity.

## Worst-case exemplars (for the fix to target)

Concrete q-by-q traces at sz256, cell `420/effort0/no-prog/no-sharp`:

**Photo** (`5e5ce43575fa67fdc0dd37146d7f479e_1024sq` @ sz256) — the cleanest
demonstration (ssim2 reference is well-behaved):

```
q:     5    10    15    20    25    30 ... 70    80    90    96   100
zen: 100.0 100.0 100.0 96.0  88.6  83.6 ... -13.7 -39.4 -87.8 -118.5 -123.1
ssim2:29.0 29.0  31.3  39.6  44.8  50.0 ... 68.8  73.7  79.7  84.6  86.0
```

zensim runs from **100 at q5 to −123 at q100** (monotone *backwards*, a 223-pt
inversion) while ssim2 correctly climbs **29 → 86**.

Largest single-step drops (from `summary.txt`):

| max single-step drop | class | size | source / cell |
|--:|---|---|---|
| 119.10 | mixed | sz96 | `gen-mixed__00113_s0718c85b` / 444 |
| 101.40 | mixed | sz48 | `gen-mixed__00134_s669bfc42` / 422 |
| 100.86 | doc | sz192 | `gen-doc__00191_s3fb88ea0` / 420 |

These curves swing through 180–199 score points within a single curve
(e.g. mixed sz48 spans −100.6 → +100.0).

## What this is — and what's NEW vs the recovery-cycle docs

### Relation to the existing axiom-violation finding

`benchmarks/ROOT_CAUSE_v39_invariant_violations_2026-05-26.md` and
`docs/METRIC_INVARIANTS_MECHANISM_AND_REDESIGN_2026-05-26.md` already
characterize a related defect: V39's unconstrained MLP *inverts off-manifold*
(blur5 > blur1 > identical) and emits scores >100 / <0 via an unbounded
~1000×-extrapolating spline. That analysis was on **controlled-degradation
synthetic ladders** (mandelbrot/checkerboard + blur) — the ROOT_CAUSE doc
explicitly flags that "no controlled-degradation monotonicity sweeps" and
"natural photos only" validation let the defect ship.

**This doc is the in-distribution, real-codec manifestation of that same
mechanism — and it is NEW evidence in three ways:**

1. **It's on actual compression output, dense across quality**, not synthetic
   blur ladders. The defect is not an off-manifold corner case — it is the
   metric's behavior on the exact thing it's a dial *for* (zenjpeg q-sweeps).
2. **It quantifies the across-quality dimension** the prior docs lacked:
   reversal rate by q-band, by size, by content class, with magnitude, plus
   the direct q90-vs-q30 inversion (86% / −143 pts). The SESSION-RESUME notes
   a V39 "67.7% / 53.6% tied" number from the 50-image × 19-q JPEG sweep;
   this 334k-row sweep confirms it at far greater resolution (69.11% reversal)
   and adds the per-content/size/q/cell breakdown + severity that sweep
   lacked.
3. **The same out-of-bounds spline mechanism is visible here**: scores reach
   −132 and the curves run monotone-backwards, exactly matching the
   ROOT_CAUSE finding that the spline extrapolates unbounded once the MLP
   leaves its narrow trained band. The *mechanism is confirmed identical*;
   the *surface (in-distribution codec q-sweeps)* is new.

### Relation to G5 (KonJND HF rank)

`v39_verdict_2026-05-27.md` shows V39 fails **G5** (KonJND+AIC-3 HF SROCC,
0.42/0.80) — a human-MOS *rank* failure at near-lossless quality.
`SOTA_TRAILS.md` / CLAUDE.md characterize G5 as a Pareto limit needing a
better HF representation. **This monotonicity defect is the within-curve
shadow of the same high-q weakness**: the q≥70 band is both where KonJND HF
rank collapses (G5) AND where the q-step reversal rate peaks at 80%. They are
the rank-vs-absolute two views of one high-quality-regime failure. G5 says
"can't rank near-lossless pairs across images"; this says "can't even order
its own quality ladder within one image at high q."

## Implications

### (a) What the v39 fix must target

- **Regime: any non-tiny image (≥64px) at mid-or-high q (q≥30).** That's
  where 82–93% of reversals live and where every product decision is made.
  The tiny/low-q corner is already fine — the fix should not regress it but
  need not focus there.
- **Content: all classes**, not a synthetic-only patch. Photo is at 71.5%;
  a fix that only addresses chart/line/screen would leave the core
  product (photo web compression) broken.
- **Mechanism: the unconstrained MLP's inversion + unbounded spline
  extrapolation** (per ROOT_CAUSE). The fix candidates already named there —
  partial-monotone-by-construction MLP + bounded squash + clamped-extrapolation
  calibration, plus an invariant/degradation-ladder gate in `bake_verdict` —
  are exactly what this evidence demands. **This dense q-sweep should become a
  monotonicity acceptance gate**: a candidate bake's per-curve q-step reversal
  rate must approach ssim2's (~20%, mostly sub-point) before it can ship as the
  product dial. The current `tests/metric_invariants.rs` controlled-ladder gate
  is necessary but this real-codec q-sweep is the in-distribution complement.

### (b) Can the picker ever target zensim instead of ssim2?

**Not with V39. The picker was correct to target ssim2.** A picker that
binary-searches q for a target zensim score is searching a function that
decreases with q on 86% of curves — there is no usable monotone region above
sz48/q30 to invert. Until a candidate bake brings the q-step reversal rate
down to roughly ssim2's level (≈20%, sub-point magnitude) on this same dense
sweep — measurable instantly by re-running the script's q-step accounting on
the candidate's scores — **the picker must keep targeting ssim2** (or a
provably-monotone zensim variant such as `PreviewV0_2`, the linear
no-MLP profile, which the ROOT_CAUSE doc confirms stays monotone+bounded).
The path to "picker targets the product dial" runs *through* the
monotone-by-construction V39 retrain, not around it.

### (c) Net

This is the #1 non-speed goal restated as a measurable acceptance criterion:
**reversal rate ≤ ~ssim2 on the 2026-05-27 dense sweep, all content/size/q.**
The defect is in-distribution, severe (median −143 pt q90-vs-q30), and
content-universal; the existing monotone-by-construction retrain direction
(SESSION-RESUME "Monotone-by-construction A retrain") is the correct fix, and
this sweep is the gate that will tell us when it's done.
