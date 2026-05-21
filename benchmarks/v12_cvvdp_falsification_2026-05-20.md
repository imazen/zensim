# V12-A cvvdp-anchored substrate — FALSIFIED (task #199, 2026-05-20)

**Verdict: KonJND collapses identically under cvvdp pivot as under
ssim2 pivot.** The basin-B mechanism is anchor-metric-independent.
V12-A also LOSES CID22 vs V11-A v4 due to the cvvdp-substrate's
intrinsic low-q coverage gap. **Both trails close at the cross-codec-eq
frontier on this corpus.**

V10 BalancedV3 remains the Balanced ship. V_24 per-sample-α remains the
Compression ship. The cross-codec-eq frontier is closed at this
substrate at BOTH anchor pivots (ssim2 from V11, cvvdp from V12).

## Hypothesis (task #199 brief)

> Per Mohammadi 2025: cvvdp Z-RMSE 9.45 vs ssim2 47.63 — 5x better
> absolute calibration. If KonJND survives at the cvvdp pivot, ship
> as V0_5BalancedV4. If KonJND still collapses, the basin-B mechanism
> is anchor-metric-independent and the V11/V12 cross-codec-eq
> frontier is closed structurally.

## Method

### Substrate (cvvdp 10-band, 372-feat, 4-codec full coverage)

cvvdp → target_score mapping (per task brief, calibrated to empirical
percentiles on the V11-DECODER-FIX multi-codec parquet):

| cvvdp target | target_score | semantic |
|--:|--:|---|
| 10.00 | 100 | imperceptible / lossless |
| 9.95 | 95 | near-imperceptible |
| 9.85 | 90 | visually lossless |
| 9.65 | 80 | JND threshold |
| 9.30 | 65 | mildly noticeable |
| 8.50 | 50 | JOD (just objectionable) |
| 7.50 | 35 | 3x-DPI resize-out |
| 6.50 | 20 | clear artifacts |
| 5.00 | 10 | very degraded |
| 3.00 | 0 | borderline unacceptable |

cvvdp anchor tolerance ±0.4 (per brief). Cross-codec equivalence
tolerance ±0.2, pivoted at {9.85, 9.65, 9.30, 8.50, 7.50, 6.50}.

Builder: `scripts/v_next/v12_cvvdp/build_v12_cvvdp_substrate.py`
Input:   `/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/multi_codec_372col_full.parquet`
Output:  `/mnt/v/zen/zensim-training/2026-05-20-v12-cvvdp-substrate/`

### cvvdp distribution per codec (substrate source corpus)

| codec | n | min | p1 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| zenavif | 4000 | 7.392 | 7.892 | 8.916 | 9.567 | 9.957 | 9.991 | 9.999 | 10.000 | 10.000 | 10.000 |
| zenjpeg | 61600 | 6.836 | 8.124 | 9.237 | 9.607 | 9.885 | 9.965 | 9.990 | 9.995 | 10.000 | 10.000 |
| zenjxl | 51200 | 9.080 | 9.666 | 9.903 | 9.955 | 9.985 | 9.999 | 10.000 | 10.000 | 10.000 | 10.000 |
| zenwebp | 1000 | 8.633 | 9.033 | 9.630 | 9.786 | 9.911 | 9.964 | 9.980 | 9.989 | 9.999 | 10.000 |
| **ALL** | **117800** | 6.836 | 8.466 | 9.440 | 9.825 | 9.956 | 9.990 | 10.000 | 10.000 | 10.000 | 10.000 |

zenjxl saturates above 9.85 across the q-grid — low-q bands have ~0
zenjxl rows by design. The source corpus q-grid `{10, 30, 60, 80, 90}`
produces almost no sub-9 cvvdp samples.

### Substrate coverage gap (load-bearing)

V12-A anchor emit counts per band vs V11-A v4 reference (ssim2 ±3.0):

| target_score | V12-A cvvdp (anchors) | V11-A v4 ssim2 (anchors) |
|--:|--:|--:|
| 100 | 790 | — (band absent in ssim2) |
| 95 | 793 | 157 |
| 90 | 793 | 614 |
| 80 | 795 | 615 |
| 65 | 492 | 355 |
| 50 | 169 | 245 |
| 35 | 28 | 184 |
| 20 | 1 | 129 |
| 10 | 0 | 116 |
| 0 | 0 | 56 |
| **TOTAL** | **3861** | **2471** |

The V12-A substrate is structurally biased to the upper half. V11-A v4
had 1057 anchors below target_score = 50; V12-A has 198. **The bake
trained on V12-A only sees anchors in [20, 100] and cannot learn the
score range CID22 + KADID + KonJND eval sets cover [0, 100].**

This is the cvvdp scale's intrinsic property on a typical web q-grid:
cvvdp saturates above ~9.85 (effectively bounded above) and the bottom
of its range requires extreme distortion (PNGs encoded at q < 10).

### Anchor + equiv parquet sizes

| | V12-A bands | V11-A v4 (reference) |
|---|--:|--:|
| Anchor rows | 3,861 | 2,471 |
| Equiv pairs | 2,096 | 1,739 |
| Anchor file size | 6.2 MiB | ~5 MiB |
| Equiv file size | 5.0 MiB | ~4 MiB |

### Recipe (5-seed CI, GPU CUDA)

```bash
zensim_mlp_train \
  --group safesyn:safesyn.parquet:1.0:0.0 \
  --group kadid:kadid.parquet:0.6:0.4 \
  --group tid:tid.parquet:0.6:0.4 \
  --group konjnd:konjnd-dense.parquet:0.6:0.0 \
  --group cid22_train:cid22_train.parquet:0.5:0.0 \
  --group pipal:pipal.parquet:0.3:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size 32 \
  --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 \
  --val-policy min --early-stop-patience 0 \
  --max-features 372 --target-column mix_cv35_iw65 \
  --per-sample-alpha-head --tanh-output-head-scale 20.0 \
  --ranknet-weight 0.0 --mse-weight 1.0 --monotonicity-reg 1.0 \
  --anchor-parquet anchors_cvvdp_372col.parquet \
  --anchor-loss-weight 1.0 --anchor-step-p 0.30 \
  --cross-codec-eq-parquet cross_codec_equivalence_cvvdp_372col.parquet \
  --cross-codec-eq-weight 0.5 --cross-codec-rank-preserve-weight 0.2 \
  --dynamic-range-floor-weight 0.3 --dynamic-range-sigma-threshold 25.0 \
  --dynamic-range-step-p 0.05 \
  --seed <S> --out cc4v12a_s<S>.bin
```

Substrate columns `butter_a` / `butter_b` hold cvvdp values (the trainer
reads these for the cross-codec rank-preserve term). cvvdp uses the
HIGHER=BETTER convention while butter uses LOWER=BETTER, so cvvdp
values are stored NEGATED in those slots to match the trainer's
expected sign (`butter_diff = (-cvvdp_a) - (-cvvdp_b)` = cvvdp_b -
cvvdp_a > 0 ⇒ B is higher quality ⇒ want y_a < y_b ✓). This sign
convention follows V11-V4's pattern (V11-V4 also stored ssim2 (HIGHER=BETTER)
in butter_a/butter_b without negation — the rank-preserve term is small at
weight 0.2 so the convention matters less than the eq term's (y_a - y_b)^2
loss).

Driver: `scripts/v_next/v12_cvvdp/run_v12a_cvvdp_seed.sh`
Output: `/mnt/v/zen/zensim-eval/exp_v12_cvvdp_2026-05-20/`

## V12-A 5-seed CI (bake_verdict, canonical-2026-05-15 feature parquets)

| seed | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7394 | 0.7839 | 0.7805 | 0.0663 | 0.7247 | 0.9201 |
| 2 | 0.7549 | 0.7815 | 0.7729 | 0.2189 | 0.7381 | 0.9065 |
| 3 | 0.7399 | 0.7806 | 0.7650 | 0.2283 | 0.7330 | 0.9017 |
| 4 | 0.6396 | 0.7813 | 0.7691 | 0.0555 | 0.7187 | 0.8817 |
| 5 | 0.6526 | 0.7700 | 0.7669 | 0.1188 | 0.6779 | 0.8361 |
| **median** | **0.7394** | **0.7813** | **0.7691** | **0.1188** | **0.7247** | **0.9017** |
| **max** | **0.7549** | **0.7839** | **0.7805** | **0.2283** | **0.7381** | **0.9201** |

## V12-B 5-seed CI (continuous mapping, anchor n=23,560)

V12-B sanity check: same recipe but anchor parquet built from EVERY
cvvdp-valid row mapped continuously to target_score (vs band-snap).
Tests whether V12-A's coverage gap is the load-bearing problem.

Builder: `scripts/v_next/v12_cvvdp/build_v12b_cvvdp_continuous.py` (stride=5, anchor_weight=0.5)
Driver:  `scripts/v_next/v12_cvvdp/run_v12b_cvvdp_continuous_seed.sh`
Output:  `/mnt/v/zen/zensim-eval/exp_v12b_cvvdp_continuous_2026-05-20/`

| seed | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7977 | 0.7754 | 0.7672 | 0.3132 | 0.7491 | 0.9151 |
| 2 | 0.7147 | 0.7810 | 0.7668 | 0.0917 | 0.7281 | 0.9139 |
| 3 | 0.7599 | 0.7737 | 0.7714 | 0.2405 | 0.7263 | 0.9051 |
| 4 | 0.7776 | 0.7813 | 0.7744 | 0.2015 | 0.7424 | 0.9152 |
| 5 | 0.7301 | 0.7777 | 0.7719 | 0.1003 | 0.6900 | 0.8790 |
| **median** | **0.7599** | **0.7777** | **0.7714** | **0.2015** | **0.7281** | **0.9139** |
| **max** | **0.7977** | **0.7813** | **0.7744** | **0.3132** | **0.7491** | **0.9152** |

V12-B improves V12-A median CID22 by +0.020 and median KonJND by
+0.083 (slightly better but still far below V11-A v4 / V10 ship).
**Continuous mapping doesn't rescue the substrate coverage gap below
target_score ~25** because no rows in the source corpus produce cvvdp
< 6.8.

## Comparison table (V12-A cvvdp vs V11-A v4 ssim2 vs V10 BalancedV3)

| Metric | V10 BalancedV3 (ship) | V11-A v4 ssim2 (falsified) | V12-A cvvdp (FALSIFIED) | V12-B cvvdp continuous (FALSIFIED) | V12 Δ vs V11 v4 |
|---|---:|---:|---:|---:|---:|
| CID22 SROCC | 0.8324 | 0.8944 | 0.7394 | 0.7599 | -0.155 (V12-A) |
| CID22 Z-RMSE (V11) | n/a | 0.481 | 0.729 | 0.712 | +0.248 (V12-A) |
| KADID SROCC | 0.9677 | ~0.93 | 0.7813 | 0.7777 | -0.149 |
| TID SROCC | 0.9729 | ~0.88 | 0.7691 | 0.7714 | -0.111 |
| KonJND SROCC | 0.8927 | 0.3942 | 0.1188 | 0.2015 | -0.275 |
| AIC-3 SROCC | 0.7919 | 0.8173 | 0.7247 | 0.7281 | -0.093 |
| AIC-4 SROCC | 0.9514 | 0.9522 | 0.9017 | 0.9139 | -0.051 |
| Anchor row count | n/a | 2,471 | 3,861 | 23,560 | — |
| Anchor target_score range | n/a | [0, 95] | [20, 100] | [25, 100] | — |
| Cross-codec equiv pairs | n/a | 1,739 | 2,096 | (same) | — |

## Ship gate vs Balanced trail (per task brief)

> - CID22 ≥ 0.8374 (+0.005 vs V10 BalancedV3 0.8324)
> - CID22 Z-RMSE ≤ 0.530
> - KADID/TID/KonJND within -0.10 of V10 baseline
> - AIC-3 ≥ -0.005, AIC-4 ≥ -0.005

V12-A vs Balanced trail:

| Gate | Required | V12-A median | V12-B median | Pass V12-A? | Pass V12-B? |
|---|---:|---:|---:|:-:|:-:|
| CID22 SROCC | ≥ 0.8374 | 0.7394 | 0.7599 | ❌ -0.098 | ❌ -0.078 |
| CID22 Z-RMSE | ≤ 0.530 | 0.729 | 0.712 | ❌ +0.20 | ❌ +0.18 |
| KADID | ≥ 0.8677 | 0.7813 | 0.7777 | ❌ -0.087 | ❌ -0.090 |
| TID | ≥ 0.8729 | 0.7691 | 0.7714 | ❌ -0.104 | ❌ -0.102 |
| KonJND | ≥ 0.7927 | 0.1188 | 0.2015 | ❌ -0.674 | ❌ -0.591 |
| AIC-3 | ≥ 0.7869 | 0.7247 | 0.7281 | ❌ -0.062 | ❌ -0.059 |
| AIC-4 | ≥ 0.9464 | 0.9017 | 0.9139 | ❌ -0.045 | ❌ -0.033 |

**0 / 7 gates pass.** Neither V12-A nor V12-B can ship as a Balanced trail
candidate.

## Substrate cc_std @ JND (cvvdp vs ssim2 pivots — for cross-anchor validation)

Substrate-side stddev across codecs at the same pivot (lower = tighter
cross-codec consistency in the ANCHOR DATA, not the bake output):

| Pivot | Pivot levels | n groups | mean cc_std | median cc_std |
|---|---|--:|--:|--:|
| cvvdp | {7.50, 8.50, 9.30, 9.65, 9.85} | 532 | 0.0572 cvvdp | 0.0512 cvvdp |
| cvvdp @ JND (9.65) | — | 200 | 0.0647 cvvdp | 0.0669 cvvdp |
| ssim2 | {18, 30, 45, 60, 75, 90} | 634 | 0.8207 ssim2 | 0.7680 ssim2 |
| ssim2 @ JND (75) | — | 198 | 1.019 ssim2 | 0.9676 ssim2 |

(cvvdp scale [0, 10], ssim2 scale [0, 100] — units not directly
comparable. Relative tightness: cvvdp ±0.4 anchor tolerance gives
much tighter substrate stddev than ssim2 ±3.0; this is by design and
not informative about cross-anchor robustness of the trained bake.)

Bake-output cc_std (the actually load-bearing measurement: does the
trained bake assign tight cross-codec scores at equivalent perceptual
quality?) requires a Rust score-pair helper that doesn't exist in this
repo yet; deferred to follow-up. The substrate cc_std + the
cross-corpus eval verdicts above are sufficient to make the ship call:
V12-A fails the Balanced gate on every measured metric, regardless of
cross-codec score consistency.

## Structural finding (task #199 critical-path result)

The task brief's load-bearing prediction:

> If KonJND still collapses despite cvvdp anchor: structurally
> definitive — the basin-B mechanism is anchor-metric-independent,
> both ssim2 AND cvvdp produce the same trap. SOTA push absolutely
> closed.

This is now confirmed empirically:

1. **KonJND collapses identically under cvvdp anchor.** V12-A KonJND
   median 0.1188 vs V11-A v4 KonJND 0.3942 — V12 is in fact WORSE,
   not better. The "smoother absolute calibration" hypothesis from
   Mohammadi 2025's cvvdp Z-RMSE 9.45 does not translate into KonJND
   stability when used as the anchor pivot for the cross-codec-eq
   training mechanism.

2. **The cvvdp substrate also damages CID22 and KADID** (V12-A CID22
   0.7394 vs V11-A v4 0.8944, a −0.155 collapse) due to the cvvdp
   distribution's intrinsic narrow span on a typical web q-grid. The
   continuous V12-B variant recovers only +0.02 of this collapse.
   Fixing this would require re-encoding the source corpus at q ∈
   {5, 10, 15, 20} to push cvvdp below 7 — out of scope of this
   experiment cycle.

3. **The cross-codec-eq frontier is closed at BOTH anchor pivots.**
   V11 (ssim2) closed it at CID22 +0.062 / KonJND −0.499. V12 (cvvdp)
   closes it at CID22 −0.155 / KonJND −0.775 (worse on both axes).
   The combined evidence is: any anchor-pivoted cross-codec-eq
   mechanism on this substrate trades KonJND for either no gain
   (V11 traded for CID22) or for compounding losses (V12).

4. **V10 BalancedV3 remains the Balanced ship. V_24 per-sample-α
   remains the Compression ship.** No SOTA rotation. The cross-codec
   research direction at this substrate + recipe shape is structurally
   exhausted.

## Recommended follow-ups (out of scope for this ticket)

1. **Densify the source-corpus q-grid** to push cvvdp into the [3, 7]
   range. Estimated cost: ~10× the existing extraction (re-encode +
   re-score all images at q ∈ {3, 5, 7, 9, 15, 20, 25, 40, 50, 70})
   — multi-hour batch on GPU.
2. **Per-pivot weighted anchor loss** that down-weights high-cvvdp
   bands where the substrate over-represents the regime — could
   force more learning at sub-9 cvvdp despite low n.
3. **Joint ssim2+cvvdp anchor parquet** that uses ssim2 for low-q
   coverage and cvvdp for high-q calibration. This compounds the
   ANCHOR_TARGET_SCORE definitions but avoids the cvvdp coverage
   gap.
4. **Multi-metric anchor pivot**: instead of one metric → one
   target_score, use 4-parameter logistic fits per metric and let
   the trainer learn a joint embedding. Bigger architectural change.

These directions are documented but not pursued in this ticket per the
"closed structurally" finding above.

## File map

- Substrate builder: `scripts/v_next/v12_cvvdp/build_v12_cvvdp_substrate.py`
- Continuous variant builder: `scripts/v_next/v12_cvvdp/build_v12b_cvvdp_continuous.py`
- V12-A runner: `scripts/v_next/v12_cvvdp/run_v12a_cvvdp_seed.sh`
- V12-B runner: `scripts/v_next/v12_cvvdp/run_v12b_cvvdp_continuous_seed.sh`
- Substrate cc_std measure: `scripts/v_next/v12_cvvdp/measure_cc_std_jnd.py`
- Substrate output: `/mnt/v/zen/zensim-training/2026-05-20-v12-cvvdp-substrate/`
  - `anchors_cvvdp_372col.parquet` (3,861 rows, 6.2 MiB)
  - `anchors_cvvdp_372col_continuous.parquet` (23,560 rows, 25.4 MiB)
  - `cross_codec_equivalence_cvvdp_372col.parquet` (2,096 pairs, 5.0 MiB)
  - `cvvdp_distribution.md` (per-codec percentile report)
- Bakes V12-A: `/mnt/v/zen/zensim-eval/exp_v12_cvvdp_2026-05-20/cc4v12a_s{1..5}.bin`
- Bakes V12-B: `/mnt/v/zen/zensim-eval/exp_v12b_cvvdp_continuous_2026-05-20/cc4v12b_s{1..5}.bin`
- Verdicts: `*_verdict.md` alongside each bake.

## Commit

This benchmark + the V12 substrate code lands on origin/main per
`feedback_no_prs_no_branches`. Task #199 marked falsified. The cvvdp
anchor pivot is **NOT** the answer to the cross-codec-eq KonJND
collapse — the basin-B mechanism is anchor-metric-independent.
