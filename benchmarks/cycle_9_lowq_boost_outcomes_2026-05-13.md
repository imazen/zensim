# Cycle-9 low-q boost experiments — outcomes (2026-05-13)

## Summary

Cycle-9 added a `--low-q-boost` flag to `train_v_next_mlp.py` to
test whether per-row weighting of low-quality training pairs
(score<50 = B0, score<65 = B1) could close the CID22 B0/B1
SROCC ceiling identified in cycle-8 closure.

The lever was hypothesized to address the structural gap where
all KonJND-trained bakes (and V0_16 SHIP) score B0≈0.40, B1≈0.40
against ssim2's 0.44/0.47 — a ~0.05 SROCC gap that doesn't yield
to any cycle-7/8 recipe variation.

**Result: FALSIFIED.** 5-seed sweep at boost=1.5 shows no
statistically significant gain over baseline boost=1.0. The
earlier single-seed "wins" (V0_34, V0_36) were upper-tail
outliers in the seed distribution.

## Trainer change

Added flag in `scripts/v_next/train_v_next_mlp.py` (zensim commit
`4b998258`):

```python
ap.add_argument("--low-q-boost", type=float, default=1.0,
                help="Multiply train_weight for low-quality rows by this "
                     "factor. Bins by target column: score<50 (B0) gets "
                     "the full multiplier, 50<=score<65 (B1) gets "
                     "sqrt(multiplier). Default 1.0 = no boost.")
```

Mechanic: applied to `df["train_weight"]` after class-balance and
before train/val/test split. The `train()` function's existing
`wt` tensor path then carries the weights into the MSE loss term.
RankNet loss is unweighted (pair sampling already over-represents
larger groups).

## Experimental sweep

5 boost levels × 1 seed (initial discovery sweep):

| Boost | Seed | CID22 SROCC | AIC-3 | AIC-4 | Tag |
|--:|--:|--:|--:|--:|---|
| 1.0 (V0_31) | 1 | 0.8628 | 0.8031 | 0.9176 | konjnd_w05 |
| 1.5 (V0_34) | 1 | 0.8635 | 0.8135 | **0.9252** | konjnd_w05_lowqboost15 |
| 2.0 (V0_35) | 1 | **0.8468** | — | 0.9139 | konjnd_w05_lowqboost2 |
| 3.0 (V0_33) | 1 | 0.8552 | 0.8043 | 0.9176 | konjnd_w05_lowqboost3 |

V0_34 (boost 1.5) appeared to dominate. V0_35 (boost 2.0) was
anomalously bad. V0_33 (boost 3.0) cleanly traded off (-CID22,
+B0 SROCC) as expected.

Then ran 5 seeds × 1 boost level (1.5) for variance estimation:

| Seed | CID22 | AIC-4 | Tag |
|--:|--:|--:|---|
| 1 | **0.8635** | **0.9252** | (V0_34, original) |
| 2 | 0.8573 | 0.9050 | v0_2_boost15 |
| 3 | 0.8585 | 0.9076 | v0_3_boost15 |
| 7 | 0.8555 | 0.9107 | v0_7_boost15 |
| 42 | **0.8639** | 0.9124 | (V0_36) |
| **Mean** | **0.8597** | **0.9122** | — |
| Std (n=5) | 0.0038 | 0.0078 | — |

## Statistical comparison

| Config | n | Mean CID22 | Mean AIC-4 |
|---|--:|--:|--:|
| boost 1.0 (V0_31, V0_32) | 2 | 0.8603 | 0.9180 |
| boost 1.5 (this sweep) | 5 | 0.8597 | 0.9122 |
| **Δ (1.5 vs 1.0)** | | **-0.0006** | **-0.0058** |

Welch's t-test on AIC-4:
- t = (0.9122 - 0.9180) / sqrt(σ²(1.5)/5 + σ²(1.0)/2)
- σ²(1.5) ≈ 0.000061, σ²(1.0) ≈ 0.0000004 (n=2 is essentially a point estimate)
- t ≈ -1.7, df ≈ 4 → p ≈ 0.17 (not significant)

Without more boost-1.0 seeds we can't establish statistical
significance, but the direction is clearly NEGATIVE (boost 1.5
slightly underperforms boost 1.0 on average).

## Why we were fooled

V0_34 (seed=1) hit AIC-4 0.9252, which is +0.4σ above the
boost-1.5 mean. V0_36 (seed=42) hit CID22 0.8639, which is +1.1σ
above the boost-1.5 mean. Both were positive outliers in their
respective directions, and we discovered them in that order.

When we ran V0_35 (seed=1, boost=2.0) and saw 0.8468 (= -3.4σ on
boost-1.5 distribution), we interpreted it as "2.0 is too high"
when it might just have been an unlucky seed at boost 2.0.

**Single-seed comparisons are unreliable when the recipe effect
is < 1× seed std.** The boost-axis effect on AIC-4 is
≈ 0 ± 0.008; we need >5 seeds per boost level to detect any
sub-noise signal.

## Cycle-9 verdict

**Low-q row-weight boosting is NOT a useful lever** for breaking
the V0_16 CID22 SHIP ceiling. The 0.05 SROCC gap in B0/B1 against
ssim2 is structural to the V_X 228-feat MLP architecture + the
synth+KonJND data regime; it doesn't yield to training-time row
reweighting.

Conjectured reasons:
1. **MSE loss weighting affects predictions linearly, but rank
   correlation is invariant to monotonic predictor scaling.**
   Upweighting B0 pairs shifts the predictor mean for B0 (lower
   MSE in that band) but doesn't necessarily improve B0 internal
   ranking.
2. **RankNet loss is unweighted** in the current implementation
   — only MSE carries the boost. RankNet contributes 0.5× weight
   to total loss and is responsible for most of the rank-correlation
   signal. So even at boost=3.0, the rank loss sees the same
   B0/B1 pairs as boost=1.0.

A future cycle-9b experiment could:
- Apply the boost to RankNet's pair sampling (oversample B0/B1
  pairs in groups) rather than to MSE row weights.
- Or weight pair contributions to the RankNet logistic loss.

These would test the same hypothesis but at the right loss term.
Not pursued in cycle-9 (out of scope without further user direction).

## Lessons learned (recorded for future cycles)

1. **The V_X 228-feat MLP recipe seed std on this data**:
   - CID22 SROCC: σ ≈ 0.004
   - AIC-4 SROCC: σ ≈ 0.008
   - AIC-3 SROCC: not measured at 5 seeds, but tick 504 V0_31 vs
     V0_32 showed Δ ≈ 0.005

2. **All cycle-7/8/9 recipe variations sit within ~0.01 of each
   other on CID22 SROCC.** They are likely **a single noisy
   plateau**. V0_16 SHIP at 0.8919 is ~6σ above the cycle-7/8/9
   recipe family — either V0_16 used a materially different
   recipe we haven't reproduced, or V0_16 is itself a lucky
   outlier of its own training distribution.

3. **Confidence intervals for any single-seed claim need
   width ~ 2σ.** Before declaring "recipe X improves metric Y",
   run ≥3 seeds and check that Δ_mean > 2σ_combined.

## Artifacts

Bakes (all 120,710-120,714 bytes, ZNPR v2 228→128→1):
- `/tmp/zensim_loop/bakes/v0_33_konjnd_w05_lowqboost3_2026-05-13.bin` (seed=1, boost=3.0)
- `/tmp/zensim_loop/bakes/v0_34_konjnd_w05_lowqboost15_2026-05-13.bin` (seed=1, boost=1.5)
- `/tmp/zensim_loop/bakes/v0_35_konjnd_w05_lowqboost2_2026-05-13.bin` (seed=1, boost=2.0)
- `/tmp/zensim_loop/bakes/v0_36_konjnd_w05_lowqboost15_seed42_2026-05-13.bin` (seed=42, boost=1.5)
- `/tmp/zensim_loop/bakes/v0_{2,3,7}_boost15_2026-05-13.bin` (3 seeds at boost=1.5)

Per-pair CSVs:
- `/tmp/zensim_loop/v0_{33,34,35,36}_per_pair.csv`
- `/tmp/zensim_loop/v0_{2,3,7}_boost15_2026-05-13_per_pair.csv`

Run directories: `/mnt/v/zen/zensim-training/2026-05-07/runs/*v0_{33,34,35,36}*` plus 3 seed dirs

Tick log entries: 508, 509, 510, 511 in
`~/work/zen/zenanalyze/zensim_champion_log.md`.

## Cycle status overall

| Cycle | Hypothesis | Verdict |
|---|---|---|
| 7 | dssim co-training closes JPEG-AI gap | FALSIFIED (V0_27 -0.060 JPEG-AI) |
| 7 | cosine LR schedule helps | FALSIFIED (V0_28 -0.0089 CID22) |
| 7 | smaller LR finds flatter min | FALSIFIED (V0_29 underconverged) |
| 8 | KonJND weight tuning yields Pareto | PARTIAL (V0_31 wins AIC-4 only) |
| 9 | Low-q row boost closes B0/B1 ceiling | FALSIFIED (5-seed mean no gain) |

**V0_16 SHIP remains unchanged.** V0_26 + V0_31 preserved as
cycle-7/8 alternatives on the live comparison site. Cycle-10
needs user-directed strategy change (data axis vs architecture
axis vs loss-axis fix from cycle-9b).
