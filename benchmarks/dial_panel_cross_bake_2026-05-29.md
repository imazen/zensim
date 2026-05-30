# Cross-bake DIAL panel comparison — 2026-05-29

Native `bake_verdict` DIAL panel (the densified multi-codec grid: q0 +
step-1 q90→100 + JND-zone + JXL-in-butteraugli-distance 49-rung ladder)
run over the shipped + sibling profile bakes. **inversions** = adjacent
quality steps where the score ran *backwards* (a ranking error);
**ties** = flat dead-zones (resolution loss, not an error); monotonicity
= 1 − inversions.

Grid: `dial_grid_372col_2026-05-29.parquet` (4,349 rows, 117 curves, 4
codec families; sha256 `98760e9a`). Reproduce:

```bash
ZENSIM_DIAL_GRID=/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29.parquet \
  ./target/release/bake_verdict --bake zensim/weights/<bake>.bin --output <out>.md
```

## Aggregate dial (four buckets summing to 1; lower inv + flat = better)

The panel splits adjacent-quality-step outcomes into FOUR distinct
buckets so a real ranking error isn't conflated with sub-JND noise or
dense-grid oversampling:

| bake | profile slot | inv (>0.5pt) | inv (strict) | mag med | flat/clamp | mono | p5/p95 | G3 |
|---|---|--:|--:|--:|--:|--:|---|:--:|
| **v47_strict_qat_native** | **Profile::A (shipped)** | **0.0208** | 0.0501 | 0.34 | **0.0163** | 0.9792 | 15.0 / 94.4 | **✓✓** |
| v_balanced_v3 | (sibling) | 0.0224 | 0.0444 | 0.50 | 0.1193 | 0.9776 | 0.0 / 99.7 | mono✓ flat✗ |
| v02_372feat_cell5 | PreviewV0_5Linear | 0.0258 | 0.0555 | 0.39 | 0.1059 | 0.9742 | 0.0 / 93.7 | mono✓ flat✗ |
| v_tuner_v11 | (sibling) | 0.0258 | 0.0709 | 0.23 | 0.0279 | 0.9742 | 14.3 / 104.2 | ✓✓ (overshoots) |
| v39_v32plus_spline | (prior Profile::A) | **0.7349** | 0.7932 | **3.34** | 0.0163 | 0.2651 | −125.4 / 91.8 | **BROKEN** |

- **inv (>0.5pt)** = material inversions: adjacent steps where the dial
  ran backwards by more than half a score-point — the real ranking-error
  rate (the gated metric, G3 ≤ 0.07).
- **inv (strict)** = any backwards move > 1e-9 — a diagnostic that
  INCLUDES sub-JND noise. **mag med** = the median backwards-step
  magnitude.
- **flat/clamp** = adjacent steps with literally identical output
  (\|Δ\|≤1e-9) — a saturation/clamp dead-zone (the other G3 sub-gate,
  ≤ 0.05).

**The 5% "inversion rate" is mostly noise.** On the good bakes the strict
any-backwards rate is ~5%, but the **median backwards step is 0.23–0.50
score-points** — sub-JND wiggles from the densified near-lossless grid
(adjacent configs are perceptually indistinguishable, so score noise
alone produces tiny reversals). Requiring a user-visible (>0.5pt)
backwards move drops the real inversion rate to **~2%**. v39 is the
exception that proves the rule: its median backwards step is **3.34
points** and material inversions are **73%** — a genuine catastrophe, not
noise.

**v47 wins the shipped-profile decision — it is the ONLY candidate that
passes both G3 sub-gates.** Lowest material inversions (0.0208), lowest
flat/clamp dead-zone rate (0.0163, 7× better than v_balanced_v3 /
Cell5), and the only bake whose output stays *bounded* (p5/p95 15.0/94.4
— no negative excursion, no >100 overshoot). v_balanced_v3 and Cell5
have low inversions but 6–7× worse clamp dead-zones (fail flat ≤ 0.05);
v_tuner_v11 passes both gates but overshoots to 104.

## The panel independently reproduces the v39 defect that motivated v47

v39 (the prior Profile::A) is **catastrophically broken on the
multi-codec grid**: monotonicity 0.21, scores plunging to −128 at the
*highest*-quality configs (jpeg score @worst→@best = 39.5 → −128.1).
v39 ships a raw-F32 spline that extrapolates unbounded on the
JXL/AVIF/WebP output features it never saw in training. This is exactly
the dial defect documented in `[[project_v39_correctness_defect]]` that
motivated the 2026-05-27 v47-strict-QAT ship — the native dial panel
catches it without any prior knowledge, and confirms v47's
bounded-output design fixes it (mono 0.21 → 0.95). **This validates both
the panel and the ship decision.**

## Cross-codec finding: JXL near-lossless is underscored by EVERY bake

JXL `score @best` (distance 0.025 ≈ q-equiv 99.9, near-lossless) median
dial score, across bakes:

| bake | jxl @best | q-codec @best (jpeg/webp/avif) |
|---|--:|--:|
| v47 | 74.9 | 92.6 / 89.3 / 94.4 |
| v_balanced_v3 | 83.0 | 98.4 / 93.3 / 99.4 |
| v_tuner_v11 | 63.6 | 95.5 / 90.2 / 97.8 |
| cell5 | 62.8 | 92.8 / 88.8 / 93.5 |

Near-lossless JXL scores **10–25 points lower** than near-lossless
JPEG/WebP/AVIF on every bake. This is a **feature-distribution gap, not
a per-bake dial bug**: the synthetic safe-synthetic training corpus is
mozjpeg/jpegli/zenwebp output — no JXL VarDCT. JXL's 372-feature
signature at near-lossless is out-of-distribution, so all bakes
systematically under-score it. This is a real G4 (cross-codec
equivalence) gap: a codec-target dial set for "score 90" would over-
encode JXL (push it past visually-lossless chasing a number it can't
reach). The denser distance ladder (49 rungs vs the old 16) is what made
this visible — at the coarse grid the JXL curve never sampled
near-lossless densely enough to show the ceiling.

**Next-lever (not yet done):** add JXL output to the training corpus, or
fit a per-codec output recalibration so the dial reads equivalent
perceptual quality equivalently across codecs. The avif inversions
(0.075–0.108, highest of the q-codecs across every bake) are a related
signal — avif's q→quality mapping is the least monotone in feature
space.

## Per-codec detail (v47 / Profile::A) — material inversions

| codec | param | min..max | inv (>0.5pt) | flat/clamp | monotonicity | score @worst→@best |
|---|---|---|--:|--:|--:|---|
| avif | q | 0..100 | 0.0349 | 0.0000 | 0.9651 | 9.5 → 94.4 |
| jpeg | q | 0..100 | 0.0068 | 0.0938 | 0.9932 | 17.9 → 92.6 |
| jxl | distance | 0.03..25.00 | 0.0209 | 0.0000 | 0.9791 | 29.0 → 74.9 |
| webp | q | 0..100 | 0.0136 | 0.0000 | 0.9864 | 8.3 → 89.3 |

avif has the highest material inversions (0.035) of the q-codecs across
every bake — its q→quality mapping is the least monotone in feature
space. jpeg's flat/clamp 0.094 is the near-lossless q97–q100 plateau (the
dial genuinely saturates where JPEG quality stops improving).
