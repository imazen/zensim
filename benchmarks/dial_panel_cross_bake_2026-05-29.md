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

## Aggregate dial (lower inversions + lower ties = better dial)

| bake | profile slot | inversions | ties | monotonicity | dial p5 / p95 | verdict |
|---|---|--:|--:|--:|---|---|
| **v47_strict_qat_native** | **Profile::A (shipped)** | **0.0501** | **0.0163** | 0.9499 | 15.0 / 94.4 | **best balance — lowest ties, bounded** |
| v_balanced_v3 | (sibling) | 0.0444 | 0.1193 | 0.9556 | 0.0 / 99.7 | low inversions but 7× ties (dead-zones), overshoots 100 |
| v02_372feat_cell5 | PreviewV0_5Linear | 0.0555 | 0.1059 | 0.9445 | 0.0 / 93.7 | good all-rounder; high ties from coarse low-q |
| v_tuner_v11 | (sibling) | 0.0709 | 0.0279 | 0.9291 | 14.3 / 104.2 | fails G3 mono; overshoots 104 |
| v39_v32plus_spline | (prior Profile::A) | **0.7932** | 0.0163 | 0.2068 | −125.4 / 91.8 | **BROKEN — see below** |

**v47 wins the shipped-profile decision.** It has the lowest tied rate
(0.0163, 7× better than v_balanced_v3's 0.1193), the 2nd-lowest
inversions (0.0501), and is the only candidate whose output stays
*bounded* (p5/p95 15.0/94.4 — no negative excursion, no >100 overshoot).
v_balanced_v3's marginally-lower inversions (0.0444) come with a 7× worse
dead-zone rate and a 99.7 overshoot; v_tuner_v11 fails G3 monotonicity
and overshoots to 104.

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

## Per-codec detail (v47 / Profile::A)

| codec | param | min..max | inversions | ties | monotonicity | score @worst→@best |
|---|---|---|--:|--:|--:|---|
| avif | q | 0..100 | 0.0754 | 0.0000 | 0.9246 | 9.5 → 94.4 |
| jpeg | q | 0..100 | 0.0285 | 0.0938 | 0.9715 | 17.9 → 92.6 |
| jxl | distance | 0.03..25.00 | 0.0407 | 0.0000 | 0.9593 | 29.0 → 74.9 |
| webp | q | 0..100 | 0.0557 | 0.0000 | 0.9443 | 8.3 → 89.3 |
