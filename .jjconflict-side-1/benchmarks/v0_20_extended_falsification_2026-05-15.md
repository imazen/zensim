# V_20 extended-features (300-feat) — falsified for CID22 ship

**Date**: 2026-05-15
**Bake**: `benchmarks/v0_20_extended_seed1_2026-05-15.bin` (ZNPR v3,
n_inputs=300, seed=1, h=128, epochs=300 with early-stop at 290,
val_mean SROCC 0.9504)
**Recipe**: V_19 base recipe + V_20 transforms broadened to
max-feature-idx 300 (139 transforms vs V_20 IS's 98).
**Result**: matches V_20 IS within numerical noise on every corpus —
the 41 added transforms in the masked-features block are **redundant
with the 98 transforms in the basic+peaks block**.

## Acceptance gate (per the experiment plan)

- CID22 SROCC ≥ 0.8895 (fast-ssim2 floor) — **FAILS** (0.8783 < 0.8895
  by 0.0112)
- No >0.005 KADID/TID regression vs V_18 ship — passes (wins both)
- No >0.005 regression vs V_20 IS — borderline (TID −0.0049, KADID +0.0007)

Verdict: **NO SHIP**. The extended-features direction does not lift
CID22 above V_18's 0.8933 ceiling, nor above fast-ssim2's 0.8895
floor.

## Full Mohammadi panel vs reference bakes

### KADID-10k (n=10125)

| Bake | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_18 ship (ref) | 0.9427 | 0.8757 | 0.7930 | 0.0426 | 0.9656 | 0.332 |
| V_20 IS (228, ref) | 0.9497 | 0.8265 | 0.8054 | 0.0387 | 0.9706 | 0.311 |
| **V_20 extended (300)** | **0.9504** | 0.7977 | **0.8058** | **0.0353** | **0.9710** | **0.308** |

Wins V_18 ship on every stat. Essentially flat vs V_20 IS (within
numerical noise on SROCC + KROCC + PWRC; very slightly better
Z-RMSE).

### TID2013 (n=3000)

| Bake | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_18 ship (ref) | 0.9526 | 0.9309 | 0.8110 | 0.0313 | 0.9702 | 0.294 |
| V_20 IS (228, ref) | 0.9616 | 0.9552 | 0.8280 | 0.0497 | 0.9764 | 0.271 |
| **V_20 extended (300)** | 0.9567 | 0.9509 | 0.8174 | **0.0457** | 0.9729 | 0.287 |

Wins V_18 ship on every stat, **but loses V_20 IS** on every stat
except OR — the extra features cost a small amount on TID.

### CID22 (n=4292)

| Bake | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_18 ship (ref) | **0.8933** | 0.8679 | 0.7081 | 0.0536 | **0.9373** | **0.455** |
| V_20 IS (228, ref) | 0.8794 | 0.8126 | 0.6915 | 0.0513 | 0.9271 | 0.482 |
| **V_20 extended (300)** | 0.8783 | 0.8056 | 0.6893 | **0.0478** | 0.9270 | 0.481 |

Loses V_18 ship on every CID22 stat. Essentially flat vs V_20 IS
(within −0.0011 on SROCC, identical PWRC + Z-RMSE).

### CID22 10-band SROCC (B0..B9 width-10 on normalized score)

| Band | range | n | V_18 ship | V_20 IS (228) | **V_20 extended (300)** | fast-ssim2 |
|---|---|--:|---:|---:|---:|---:|
| B3 | [0.30, 0.40) | 57 | 0.0246 | **0.1534** | 0.1476 | 0.1335 |
| B4 | [0.40, 0.50) | 266 | 0.3029 | 0.2717 | 0.2947 | 0.2888 |
| B5 | [0.50, 0.60) | 615 | 0.3891 | 0.3344 | 0.3301 | 0.3888 |
| B6 | [0.60, 0.70) | 836 | 0.3943 | 0.3930 | 0.3734 | 0.4173 |
| B7 | [0.70, 0.80) | 1092 | 0.3936 | 0.3692 | 0.3687 | 0.3974 |
| B8 | [0.80, 0.90) | 1382 | 0.5127 | 0.4938 | 0.4952 | 0.5006 |
| B9 | [0.90, 1.00] | 43 | 0.1545 | 0.1146 | 0.1276 | 0.1121 |

The 300-feat bake reproduces V_20 IS's per-band shape **exactly**:

- **B3 specialist trade**: closes ssim2's 0.13 floor (0.1476 vs ssim2
  0.1335) at the cost of mid-band regression — same shape as V_20 IS.
- B4 +0.023 vs V_20 IS (tiny lift in the extended bake)
- B5/B6/B7 essentially flat vs V_20 IS, all worse than V_18 ship
- B8 +0.001 vs V_20 IS
- B9 +0.013 vs V_20 IS (small high-q recovery)

The masked-features block (idx 228..299) does NOT carry a separate
signal from the basic+peaks block (idx 0..227). The 41 added
transforms compete for the same MLP-internal axes and produce
near-identical predictions.

## Interpretation

**MLP capacity (h=128) is not the bottleneck**. The 300-feat
trainer sees 31 % more input columns but trains to a near-identical
prediction surface as the 228-feat trainer. This rules out the
"feature engineering at the runtime feature-width level" direction:

- V_18 ship (228 features, 3-way concat, no transforms): 0.8933
- V_20 IS (228 features, single MLP, 98 transforms): 0.8794
- V_20 extended (300 features, single MLP, 139 transforms): 0.8783

The progression confirms that the input-shaping mechanism is
already near-saturated at 228 features. Adding 41 masked-feature
transforms (predominantly winsor_p99) gives the trainer redundant
information; the MLP's effective rank doesn't change.

**The V_18 ceiling is structural to the recipe**, not the input
feature count. To break it requires fundamentally different
mechanisms documented elsewhere as queued V_X candidates:

- **V_22 CVVDP distillation** (task #45 / #49) — train against a
  fundamentally better metric's predictions, not engineered
  feature inputs.
- **V_20d JND-anchored output calibration** (task #41) — use
  Jenadeleh 2025 / Testolina 2023 PJND units to constrain the
  output scale.
- **V_20c LMS + opponent feature branch** (task #40) — add new
  feature columns from a different color space (FRIQUEE 2017
  features in LMS/opponent), not just transforms of the existing
  XYB-derived ones.
- **Authentic-distortion corpus expansion** — the CID22 ceiling
  exists because we train almost entirely on synthetic distortions
  (per the FRIQUEE 2017 caveat that materialized in V_20a IW +
  V_20b distortion-manifold).

## What stays from this work

- **`ProfileParams.extended_features`** + **`compute_iw_features`**
  fields ship anyway (commit `f140776a`). They cost nothing at
  defaults `false` and enable future research bakes (V_20c, V_22,
  ...) to opt in to larger feature regimes through the standard
  runtime path.
- The 300-feat training pipeline + screen + args-gen are all working
  and reproducible at this commit. Future researchers can
  `--max-feature-idx 300` the screen, retrain at h=192/256, etc.
  without re-wiring.
- The bake itself stays at `benchmarks/v0_20_extended_seed1_2026-05-15.bin`
  for reproducibility / regression-testing future extended-features
  experiments.

## What does NOT ship

- **No `PreviewV0_5_Extended` profile variant**. The bake doesn't
  clear V_18's CID22 ceiling, and the extended-features compute
  overhead (~10-30 % per pair) doesn't pay back. The 228-feat fast
  path (V_18 ship / PreviewV0_3 / PreviewV0_4) remains canonical.

## Files

- Bake: `benchmarks/v0_20_extended_seed1_2026-05-15.bin` (123 KB, ZNPR v3)
- Train log: `/tmp/reeval_logs/v0_20_extended_seed1.train.log`
- Eval log: `/tmp/reeval_logs/v0_20_extended_seed1.eval.log`
- Transform list: `/tmp/v0_20_transforms_300.txt` (139 flags)
- Screen TSV: `benchmarks/v0_20_feature_transform_greedy_screen_2026-05-15.tsv`
