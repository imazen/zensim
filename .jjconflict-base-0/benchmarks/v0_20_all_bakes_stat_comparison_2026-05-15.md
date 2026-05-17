# V_20 full statistical-rigor comparison across all bakes (2026-05-15)

Per CLAUDE.md mandate, every eval emits SROCC + PLCC + KROCC + 
OR + PWRC + Z-RMSE. This consolidates the V_X bake row from each
eval log into a single comparison.

Static baselines for reference: V_2 (linear) + fast-ssim2 +
butteraugli (Z-RMSE notes: corpus-wide σ on KADID/TID/CID22 since
they don't carry bootstrap σ; AIC-3/AIC-4 have per-stimulus σ).

## KADIK10k

| Bake | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V0_2 (static) | 0.8192 | 0.7136 | 0.6230 | 0.0430 | 0.8879 | 0.577 |
| fast-ssim2 (static) | 0.8133 | 0.8030 | 0.6174 | 0.0516 | 0.8828 | 0.585 |
| butteraugli (static) | 0.6062 | 0.4451 | 0.4304 | 0.0381 | 0.7041 | 0.778 |
| **V_18 ship (3-way concat)** | **0.9427** | 0.8757 | 0.7930 | 0.0426 | 0.9656 | 0.332 |
| **V_18 base seed=1 (single MLP)** | **0.9464** | 0.8535 | 0.7995 | 0.0388 | 0.9682 | 0.321 |
| **V_20 IS (98 transforms, single MLP)** | **0.9497** | 0.8265 | 0.8054 | 0.0387 | 0.9706 | 0.311 |
| **V_20b distortion manifold** | **0.9656** | 0.9515 | 0.8390 | 0.0450 | 0.9795 | 0.256 |
| **D1 3-way concat with transforms** | **0.9504** | 0.8562 | 0.8066 | 0.0411 | 0.9708 | 0.309 |
| **D3 tighter transforms (lift>=0.10)** | **0.9471** | 0.8995 | 0.8005 | 0.0285 | 0.9684 | 0.319 |

## TID2013

| Bake | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V0_2 (static) | 0.8427 | 0.8345 | 0.6664 | 0.0397 | 0.8781 | 0.511 |
| fast-ssim2 (static) | 0.8460 | 0.8486 | 0.6614 | 0.0467 | 0.8846 | 0.526 |
| butteraugli (static) | 0.6696 | 0.4884 | 0.4922 | 0.0393 | 0.7276 | 0.750 |
| **V_18 ship (3-way concat)** | **0.9526** | 0.9309 | 0.8110 | 0.0313 | 0.9702 | 0.294 |
| **V_18 base seed=1 (single MLP)** | **0.9568** | 0.9345 | 0.8185 | 0.0257 | 0.9731 | 0.289 |
| **V_20 IS (98 transforms, single MLP)** | **0.9616** | 0.9552 | 0.8280 | 0.0497 | 0.9764 | 0.271 |
| **V_20b distortion manifold** | **0.9793** | 0.9784 | 0.8746 | 0.0480 | 0.9874 | 0.204 |
| **D1 3-way concat with transforms** | **0.9616** | 0.9568 | 0.8283 | 0.0480 | 0.9764 | 0.269 |
| **D3 tighter transforms (lift>=0.10)** | **0.9568** | 0.9528 | 0.8176 | 0.0477 | 0.9733 | 0.282 |

## CID22

| Bake | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V0_2 (static) | 0.8676 | 0.8561 | 0.6786 | 0.0478 | 0.9174 | 0.498 |
| fast-ssim2 (static) | 0.8895 | 0.8778 | 0.7062 | 0.0424 | 0.9351 | 0.460 |
| butteraugli (static) | 0.7412 | 0.6841 | 0.5508 | 0.0450 | 0.8124 | 0.681 |
| **V_18 ship (3-way concat)** | **0.8933** | 0.8679 | 0.7081 | 0.0536 | 0.9373 | 0.455 |
| **V_18 base seed=1 (single MLP)** | **0.8880** | 0.8522 | 0.7040 | 0.0566 | 0.9326 | 0.465 |
| **V_20 IS (98 transforms, single MLP)** | **0.8794** | 0.8126 | 0.6915 | 0.0513 | 0.9271 | 0.482 |
| **V_20b distortion manifold** | **0.8660** | 0.8527 | 0.6763 | 0.0482 | 0.9126 | 0.484 |
| **D1 3-way concat with transforms** | **0.8794** | 0.8213 | 0.6922 | 0.0492 | 0.9269 | 0.480 |
| **D3 tighter transforms (lift>=0.10)** | **0.8782** | 0.7996 | 0.6901 | 0.0475 | 0.9273 | 0.481 |

## Reading notes

- **SROCC** is rank correlation. Calibration-invariant.
- **PLCC** is Pearson on calibrated outputs vs MOS. Sensitive to
  output scale — V_20 IS bakes are RAW (no affine calibration),
  so their PLCC can mislead. V_18 ship is affine-calibrated.
- **KROCC** is Kendall-τ — alternative to SROCC; sometimes more
  stable at low n.
- **OR** = outlier ratio (fraction of predictions outside ±2σ of
  subjective). Lower is better.
- **PWRC** = Pearson-weighted rank correlation (Mohammadi 2025).
- **Z-RMSE** = σ-normalized RMSE on calibrated outputs. Lower is
  better. On KADID/TID/CID22 this uses corpus-wide σ (less
  informative than the AIC-3 per-stimulus form).

**Caveat for V_20 IS / V_20b / D1 / D3**: PLCC + Z-RMSE on these
rows reflect the bake's RAW output range, not a calibrated 0..100
score. For direct comparison with V_18 ship's PLCC, the V_X bakes
would need affine calibration via
`scripts/v_next/affine_calibrate_znpr_v2.py`. SROCC + KROCC +
PWRC are calibration-invariant and tell the true ranking story.
