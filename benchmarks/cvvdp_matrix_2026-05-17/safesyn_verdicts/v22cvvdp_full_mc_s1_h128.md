# bake_verdict — instant V_X eval

- Bake: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22cvvdp_full_mc_s1_h128.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: yes (uses predict_transformed)

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8425 | 0.8429 | 0.6476 | 0.0496 | 0.9020 | 0.538 |
| KADIK10k | 10125 | 0.8341 | 0.8335 | 0.6400 | 0.0336 | 0.9009 | 0.552 |
| TID2013 | 3000 | 0.8581 | 0.8696 | 0.6784 | 0.0433 | 0.8896 | 0.494 |
| KonJND-1k (full) | 1008 | 0.2131 | 0.1398 | 0.1463 | 0.0357 | 0.3430 | 0.990 |
| AIC-3 CTC | 600 | 0.8313 | 0.8360 | 0.6678 | 0.0583 | 0.8974 | 0.549 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8425 | 0.8429 | 0.6476 | 0.0496 | 0.9020 | 0.538 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1552 | 0.2607 | 0.1015 | 0.0351 | 0.1657 | 0.965 | 0.0205 |
| B4 | [0.40, 0.50) | 266 | 0.2363 | 0.2467 | 0.1535 | 0.0414 | 0.3096 | 0.969 | 0.0223 |
| B5 | [0.50, 0.60) | 615 | 0.3460 | 0.3633 | 0.2331 | 0.0472 | 0.4129 | 0.932 | 0.0220 |
| B6 | [0.60, 0.70) | 836 | 0.3455 | 0.3479 | 0.2334 | 0.0455 | 0.4066 | 0.938 | 0.0231 |
| B7 | [0.70, 0.80) | 1092 | 0.3341 | 0.3391 | 0.2291 | 0.0504 | 0.3983 | 0.941 | 0.0231 |
| B8 | [0.80, 0.90) | 1382 | 0.3915 | 0.3935 | 0.2622 | 0.0441 | 0.4633 | 0.919 | 0.0208 |
| B9 | [0.90, 1.00] | 43 | 0.1057 | 0.1805 | 0.0831 | 0.0698 | 0.2030 | 0.984 | 0.0052 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8341 | 0.8335 | 0.6400 | 0.0336 | 0.9009 | 0.552 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.1673 | 0.1784 | 0.1172 | 0.0411 | 0.2182 | 0.984 | 0.0198 |
| B1 | [0.10, 0.20) | 910 | 0.1716 | 0.2030 | 0.1188 | 0.0385 | 0.2257 | 0.979 | 0.0220 |
| B2 | [0.20, 0.30) | 1111 | 0.0692 | 0.0731 | 0.0479 | 0.0351 | 0.0728 | 0.997 | 0.0248 |
| B3 | [0.30, 0.40) | 1291 | 0.1503 | 0.1716 | 0.1050 | 0.0442 | 0.1748 | 0.985 | 0.0240 |
| B4 | [0.40, 0.50) | 1013 | 0.1720 | 0.2054 | 0.1223 | 0.0444 | 0.2056 | 0.979 | 0.0247 |
| B5 | [0.50, 0.60) | 919 | 0.1106 | 0.1680 | 0.0767 | 0.0424 | 0.1387 | 0.986 | 0.0246 |
| B6 | [0.60, 0.70) | 936 | 0.1082 | 0.1505 | 0.0740 | 0.0374 | 0.1296 | 0.989 | 0.0249 |
| B7 | [0.70, 0.80) | 985 | 0.1513 | 0.1691 | 0.1044 | 0.0437 | 0.1642 | 0.986 | 0.0220 |
| B8 | [0.80, 0.90) | 1699 | 0.3902 | 0.3961 | 0.2723 | 0.0341 | 0.4627 | 0.918 | 0.0234 |
| B9 | [0.90, 1.00] | 486 | 0.1639 | 0.2150 | 0.1128 | 0.0391 | 0.1866 | 0.977 | 0.0122 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8581 | 0.8696 | 0.6784 | 0.0433 | 0.8896 | 0.494 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.1875 | 0.2377 | 0.1455 | 0.0345 | 0.1736 | 0.971 | 0.0190 |
| B1 | [0.10, 0.20) | 34 | 0.4613 | 0.5906 | 0.3161 | 0.0588 | 0.5849 | 0.807 | 0.0217 |
| B2 | [0.20, 0.30) | 185 | 0.2659 | 0.2945 | 0.1824 | 0.0649 | 0.3343 | 0.956 | 0.0230 |
| B3 | [0.30, 0.40) | 493 | 0.4160 | 0.4328 | 0.2869 | 0.0487 | 0.5216 | 0.901 | 0.0217 |
| B4 | [0.40, 0.50) | 677 | 0.5026 | 0.5150 | 0.3489 | 0.0443 | 0.5920 | 0.857 | 0.0209 |
| B5 | [0.50, 0.60) | 705 | 0.4228 | 0.4601 | 0.2903 | 0.0454 | 0.5067 | 0.888 | 0.0221 |
| B6 | [0.60, 0.70) | 809 | 0.1608 | 0.2211 | 0.1089 | 0.0346 | 0.1975 | 0.975 | 0.0209 |
| B7 | [0.70, 0.80) | 67 | 0.4106 | 0.4804 | 0.2818 | 0.0896 | 0.4761 | 0.877 | 0.0152 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.2131 | 0.1398 | 0.1463 | 0.0357 | 0.3430 | 0.990 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8313 | 0.8360 | 0.6678 | 0.0583 | 0.8974 | 0.549 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 3.49s (19025 pair rows scored across 5 corpora).
