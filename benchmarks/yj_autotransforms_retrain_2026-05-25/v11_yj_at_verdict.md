# bake_verdict — instant V_X eval

- Bake: `zensim/weights/v_tuner_v11_yj_autotransforms_2026-05-25.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: yes (uses predict_transformed)

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8155 | 0.8157 | 0.6129 | 0.0415 | 0.8809 | 0.578 |
| KADIK10k | 10125 | 0.8977 | 0.8977 | 0.7183 | 0.0449 | 0.9378 | 0.441 |
| TID2013 | 3000 | 0.8812 | 0.8937 | 0.7004 | 0.0480 | 0.9131 | 0.449 |
| KonJND-1k (full) | 1008 | 0.6656 | 0.7284 | 0.4571 | 0.0427 | 0.7202 | 0.685 |
| AIC-3 CTC | 600 | 0.7494 | 0.7431 | 0.5794 | 0.0467 | 0.8257 | 0.669 |
| AIC-4 sample | 300 | 0.9193 | 0.9147 | 0.7585 | 0.0567 | 0.9523 | 0.404 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8155 | 0.8157 | 0.6129 | 0.0415 | 0.8809 | 0.578 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0656 | 0.2547 | 0.0401 | 0.0526 | 0.0476 | 0.967 | 0.0206 |
| B4 | [0.40, 0.50) | 266 | 0.1518 | 0.1463 | 0.1008 | 0.0414 | 0.1971 | 0.989 | 0.0229 |
| B5 | [0.50, 0.60) | 615 | 0.2396 | 0.2588 | 0.1627 | 0.0423 | 0.2835 | 0.966 | 0.0230 |
| B6 | [0.60, 0.70) | 836 | 0.2569 | 0.2728 | 0.1726 | 0.0467 | 0.3007 | 0.962 | 0.0240 |
| B7 | [0.70, 0.80) | 1092 | 0.3016 | 0.3179 | 0.2045 | 0.0458 | 0.3579 | 0.948 | 0.0234 |
| B8 | [0.80, 0.90) | 1382 | 0.3597 | 0.3664 | 0.2407 | 0.0369 | 0.4259 | 0.930 | 0.0211 |
| B9 | [0.90, 1.00] | 43 | 0.0947 | 0.2736 | 0.0543 | 0.0233 | 0.1682 | 0.962 | 0.0050 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8977 | 0.8977 | 0.7183 | 0.0449 | 0.9378 | 0.441 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2022 | 0.2154 | 0.1409 | 0.0440 | 0.2471 | 0.977 | 0.0195 |
| B1 | [0.10, 0.20) | 910 | 0.2056 | 0.2187 | 0.1437 | 0.0418 | 0.2486 | 0.976 | 0.0217 |
| B2 | [0.20, 0.30) | 1111 | 0.1623 | 0.1700 | 0.1111 | 0.0405 | 0.1838 | 0.985 | 0.0244 |
| B3 | [0.30, 0.40) | 1291 | 0.1829 | 0.1838 | 0.1278 | 0.0411 | 0.2173 | 0.983 | 0.0239 |
| B4 | [0.40, 0.50) | 1013 | 0.2030 | 0.2111 | 0.1427 | 0.0415 | 0.2422 | 0.977 | 0.0247 |
| B5 | [0.50, 0.60) | 919 | 0.1539 | 0.1783 | 0.1068 | 0.0424 | 0.1917 | 0.984 | 0.0245 |
| B6 | [0.60, 0.70) | 936 | 0.1823 | 0.1909 | 0.1272 | 0.0449 | 0.2282 | 0.982 | 0.0248 |
| B7 | [0.70, 0.80) | 985 | 0.1769 | 0.1815 | 0.1234 | 0.0426 | 0.2023 | 0.983 | 0.0220 |
| B8 | [0.80, 0.90) | 1699 | 0.3684 | 0.3722 | 0.2570 | 0.0383 | 0.4410 | 0.928 | 0.0237 |
| B9 | [0.90, 1.00] | 486 | 0.1627 | 0.1517 | 0.1141 | 0.0370 | 0.1803 | 0.988 | 0.0125 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8812 | 0.8937 | 0.7004 | 0.0480 | 0.9131 | 0.449 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.4200 | 0.5334 | 0.2885 | 0.0345 | 0.4871 | 0.846 | 0.0150 |
| B1 | [0.10, 0.20) | 34 | 0.6232 | 0.6682 | 0.4339 | 0.0588 | 0.7425 | 0.744 | 0.0206 |
| B2 | [0.20, 0.30) | 185 | 0.2866 | 0.4390 | 0.2023 | 0.0432 | 0.3436 | 0.898 | 0.0206 |
| B3 | [0.30, 0.40) | 493 | 0.4498 | 0.4537 | 0.3109 | 0.0467 | 0.5379 | 0.891 | 0.0211 |
| B4 | [0.40, 0.50) | 677 | 0.4869 | 0.4937 | 0.3381 | 0.0458 | 0.5749 | 0.870 | 0.0213 |
| B5 | [0.50, 0.60) | 705 | 0.4443 | 0.4687 | 0.3073 | 0.0454 | 0.5252 | 0.883 | 0.0218 |
| B6 | [0.60, 0.70) | 809 | 0.1915 | 0.2331 | 0.1310 | 0.0457 | 0.2313 | 0.972 | 0.0208 |
| B7 | [0.70, 0.80) | 67 | 0.3801 | 0.4964 | 0.2627 | 0.0448 | 0.4353 | 0.868 | 0.0153 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.6656 | 0.7284 | 0.4571 | 0.0427 | 0.7202 | 0.685 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7494 | 0.7431 | 0.5794 | 0.0467 | 0.8257 | 0.669 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-4 sample (n=300)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9193 | 0.9147 | 0.7585 | 0.0567 | 0.9523 | 0.404 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-4 sample — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 5.66s (19325 pair rows scored across 6 corpora).
