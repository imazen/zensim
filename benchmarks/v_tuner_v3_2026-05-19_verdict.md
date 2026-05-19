# bake_verdict — instant V_X eval

- Bake: `/mnt/v/zen/zensim-eval/exp_tuner_2026-05-18/tuner_v3_s1_calibrated.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: no

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8454 | 0.8489 | 0.6571 | 0.0412 | 0.8979 | 0.528 |
| KADIK10k | 10125 | 0.3711 | 0.4460 | 0.2489 | 0.0301 | 0.5299 | 0.895 |
| TID2013 | 3000 | 0.3702 | 0.6088 | 0.2495 | 0.0250 | 0.4735 | 0.793 |
| KonJND-1k (full) | 1008 | 0.1479 | 0.2423 | 0.1065 | 0.0367 | 0.2792 | 0.970 |
| AIC-3 CTC | 600 | 0.8127 | 0.8224 | 0.6475 | 0.0450 | 0.8832 | 0.569 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8454 | 0.8489 | 0.6571 | 0.0412 | 0.8979 | 0.528 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0570 | 0.1522 | 0.0388 | 0.0526 | 0.0031 | 0.988 | 0.0210 |
| B4 | [0.40, 0.50) | 266 | 0.2802 | 0.2743 | 0.1912 | 0.0414 | 0.3374 | 0.962 | 0.0218 |
| B5 | [0.50, 0.60) | 615 | 0.2667 | 0.2852 | 0.1819 | 0.0472 | 0.3208 | 0.958 | 0.0227 |
| B6 | [0.60, 0.70) | 836 | 0.3108 | 0.3244 | 0.2080 | 0.0299 | 0.3632 | 0.946 | 0.0234 |
| B7 | [0.70, 0.80) | 1092 | 0.3491 | 0.3502 | 0.2387 | 0.0385 | 0.4074 | 0.937 | 0.0229 |
| B8 | [0.80, 0.90) | 1382 | 0.4284 | 0.4300 | 0.2905 | 0.0427 | 0.5010 | 0.903 | 0.0204 |
| B9 | [0.90, 1.00] | 43 | 0.1543 | 0.3333 | 0.1074 | 0.0465 | 0.2894 | 0.943 | 0.0050 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.3711 | 0.4460 | 0.2489 | 0.0301 | 0.5299 | 0.895 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.1891 | 0.1972 | 0.1322 | 0.0482 | 0.2122 | 0.980 | 0.0196 |
| B1 | [0.10, 0.20) | 910 | 0.1418 | 0.1524 | 0.0991 | 0.0352 | 0.1656 | 0.988 | 0.0220 |
| B2 | [0.20, 0.30) | 1111 | 0.1251 | 0.1815 | 0.0864 | 0.0405 | 0.1578 | 0.983 | 0.0244 |
| B3 | [0.30, 0.40) | 1291 | 0.0359 | 0.1378 | 0.0256 | 0.0333 | 0.0438 | 0.990 | 0.0242 |
| B4 | [0.40, 0.50) | 1013 | 0.0153 | 0.1233 | 0.0116 | 0.0316 | 0.0288 | 0.992 | 0.0252 |
| B5 | [0.50, 0.60) | 919 | 0.0013 | 0.0606 | 0.0010 | 0.0392 | 0.0037 | 0.998 | 0.0249 |
| B6 | [0.60, 0.70) | 936 | 0.0071 | 0.0626 | 0.0054 | 0.0342 | 0.0067 | 0.998 | 0.0253 |
| B7 | [0.70, 0.80) | 985 | 0.0654 | 0.1316 | 0.0447 | 0.0457 | 0.0605 | 0.991 | 0.0222 |
| B8 | [0.80, 0.90) | 1699 | 0.2927 | 0.3077 | 0.2039 | 0.0306 | 0.3440 | 0.951 | 0.0244 |
| B9 | [0.90, 1.00] | 486 | 0.1554 | 0.1722 | 0.1080 | 0.0288 | 0.1882 | 0.985 | 0.0125 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.3702 | 0.6088 | 0.2495 | 0.0250 | 0.4735 | 0.793 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.3826 | 0.4735 | 0.2639 | 0.0345 | 0.4137 | 0.881 | 0.0164 |
| B1 | [0.10, 0.20) | 34 | 0.6381 | 0.7163 | 0.4375 | 0.0294 | 0.7643 | 0.698 | 0.0179 |
| B2 | [0.20, 0.30) | 185 | 0.2536 | 0.2749 | 0.1738 | 0.0541 | 0.3213 | 0.961 | 0.0232 |
| B3 | [0.30, 0.40) | 493 | 0.2564 | 0.2854 | 0.1728 | 0.0507 | 0.3186 | 0.958 | 0.0231 |
| B4 | [0.40, 0.50) | 677 | 0.1373 | 0.2087 | 0.0893 | 0.0222 | 0.1691 | 0.978 | 0.0245 |
| B5 | [0.50, 0.60) | 705 | 0.0454 | 0.1804 | 0.0303 | 0.0326 | 0.0615 | 0.984 | 0.0253 |
| B6 | [0.60, 0.70) | 809 | 0.0564 | 0.1522 | 0.0385 | 0.0396 | 0.0691 | 0.988 | 0.0212 |
| B7 | [0.70, 0.80) | 67 | 0.3526 | 0.4453 | 0.2464 | 0.0299 | 0.4157 | 0.895 | 0.0156 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.1479 | 0.2423 | 0.1065 | 0.0367 | 0.2792 | 0.970 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8127 | 0.8224 | 0.6475 | 0.0450 | 0.8832 | 0.569 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 26.53s (19025 pair rows scored across 5 corpora).
