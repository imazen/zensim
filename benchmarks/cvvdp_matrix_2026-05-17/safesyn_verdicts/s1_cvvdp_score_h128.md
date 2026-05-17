# bake_verdict — instant V_X eval

- Bake: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/s1_cvvdp_score_h128.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: yes (uses predict_transformed)

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.7734 | 0.7469 | 0.5792 | 0.0582 | 0.8401 | 0.665 |
| KADIK10k | 10125 | 0.7475 | 0.7471 | 0.5583 | 0.0596 | 0.8151 | 0.665 |
| TID2013 | 3000 | 0.7486 | 0.7400 | 0.5468 | 0.0540 | 0.8127 | 0.673 |
| KonJND-1k (full) | 1008 | 0.4999 | 0.4486 | 0.3414 | 0.0387 | 0.5951 | 0.894 |
| AIC-3 CTC | 600 | 0.7140 | 0.7320 | 0.5436 | 0.0450 | 0.8063 | 0.681 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7734 | 0.7469 | 0.5792 | 0.0582 | 0.8401 | 0.665 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.2717 | 0.2962 | 0.1855 | 0.0351 | 0.2224 | 0.955 | 0.0195 |
| B4 | [0.40, 0.50) | 266 | 0.1162 | 0.1807 | 0.0766 | 0.0414 | 0.1665 | 0.984 | 0.0228 |
| B5 | [0.50, 0.60) | 615 | 0.2471 | 0.2527 | 0.1662 | 0.0488 | 0.3102 | 0.968 | 0.0232 |
| B6 | [0.60, 0.70) | 836 | 0.2694 | 0.2751 | 0.1814 | 0.0419 | 0.3170 | 0.961 | 0.0239 |
| B7 | [0.70, 0.80) | 1092 | 0.3216 | 0.3432 | 0.2184 | 0.0467 | 0.3946 | 0.939 | 0.0232 |
| B8 | [0.80, 0.90) | 1382 | 0.4200 | 0.4286 | 0.2838 | 0.0398 | 0.4982 | 0.903 | 0.0203 |
| B9 | [0.90, 1.00] | 43 | 0.0083 | 0.2107 | 0.0055 | 0.0233 | 0.0908 | 0.978 | 0.0051 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7475 | 0.7471 | 0.5583 | 0.0596 | 0.8151 | 0.665 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.0965 | 0.1600 | 0.0673 | 0.0369 | 0.1364 | 0.987 | 0.0199 |
| B1 | [0.10, 0.20) | 910 | 0.0697 | 0.1035 | 0.0485 | 0.0352 | 0.1012 | 0.995 | 0.0222 |
| B2 | [0.20, 0.30) | 1111 | 0.0203 | 0.0773 | 0.0136 | 0.0459 | 0.0263 | 0.997 | 0.0248 |
| B3 | [0.30, 0.40) | 1291 | 0.1672 | 0.1847 | 0.1170 | 0.0480 | 0.2045 | 0.983 | 0.0238 |
| B4 | [0.40, 0.50) | 1013 | 0.1418 | 0.1605 | 0.0979 | 0.0474 | 0.1643 | 0.987 | 0.0249 |
| B5 | [0.50, 0.60) | 919 | 0.1082 | 0.1414 | 0.0746 | 0.0468 | 0.1387 | 0.990 | 0.0247 |
| B6 | [0.60, 0.70) | 936 | 0.0819 | 0.0909 | 0.0568 | 0.0459 | 0.1043 | 0.996 | 0.0252 |
| B7 | [0.70, 0.80) | 985 | 0.1635 | 0.1838 | 0.1144 | 0.0497 | 0.1830 | 0.983 | 0.0220 |
| B8 | [0.80, 0.90) | 1699 | 0.3571 | 0.3587 | 0.2497 | 0.0447 | 0.4244 | 0.933 | 0.0239 |
| B9 | [0.90, 1.00] | 486 | 0.1488 | 0.1373 | 0.1072 | 0.0412 | 0.1582 | 0.991 | 0.0126 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7486 | 0.7400 | 0.5468 | 0.0540 | 0.8127 | 0.673 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.1882 | 0.3288 | 0.1307 | 0.0345 | 0.2509 | 0.944 | 0.0185 |
| B1 | [0.10, 0.20) | 34 | 0.0584 | 0.3109 | 0.0161 | 0.0000 | 0.0427 | 0.950 | 0.0258 |
| B2 | [0.20, 0.30) | 185 | 0.0506 | 0.2367 | 0.0324 | 0.0324 | 0.0539 | 0.972 | 0.0233 |
| B3 | [0.30, 0.40) | 493 | 0.1032 | 0.2170 | 0.0700 | 0.0426 | 0.1108 | 0.976 | 0.0236 |
| B4 | [0.40, 0.50) | 677 | 0.2756 | 0.2859 | 0.1850 | 0.0414 | 0.3333 | 0.958 | 0.0239 |
| B5 | [0.50, 0.60) | 705 | 0.2478 | 0.2487 | 0.1707 | 0.0468 | 0.2984 | 0.969 | 0.0248 |
| B6 | [0.60, 0.70) | 809 | 0.1598 | 0.2108 | 0.1104 | 0.0420 | 0.1841 | 0.978 | 0.0208 |
| B7 | [0.70, 0.80) | 67 | 0.3428 | 0.4612 | 0.2337 | 0.0746 | 0.3940 | 0.887 | 0.0157 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.4999 | 0.4486 | 0.3414 | 0.0387 | 0.5951 | 0.894 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7140 | 0.7320 | 0.5436 | 0.0450 | 0.8063 | 0.681 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 5.01s (19025 pair rows scored across 5 corpora).
