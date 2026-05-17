# bake_verdict — instant V_X eval

- Bake: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/s1_cvvdp_log_h128.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: yes (uses predict_transformed)

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8471 | 0.8506 | 0.6534 | 0.0545 | 0.9043 | 0.526 |
| KADIK10k | 10125 | 0.7969 | 0.7951 | 0.5950 | 0.0242 | 0.8783 | 0.607 |
| TID2013 | 3000 | 0.8140 | 0.8326 | 0.6229 | 0.0417 | 0.8599 | 0.554 |
| KonJND-1k (full) | 1008 | 0.2633 | 0.2087 | 0.1796 | 0.0367 | 0.3831 | 0.978 |
| AIC-3 CTC | 600 | 0.8154 | 0.8273 | 0.6516 | 0.0550 | 0.8863 | 0.562 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8471 | 0.8506 | 0.6534 | 0.0545 | 0.9043 | 0.526 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0425 | 0.1922 | 0.0301 | 0.0526 | 0.0326 | 0.981 | 0.0210 |
| B4 | [0.40, 0.50) | 266 | 0.2553 | 0.2860 | 0.1697 | 0.0451 | 0.3176 | 0.958 | 0.0218 |
| B5 | [0.50, 0.60) | 615 | 0.3614 | 0.3628 | 0.2447 | 0.0439 | 0.4298 | 0.932 | 0.0219 |
| B6 | [0.60, 0.70) | 836 | 0.3637 | 0.3643 | 0.2451 | 0.0490 | 0.4315 | 0.931 | 0.0230 |
| B7 | [0.70, 0.80) | 1092 | 0.3376 | 0.3439 | 0.2313 | 0.0531 | 0.4008 | 0.939 | 0.0231 |
| B8 | [0.80, 0.90) | 1382 | 0.3876 | 0.3874 | 0.2588 | 0.0405 | 0.4611 | 0.922 | 0.0210 |
| B9 | [0.90, 1.00] | 43 | 0.1210 | 0.1805 | 0.0831 | 0.0233 | 0.2190 | 0.984 | 0.0052 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7969 | 0.7951 | 0.5950 | 0.0242 | 0.8783 | 0.607 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.1402 | 0.1432 | 0.0981 | 0.0482 | 0.1927 | 0.990 | 0.0199 |
| B1 | [0.10, 0.20) | 910 | 0.1453 | 0.1492 | 0.1010 | 0.0352 | 0.1984 | 0.989 | 0.0222 |
| B2 | [0.20, 0.30) | 1111 | 0.0731 | 0.0865 | 0.0502 | 0.0279 | 0.0825 | 0.996 | 0.0248 |
| B3 | [0.30, 0.40) | 1291 | 0.1580 | 0.1806 | 0.1101 | 0.0356 | 0.1870 | 0.984 | 0.0239 |
| B4 | [0.40, 0.50) | 1013 | 0.1008 | 0.1272 | 0.0706 | 0.0346 | 0.1262 | 0.992 | 0.0251 |
| B5 | [0.50, 0.60) | 919 | 0.0652 | 0.0816 | 0.0453 | 0.0359 | 0.0805 | 0.997 | 0.0248 |
| B6 | [0.60, 0.70) | 936 | 0.0835 | 0.0957 | 0.0572 | 0.0342 | 0.1053 | 0.995 | 0.0252 |
| B7 | [0.70, 0.80) | 985 | 0.1215 | 0.1386 | 0.0845 | 0.0477 | 0.1333 | 0.990 | 0.0222 |
| B8 | [0.80, 0.90) | 1699 | 0.2972 | 0.3149 | 0.2064 | 0.0347 | 0.3518 | 0.949 | 0.0244 |
| B9 | [0.90, 1.00] | 486 | 0.1372 | 0.1691 | 0.0940 | 0.0309 | 0.1581 | 0.986 | 0.0124 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8140 | 0.8326 | 0.6229 | 0.0417 | 0.8599 | 0.554 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0365 | 0.2495 | 0.0074 | 0.0345 | 0.0902 | 0.968 | 0.0187 |
| B1 | [0.10, 0.20) | 34 | 0.3666 | 0.3967 | 0.2554 | 0.0294 | 0.5057 | 0.918 | 0.0258 |
| B2 | [0.20, 0.30) | 185 | 0.2289 | 0.2502 | 0.1553 | 0.0378 | 0.2949 | 0.968 | 0.0237 |
| B3 | [0.30, 0.40) | 493 | 0.3325 | 0.3362 | 0.2266 | 0.0304 | 0.4297 | 0.942 | 0.0227 |
| B4 | [0.40, 0.50) | 677 | 0.4370 | 0.4431 | 0.2989 | 0.0473 | 0.5138 | 0.896 | 0.0220 |
| B5 | [0.50, 0.60) | 705 | 0.3296 | 0.3350 | 0.2231 | 0.0496 | 0.3991 | 0.942 | 0.0238 |
| B6 | [0.60, 0.70) | 809 | 0.1416 | 0.1606 | 0.0960 | 0.0358 | 0.1690 | 0.987 | 0.0211 |
| B7 | [0.70, 0.80) | 67 | 0.4510 | 0.4943 | 0.3072 | 0.0746 | 0.5187 | 0.869 | 0.0152 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.2633 | 0.2087 | 0.1796 | 0.0367 | 0.3831 | 0.978 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8154 | 0.8273 | 0.6516 | 0.0550 | 0.8863 | 0.562 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 3.75s (19025 pair rows scored across 5 corpora).
