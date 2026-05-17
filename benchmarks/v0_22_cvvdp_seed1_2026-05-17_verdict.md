# bake_verdict — instant V_X eval

- Bake: `benchmarks/v0_22_cvvdp_seed1_2026-05-17.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 300
- Feature transforms: no

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8348 | 0.8331 | 0.6408 | 0.0541 | 0.8955 | 0.553 |
| KADIK10k | 10125 | 0.5788 | 0.5925 | 0.4249 | 0.0277 | 0.6826 | 0.806 |
| TID2013 | 3000 | 0.7331 | 0.7436 | 0.5656 | 0.0340 | 0.7923 | 0.669 |
| KonJND-1k (full) | 1008 | 0.2770 | 0.1995 | 0.1889 | 0.0377 | 0.4005 | 0.980 |
| AIC-3 CTC | 600 | 0.8202 | 0.8302 | 0.6524 | 0.0617 | 0.8895 | 0.557 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8348 | 0.8331 | 0.6408 | 0.0541 | 0.8955 | 0.553 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1941 | 0.2132 | 0.1328 | 0.0526 | 0.2129 | 0.977 | 0.0205 |
| B4 | [0.40, 0.50) | 266 | 0.2429 | 0.2386 | 0.1638 | 0.0414 | 0.3014 | 0.971 | 0.0223 |
| B5 | [0.50, 0.60) | 615 | 0.3390 | 0.3342 | 0.2307 | 0.0455 | 0.4186 | 0.943 | 0.0225 |
| B6 | [0.60, 0.70) | 836 | 0.3495 | 0.3524 | 0.2354 | 0.0431 | 0.4156 | 0.936 | 0.0231 |
| B7 | [0.70, 0.80) | 1092 | 0.3042 | 0.3148 | 0.2068 | 0.0440 | 0.3697 | 0.949 | 0.0235 |
| B8 | [0.80, 0.90) | 1382 | 0.4386 | 0.4394 | 0.2955 | 0.0449 | 0.5121 | 0.898 | 0.0203 |
| B9 | [0.90, 1.00] | 43 | 0.0852 | 0.2439 | 0.0609 | 0.0465 | 0.1382 | 0.970 | 0.0050 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.5788 | 0.5925 | 0.4249 | 0.0277 | 0.6826 | 0.806 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2102 | 0.2337 | 0.1477 | 0.0355 | 0.2595 | 0.972 | 0.0196 |
| B1 | [0.10, 0.20) | 910 | 0.0982 | 0.0588 | 0.0675 | 0.0363 | 0.1357 | 0.998 | 0.0224 |
| B2 | [0.20, 0.30) | 1111 | 0.0652 | 0.0369 | 0.0453 | 0.0324 | 0.0673 | 0.999 | 0.0249 |
| B3 | [0.30, 0.40) | 1291 | 0.1130 | 0.1441 | 0.0791 | 0.0395 | 0.1366 | 0.990 | 0.0241 |
| B4 | [0.40, 0.50) | 1013 | 0.1164 | 0.0478 | 0.0825 | 0.0296 | 0.1266 | 0.999 | 0.0253 |
| B5 | [0.50, 0.60) | 919 | 0.0613 | 0.0504 | 0.0421 | 0.0337 | 0.0858 | 0.999 | 0.0249 |
| B6 | [0.60, 0.70) | 936 | 0.0643 | 0.0957 | 0.0436 | 0.0524 | 0.0790 | 0.995 | 0.0252 |
| B7 | [0.70, 0.80) | 985 | 0.0557 | 0.1532 | 0.0388 | 0.0355 | 0.0491 | 0.988 | 0.0221 |
| B8 | [0.80, 0.90) | 1699 | 0.2898 | 0.3052 | 0.2013 | 0.0353 | 0.3430 | 0.952 | 0.0245 |
| B9 | [0.90, 1.00] | 486 | 0.1626 | 0.1566 | 0.1150 | 0.0370 | 0.1705 | 0.988 | 0.0125 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7331 | 0.7436 | 0.5656 | 0.0340 | 0.7923 | 0.669 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0515 | 0.2398 | 0.0567 | 0.0000 | 0.0226 | 0.971 | 0.0192 |
| B1 | [0.10, 0.20) | 34 | 0.3545 | 0.4640 | 0.2232 | 0.0294 | 0.4604 | 0.886 | 0.0243 |
| B2 | [0.20, 0.30) | 185 | 0.1758 | 0.2289 | 0.1174 | 0.0324 | 0.2373 | 0.973 | 0.0235 |
| B3 | [0.30, 0.40) | 493 | 0.3710 | 0.3846 | 0.2548 | 0.0385 | 0.4330 | 0.923 | 0.0219 |
| B4 | [0.40, 0.50) | 677 | 0.3768 | 0.3925 | 0.2585 | 0.0222 | 0.4369 | 0.920 | 0.0225 |
| B5 | [0.50, 0.60) | 705 | 0.3099 | 0.3177 | 0.2106 | 0.0340 | 0.3723 | 0.948 | 0.0240 |
| B6 | [0.60, 0.70) | 809 | 0.0912 | 0.1655 | 0.0630 | 0.0309 | 0.1168 | 0.986 | 0.0212 |
| B7 | [0.70, 0.80) | 67 | 0.3605 | 0.4027 | 0.2482 | 0.0597 | 0.4252 | 0.915 | 0.0169 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.2770 | 0.1995 | 0.1889 | 0.0377 | 0.4005 | 0.980 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8202 | 0.8302 | 0.6524 | 0.0617 | 0.8895 | 0.557 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 3.68s (19025 pair rows scored across 5 corpora).
