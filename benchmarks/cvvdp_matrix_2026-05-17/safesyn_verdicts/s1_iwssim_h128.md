# bake_verdict — instant V_X eval

- Bake: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/s1_iwssim_h128.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: yes (uses predict_transformed)

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8319 | 0.8392 | 0.6469 | 0.0431 | 0.8872 | 0.544 |
| KADIK10k | 10125 | 0.7501 | 0.7497 | 0.5646 | 0.0491 | 0.8291 | 0.662 |
| TID2013 | 3000 | 0.7439 | 0.7786 | 0.5604 | 0.0580 | 0.8117 | 0.628 |
| KonJND-1k (full) | 1008 | 0.0871 | 0.1725 | 0.0626 | 0.0298 | 0.1485 | 0.985 |
| AIC-3 CTC | 600 | 0.7775 | 0.7935 | 0.6141 | 0.0667 | 0.8545 | 0.609 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8319 | 0.8392 | 0.6469 | 0.0431 | 0.8872 | 0.544 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1135 | 0.1733 | 0.0727 | 0.0702 | 0.0440 | 0.985 | 0.0204 |
| B4 | [0.40, 0.50) | 266 | 0.2114 | 0.2696 | 0.1419 | 0.0414 | 0.2753 | 0.963 | 0.0221 |
| B5 | [0.50, 0.60) | 615 | 0.2833 | 0.2937 | 0.1891 | 0.0488 | 0.3389 | 0.956 | 0.0227 |
| B6 | [0.60, 0.70) | 836 | 0.2761 | 0.2912 | 0.1846 | 0.0443 | 0.3254 | 0.957 | 0.0237 |
| B7 | [0.70, 0.80) | 1092 | 0.3275 | 0.3339 | 0.2253 | 0.0458 | 0.3741 | 0.943 | 0.0232 |
| B8 | [0.80, 0.90) | 1382 | 0.4531 | 0.4629 | 0.3115 | 0.0499 | 0.5316 | 0.886 | 0.0200 |
| B9 | [0.90, 1.00] | 43 | 0.0495 | 0.4694 | 0.0078 | 0.0698 | 0.1645 | 0.883 | 0.0045 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7501 | 0.7497 | 0.5646 | 0.0491 | 0.8291 | 0.662 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.0888 | 0.1304 | 0.0612 | 0.0539 | 0.1089 | 0.991 | 0.0199 |
| B1 | [0.10, 0.20) | 910 | 0.0793 | 0.0952 | 0.0544 | 0.0352 | 0.1084 | 0.995 | 0.0223 |
| B2 | [0.20, 0.30) | 1111 | 0.0819 | 0.0958 | 0.0561 | 0.0342 | 0.1088 | 0.995 | 0.0247 |
| B3 | [0.30, 0.40) | 1291 | 0.1746 | 0.1989 | 0.1209 | 0.0434 | 0.2160 | 0.980 | 0.0237 |
| B4 | [0.40, 0.50) | 1013 | 0.1405 | 0.1663 | 0.0987 | 0.0494 | 0.1626 | 0.986 | 0.0250 |
| B5 | [0.50, 0.60) | 919 | 0.1127 | 0.1194 | 0.0785 | 0.0424 | 0.1546 | 0.993 | 0.0248 |
| B6 | [0.60, 0.70) | 936 | 0.0775 | 0.0904 | 0.0528 | 0.0417 | 0.0911 | 0.996 | 0.0252 |
| B7 | [0.70, 0.80) | 985 | 0.2074 | 0.2183 | 0.1473 | 0.0518 | 0.2406 | 0.976 | 0.0218 |
| B8 | [0.80, 0.90) | 1699 | 0.3723 | 0.3799 | 0.2610 | 0.0418 | 0.4478 | 0.925 | 0.0236 |
| B9 | [0.90, 1.00] | 486 | 0.1480 | 0.1712 | 0.1055 | 0.0412 | 0.1616 | 0.985 | 0.0124 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7439 | 0.7786 | 0.5604 | 0.0580 | 0.8117 | 0.628 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0374 | 0.2269 | 0.0518 | 0.0345 | 0.0195 | 0.974 | 0.0190 |
| B1 | [0.10, 0.20) | 34 | 0.2639 | 0.3352 | 0.1911 | 0.0588 | 0.4043 | 0.942 | 0.0269 |
| B2 | [0.20, 0.30) | 185 | 0.0484 | 0.1881 | 0.0325 | 0.0541 | 0.0696 | 0.982 | 0.0238 |
| B3 | [0.30, 0.40) | 493 | 0.2385 | 0.2692 | 0.1619 | 0.0385 | 0.2859 | 0.963 | 0.0231 |
| B4 | [0.40, 0.50) | 677 | 0.3383 | 0.3577 | 0.2285 | 0.0428 | 0.4059 | 0.934 | 0.0231 |
| B5 | [0.50, 0.60) | 705 | 0.3235 | 0.3300 | 0.2236 | 0.0482 | 0.3918 | 0.944 | 0.0240 |
| B6 | [0.60, 0.70) | 809 | 0.0930 | 0.1259 | 0.0634 | 0.0396 | 0.1124 | 0.992 | 0.0213 |
| B7 | [0.70, 0.80) | 67 | 0.3850 | 0.3984 | 0.2636 | 0.0448 | 0.4320 | 0.917 | 0.0166 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.0871 | 0.1725 | 0.0626 | 0.0298 | 0.1485 | 0.985 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7775 | 0.7935 | 0.6141 | 0.0667 | 0.8545 | 0.609 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 3.45s (19025 pair rows scored across 5 corpora).
