# bake_verdict — instant V_X eval

- Bake: `zensim-experimental/weights/v_tuner_v11_2026-05-24.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: no

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8604 | 0.8525 | 0.6728 | 0.0454 | 0.9089 | 0.523 |
| KADIK10k | 10125 | 0.9237 | 0.9229 | 0.7554 | 0.0551 | 0.9550 | 0.385 |
| TID2013 | 3000 | 0.8849 | 0.8886 | 0.7085 | 0.0517 | 0.9146 | 0.459 |
| KonJND-1k (full) | 1008 | 0.2888 | 0.2586 | 0.1971 | 0.0476 | 0.4043 | 0.966 |
| AIC-3 CTC | 600 | 0.7761 | 0.7877 | 0.6074 | 0.0417 | 0.8538 | 0.616 |
| AIC-4 sample | 300 | 0.9284 | 0.9236 | 0.7717 | 0.0367 | 0.9620 | 0.383 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8604 | 0.8525 | 0.6728 | 0.0454 | 0.9089 | 0.523 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0510 | 0.3128 | 0.0213 | 0.0702 | 0.0940 | 0.950 | 0.0203 |
| B4 | [0.40, 0.50) | 266 | 0.2303 | 0.2835 | 0.1554 | 0.0564 | 0.2864 | 0.959 | 0.0218 |
| B5 | [0.50, 0.60) | 615 | 0.2728 | 0.2840 | 0.1857 | 0.0390 | 0.3220 | 0.959 | 0.0228 |
| B6 | [0.60, 0.70) | 836 | 0.2868 | 0.2917 | 0.1922 | 0.0395 | 0.3305 | 0.957 | 0.0237 |
| B7 | [0.70, 0.80) | 1092 | 0.4077 | 0.4093 | 0.2786 | 0.0476 | 0.4820 | 0.912 | 0.0222 |
| B8 | [0.80, 0.90) | 1382 | 0.4996 | 0.5003 | 0.3389 | 0.0398 | 0.5901 | 0.866 | 0.0196 |
| B9 | [0.90, 1.00] | 43 | 0.2197 | 0.3494 | 0.1429 | 0.0465 | 0.1581 | 0.937 | 0.0048 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9237 | 0.9229 | 0.7554 | 0.0551 | 0.9550 | 0.385 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2446 | 0.2480 | 0.1698 | 0.0227 | 0.3027 | 0.969 | 0.0194 |
| B1 | [0.10, 0.20) | 910 | 0.2502 | 0.2511 | 0.1744 | 0.0429 | 0.3121 | 0.968 | 0.0216 |
| B2 | [0.20, 0.30) | 1111 | 0.1700 | 0.1761 | 0.1176 | 0.0360 | 0.1941 | 0.984 | 0.0244 |
| B3 | [0.30, 0.40) | 1291 | 0.2029 | 0.2119 | 0.1417 | 0.0418 | 0.2409 | 0.977 | 0.0237 |
| B4 | [0.40, 0.50) | 1013 | 0.2225 | 0.2219 | 0.1558 | 0.0395 | 0.2699 | 0.975 | 0.0246 |
| B5 | [0.50, 0.60) | 919 | 0.1999 | 0.2143 | 0.1394 | 0.0392 | 0.2549 | 0.977 | 0.0243 |
| B6 | [0.60, 0.70) | 936 | 0.2151 | 0.2450 | 0.1480 | 0.0406 | 0.2649 | 0.970 | 0.0243 |
| B7 | [0.70, 0.80) | 985 | 0.2403 | 0.2424 | 0.1686 | 0.0457 | 0.2864 | 0.970 | 0.0217 |
| B8 | [0.80, 0.90) | 1699 | 0.4176 | 0.4174 | 0.2929 | 0.0406 | 0.4950 | 0.909 | 0.0231 |
| B9 | [0.90, 1.00] | 486 | 0.1809 | 0.1858 | 0.1247 | 0.0370 | 0.2103 | 0.983 | 0.0124 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8849 | 0.8886 | 0.7085 | 0.0517 | 0.9146 | 0.459 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.3286 | 0.3846 | 0.2343 | 0.0345 | 0.3920 | 0.923 | 0.0173 |
| B1 | [0.10, 0.20) | 34 | 0.6282 | 0.6776 | 0.4268 | 0.0294 | 0.7296 | 0.735 | 0.0193 |
| B2 | [0.20, 0.30) | 185 | 0.3712 | 0.3908 | 0.2507 | 0.0324 | 0.4657 | 0.920 | 0.0219 |
| B3 | [0.30, 0.40) | 493 | 0.4543 | 0.4624 | 0.3130 | 0.0446 | 0.5535 | 0.887 | 0.0211 |
| B4 | [0.40, 0.50) | 677 | 0.5306 | 0.5414 | 0.3666 | 0.0502 | 0.6193 | 0.841 | 0.0204 |
| B5 | [0.50, 0.60) | 705 | 0.4672 | 0.4903 | 0.3231 | 0.0525 | 0.5490 | 0.872 | 0.0215 |
| B6 | [0.60, 0.70) | 809 | 0.2069 | 0.2481 | 0.1408 | 0.0420 | 0.2508 | 0.969 | 0.0207 |
| B7 | [0.70, 0.80) | 67 | 0.3674 | 0.5134 | 0.2518 | 0.0597 | 0.4378 | 0.858 | 0.0149 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.2888 | 0.2586 | 0.1971 | 0.0476 | 0.4043 | 0.966 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7761 | 0.7877 | 0.6074 | 0.0417 | 0.8538 | 0.616 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-4 sample (n=300)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9284 | 0.9236 | 0.7717 | 0.0367 | 0.9620 | 0.383 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-4 sample — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 9.31s (19325 pair rows scored across 6 corpora).
