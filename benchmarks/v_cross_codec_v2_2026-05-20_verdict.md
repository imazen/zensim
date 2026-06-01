# bake_verdict — instant V_X eval

- Bake: `zensim-experimental/weights/v_cross_codec_v2_2026-05-20.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: no

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8797 | 0.8804 | 0.6908 | 0.0422 | 0.9306 | 0.474 |
| KADIK10k | 10125 | 0.8003 | 0.7578 | 0.5997 | 0.0333 | 0.8757 | 0.653 |
| TID2013 | 3000 | 0.8215 | 0.8027 | 0.6282 | 0.0303 | 0.8707 | 0.596 |
| KonJND-1k (full) | 1008 | 0.3269 | 0.2797 | 0.2288 | 0.0308 | 0.4891 | 0.960 |
| AIC-3 CTC | 600 | 0.8060 | 0.8220 | 0.6462 | 0.0533 | 0.8791 | 0.570 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8797 | 0.8804 | 0.6908 | 0.0422 | 0.9306 | 0.474 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0200 | 0.2057 | 0.0088 | 0.0526 | 0.0620 | 0.979 | 0.0213 |
| B4 | [0.40, 0.50) | 266 | 0.2596 | 0.2623 | 0.1787 | 0.0489 | 0.3252 | 0.965 | 0.0221 |
| B5 | [0.50, 0.60) | 615 | 0.3575 | 0.3598 | 0.2433 | 0.0504 | 0.4172 | 0.933 | 0.0219 |
| B6 | [0.60, 0.70) | 836 | 0.3629 | 0.3684 | 0.2444 | 0.0311 | 0.4316 | 0.930 | 0.0230 |
| B7 | [0.70, 0.80) | 1092 | 0.3607 | 0.3660 | 0.2474 | 0.0385 | 0.4264 | 0.931 | 0.0227 |
| B8 | [0.80, 0.90) | 1382 | 0.4566 | 0.4616 | 0.3089 | 0.0470 | 0.5397 | 0.887 | 0.0201 |
| B9 | [0.90, 1.00] | 43 | 0.0526 | 0.2823 | 0.0343 | 0.0698 | 0.0088 | 0.959 | 0.0050 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8003 | 0.7578 | 0.5997 | 0.0333 | 0.8757 | 0.653 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.1793 | 0.1852 | 0.1257 | 0.0383 | 0.2364 | 0.983 | 0.0198 |
| B1 | [0.10, 0.20) | 910 | 0.1424 | 0.1647 | 0.0990 | 0.0374 | 0.1944 | 0.986 | 0.0222 |
| B2 | [0.20, 0.30) | 1111 | 0.0382 | 0.0982 | 0.0263 | 0.0315 | 0.0363 | 0.995 | 0.0248 |
| B3 | [0.30, 0.40) | 1291 | 0.1252 | 0.1210 | 0.0869 | 0.0356 | 0.1535 | 0.993 | 0.0242 |
| B4 | [0.40, 0.50) | 1013 | 0.1254 | 0.1391 | 0.0872 | 0.0336 | 0.1471 | 0.990 | 0.0251 |
| B5 | [0.50, 0.60) | 919 | 0.0791 | 0.0669 | 0.0549 | 0.0316 | 0.1047 | 0.998 | 0.0249 |
| B6 | [0.60, 0.70) | 936 | 0.0933 | 0.0872 | 0.0643 | 0.0331 | 0.1137 | 0.996 | 0.0252 |
| B7 | [0.70, 0.80) | 985 | 0.1495 | 0.1553 | 0.1051 | 0.0386 | 0.1598 | 0.988 | 0.0221 |
| B8 | [0.80, 0.90) | 1699 | 0.3425 | 0.3428 | 0.2390 | 0.0235 | 0.4019 | 0.939 | 0.0240 |
| B9 | [0.90, 1.00] | 486 | 0.1468 | 0.1889 | 0.1035 | 0.0165 | 0.1544 | 0.982 | 0.0125 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8215 | 0.8027 | 0.6282 | 0.0303 | 0.8707 | 0.596 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0192 | 0.2398 | 0.0123 | 0.0000 | 0.0086 | 0.971 | 0.0192 |
| B1 | [0.10, 0.20) | 34 | 0.4535 | 0.5556 | 0.3196 | 0.0588 | 0.5998 | 0.831 | 0.0231 |
| B2 | [0.20, 0.30) | 185 | 0.2475 | 0.2735 | 0.1657 | 0.0486 | 0.3152 | 0.962 | 0.0232 |
| B3 | [0.30, 0.40) | 493 | 0.3133 | 0.3383 | 0.2131 | 0.0365 | 0.3868 | 0.941 | 0.0227 |
| B4 | [0.40, 0.50) | 677 | 0.4029 | 0.3906 | 0.2757 | 0.0428 | 0.4772 | 0.921 | 0.0227 |
| B5 | [0.50, 0.60) | 705 | 0.3147 | 0.3200 | 0.2129 | 0.0213 | 0.3851 | 0.947 | 0.0240 |
| B6 | [0.60, 0.70) | 809 | 0.1036 | 0.1500 | 0.0702 | 0.0173 | 0.1281 | 0.989 | 0.0212 |
| B7 | [0.70, 0.80) | 67 | 0.3858 | 0.4249 | 0.2645 | 0.0597 | 0.4600 | 0.905 | 0.0165 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.3269 | 0.2797 | 0.2288 | 0.0308 | 0.4891 | 0.960 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8060 | 0.8220 | 0.6462 | 0.0533 | 0.8791 | 0.570 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 4.77s (19025 pair rows scored across 5 corpora).
