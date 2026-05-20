# bake_verdict — instant V_X eval

- Bake: `/home/lilith/work/zen/zensim--cross-codec-v8/zensim/weights/v_balanced_v3_2026-05-20.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 300
- Feature transforms: no

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8324 | 0.8256 | 0.6340 | 0.0496 | 0.9006 | 0.564 |
| KADIK10k | 10125 | 0.9664 | 0.9562 | 0.8420 | 0.0427 | 0.9793 | 0.293 |
| TID2013 | 3000 | 0.9712 | 0.9379 | 0.8521 | 0.0353 | 0.9815 | 0.347 |
| KonJND-1k (full) | 1008 | 0.8927 | 0.9270 | 0.7070 | 0.0437 | 0.9178 | 0.375 |
| AIC-3 CTC | 600 | 0.7845 | 0.7952 | 0.6155 | 0.0467 | 0.8630 | 0.606 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8324 | 0.8256 | 0.6340 | 0.0496 | 0.9006 | 0.564 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0487 | 0.1900 | 0.0276 | 0.0351 | 0.0703 | 0.982 | 0.0207 |
| B4 | [0.40, 0.50) | 266 | 0.2510 | 0.2456 | 0.1658 | 0.0489 | 0.3169 | 0.969 | 0.0223 |
| B5 | [0.50, 0.60) | 615 | 0.2466 | 0.2592 | 0.1655 | 0.0407 | 0.3049 | 0.966 | 0.0230 |
| B6 | [0.60, 0.70) | 836 | 0.2220 | 0.2316 | 0.1484 | 0.0335 | 0.2581 | 0.973 | 0.0243 |
| B7 | [0.70, 0.80) | 1092 | 0.3192 | 0.3245 | 0.2173 | 0.0495 | 0.3766 | 0.946 | 0.0233 |
| B8 | [0.80, 0.90) | 1382 | 0.4958 | 0.5033 | 0.3354 | 0.0384 | 0.5846 | 0.864 | 0.0195 |
| B9 | [0.90, 1.00] | 43 | 0.1056 | 0.1831 | 0.0698 | 0.0465 | 0.2725 | 0.983 | 0.0054 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9664 | 0.9562 | 0.8420 | 0.0427 | 0.9793 | 0.293 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.1274 | 0.1174 | 0.1077 | 0.0057 | 0.1547 | 0.993 | 0.0198 |
| B1 | [0.10, 0.20) | 910 | 0.3127 | 0.3072 | 0.2484 | 0.0253 | 0.3601 | 0.952 | 0.0209 |
| B2 | [0.20, 0.30) | 1111 | 0.3982 | 0.3906 | 0.2821 | 0.0477 | 0.4729 | 0.921 | 0.0225 |
| B3 | [0.30, 0.40) | 1291 | 0.3342 | 0.3235 | 0.2352 | 0.0442 | 0.3953 | 0.946 | 0.0226 |
| B4 | [0.40, 0.50) | 1013 | 0.3754 | 0.3793 | 0.2666 | 0.0326 | 0.4405 | 0.925 | 0.0228 |
| B5 | [0.50, 0.60) | 919 | 0.3454 | 0.3536 | 0.2442 | 0.0305 | 0.4190 | 0.935 | 0.0229 |
| B6 | [0.60, 0.70) | 936 | 0.3649 | 0.3649 | 0.2549 | 0.0417 | 0.4434 | 0.931 | 0.0232 |
| B7 | [0.70, 0.80) | 985 | 0.3603 | 0.3666 | 0.2552 | 0.0477 | 0.4420 | 0.930 | 0.0207 |
| B8 | [0.80, 0.90) | 1699 | 0.5019 | 0.5013 | 0.3554 | 0.0377 | 0.5871 | 0.865 | 0.0217 |
| B9 | [0.90, 1.00] | 486 | 0.1818 | 0.2299 | 0.1248 | 0.0329 | 0.2158 | 0.973 | 0.0124 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9712 | 0.9379 | 0.8521 | 0.0353 | 0.9815 | 0.347 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0000 | 0.0000 | 0.0000 | 0.0690 | 0.0000 | 1.000 | 0.0201 |
| B1 | [0.10, 0.20) | 34 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.0275 |
| B2 | [0.20, 0.30) | 185 | 0.0599 | 0.0728 | 0.0492 | 0.0108 | 0.0484 | 0.997 | 0.0241 |
| B3 | [0.30, 0.40) | 493 | 0.7146 | 0.6865 | 0.5340 | 0.0365 | 0.8083 | 0.727 | 0.0165 |
| B4 | [0.40, 0.50) | 677 | 0.7625 | 0.7572 | 0.5598 | 0.0547 | 0.8370 | 0.653 | 0.0154 |
| B5 | [0.50, 0.60) | 705 | 0.7077 | 0.7067 | 0.5096 | 0.0511 | 0.7907 | 0.708 | 0.0166 |
| B6 | [0.60, 0.70) | 809 | 0.5860 | 0.5819 | 0.4099 | 0.0396 | 0.6825 | 0.813 | 0.0169 |
| B7 | [0.70, 0.80) | 67 | 0.2842 | 0.4937 | 0.1928 | 0.0746 | 0.2801 | 0.870 | 0.0152 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8927 | 0.9270 | 0.7070 | 0.0437 | 0.9178 | 0.375 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7845 | 0.7952 | 0.6155 | 0.0467 | 0.8630 | 0.606 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 3.26s (19025 pair rows scored across 5 corpora).
