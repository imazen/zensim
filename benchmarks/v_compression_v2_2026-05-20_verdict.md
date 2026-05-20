# bake_verdict — instant V_X eval

- Bake: `zensim/weights/v_compression_v2_2026-05-20.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 300
- Feature transforms: no

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8641 | 0.8611 | 0.6742 | 0.0499 | 0.9157 | 0.508 |
| KADIK10k | 10125 | 0.9316 | 0.9187 | 0.7684 | 0.0389 | 0.9602 | 0.395 |
| TID2013 | 3000 | 0.8893 | 0.8945 | 0.7130 | 0.0400 | 0.9173 | 0.447 |
| KonJND-1k (full) | 1008 | 0.8080 | 0.8649 | 0.5935 | 0.0437 | 0.8505 | 0.502 |
| AIC-3 CTC | 600 | 0.8183 | 0.8244 | 0.6527 | 0.0550 | 0.8856 | 0.566 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8641 | 0.8611 | 0.6742 | 0.0499 | 0.9157 | 0.508 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0825 | 0.2911 | 0.0564 | 0.0175 | 0.0373 | 0.957 | 0.0204 |
| B4 | [0.40, 0.50) | 266 | 0.2699 | 0.2645 | 0.1801 | 0.0414 | 0.3326 | 0.964 | 0.0220 |
| B5 | [0.50, 0.60) | 615 | 0.2656 | 0.2736 | 0.1803 | 0.0504 | 0.3170 | 0.962 | 0.0228 |
| B6 | [0.60, 0.70) | 836 | 0.2729 | 0.2756 | 0.1817 | 0.0431 | 0.3195 | 0.961 | 0.0239 |
| B7 | [0.70, 0.80) | 1092 | 0.3792 | 0.3863 | 0.2602 | 0.0403 | 0.4420 | 0.922 | 0.0225 |
| B8 | [0.80, 0.90) | 1382 | 0.4971 | 0.5005 | 0.3389 | 0.0441 | 0.5801 | 0.866 | 0.0195 |
| B9 | [0.90, 1.00] | 43 | 0.1401 | 0.4356 | 0.1030 | 0.0465 | 0.2949 | 0.900 | 0.0047 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9316 | 0.9187 | 0.7684 | 0.0389 | 0.9602 | 0.395 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2461 | 0.2629 | 0.1728 | 0.0539 | 0.3139 | 0.965 | 0.0194 |
| B1 | [0.10, 0.20) | 910 | 0.2599 | 0.2620 | 0.1803 | 0.0429 | 0.3219 | 0.965 | 0.0215 |
| B2 | [0.20, 0.30) | 1111 | 0.1918 | 0.2016 | 0.1324 | 0.0351 | 0.2230 | 0.979 | 0.0242 |
| B3 | [0.30, 0.40) | 1291 | 0.2201 | 0.2312 | 0.1539 | 0.0418 | 0.2601 | 0.973 | 0.0236 |
| B4 | [0.40, 0.50) | 1013 | 0.2516 | 0.2502 | 0.1779 | 0.0355 | 0.3000 | 0.968 | 0.0244 |
| B5 | [0.50, 0.60) | 919 | 0.2035 | 0.2302 | 0.1421 | 0.0359 | 0.2513 | 0.973 | 0.0241 |
| B6 | [0.60, 0.70) | 936 | 0.2290 | 0.2420 | 0.1583 | 0.0321 | 0.2803 | 0.970 | 0.0243 |
| B7 | [0.70, 0.80) | 985 | 0.2718 | 0.2729 | 0.1900 | 0.0376 | 0.3177 | 0.962 | 0.0214 |
| B8 | [0.80, 0.90) | 1699 | 0.4514 | 0.4509 | 0.3169 | 0.0365 | 0.5343 | 0.893 | 0.0226 |
| B9 | [0.90, 1.00] | 486 | 0.1708 | 0.1817 | 0.1168 | 0.0391 | 0.2037 | 0.983 | 0.0124 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8893 | 0.8945 | 0.7130 | 0.0400 | 0.9173 | 0.447 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.1857 | 0.3541 | 0.1356 | 0.0345 | 0.2015 | 0.935 | 0.0179 |
| B1 | [0.10, 0.20) | 34 | 0.4807 | 0.4927 | 0.3411 | 0.0294 | 0.6201 | 0.870 | 0.0245 |
| B2 | [0.20, 0.30) | 185 | 0.3282 | 0.3478 | 0.2237 | 0.0378 | 0.4069 | 0.938 | 0.0224 |
| B3 | [0.30, 0.40) | 493 | 0.4693 | 0.4825 | 0.3256 | 0.0365 | 0.5672 | 0.876 | 0.0208 |
| B4 | [0.40, 0.50) | 677 | 0.5339 | 0.5410 | 0.3729 | 0.0414 | 0.6269 | 0.841 | 0.0204 |
| B5 | [0.50, 0.60) | 705 | 0.4687 | 0.4938 | 0.3245 | 0.0369 | 0.5541 | 0.870 | 0.0215 |
| B6 | [0.60, 0.70) | 809 | 0.1864 | 0.2225 | 0.1268 | 0.0358 | 0.2322 | 0.975 | 0.0209 |
| B7 | [0.70, 0.80) | 67 | 0.3474 | 0.4519 | 0.2291 | 0.0746 | 0.4019 | 0.892 | 0.0156 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8080 | 0.8649 | 0.5935 | 0.0437 | 0.8505 | 0.502 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8183 | 0.8244 | 0.6527 | 0.0550 | 0.8856 | 0.566 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 3.89s (19025 pair rows scored across 5 corpora).
