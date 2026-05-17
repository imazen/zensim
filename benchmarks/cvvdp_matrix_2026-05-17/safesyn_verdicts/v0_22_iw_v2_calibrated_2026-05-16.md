# bake_verdict — instant V_X eval

- Bake: `zensim/weights/v0_22_iw_v2_calibrated_2026-05-16.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: yes (uses predict_transformed)

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8163 | 0.8226 | 0.6317 | 0.0473 | 0.8754 | 0.569 |
| KADIK10k | 10125 | 0.9506 | 0.9517 | 0.8068 | 0.0430 | 0.9710 | 0.307 |
| TID2013 | 3000 | 0.9617 | 0.9623 | 0.8280 | 0.0487 | 0.9766 | 0.272 |
| KonJND-1k (full) | 1008 | 0.0303 | 0.1059 | 0.0229 | 0.0387 | 0.0883 | 0.994 |
| AIC-3 CTC | 600 | 0.8070 | 0.8162 | 0.6404 | 0.0517 | 0.8785 | 0.578 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8163 | 0.8226 | 0.6317 | 0.0473 | 0.8754 | 0.569 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1180 | 0.1640 | 0.0677 | 0.0526 | 0.0399 | 0.986 | 0.0214 |
| B4 | [0.40, 0.50) | 266 | 0.2613 | 0.2989 | 0.1743 | 0.0489 | 0.3169 | 0.954 | 0.0217 |
| B5 | [0.50, 0.60) | 615 | 0.2566 | 0.2738 | 0.1741 | 0.0520 | 0.3058 | 0.962 | 0.0228 |
| B6 | [0.60, 0.70) | 836 | 0.2351 | 0.2549 | 0.1562 | 0.0431 | 0.2807 | 0.967 | 0.0241 |
| B7 | [0.70, 0.80) | 1092 | 0.3415 | 0.3507 | 0.2364 | 0.0449 | 0.3824 | 0.936 | 0.0229 |
| B8 | [0.80, 0.90) | 1382 | 0.4522 | 0.4541 | 0.3091 | 0.0449 | 0.5303 | 0.891 | 0.0201 |
| B9 | [0.90, 1.00] | 43 | 0.0590 | 0.4033 | 0.0188 | 0.0465 | 0.1713 | 0.915 | 0.0047 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9506 | 0.9517 | 0.8068 | 0.0430 | 0.9710 | 0.307 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.3782 | 0.3833 | 0.2643 | 0.0440 | 0.4627 | 0.924 | 0.0183 |
| B1 | [0.10, 0.20) | 910 | 0.3386 | 0.3491 | 0.2379 | 0.0396 | 0.3945 | 0.937 | 0.0205 |
| B2 | [0.20, 0.30) | 1111 | 0.3270 | 0.3339 | 0.2269 | 0.0396 | 0.3864 | 0.943 | 0.0231 |
| B3 | [0.30, 0.40) | 1291 | 0.2938 | 0.2927 | 0.2063 | 0.0488 | 0.3522 | 0.956 | 0.0230 |
| B4 | [0.40, 0.50) | 1013 | 0.2739 | 0.2787 | 0.1941 | 0.0375 | 0.3243 | 0.960 | 0.0240 |
| B5 | [0.50, 0.60) | 919 | 0.2641 | 0.2666 | 0.1862 | 0.0457 | 0.3229 | 0.964 | 0.0238 |
| B6 | [0.60, 0.70) | 936 | 0.3068 | 0.3074 | 0.2121 | 0.0395 | 0.3833 | 0.952 | 0.0239 |
| B7 | [0.70, 0.80) | 985 | 0.2845 | 0.2893 | 0.1999 | 0.0457 | 0.3513 | 0.957 | 0.0214 |
| B8 | [0.80, 0.90) | 1699 | 0.4445 | 0.4475 | 0.3117 | 0.0453 | 0.5275 | 0.894 | 0.0227 |
| B9 | [0.90, 1.00] | 486 | 0.1361 | 0.1451 | 0.0965 | 0.0535 | 0.1314 | 0.989 | 0.0125 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9617 | 0.9623 | 0.8280 | 0.0487 | 0.9766 | 0.272 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.2035 | 0.3588 | 0.1603 | 0.0690 | 0.2138 | 0.933 | 0.0175 |
| B1 | [0.10, 0.20) | 34 | 0.3807 | 0.3956 | 0.2375 | 0.0000 | 0.4635 | 0.918 | 0.0257 |
| B2 | [0.20, 0.30) | 185 | 0.5630 | 0.5804 | 0.3902 | 0.0595 | 0.6570 | 0.814 | 0.0187 |
| B3 | [0.30, 0.40) | 493 | 0.6408 | 0.6457 | 0.4623 | 0.0507 | 0.7113 | 0.764 | 0.0172 |
| B4 | [0.40, 0.50) | 677 | 0.6732 | 0.6741 | 0.4859 | 0.0502 | 0.7479 | 0.739 | 0.0172 |
| B5 | [0.50, 0.60) | 705 | 0.6229 | 0.6225 | 0.4415 | 0.0369 | 0.7122 | 0.783 | 0.0188 |
| B6 | [0.60, 0.70) | 809 | 0.5505 | 0.5501 | 0.3813 | 0.0396 | 0.6543 | 0.835 | 0.0175 |
| B7 | [0.70, 0.80) | 67 | 0.0348 | 0.1235 | 0.0195 | 0.0299 | 0.0172 | 0.992 | 0.0189 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.0303 | 0.1059 | 0.0229 | 0.0387 | 0.0883 | 0.994 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8070 | 0.8162 | 0.6404 | 0.0517 | 0.8785 | 0.578 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 3.31s (19025 pair rows scored across 5 corpora).
