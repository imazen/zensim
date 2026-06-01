# bake_verdict — instant V_X eval

- Bake: `/home/lilith/work/zen/zensim--cross-codec-v8/zensim-experimental/weights/v11_candidates/v_balanced_v11a_s1_spline_2026-05-20.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 300
- Feature transforms: no

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8157 | 0.8139 | 0.6221 | 0.0107 | 0.8780 | 0.581 |
| KADIK10k | 10125 | 0.9139 | 0.9077 | 0.7419 | 0.0351 | 0.9485 | 0.420 |
| TID2013 | 3000 | 0.8908 | 0.8886 | 0.7099 | 0.0170 | 0.9204 | 0.459 |
| KonJND-1k (full) | 1008 | 0.4306 | 0.3861 | 0.2970 | 0.0437 | 0.5655 | 0.922 |
| AIC-3 CTC | 600 | 0.8102 | 0.8177 | 0.6431 | 0.0250 | 0.8873 | 0.576 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8157 | 0.8139 | 0.6221 | 0.0107 | 0.8780 | 0.581 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1714 | 0.2897 | 0.1228 | 0.0351 | 0.2517 | 0.957 | 0.0205 |
| B4 | [0.40, 0.50) | 266 | 0.2060 | 0.2154 | 0.1376 | 0.0451 | 0.2603 | 0.977 | 0.0226 |
| B5 | [0.50, 0.60) | 615 | 0.2822 | 0.2927 | 0.1899 | 0.0423 | 0.3206 | 0.956 | 0.0226 |
| B6 | [0.60, 0.70) | 836 | 0.3019 | 0.3013 | 0.2040 | 0.0120 | 0.3580 | 0.954 | 0.0237 |
| B7 | [0.70, 0.80) | 1092 | 0.3145 | 0.3206 | 0.2140 | 0.0247 | 0.3719 | 0.947 | 0.0233 |
| B8 | [0.80, 0.90) | 1382 | 0.4097 | 0.4088 | 0.2758 | 0.0195 | 0.4789 | 0.913 | 0.0207 |
| B9 | [0.90, 1.00] | 43 | 0.1628 | 0.3059 | 0.1141 | 0.0465 | 0.0803 | 0.952 | 0.0049 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9139 | 0.9077 | 0.7419 | 0.0351 | 0.9485 | 0.420 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2346 | 0.2685 | 0.1638 | 0.0355 | 0.2891 | 0.963 | 0.0192 |
| B1 | [0.10, 0.20) | 910 | 0.2372 | 0.2436 | 0.1652 | 0.0341 | 0.2979 | 0.970 | 0.0216 |
| B2 | [0.20, 0.30) | 1111 | 0.1594 | 0.1700 | 0.1111 | 0.0441 | 0.1877 | 0.985 | 0.0244 |
| B3 | [0.30, 0.40) | 1291 | 0.2146 | 0.2176 | 0.1491 | 0.0418 | 0.2592 | 0.976 | 0.0237 |
| B4 | [0.40, 0.50) | 1013 | 0.2283 | 0.2283 | 0.1603 | 0.0395 | 0.2701 | 0.974 | 0.0245 |
| B5 | [0.50, 0.60) | 919 | 0.1863 | 0.1989 | 0.1300 | 0.0424 | 0.2297 | 0.980 | 0.0243 |
| B6 | [0.60, 0.70) | 936 | 0.1765 | 0.1850 | 0.1218 | 0.0288 | 0.2073 | 0.983 | 0.0247 |
| B7 | [0.70, 0.80) | 985 | 0.2379 | 0.2368 | 0.1665 | 0.0071 | 0.2824 | 0.972 | 0.0217 |
| B8 | [0.80, 0.90) | 1699 | 0.4252 | 0.4244 | 0.2980 | 0.0312 | 0.5027 | 0.905 | 0.0230 |
| B9 | [0.90, 1.00] | 486 | 0.1796 | 0.2299 | 0.1262 | 0.0412 | 0.2037 | 0.973 | 0.0124 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8908 | 0.8886 | 0.7099 | 0.0170 | 0.9204 | 0.459 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0010 | 0.2398 | 0.0025 | 0.0345 | 0.0476 | 0.971 | 0.0192 |
| B1 | [0.10, 0.20) | 34 | 0.5339 | 0.6865 | 0.3589 | 0.0294 | 0.7104 | 0.727 | 0.0202 |
| B2 | [0.20, 0.30) | 185 | 0.3432 | 0.3805 | 0.2350 | 0.0432 | 0.4138 | 0.925 | 0.0218 |
| B3 | [0.30, 0.40) | 493 | 0.3796 | 0.3816 | 0.2625 | 0.0446 | 0.4571 | 0.924 | 0.0220 |
| B4 | [0.40, 0.50) | 677 | 0.5416 | 0.5444 | 0.3756 | 0.0369 | 0.6293 | 0.839 | 0.0202 |
| B5 | [0.50, 0.60) | 705 | 0.4622 | 0.4830 | 0.3174 | 0.0043 | 0.5507 | 0.876 | 0.0216 |
| B6 | [0.60, 0.70) | 809 | 0.1932 | 0.2033 | 0.1306 | 0.0309 | 0.2381 | 0.979 | 0.0209 |
| B7 | [0.70, 0.80) | 67 | 0.3034 | 0.4449 | 0.2010 | 0.0299 | 0.3468 | 0.896 | 0.0157 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.4306 | 0.3861 | 0.2970 | 0.0437 | 0.5655 | 0.922 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8102 | 0.8177 | 0.6431 | 0.0250 | 0.8873 | 0.576 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 3.19s (19025 pair rows scored across 5 corpora).
