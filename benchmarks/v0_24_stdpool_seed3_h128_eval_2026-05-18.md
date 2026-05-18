# bake_verdict — instant V_X eval

- Bake: `benchmarks/v0_24_stdpool_seed3_h128_2026-05-18.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 228
- Feature transforms: no

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.6046 | 0.5988 | 0.4287 | 0.0545 | 0.6909 | 0.801 |
| KADIK10k | 10125 | 0.6986 | 0.6964 | 0.5260 | 0.0559 | 0.7618 | 0.718 |
| TID2013 | 3000 | 0.7699 | 0.7949 | 0.5747 | 0.0500 | 0.8370 | 0.607 |
| KonJND-1k (full) | 1008 | 0.0520 | 0.1808 | 0.0235 | 0.0357 | 0.0748 | 0.984 |
| AIC-3 CTC | 600 | 0.6205 | 0.6138 | 0.4601 | 0.0567 | 0.7144 | 0.789 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.6046 | 0.5988 | 0.4287 | 0.0545 | 0.6909 | 0.801 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1471 | 0.2543 | 0.0802 | 0.0351 | 0.0740 | 0.967 | 0.0195 |
| B4 | [0.40, 0.50) | 266 | 0.0756 | 0.1290 | 0.0521 | 0.0414 | 0.1054 | 0.992 | 0.0228 |
| B5 | [0.50, 0.60) | 615 | 0.1939 | 0.1977 | 0.1301 | 0.0423 | 0.2406 | 0.980 | 0.0236 |
| B6 | [0.60, 0.70) | 836 | 0.2042 | 0.2106 | 0.1368 | 0.0371 | 0.2491 | 0.978 | 0.0245 |
| B7 | [0.70, 0.80) | 1092 | 0.1924 | 0.1985 | 0.1297 | 0.0394 | 0.2442 | 0.980 | 0.0246 |
| B8 | [0.80, 0.90) | 1382 | 0.2575 | 0.2637 | 0.1707 | 0.0355 | 0.3095 | 0.965 | 0.0222 |
| B9 | [0.90, 1.00] | 43 | 0.1264 | 0.1805 | 0.0808 | 0.0465 | 0.1493 | 0.984 | 0.0052 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.6986 | 0.6964 | 0.5260 | 0.0559 | 0.7618 | 0.718 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.1086 | 0.1406 | 0.0746 | 0.0397 | 0.1386 | 0.990 | 0.0200 |
| B1 | [0.10, 0.20) | 910 | 0.1386 | 0.1464 | 0.0973 | 0.0593 | 0.1565 | 0.989 | 0.0220 |
| B2 | [0.20, 0.30) | 1111 | 0.0435 | 0.0581 | 0.0301 | 0.0360 | 0.0648 | 0.998 | 0.0249 |
| B3 | [0.30, 0.40) | 1291 | 0.1246 | 0.1256 | 0.0862 | 0.0442 | 0.1539 | 0.992 | 0.0242 |
| B4 | [0.40, 0.50) | 1013 | 0.1692 | 0.1736 | 0.1177 | 0.0464 | 0.1948 | 0.985 | 0.0249 |
| B5 | [0.50, 0.60) | 919 | 0.1124 | 0.1252 | 0.0780 | 0.0381 | 0.1359 | 0.992 | 0.0247 |
| B6 | [0.60, 0.70) | 936 | 0.0786 | 0.1215 | 0.0553 | 0.0406 | 0.0867 | 0.993 | 0.0251 |
| B7 | [0.70, 0.80) | 985 | 0.0527 | 0.0605 | 0.0366 | 0.0355 | 0.0578 | 0.998 | 0.0225 |
| B8 | [0.80, 0.90) | 1699 | 0.2953 | 0.3156 | 0.2055 | 0.0388 | 0.3580 | 0.949 | 0.0243 |
| B9 | [0.90, 1.00] | 486 | 0.1423 | 0.1420 | 0.1020 | 0.0453 | 0.1565 | 0.990 | 0.0125 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7699 | 0.7949 | 0.5747 | 0.0500 | 0.8370 | 0.607 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0485 | 0.2398 | 0.0419 | 0.0000 | 0.0569 | 0.971 | 0.0192 |
| B1 | [0.10, 0.20) | 34 | 0.5266 | 0.6162 | 0.3554 | 0.0588 | 0.6788 | 0.788 | 0.0222 |
| B2 | [0.20, 0.30) | 185 | 0.2842 | 0.4087 | 0.1968 | 0.0541 | 0.3388 | 0.913 | 0.0214 |
| B3 | [0.30, 0.40) | 493 | 0.1500 | 0.1907 | 0.1012 | 0.0406 | 0.1787 | 0.982 | 0.0237 |
| B4 | [0.40, 0.50) | 677 | 0.2771 | 0.2807 | 0.1889 | 0.0443 | 0.3336 | 0.960 | 0.0239 |
| B5 | [0.50, 0.60) | 705 | 0.2490 | 0.2443 | 0.1671 | 0.0397 | 0.3048 | 0.970 | 0.0249 |
| B6 | [0.60, 0.70) | 809 | 0.1146 | 0.1984 | 0.0780 | 0.0383 | 0.1456 | 0.980 | 0.0210 |
| B7 | [0.70, 0.80) | 67 | 0.1431 | 0.2723 | 0.0939 | 0.0299 | 0.1700 | 0.962 | 0.0178 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.0520 | 0.1808 | 0.0235 | 0.0357 | 0.0748 | 0.984 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.6205 | 0.6138 | 0.4601 | 0.0567 | 0.7144 | 0.789 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 6.90s (19025 pair rows scored across 5 corpora).
