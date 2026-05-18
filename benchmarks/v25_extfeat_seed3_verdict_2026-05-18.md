# bake_verdict — instant V_X eval

- Bake: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v25_extfeat_mix_cv40_konjnd_0_02_LARGE_iwssim_h128_s3.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-18-extfeat`
- Bake n_inputs: 324
- Feature transforms: no

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8171 | 0.7948 | 0.6180 | 0.0440 | 0.8795 | 0.607 |
| KADIK10k | 10125 | 0.8999 | 0.9000 | 0.7208 | 0.0330 | 0.9418 | 0.436 |
| TID2013 | 3000 | 0.8822 | 0.8929 | 0.7032 | 0.0420 | 0.9128 | 0.450 |
| KonJND-1k (full) | 1008 | 0.8108 | 0.9143 | 0.5991 | 0.0337 | 0.8574 | 0.405 |
| AIC-3 CTC | 600 | 0.7701 | 0.7803 | 0.6028 | 0.0533 | 0.8430 | 0.625 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8171 | 0.7948 | 0.6180 | 0.0440 | 0.8795 | 0.607 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.2344 | 0.3826 | 0.1566 | 0.0351 | 0.2329 | 0.924 | 0.0190 |
| B4 | [0.40, 0.50) | 266 | 0.1257 | 0.1746 | 0.0864 | 0.0376 | 0.1486 | 0.985 | 0.0224 |
| B5 | [0.50, 0.60) | 615 | 0.1392 | 0.2152 | 0.0940 | 0.0423 | 0.1585 | 0.977 | 0.0232 |
| B6 | [0.60, 0.70) | 836 | 0.1949 | 0.2121 | 0.1292 | 0.0443 | 0.2243 | 0.977 | 0.0245 |
| B7 | [0.70, 0.80) | 1092 | 0.3273 | 0.3277 | 0.2242 | 0.0421 | 0.3810 | 0.945 | 0.0232 |
| B8 | [0.80, 0.90) | 1382 | 0.5065 | 0.5128 | 0.3446 | 0.0398 | 0.5935 | 0.858 | 0.0193 |
| B9 | [0.90, 1.00] | 43 | 0.0892 | 0.2945 | 0.0498 | 0.0233 | 0.2231 | 0.956 | 0.0049 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8999 | 0.9000 | 0.7208 | 0.0330 | 0.9418 | 0.436 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2284 | 0.2593 | 0.1599 | 0.0440 | 0.2833 | 0.966 | 0.0193 |
| B1 | [0.10, 0.20) | 910 | 0.1866 | 0.1993 | 0.1304 | 0.0484 | 0.2451 | 0.980 | 0.0219 |
| B2 | [0.20, 0.30) | 1111 | 0.1018 | 0.1078 | 0.0697 | 0.0351 | 0.1126 | 0.994 | 0.0247 |
| B3 | [0.30, 0.40) | 1291 | 0.1904 | 0.2025 | 0.1324 | 0.0434 | 0.2301 | 0.979 | 0.0238 |
| B4 | [0.40, 0.50) | 1013 | 0.2229 | 0.2312 | 0.1571 | 0.0336 | 0.2649 | 0.973 | 0.0245 |
| B5 | [0.50, 0.60) | 919 | 0.1657 | 0.2101 | 0.1149 | 0.0316 | 0.2092 | 0.978 | 0.0243 |
| B6 | [0.60, 0.70) | 936 | 0.1838 | 0.1963 | 0.1269 | 0.0299 | 0.2357 | 0.981 | 0.0248 |
| B7 | [0.70, 0.80) | 985 | 0.2371 | 0.2475 | 0.1662 | 0.0406 | 0.2806 | 0.969 | 0.0216 |
| B8 | [0.80, 0.90) | 1699 | 0.4192 | 0.4213 | 0.2937 | 0.0394 | 0.4985 | 0.907 | 0.0230 |
| B9 | [0.90, 1.00] | 486 | 0.1801 | 0.1961 | 0.1265 | 0.0391 | 0.2108 | 0.981 | 0.0124 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8822 | 0.8929 | 0.7032 | 0.0420 | 0.9128 | 0.450 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.1835 | 0.2649 | 0.1110 | 0.0345 | 0.1872 | 0.964 | 0.0191 |
| B1 | [0.10, 0.20) | 34 | 0.4274 | 0.5581 | 0.3161 | 0.0588 | 0.5525 | 0.830 | 0.0230 |
| B2 | [0.20, 0.30) | 185 | 0.3188 | 0.3484 | 0.2181 | 0.0432 | 0.4028 | 0.937 | 0.0223 |
| B3 | [0.30, 0.40) | 493 | 0.4606 | 0.4676 | 0.3184 | 0.0467 | 0.5567 | 0.884 | 0.0210 |
| B4 | [0.40, 0.50) | 677 | 0.5176 | 0.5184 | 0.3617 | 0.0502 | 0.6008 | 0.855 | 0.0207 |
| B5 | [0.50, 0.60) | 705 | 0.4562 | 0.4796 | 0.3157 | 0.0426 | 0.5373 | 0.877 | 0.0217 |
| B6 | [0.60, 0.70) | 809 | 0.1722 | 0.1903 | 0.1157 | 0.0408 | 0.2190 | 0.982 | 0.0211 |
| B7 | [0.70, 0.80) | 67 | 0.3571 | 0.4508 | 0.2409 | 0.0448 | 0.4200 | 0.893 | 0.0156 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8108 | 0.9143 | 0.5991 | 0.0337 | 0.8574 | 0.405 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7701 | 0.7803 | 0.6028 | 0.0533 | 0.8430 | 0.625 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 6.53s (19025 pair rows scored across 5 corpora).
