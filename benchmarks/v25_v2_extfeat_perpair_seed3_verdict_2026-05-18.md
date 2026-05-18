# bake_verdict — instant V_X eval

- Bake: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v25_v2_extfeat_mix_cv40_konjnd_0_02_LARGE_iwssim_perpair_h128_s3.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-18-extfeat`
- Bake n_inputs: 343
- Feature transforms: no

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.5919 | 0.5821 | 0.4162 | 0.0422 | 0.6797 | 0.813 |
| KADIK10k | 10125 | 0.8997 | 0.8985 | 0.7179 | 0.0314 | 0.9414 | 0.439 |
| TID2013 | 3000 | 0.8714 | 0.8853 | 0.6891 | 0.0343 | 0.9048 | 0.465 |
| KonJND-1k (full) | 1008 | 0.8570 | 0.9545 | 0.6578 | 0.0278 | 0.9028 | 0.298 |
| AIC-3 CTC | 600 | 0.7294 | 0.7395 | 0.5617 | 0.0517 | 0.8049 | 0.673 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.5919 | 0.5821 | 0.4162 | 0.0422 | 0.6797 | 0.813 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1510 | 0.3296 | 0.1053 | 0.0351 | 0.1560 | 0.944 | 0.0198 |
| B4 | [0.40, 0.50) | 266 | 0.0722 | 0.1568 | 0.0486 | 0.0451 | 0.0861 | 0.988 | 0.0226 |
| B5 | [0.50, 0.60) | 615 | 0.0547 | 0.1064 | 0.0379 | 0.0455 | 0.0702 | 0.994 | 0.0239 |
| B6 | [0.60, 0.70) | 836 | 0.0728 | 0.1021 | 0.0484 | 0.0407 | 0.0767 | 0.995 | 0.0251 |
| B7 | [0.70, 0.80) | 1092 | 0.1749 | 0.1895 | 0.1174 | 0.0412 | 0.2128 | 0.982 | 0.0246 |
| B8 | [0.80, 0.90) | 1382 | 0.3619 | 0.3674 | 0.2427 | 0.0384 | 0.4376 | 0.930 | 0.0213 |
| B9 | [0.90, 1.00] | 43 | 0.0311 | 0.2463 | 0.0166 | 0.0233 | 0.0045 | 0.969 | 0.0051 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8997 | 0.8985 | 0.7179 | 0.0314 | 0.9414 | 0.439 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2145 | 0.2326 | 0.1513 | 0.0454 | 0.2681 | 0.973 | 0.0195 |
| B1 | [0.10, 0.20) | 910 | 0.2019 | 0.2103 | 0.1406 | 0.0473 | 0.2606 | 0.978 | 0.0219 |
| B2 | [0.20, 0.30) | 1111 | 0.1125 | 0.1335 | 0.0781 | 0.0351 | 0.1235 | 0.991 | 0.0247 |
| B3 | [0.30, 0.40) | 1291 | 0.1836 | 0.1908 | 0.1273 | 0.0418 | 0.2232 | 0.982 | 0.0239 |
| B4 | [0.40, 0.50) | 1013 | 0.2080 | 0.2184 | 0.1458 | 0.0424 | 0.2488 | 0.976 | 0.0247 |
| B5 | [0.50, 0.60) | 919 | 0.1568 | 0.2065 | 0.1094 | 0.0348 | 0.1982 | 0.978 | 0.0243 |
| B6 | [0.60, 0.70) | 936 | 0.1717 | 0.1852 | 0.1184 | 0.0288 | 0.2194 | 0.983 | 0.0248 |
| B7 | [0.70, 0.80) | 985 | 0.2092 | 0.2164 | 0.1465 | 0.0376 | 0.2382 | 0.976 | 0.0217 |
| B8 | [0.80, 0.90) | 1699 | 0.4274 | 0.4295 | 0.3000 | 0.0424 | 0.5021 | 0.903 | 0.0229 |
| B9 | [0.90, 1.00] | 486 | 0.1822 | 0.1989 | 0.1269 | 0.0350 | 0.2169 | 0.980 | 0.0124 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8714 | 0.8853 | 0.6891 | 0.0343 | 0.9048 | 0.465 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0943 | 0.2129 | 0.0962 | 0.0345 | 0.0628 | 0.977 | 0.0192 |
| B1 | [0.10, 0.20) | 34 | 0.4184 | 0.4654 | 0.2875 | 0.0588 | 0.5535 | 0.885 | 0.0253 |
| B2 | [0.20, 0.30) | 185 | 0.2454 | 0.2541 | 0.1627 | 0.0270 | 0.3157 | 0.967 | 0.0234 |
| B3 | [0.30, 0.40) | 493 | 0.4635 | 0.4700 | 0.3185 | 0.0385 | 0.5684 | 0.883 | 0.0211 |
| B4 | [0.40, 0.50) | 677 | 0.5266 | 0.5341 | 0.3681 | 0.0458 | 0.6137 | 0.845 | 0.0205 |
| B5 | [0.50, 0.60) | 705 | 0.4233 | 0.4450 | 0.2903 | 0.0468 | 0.5008 | 0.896 | 0.0223 |
| B6 | [0.60, 0.70) | 809 | 0.1511 | 0.1813 | 0.1014 | 0.0346 | 0.1892 | 0.983 | 0.0210 |
| B7 | [0.70, 0.80) | 67 | 0.3755 | 0.4290 | 0.2554 | 0.0746 | 0.4323 | 0.903 | 0.0160 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8570 | 0.9545 | 0.6578 | 0.0278 | 0.9028 | 0.298 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7294 | 0.7395 | 0.5617 | 0.0517 | 0.8049 | 0.673 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 11.00s (19025 pair rows scored across 5 corpora).
