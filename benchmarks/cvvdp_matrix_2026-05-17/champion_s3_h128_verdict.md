# bake_verdict — instant V_X eval

- Bake: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22cvvdp_full_mc_s3_h128.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: yes (uses predict_transformed)

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8572 | 0.8574 | 0.6654 | 0.0499 | 0.9110 | 0.515 |
| KADIK10k | 10125 | 0.8306 | 0.8306 | 0.6365 | 0.0328 | 0.8985 | 0.557 |
| TID2013 | 3000 | 0.8562 | 0.8682 | 0.6762 | 0.0423 | 0.8876 | 0.496 |
| KonJND-1k (full) | 1008 | 0.1950 | 0.1505 | 0.1340 | 0.0377 | 0.3159 | 0.989 |
| AIC-3 CTC | 600 | 0.8494 | 0.8529 | 0.6883 | 0.0567 | 0.9095 | 0.522 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8572 | 0.8574 | 0.6654 | 0.0499 | 0.9110 | 0.515 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1600 | 0.2538 | 0.1078 | 0.0526 | 0.1676 | 0.967 | 0.0202 |
| B4 | [0.40, 0.50) | 266 | 0.2534 | 0.2631 | 0.1671 | 0.0451 | 0.3155 | 0.965 | 0.0220 |
| B5 | [0.50, 0.60) | 615 | 0.3468 | 0.3575 | 0.2364 | 0.0504 | 0.4112 | 0.934 | 0.0219 |
| B6 | [0.60, 0.70) | 836 | 0.3678 | 0.3698 | 0.2489 | 0.0490 | 0.4324 | 0.929 | 0.0228 |
| B7 | [0.70, 0.80) | 1092 | 0.3551 | 0.3604 | 0.2437 | 0.0531 | 0.4193 | 0.933 | 0.0228 |
| B8 | [0.80, 0.90) | 1382 | 0.4001 | 0.4013 | 0.2679 | 0.0456 | 0.4744 | 0.916 | 0.0207 |
| B9 | [0.90, 1.00] | 43 | 0.1208 | 0.1805 | 0.0875 | 0.0698 | 0.2186 | 0.984 | 0.0052 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8306 | 0.8306 | 0.6365 | 0.0328 | 0.8985 | 0.557 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.1773 | 0.1963 | 0.1239 | 0.0355 | 0.2357 | 0.981 | 0.0198 |
| B1 | [0.10, 0.20) | 910 | 0.1727 | 0.2111 | 0.1201 | 0.0418 | 0.2313 | 0.977 | 0.0220 |
| B2 | [0.20, 0.30) | 1111 | 0.0650 | 0.0664 | 0.0451 | 0.0369 | 0.0681 | 0.998 | 0.0249 |
| B3 | [0.30, 0.40) | 1291 | 0.1565 | 0.1788 | 0.1090 | 0.0457 | 0.1803 | 0.984 | 0.0239 |
| B4 | [0.40, 0.50) | 1013 | 0.1753 | 0.2120 | 0.1245 | 0.0434 | 0.2088 | 0.977 | 0.0247 |
| B5 | [0.50, 0.60) | 919 | 0.1071 | 0.1628 | 0.0749 | 0.0403 | 0.1325 | 0.987 | 0.0246 |
| B6 | [0.60, 0.70) | 936 | 0.1137 | 0.1556 | 0.0777 | 0.0438 | 0.1370 | 0.988 | 0.0249 |
| B7 | [0.70, 0.80) | 985 | 0.1458 | 0.1648 | 0.1008 | 0.0406 | 0.1560 | 0.986 | 0.0221 |
| B8 | [0.80, 0.90) | 1699 | 0.3861 | 0.3916 | 0.2693 | 0.0347 | 0.4570 | 0.920 | 0.0234 |
| B9 | [0.90, 1.00] | 486 | 0.1623 | 0.2022 | 0.1111 | 0.0370 | 0.1909 | 0.979 | 0.0123 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8562 | 0.8682 | 0.6762 | 0.0423 | 0.8876 | 0.496 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.1495 | 0.2377 | 0.1258 | 0.0345 | 0.1423 | 0.971 | 0.0190 |
| B1 | [0.10, 0.20) | 34 | 0.4830 | 0.5906 | 0.3411 | 0.0882 | 0.6219 | 0.807 | 0.0217 |
| B2 | [0.20, 0.30) | 185 | 0.2455 | 0.2772 | 0.1643 | 0.0595 | 0.3052 | 0.961 | 0.0231 |
| B3 | [0.30, 0.40) | 493 | 0.4066 | 0.4237 | 0.2794 | 0.0487 | 0.5037 | 0.906 | 0.0217 |
| B4 | [0.40, 0.50) | 677 | 0.5014 | 0.5160 | 0.3469 | 0.0443 | 0.5910 | 0.857 | 0.0209 |
| B5 | [0.50, 0.60) | 705 | 0.4275 | 0.4619 | 0.2938 | 0.0440 | 0.5106 | 0.887 | 0.0221 |
| B6 | [0.60, 0.70) | 809 | 0.1534 | 0.2083 | 0.1043 | 0.0346 | 0.1881 | 0.978 | 0.0209 |
| B7 | [0.70, 0.80) | 67 | 0.4074 | 0.4508 | 0.2772 | 0.0746 | 0.4661 | 0.893 | 0.0160 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.1950 | 0.1505 | 0.1340 | 0.0377 | 0.3159 | 0.989 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8494 | 0.8529 | 0.6883 | 0.0567 | 0.9095 | 0.522 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 3.55s (19025 pair rows scored across 5 corpora).
