# bake_verdict — instant V_X eval

- Bake: `/mnt/v/output/zensim/bakes/v47_qat_recal_2026-05-27.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: yes (uses predict_transformed)

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC | geomean3 |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8657 | 0.8580 | 0.6742 | 0.0012 | 0.9782 | 0.514 | 0.8135 | 0.8990 |
| KADIK10k | 10125 | 0.7933 | 0.7914 | 0.5959 | 0.0003 | 0.9494 | 0.611 | 0.7293 | 0.8416 |
| TID2013 | 3000 | 0.7927 | 0.8171 | 0.6024 | 0.0053 | 0.9685 | 0.576 | 0.7778 | 0.8561 |
| KonJND-1k (full) | 1008 | 0.4185 | 0.3606 | 0.2872 | 0.0010 | 0.7915 | 0.933 | 0.5416 | 0.4925 |
| AIC-3 CTC | 600 | 0.7680 | 0.7847 | 0.5977 | 0.0000 | 0.9334 | 0.620 | 0.7074 | 0.8255 |
| AIC-4 sample | 300 | 0.8854 | 0.8770 | 0.7051 | 0.0000 | 0.9756 | 0.480 | 0.8351 | 0.9116 |

## CODEC_TARGET_GOALS.md scorecard (measurable subset)

| Goal | Measure | Value | Soft score |
|---|---|---:|---:|
| G1 dynamic range | pooled p5≤25 ∧ p95≥85 | p5=-32.8 p95=84.3 | 0.98 |
| G5 HF rank | KonJND+AIC-3 SROCC ≥0.70 | 0.418 / 0.768 | 0.23 |
| G7 CID22 rank | SROCC ≥0.85 (advisory) | 0.8657 | 1.00 |
| G8 Z-RMSE | AIC-3 ≤0.80 | 0.620 | 0.60 |
| G9 DS-AUC | AIC-3 ≥0.70 | 0.7074 | 0.05 |

**Weighted goal score (measurable subset): 0.627**

_G2 (JND anchor), G3 (monotonicity), G4 (cross-codec), G6 (MF band coverage), G10 (per-source), G11 (display) require external q-sweep / cross-codec / multi-PPD data not present in the held-out feature parquets. Run the dedicated q-sweep harness for those._

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8657 | 0.8580 | 0.6742 | 0.0012 | 0.9782 | 0.514 | 0.8135 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1829 | 0.2344 | 0.1165 | 0.0526 | 0.6696 | 0.972 | 0.0201 |
| B4 | [0.40, 0.50) | 266 | 0.2199 | 0.2683 | 0.1479 | 0.0301 | 0.7745 | 0.963 | 0.0220 |
| B5 | [0.50, 0.60) | 615 | 0.3482 | 0.3505 | 0.2357 | 0.0146 | 0.7281 | 0.937 | 0.0222 |
| B6 | [0.60, 0.70) | 836 | 0.3757 | 0.3720 | 0.2540 | 0.0144 | 0.7370 | 0.928 | 0.0230 |
| B7 | [0.70, 0.80) | 1092 | 0.3660 | 0.3859 | 0.2523 | 0.0165 | 0.7325 | 0.923 | 0.0226 |
| B8 | [0.80, 0.90) | 1382 | 0.4535 | 0.4598 | 0.3051 | 0.0145 | 0.8005 | 0.888 | 0.0200 |
| B9 | [0.90, 1.00] | 43 | 0.0441 | 0.1890 | 0.0321 | 0.0000 | 0.5599 | 0.982 | 0.0051 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7933 | 0.7914 | 0.5959 | 0.0003 | 0.9494 | 0.611 | 0.7293 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.1882 | 0.2020 | 0.1313 | 0.0298 | 0.6369 | 0.979 | 0.0196 |
| B1 | [0.10, 0.20) | 910 | 0.1092 | 0.1137 | 0.0760 | 0.0000 | 0.5745 | 0.994 | 0.0222 |
| B2 | [0.20, 0.30) | 1111 | 0.0831 | 0.1302 | 0.0590 | 0.0000 | 0.5508 | 0.991 | 0.0246 |
| B3 | [0.30, 0.40) | 1291 | 0.1402 | 0.1474 | 0.0977 | 0.0000 | 0.5866 | 0.989 | 0.0241 |
| B4 | [0.40, 0.50) | 1013 | 0.1535 | 0.1526 | 0.1069 | 0.0000 | 0.5890 | 0.988 | 0.0250 |
| B5 | [0.50, 0.60) | 919 | 0.0865 | 0.0936 | 0.0602 | 0.0000 | 0.5691 | 0.996 | 0.0248 |
| B6 | [0.60, 0.70) | 936 | 0.0655 | 0.0873 | 0.0452 | 0.0000 | 0.5553 | 0.996 | 0.0253 |
| B7 | [0.70, 0.80) | 985 | 0.1846 | 0.1891 | 0.1282 | 0.0041 | 0.6354 | 0.982 | 0.0219 |
| B8 | [0.80, 0.90) | 1699 | 0.3694 | 0.3772 | 0.2589 | 0.0147 | 0.7320 | 0.926 | 0.0236 |
| B9 | [0.90, 1.00] | 486 | 0.1460 | 0.1633 | 0.1052 | 0.0350 | 0.7949 | 0.987 | 0.0126 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7927 | 0.8171 | 0.6024 | 0.0053 | 0.9685 | 0.576 | 0.7778 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.4882 | 0.6417 | 0.3280 | 0.0000 | 0.9523 | 0.767 | 0.0143 |
| B1 | [0.10, 0.20) | 34 | 0.4043 | 0.4935 | 0.2768 | 0.0000 | 0.9376 | 0.870 | 0.0233 |
| B2 | [0.20, 0.30) | 185 | 0.3087 | 0.3296 | 0.2124 | 0.0108 | 0.7301 | 0.944 | 0.0224 |
| B3 | [0.30, 0.40) | 493 | 0.2728 | 0.3503 | 0.1904 | 0.0243 | 0.7679 | 0.937 | 0.0223 |
| B4 | [0.40, 0.50) | 677 | 0.2878 | 0.3254 | 0.1960 | 0.0074 | 0.6780 | 0.946 | 0.0233 |
| B5 | [0.50, 0.60) | 705 | 0.3223 | 0.3289 | 0.2254 | 0.0085 | 0.7024 | 0.944 | 0.0239 |
| B6 | [0.60, 0.70) | 809 | 0.1782 | 0.1928 | 0.1224 | 0.0272 | 0.6011 | 0.981 | 0.0210 |
| B7 | [0.70, 0.80) | 67 | 0.3613 | 0.4631 | 0.2464 | 0.0299 | 0.7463 | 0.886 | 0.0162 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.4185 | 0.3606 | 0.2872 | 0.0010 | 0.7915 | 0.933 | 0.5416 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7680 | 0.7847 | 0.5977 | 0.0000 | 0.9334 | 0.620 | 0.7074 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-4 sample (n=300)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8854 | 0.8770 | 0.7051 | 0.0000 | 0.9756 | 0.480 | 0.8351 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-4 sample — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 15.48s (19325 pair rows scored across 6 corpora).
