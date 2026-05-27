# bake_verdict — instant V_X eval

- Bake: `/mnt/v/output/zensim/bakes/v47_strict_recal_2026-05-27.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: yes (uses predict_transformed)

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC | geomean3 |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8547 | 0.8410 | 0.6619 | 0.0012 | 0.9754 | 0.541 | 0.8078 | 0.8884 |
| KADIK10k | 10125 | 0.7995 | 0.7973 | 0.6123 | 0.0007 | 0.9541 | 0.604 | 0.7430 | 0.8473 |
| TID2013 | 3000 | 0.7936 | 0.7934 | 0.6069 | 0.0063 | 0.9687 | 0.609 | 0.7571 | 0.8481 |
| KonJND-1k (full) | 1008 | 0.4850 | 0.4167 | 0.3353 | 0.0020 | 0.8144 | 0.909 | 0.5310 | 0.5480 |
| AIC-3 CTC | 600 | 0.7700 | 0.7860 | 0.5999 | 0.0000 | 0.9341 | 0.618 | 0.7222 | 0.8269 |
| AIC-4 sample | 300 | 0.8902 | 0.8774 | 0.7136 | 0.0000 | 0.9758 | 0.480 | 0.8268 | 0.9135 |

## CODEC_TARGET_GOALS.md scorecard (measurable subset)

| Goal | Measure | Value | Soft score |
|---|---|---:|---:|
| G1 dynamic range | pooled p5≤25 ∧ p95≥85 | p5=0.0 p95=84.5 | 0.99 |
| G5 HF rank | KonJND+AIC-3 SROCC ≥0.70 | 0.485 / 0.770 | 0.23 |
| G7 CID22 rank | SROCC ≥0.85 (advisory) | 0.8547 | 1.00 |
| G8 Z-RMSE | AIC-3 ≤0.80 | 0.618 | 0.61 |
| G9 DS-AUC | AIC-3 ≥0.70 | 0.7222 | 0.15 |

**Weighted goal score (measurable subset): 0.644**

_G2 (JND anchor), G3 (monotonicity), G4 (cross-codec), G6 (MF band coverage), G10 (per-source), G11 (display) require external q-sweep / cross-codec / multi-PPD data not present in the held-out feature parquets. Run the dedicated q-sweep harness for those._

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8547 | 0.8410 | 0.6619 | 0.0012 | 0.9754 | 0.541 | 0.8078 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1983 | 0.2852 | 0.1316 | 0.0526 | 0.7026 | 0.958 | 0.0195 |
| B4 | [0.40, 0.50) | 266 | 0.2096 | 0.2477 | 0.1405 | 0.0188 | 0.6496 | 0.969 | 0.0223 |
| B5 | [0.50, 0.60) | 615 | 0.3285 | 0.3334 | 0.2221 | 0.0163 | 0.7156 | 0.943 | 0.0224 |
| B6 | [0.60, 0.70) | 836 | 0.3616 | 0.3584 | 0.2444 | 0.0156 | 0.7288 | 0.934 | 0.0230 |
| B7 | [0.70, 0.80) | 1092 | 0.3555 | 0.3766 | 0.2445 | 0.0137 | 0.7260 | 0.926 | 0.0227 |
| B8 | [0.80, 0.90) | 1382 | 0.4461 | 0.4506 | 0.3001 | 0.0166 | 0.7952 | 0.893 | 0.0201 |
| B9 | [0.90, 1.00] | 43 | 0.0160 | 0.1805 | 0.0011 | 0.0000 | 0.4923 | 0.984 | 0.0052 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7995 | 0.7973 | 0.6123 | 0.0007 | 0.9541 | 0.604 | 0.7430 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.1694 | 0.1711 | 0.1371 | 0.0397 | 0.7967 | 0.985 | 0.0197 |
| B1 | [0.10, 0.20) | 910 | 0.1046 | 0.1181 | 0.0811 | 0.0000 | 0.6256 | 0.993 | 0.0223 |
| B2 | [0.20, 0.30) | 1111 | 0.0876 | 0.0998 | 0.0641 | 0.0000 | 0.5607 | 0.995 | 0.0247 |
| B3 | [0.30, 0.40) | 1291 | 0.1403 | 0.1526 | 0.1007 | 0.0000 | 0.5852 | 0.988 | 0.0241 |
| B4 | [0.40, 0.50) | 1013 | 0.1534 | 0.1555 | 0.1087 | 0.0000 | 0.5884 | 0.988 | 0.0250 |
| B5 | [0.50, 0.60) | 919 | 0.0862 | 0.0965 | 0.0604 | 0.0000 | 0.5685 | 0.995 | 0.0248 |
| B6 | [0.60, 0.70) | 936 | 0.0551 | 0.0844 | 0.0378 | 0.0000 | 0.5463 | 0.996 | 0.0253 |
| B7 | [0.70, 0.80) | 985 | 0.1791 | 0.1854 | 0.1238 | 0.0030 | 0.6268 | 0.983 | 0.0219 |
| B8 | [0.80, 0.90) | 1699 | 0.3844 | 0.3829 | 0.2694 | 0.0129 | 0.7404 | 0.924 | 0.0235 |
| B9 | [0.90, 1.00] | 486 | 0.1412 | 0.1526 | 0.1012 | 0.0350 | 0.7852 | 0.988 | 0.0126 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7936 | 0.7934 | 0.6069 | 0.0063 | 0.9687 | 0.609 | 0.7571 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0000 | 0.0000 | 0.0000 | 0.0345 | 0.0000 | 1.000 | 0.0201 |
| B1 | [0.10, 0.20) | 34 | 0.2396 | 0.2432 | 0.1988 | 0.0000 | 0.9976 | 0.970 | 0.0262 |
| B2 | [0.20, 0.30) | 185 | 0.2171 | 0.1651 | 0.1665 | 0.0000 | 0.7706 | 0.986 | 0.0239 |
| B3 | [0.30, 0.40) | 493 | 0.2491 | 0.2854 | 0.1781 | 0.0264 | 0.7523 | 0.958 | 0.0231 |
| B4 | [0.40, 0.50) | 677 | 0.2954 | 0.3454 | 0.2032 | 0.0074 | 0.6838 | 0.938 | 0.0232 |
| B5 | [0.50, 0.60) | 705 | 0.3322 | 0.3499 | 0.2324 | 0.0071 | 0.7094 | 0.937 | 0.0236 |
| B6 | [0.60, 0.70) | 809 | 0.2361 | 0.2494 | 0.1625 | 0.0222 | 0.6265 | 0.968 | 0.0206 |
| B7 | [0.70, 0.80) | 67 | 0.3851 | 0.5140 | 0.2645 | 0.0299 | 0.9476 | 0.858 | 0.0150 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.4850 | 0.4167 | 0.3353 | 0.0020 | 0.8144 | 0.909 | 0.5310 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7700 | 0.7860 | 0.5999 | 0.0000 | 0.9341 | 0.618 | 0.7222 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-4 sample (n=300)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8902 | 0.8774 | 0.7136 | 0.0000 | 0.9758 | 0.480 | 0.8268 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-4 sample — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 13.34s (19325 pair rows scored across 6 corpora).
