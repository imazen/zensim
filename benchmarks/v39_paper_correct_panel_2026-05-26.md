# bake_verdict — instant V_X eval

- Bake: `zensim/weights/v39_v32plus_spline_seed17_2026-05-25.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: yes (uses predict_transformed)

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC | geomean3 |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8793 | 0.8702 | 0.6881 | 0.0438 | 0.9264 | 0.493 | 0.8168 | 0.8916 |
| KADIK10k | 10125 | 0.9251 | 0.9262 | 0.7591 | 0.0501 | 0.9542 | 0.377 | 0.8881 | 0.9351 |
| TID2013 | 3000 | 0.9317 | 0.9263 | 0.7651 | 0.0477 | 0.9545 | 0.377 | 0.9076 | 0.9375 |
| KonJND-1k (full) | 1008 | 0.4197 | 0.3713 | 0.2910 | 0.0387 | 0.5594 | 0.929 | 0.5398 | 0.4434 |
| AIC-3 CTC | 600 | 0.8023 | 0.8116 | 0.6351 | 0.0517 | 0.8749 | 0.584 | 0.7385 | 0.8290 |
| AIC-4 sample | 300 | 0.9051 | 0.8931 | 0.7425 | 0.0500 | 0.9423 | 0.450 | 0.8486 | 0.9133 |

## CODEC_TARGET_GOALS.md scorecard (measurable subset)

| Goal | Measure | Value | Soft score |
|---|---|---:|---:|
| G1 dynamic range | pooled p5≤25 ∧ p95≥85 | p5=-89.7 p95=97.4 | 1.00 |
| G5 HF rank | KonJND+AIC-3 SROCC ≥0.70 | 0.420 / 0.802 | 0.34 |
| G7 CID22 rank | SROCC ≥0.85 (advisory) | 0.8793 | 1.00 |
| G8 Z-RMSE | AIC-3 ≤0.80 | 0.584 | 0.72 |
| G9 DS-AUC | AIC-3 ≥0.70 | 0.7385 | 0.26 |

**Weighted goal score (measurable subset): 0.714**

_G2 (JND anchor), G3 (monotonicity), G4 (cross-codec), G6 (MF band coverage), G10 (per-source), G11 (display) require external q-sweep / cross-codec / multi-PPD data not present in the held-out feature parquets. Run the dedicated q-sweep harness for those._

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8793 | 0.8702 | 0.6881 | 0.0438 | 0.9264 | 0.493 | 0.8168 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0806 | 0.2331 | 0.0614 | 0.0351 | 0.0215 | 0.972 | 0.0209 |
| B4 | [0.40, 0.50) | 266 | 0.1478 | 0.2008 | 0.0987 | 0.0414 | 0.1652 | 0.980 | 0.0229 |
| B5 | [0.50, 0.60) | 615 | 0.2999 | 0.3028 | 0.2043 | 0.0439 | 0.3643 | 0.953 | 0.0227 |
| B6 | [0.60, 0.70) | 836 | 0.3581 | 0.3644 | 0.2398 | 0.0455 | 0.4391 | 0.931 | 0.0231 |
| B7 | [0.70, 0.80) | 1092 | 0.3841 | 0.3959 | 0.2658 | 0.0458 | 0.4564 | 0.918 | 0.0224 |
| B8 | [0.80, 0.90) | 1382 | 0.5017 | 0.5005 | 0.3394 | 0.0340 | 0.5931 | 0.866 | 0.0197 |
| B9 | [0.90, 1.00] | 43 | 0.1699 | 0.2740 | 0.1274 | 0.0698 | 0.1311 | 0.962 | 0.0049 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9251 | 0.9262 | 0.7591 | 0.0501 | 0.9542 | 0.377 | 0.8881 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2120 | 0.2383 | 0.1495 | 0.0426 | 0.2838 | 0.971 | 0.0196 |
| B1 | [0.10, 0.20) | 910 | 0.2475 | 0.2607 | 0.1736 | 0.0385 | 0.2992 | 0.965 | 0.0215 |
| B2 | [0.20, 0.30) | 1111 | 0.2423 | 0.2662 | 0.1676 | 0.0441 | 0.2930 | 0.964 | 0.0238 |
| B3 | [0.30, 0.40) | 1291 | 0.2301 | 0.2299 | 0.1604 | 0.0465 | 0.2811 | 0.973 | 0.0236 |
| B4 | [0.40, 0.50) | 1013 | 0.2356 | 0.2348 | 0.1632 | 0.0405 | 0.2757 | 0.972 | 0.0244 |
| B5 | [0.50, 0.60) | 919 | 0.2291 | 0.2360 | 0.1587 | 0.0435 | 0.2885 | 0.972 | 0.0241 |
| B6 | [0.60, 0.70) | 936 | 0.2097 | 0.2130 | 0.1452 | 0.0427 | 0.2408 | 0.977 | 0.0244 |
| B7 | [0.70, 0.80) | 985 | 0.1975 | 0.2069 | 0.1381 | 0.0416 | 0.2454 | 0.978 | 0.0219 |
| B8 | [0.80, 0.90) | 1699 | 0.3918 | 0.3962 | 0.2755 | 0.0459 | 0.4647 | 0.918 | 0.0234 |
| B9 | [0.90, 1.00] | 486 | 0.1355 | 0.1471 | 0.0980 | 0.0473 | 0.1546 | 0.989 | 0.0126 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9317 | 0.9263 | 0.7651 | 0.0477 | 0.9545 | 0.377 | 0.9076 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.2944 | 0.4847 | 0.1899 | 0.0345 | 0.4311 | 0.875 | 0.0168 |
| B1 | [0.10, 0.20) | 34 | 0.3238 | 0.4522 | 0.2161 | 0.0588 | 0.4282 | 0.892 | 0.0241 |
| B2 | [0.20, 0.30) | 185 | 0.0765 | 0.1539 | 0.0496 | 0.0432 | 0.1042 | 0.988 | 0.0240 |
| B3 | [0.30, 0.40) | 493 | 0.4555 | 0.4724 | 0.3112 | 0.0467 | 0.5482 | 0.881 | 0.0208 |
| B4 | [0.40, 0.50) | 677 | 0.5557 | 0.5572 | 0.3907 | 0.0414 | 0.6409 | 0.830 | 0.0200 |
| B5 | [0.50, 0.60) | 705 | 0.5161 | 0.5195 | 0.3591 | 0.0511 | 0.6027 | 0.854 | 0.0209 |
| B6 | [0.60, 0.70) | 809 | 0.3911 | 0.3886 | 0.2673 | 0.0482 | 0.4788 | 0.921 | 0.0196 |
| B7 | [0.70, 0.80) | 67 | 0.1770 | 0.2545 | 0.1202 | 0.0448 | 0.2161 | 0.967 | 0.0181 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.4197 | 0.3713 | 0.2910 | 0.0387 | 0.5594 | 0.929 | 0.5398 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8023 | 0.8116 | 0.6351 | 0.0517 | 0.8749 | 0.584 | 0.7385 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-4 sample (n=300)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9051 | 0.8931 | 0.7425 | 0.0500 | 0.9423 | 0.450 | 0.8486 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-4 sample — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 4.63s (19325 pair rows scored across 6 corpora).
