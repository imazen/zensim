bake_verdict — bake=/mnt/v/output/zensim/bakes/v47_masked_strict_2026-05-26.bin  features-root=/mnt/v/zen/zensim-training/2026-05-15-full-features  corpora=cid22,kadid,tid,konjnd,aic3,aic4
bake: n_inputs=372  feature_transforms=yes  per_sample_alpha_head=yes  hybrid_head=no
  CID22: loaded 4292 pairs × 372 features from "/mnt/v/zen/zensim-training/2026-05-15-full-features/cid22_features_372col_2026-05-15.parquet"
  KADIK10k: loaded 10125 pairs × 372 features from "/mnt/v/zen/zensim-training/2026-05-15-full-features/kadid_features_372col_2026-05-15.parquet"
  TID2013: loaded 3000 pairs × 372 features from "/mnt/v/zen/zensim-training/2026-05-15-full-features/tid_features_372col_2026-05-15.parquet"
  KonJND-1k (full): loaded 1008 pairs × 372 features from "/mnt/v/zen/zensim-training/2026-05-15-full-features/konjnd_features_372col_2026-05-15.parquet"
  AIC-3 CTC: loaded 600 pairs × 372 features from "/mnt/v/zen/zensim-training/2026-05-15-full-features/aic3_features_372col_2026-05-15.parquet"
  AIC-4 sample: loaded 300 pairs × 372 features from "/mnt/v/zen/zensim-training/2026-05-15-full-features/aic4_features_372col_2026-05-20.parquet"
# bake_verdict — instant V_X eval

- Bake: `/mnt/v/output/zensim/bakes/v47_masked_strict_2026-05-26.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: yes (uses predict_transformed)

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC | geomean3 |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8547 | 0.7761 | 0.6619 | 0.0033 | 0.9754 | 0.631 | 0.7699 | 0.8649 |
| KADIK10k | 10125 | 0.8030 | 0.7967 | 0.6081 | 0.0006 | 0.9522 | 0.604 | 0.7440 | 0.8477 |
| TID2013 | 3000 | 0.7965 | 0.8232 | 0.6107 | 0.0037 | 0.9686 | 0.568 | 0.7856 | 0.8596 |
| KonJND-1k (full) | 1008 | 0.4850 | 0.4200 | 0.3353 | 0.0020 | 0.8144 | 0.908 | 0.5651 | 0.5495 |
| AIC-3 CTC | 600 | 0.7700 | 0.7749 | 0.5999 | 0.0000 | 0.9452 | 0.632 | 0.6135 | 0.8262 |
| AIC-4 sample | 300 | 0.8902 | 0.8733 | 0.7136 | 0.0000 | 0.9771 | 0.487 | 0.7292 | 0.9124 |

## CODEC_TARGET_GOALS.md scorecard (measurable subset)

| Goal | Measure | Value | Soft score |
|---|---|---:|---:|
| G1 dynamic range | pooled p5≤25 ∧ p95≥85 | p5=-42.4 p95=-15.7 | 0.00 |
| G5 HF rank | KonJND+AIC-3 SROCC ≥0.70 | 0.485 / 0.770 | 0.23 |
| G7 CID22 rank | SROCC ≥0.85 (advisory) | 0.8547 | 1.00 |
| G8 Z-RMSE | AIC-3 ≤0.80 | 0.632 | 0.56 |
| G9 DS-AUC | AIC-3 ≥0.70 | 0.6135 | 0.00 |

**Weighted goal score (measurable subset): 0.265**

_G2 (JND anchor), G3 (monotonicity), G4 (cross-codec), G6 (MF band coverage), G10 (per-source), G11 (display) require external q-sweep / cross-codec / multi-PPD data not present in the held-out feature parquets. Run the dedicated q-sweep harness for those._

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8547 | 0.7761 | 0.6619 | 0.0033 | 0.9754 | 0.631 | 0.7699 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1983 | 0.2852 | 0.1316 | 0.0526 | 0.7026 | 0.958 | 0.0195 |
| B4 | [0.40, 0.50) | 266 | 0.2096 | 0.2477 | 0.1405 | 0.0150 | 0.6496 | 0.969 | 0.0223 |
| B5 | [0.50, 0.60) | 615 | 0.3285 | 0.3331 | 0.2221 | 0.0163 | 0.7143 | 0.943 | 0.0224 |
| B6 | [0.60, 0.70) | 836 | 0.3616 | 0.3555 | 0.2444 | 0.0144 | 0.7288 | 0.935 | 0.0231 |
| B7 | [0.70, 0.80) | 1092 | 0.3555 | 0.3726 | 0.2445 | 0.0137 | 0.7400 | 0.928 | 0.0228 |
| B8 | [0.80, 0.90) | 1382 | 0.4461 | 0.4506 | 0.3001 | 0.0166 | 0.7952 | 0.893 | 0.0202 |
| B9 | [0.90, 1.00] | 43 | 0.0160 | 0.1281 | 0.0011 | 0.0233 | 0.5496 | 0.992 | 0.0051 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8030 | 0.7967 | 0.6081 | 0.0006 | 0.9522 | 0.604 | 0.7440 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2203 | 0.2280 | 0.1539 | 0.0284 | 0.6682 | 0.974 | 0.0195 |
| B1 | [0.10, 0.20) | 910 | 0.1283 | 0.1431 | 0.0900 | 0.0000 | 0.6694 | 0.990 | 0.0222 |
| B2 | [0.20, 0.30) | 1111 | 0.1007 | 0.1139 | 0.0705 | 0.0000 | 0.5622 | 0.993 | 0.0247 |
| B3 | [0.30, 0.40) | 1291 | 0.1385 | 0.1516 | 0.0972 | 0.0000 | 0.5773 | 0.988 | 0.0241 |
| B4 | [0.40, 0.50) | 1013 | 0.1575 | 0.1572 | 0.1106 | 0.0000 | 0.6508 | 0.988 | 0.0250 |
| B5 | [0.50, 0.60) | 919 | 0.0862 | 0.1026 | 0.0602 | 0.0000 | 0.6205 | 0.995 | 0.0248 |
| B6 | [0.60, 0.70) | 936 | 0.0551 | 0.0626 | 0.0378 | 0.0000 | 0.5464 | 0.998 | 0.0253 |
| B7 | [0.70, 0.80) | 985 | 0.1791 | 0.1864 | 0.1238 | 0.0030 | 0.6268 | 0.982 | 0.0219 |
| B8 | [0.80, 0.90) | 1699 | 0.3844 | 0.3571 | 0.2694 | 0.0188 | 0.7404 | 0.934 | 0.0239 |
| B9 | [0.90, 1.00] | 486 | 0.1412 | 0.1400 | 0.1012 | 0.0556 | 0.8342 | 0.990 | 0.0124 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7965 | 0.8232 | 0.6107 | 0.0037 | 0.9686 | 0.568 | 0.7856 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.3436 | 0.5007 | 0.2392 | 0.0000 | 0.9626 | 0.866 | 0.0168 |
| B1 | [0.10, 0.20) | 34 | 0.5385 | 0.6094 | 0.3696 | 0.0000 | 0.8631 | 0.793 | 0.0213 |
| B2 | [0.20, 0.30) | 185 | 0.3662 | 0.4271 | 0.2583 | 0.0162 | 0.7635 | 0.904 | 0.0211 |
| B3 | [0.30, 0.40) | 493 | 0.2702 | 0.3153 | 0.1870 | 0.0223 | 0.6584 | 0.949 | 0.0226 |
| B4 | [0.40, 0.50) | 677 | 0.2951 | 0.3454 | 0.2028 | 0.0074 | 0.6829 | 0.938 | 0.0232 |
| B5 | [0.50, 0.60) | 705 | 0.3323 | 0.3499 | 0.2324 | 0.0057 | 0.7094 | 0.937 | 0.0236 |
| B6 | [0.60, 0.70) | 809 | 0.2361 | 0.2469 | 0.1625 | 0.0222 | 0.6684 | 0.969 | 0.0206 |
| B7 | [0.70, 0.80) | 67 | 0.3851 | 0.5140 | 0.2645 | 0.0299 | 0.9476 | 0.858 | 0.0150 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.4850 | 0.4200 | 0.3353 | 0.0020 | 0.8144 | 0.908 | 0.5651 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7700 | 0.7749 | 0.5999 | 0.0000 | 0.9452 | 0.632 | 0.6135 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-4 sample (n=300)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8902 | 0.8733 | 0.7136 | 0.0000 | 0.9771 | 0.487 | 0.7292 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-4 sample — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 50.41s (19325 pair rows scored across 6 corpora).
bake_verdict: complete in 50.41s
