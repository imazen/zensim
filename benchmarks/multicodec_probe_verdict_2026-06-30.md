# bake_verdict — instant V_X eval

- Bake: `/mnt/v/output/zensim-multicodec-probe/probe_372.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: no

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC | geomean3 |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8827 | 0.8843 | 0.6995 | 0.0002 | 0.9822 | 0.467 | 0.8238 | 0.9152 |
| KADIK10k | 10125 | 0.6397 | 0.6709 | 0.4643 | 0.0037 | 0.8884 | 0.742 | 0.5844 | 0.7251 |
| TID2013 | 3000 | 0.7248 | 0.7300 | 0.5370 | 0.0227 | 0.9199 | 0.683 | 0.6964 | 0.7866 |
| KonJND-1k (full) | 1008 | 0.5619 | 0.5291 | 0.3948 | 0.0040 | 0.8564 | 0.849 | 0.5895 | 0.6338 |
| AIC-3 CTC | 600 | 0.7948 | 0.8044 | 0.6252 | 0.0000 | 0.9432 | 0.594 | 0.7254 | 0.8449 |
| AIC-4 sample | 300 | 0.8921 | 0.8795 | 0.7158 | 0.0000 | 0.9779 | 0.476 | 0.8379 | 0.9155 |

## CODEC_TARGET_GOALS.md scorecard (measurable subset)

| Goal | Measure | Value | Soft score |
|---|---|---:|---:|
| G1 dynamic range | pooled p5≤25 ∧ p95≥85 | p5=0.1 p95=81.5 | 0.90 |
| G5 HF rank | KonJND+AIC-3 SROCC ≥0.70 | 0.562 / 0.795 | 0.32 |
| G7 CID22 rank | SROCC ≥0.85 (advisory) | 0.8827 | 1.00 |
| G8 Z-RMSE | AIC-3 ≤0.80 | 0.594 | 0.69 |
| G9 DS-AUC | AIC-3 ≥0.70 | 0.7254 | 0.17 |

**Weighted goal score (measurable subset): 0.654**

_G2 (JND anchor), G3 (monotonicity), G4 (cross-codec), G6 (MF band coverage), G10 (per-source), G11 (display) require external q-sweep / cross-codec / multi-PPD data not present in the held-out feature parquets. Run the dedicated q-sweep harness for those._

## DIAL panel (codec-target G1/G3 — densified multi-codec q-sweep)

Grid: `/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29.parquet` — 4817 rows, 115 curves across 4 codec families.

| metric | value | gate | pass |
|---|--:|---|:--:|
| forward strict-increase | 0.0298 | — | |
| forward sub-resolution (≤0.5pt move) | 0.6148 | — (dense-grid) | |
| **inversions** (backwards > 0.5pt) | 0.2780 | G3 ≤ 0.07 | ✗ |
| ↳ strict backwards (any > 1e-9) | 0.8162 | — (noise diag) | |
| ↳ backwards-step magnitude med / p90 | 0.32 / 1.29 | score-pts | |
| codec-saturated (identical encode) | 0.0774 | — (codec ceiling) | |
| flat / clamp dead-zone (distinct feats, \|Δ\|≤1e-9) | 0.0000 | G3 ≤ 0.05 | ✓ |
| monotonicity (1 − inversions) | 0.7220 | G3 ≥ 0.93 | ✗ |
| dial p5 / p95 | -4.0 / 101.3 | G1 p5≤25 ∧ p95≥85 | ✓ |
| G1 soft / G3 soft | 1.00 / 0.00 | (1.0 = full pass) | |

Per-codec inversions / flat-clamp + representable config range:

| codec | param | min..max | n_curves | n_pairs | inversions | flat | monotonicity | score @worst→@best |
|---|---|---|--:|--:|--:|--:|--:|---|
| avif | q | 0..100 | 35 | 1365 | 0.3436 | 0.0000 | 0.6564 | 15.8 → -3.9 |
| jpeg | q | 0..100 | 23 | 897 | 0.2363 | 0.0000 | 0.7637 | 13.4 → -2.8 |
| jxl | distance | 0.03..25.00 | 33 | 1504 | 0.2467 | 0.0000 | 0.7533 | 15.1 → 7.4 |
| webp | q | 0..100 | 24 | 936 | 0.2724 | 0.0000 | 0.7276 | 17.1 → -1.3 |

_`param`/`min..max` = the native codec config axis and its representable range in the grid (integer quality for q-codecs; butteraugli distance for JXL — lower distance = higher quality). `score @worst→@best` = median dial score at the lowest- and highest-quality representable config (for distance, worst = max distance). **inversions** = fraction of adjacent-q pairs where the score went BACKWARDS by more than 0.5 score-pt (higher quality scored materially lower — a real ranking error; the gated metric); **flat** = distinct-feature pairs with identical output (\|Δ\|≤1e-9 — a metric dead-zone). Pairs where the CODEC emitted an identical image at two q (near-identical features — zenjpeg/webp quality ceiling) are split into a separate **codec-saturated** bucket and are NOT counted as a bake dead-zone. The aggregate table additionally breaks out the strict (any-backwards) rate and the backwards-step magnitude distribution, plus a sub-resolution bucket (0<\|Δ\|≤0.5 pt) that is EXPECTED on the densified near-lossless grid (adjacent configs are sub-JND apart, so the dial correctly barely moves) and is NOT gated. monotonicity = 1 − inversions. Densified grid: q0 + step-1 q90→100 + fractional near-lossless q for q-codecs (96.5..99.9) + JND zone + jxl-in-butteraugli-distance (0→0.3 step .025, 0.3→1 step .05, 1→3 step .2, 13→25 step 2; q-equiv = 100 − 4·distance)._

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8827 | 0.8843 | 0.6995 | 0.0002 | 0.9822 | 0.467 | 0.8238 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0039 | 0.1972 | 0.0025 | 0.0175 | 0.7246 | 0.980 | 0.0217 |
| B4 | [0.40, 0.50) | 266 | 0.2720 | 0.2798 | 0.1864 | 0.0376 | 0.6583 | 0.960 | 0.0220 |
| B5 | [0.50, 0.60) | 615 | 0.4029 | 0.4042 | 0.2764 | 0.0114 | 0.7659 | 0.915 | 0.0216 |
| B6 | [0.60, 0.70) | 836 | 0.3925 | 0.3991 | 0.2646 | 0.0167 | 0.7463 | 0.917 | 0.0225 |
| B7 | [0.70, 0.80) | 1092 | 0.3963 | 0.4138 | 0.2751 | 0.0165 | 0.7497 | 0.910 | 0.0222 |
| B8 | [0.80, 0.90) | 1382 | 0.4959 | 0.5010 | 0.3359 | 0.0101 | 0.8258 | 0.865 | 0.0197 |
| B9 | [0.90, 1.00] | 43 | 0.0127 | 0.2055 | 0.0100 | 0.0465 | 0.6207 | 0.979 | 0.0050 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.6397 | 0.6709 | 0.4643 | 0.0037 | 0.8884 | 0.742 | 0.5844 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.0930 | 0.1415 | 0.0656 | 0.0255 | 0.6725 | 0.990 | 0.0199 |
| B1 | [0.10, 0.20) | 910 | 0.0303 | 0.0918 | 0.0203 | 0.0000 | 0.6113 | 0.996 | 0.0222 |
| B2 | [0.20, 0.30) | 1111 | 0.0309 | 0.0966 | 0.0217 | 0.0009 | 0.6149 | 0.995 | 0.0247 |
| B3 | [0.30, 0.40) | 1291 | 0.1375 | 0.1586 | 0.0963 | 0.0000 | 0.5806 | 0.987 | 0.0240 |
| B4 | [0.40, 0.50) | 1013 | 0.1238 | 0.1816 | 0.0867 | 0.0000 | 0.7024 | 0.983 | 0.0249 |
| B5 | [0.50, 0.60) | 919 | 0.0772 | 0.0908 | 0.0530 | 0.0000 | 0.5684 | 0.996 | 0.0248 |
| B6 | [0.60, 0.70) | 936 | 0.0636 | 0.1044 | 0.0439 | 0.0000 | 0.5670 | 0.995 | 0.0252 |
| B7 | [0.70, 0.80) | 985 | 0.1050 | 0.1185 | 0.0742 | 0.0000 | 0.5887 | 0.993 | 0.0223 |
| B8 | [0.80, 0.90) | 1699 | 0.3291 | 0.3428 | 0.2288 | 0.0059 | 0.7089 | 0.939 | 0.0241 |
| B9 | [0.90, 1.00] | 486 | 0.1732 | 0.1626 | 0.1245 | 0.0432 | 0.6845 | 0.987 | 0.0125 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7248 | 0.7300 | 0.5370 | 0.0227 | 0.9199 | 0.683 | 0.6964 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.1636 | 0.3010 | 0.1110 | 0.0000 | 0.6225 | 0.954 | 0.0190 |
| B1 | [0.10, 0.20) | 34 | 0.3654 | 0.4283 | 0.2375 | 0.0000 | 0.9088 | 0.904 | 0.0253 |
| B2 | [0.20, 0.30) | 185 | 0.0589 | 0.1264 | 0.0375 | 0.0000 | 0.7882 | 0.992 | 0.0240 |
| B3 | [0.30, 0.40) | 493 | 0.0825 | 0.1793 | 0.0555 | 0.0081 | 0.5565 | 0.984 | 0.0238 |
| B4 | [0.40, 0.50) | 677 | 0.3055 | 0.3826 | 0.2063 | 0.0015 | 0.6985 | 0.924 | 0.0228 |
| B5 | [0.50, 0.60) | 705 | 0.2546 | 0.2948 | 0.1733 | 0.0085 | 0.6663 | 0.956 | 0.0244 |
| B6 | [0.60, 0.70) | 809 | 0.1551 | 0.1774 | 0.1067 | 0.0235 | 0.6942 | 0.984 | 0.0211 |
| B7 | [0.70, 0.80) | 67 | 0.2321 | 0.3027 | 0.1583 | 0.0149 | 0.9113 | 0.953 | 0.0179 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.5619 | 0.5291 | 0.3948 | 0.0040 | 0.8564 | 0.849 | 0.5895 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7948 | 0.8044 | 0.6252 | 0.0000 | 0.9432 | 0.594 | 0.7254 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-4 sample (n=300)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8921 | 0.8795 | 0.7158 | 0.0000 | 0.9779 | 0.476 | 0.8379 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-4 sample — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 8.78s (19325 pair rows scored across 6 corpora).
