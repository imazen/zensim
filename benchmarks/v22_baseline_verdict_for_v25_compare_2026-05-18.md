# bake_verdict — instant V_X eval

- Bake: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 300
- Feature transforms: no

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8324 | 0.8289 | 0.6340 | 0.0443 | 0.9006 | 0.559 |
| KADIK10k | 10125 | 0.9677 | 0.9686 | 0.8432 | 0.0433 | 0.9804 | 0.249 |
| TID2013 | 3000 | 0.9729 | 0.9717 | 0.8571 | 0.0357 | 0.9832 | 0.236 |
| KonJND-1k (full) | 1008 | 0.8927 | 0.9265 | 0.7070 | 0.0446 | 0.9178 | 0.376 |
| AIC-3 CTC | 600 | 0.7845 | 0.7953 | 0.6155 | 0.0433 | 0.8630 | 0.606 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8324 | 0.8289 | 0.6340 | 0.0443 | 0.9006 | 0.559 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0487 | 0.1900 | 0.0276 | 0.0351 | 0.0703 | 0.982 | 0.0207 |
| B4 | [0.40, 0.50) | 266 | 0.2510 | 0.2452 | 0.1658 | 0.0489 | 0.3169 | 0.969 | 0.0223 |
| B5 | [0.50, 0.60) | 615 | 0.2466 | 0.2584 | 0.1655 | 0.0423 | 0.3049 | 0.966 | 0.0231 |
| B6 | [0.60, 0.70) | 836 | 0.2220 | 0.2317 | 0.1484 | 0.0347 | 0.2580 | 0.973 | 0.0243 |
| B7 | [0.70, 0.80) | 1092 | 0.3192 | 0.3244 | 0.2173 | 0.0513 | 0.3766 | 0.946 | 0.0233 |
| B8 | [0.80, 0.90) | 1382 | 0.4958 | 0.5026 | 0.3354 | 0.0347 | 0.5846 | 0.865 | 0.0195 |
| B9 | [0.90, 1.00] | 43 | 0.1056 | 0.1831 | 0.0698 | 0.0233 | 0.2725 | 0.983 | 0.0054 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9677 | 0.9686 | 0.8432 | 0.0433 | 0.9804 | 0.249 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.4150 | 0.4235 | 0.2939 | 0.0241 | 0.5015 | 0.906 | 0.0179 |
| B1 | [0.10, 0.20) | 910 | 0.4149 | 0.4191 | 0.2922 | 0.0374 | 0.4853 | 0.908 | 0.0199 |
| B2 | [0.20, 0.30) | 1111 | 0.3995 | 0.4047 | 0.2807 | 0.0387 | 0.4741 | 0.914 | 0.0222 |
| B3 | [0.30, 0.40) | 1291 | 0.3342 | 0.3325 | 0.2352 | 0.0418 | 0.3953 | 0.943 | 0.0225 |
| B4 | [0.40, 0.50) | 1013 | 0.3754 | 0.3801 | 0.2666 | 0.0444 | 0.4405 | 0.925 | 0.0228 |
| B5 | [0.50, 0.60) | 919 | 0.3454 | 0.3527 | 0.2442 | 0.0457 | 0.4190 | 0.936 | 0.0229 |
| B6 | [0.60, 0.70) | 936 | 0.3649 | 0.3643 | 0.2549 | 0.0438 | 0.4434 | 0.931 | 0.0232 |
| B7 | [0.70, 0.80) | 985 | 0.3603 | 0.3669 | 0.2552 | 0.0396 | 0.4420 | 0.930 | 0.0207 |
| B8 | [0.80, 0.90) | 1699 | 0.5019 | 0.5025 | 0.3554 | 0.0383 | 0.5871 | 0.865 | 0.0217 |
| B9 | [0.90, 1.00] | 486 | 0.1818 | 0.2299 | 0.1248 | 0.0370 | 0.2158 | 0.973 | 0.0124 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.9729 | 0.9717 | 0.8571 | 0.0357 | 0.9832 | 0.236 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0621 | 0.3495 | 0.0666 | 0.0345 | 0.0778 | 0.937 | 0.0181 |
| B1 | [0.10, 0.20) | 34 | 0.5319 | 0.5997 | 0.3411 | 0.0294 | 0.7017 | 0.800 | 0.0222 |
| B2 | [0.20, 0.30) | 185 | 0.6479 | 0.6523 | 0.4607 | 0.0378 | 0.7180 | 0.758 | 0.0166 |
| B3 | [0.30, 0.40) | 493 | 0.7352 | 0.7356 | 0.5396 | 0.0446 | 0.8173 | 0.677 | 0.0152 |
| B4 | [0.40, 0.50) | 677 | 0.7625 | 0.7625 | 0.5598 | 0.0502 | 0.8370 | 0.647 | 0.0151 |
| B5 | [0.50, 0.60) | 705 | 0.7077 | 0.7068 | 0.5096 | 0.0496 | 0.7907 | 0.707 | 0.0166 |
| B6 | [0.60, 0.70) | 809 | 0.5860 | 0.5854 | 0.4099 | 0.0346 | 0.6825 | 0.811 | 0.0168 |
| B7 | [0.70, 0.80) | 67 | 0.2842 | 0.4937 | 0.1928 | 0.0746 | 0.2801 | 0.870 | 0.0152 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8927 | 0.9265 | 0.7070 | 0.0446 | 0.9178 | 0.376 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.7845 | 0.7953 | 0.6155 | 0.0433 | 0.8630 | 0.606 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 8.61s (19025 pair rows scored across 5 corpora).
