# bake_verdict — instant V_X eval

- Bake: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22cvvdp_safe_kadid_s1_h128.bin`
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bake n_inputs: 372
- Feature transforms: yes (uses predict_transformed)

## Summary (one row per corpus)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8526 | 0.8536 | 0.6586 | 0.0529 | 0.9074 | 0.521 |
| KADIK10k | 10125 | 0.8362 | 0.8351 | 0.6426 | 0.0331 | 0.9020 | 0.550 |
| TID2013 | 3000 | 0.8564 | 0.8674 | 0.6748 | 0.0420 | 0.8910 | 0.498 |
| KonJND-1k (full) | 1008 | 0.1155 | 0.1469 | 0.0789 | 0.0437 | 0.2127 | 0.989 |
| AIC-3 CTC | 600 | 0.8031 | 0.8166 | 0.6379 | 0.0567 | 0.8770 | 0.577 |

## CID22 (n=4292)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8526 | 0.8536 | 0.6586 | 0.0529 | 0.9074 | 0.521 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### CID22 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0205 | 0.1629 | 0.0113 | 0.0526 | 0.0274 | 0.987 | 0.0211 |
| B4 | [0.40, 0.50) | 266 | 0.2365 | 0.3160 | 0.1563 | 0.0526 | 0.2975 | 0.949 | 0.0216 |
| B5 | [0.50, 0.60) | 615 | 0.3343 | 0.3339 | 0.2264 | 0.0455 | 0.3956 | 0.943 | 0.0222 |
| B6 | [0.60, 0.70) | 836 | 0.3499 | 0.3525 | 0.2339 | 0.0419 | 0.4211 | 0.936 | 0.0232 |
| B7 | [0.70, 0.80) | 1092 | 0.3395 | 0.3424 | 0.2322 | 0.0540 | 0.3993 | 0.940 | 0.0230 |
| B8 | [0.80, 0.90) | 1382 | 0.4014 | 0.4046 | 0.2693 | 0.0434 | 0.4760 | 0.914 | 0.0207 |
| B9 | [0.90, 1.00] | 43 | 0.1318 | 0.1805 | 0.0941 | 0.0465 | 0.2285 | 0.984 | 0.0052 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KADIK10k (n=10125)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8362 | 0.8351 | 0.6426 | 0.0331 | 0.9020 | 0.550 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### KADIK10k 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.1728 | 0.2151 | 0.1211 | 0.0397 | 0.2286 | 0.977 | 0.0197 |
| B1 | [0.10, 0.20) | 910 | 0.1691 | 0.1905 | 0.1185 | 0.0418 | 0.2216 | 0.982 | 0.0220 |
| B2 | [0.20, 0.30) | 1111 | 0.0822 | 0.0881 | 0.0566 | 0.0369 | 0.0916 | 0.996 | 0.0248 |
| B3 | [0.30, 0.40) | 1291 | 0.1688 | 0.1880 | 0.1184 | 0.0449 | 0.1944 | 0.982 | 0.0238 |
| B4 | [0.40, 0.50) | 1013 | 0.1689 | 0.1978 | 0.1193 | 0.0385 | 0.2045 | 0.980 | 0.0248 |
| B5 | [0.50, 0.60) | 919 | 0.1073 | 0.1621 | 0.0743 | 0.0392 | 0.1333 | 0.987 | 0.0246 |
| B6 | [0.60, 0.70) | 936 | 0.1118 | 0.1382 | 0.0769 | 0.0427 | 0.1367 | 0.990 | 0.0250 |
| B7 | [0.70, 0.80) | 985 | 0.1539 | 0.1638 | 0.1064 | 0.0396 | 0.1694 | 0.986 | 0.0220 |
| B8 | [0.80, 0.90) | 1699 | 0.3720 | 0.3782 | 0.2594 | 0.0347 | 0.4425 | 0.926 | 0.0236 |
| B9 | [0.90, 1.00] | 486 | 0.1557 | 0.2199 | 0.1057 | 0.0391 | 0.1826 | 0.976 | 0.0122 |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## TID2013 (n=3000)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8564 | 0.8674 | 0.6748 | 0.0420 | 0.8910 | 0.498 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

### TID2013 10-band full Mohammadi panel (PRIMARY release gate)

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
|---|---|--:|---:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.1660 | 0.3702 | 0.1159 | 0.0345 | 0.1605 | 0.929 | 0.0181 |
| B1 | [0.10, 0.20) | 34 | 0.4487 | 0.4879 | 0.3089 | 0.0588 | 0.5664 | 0.873 | 0.0241 |
| B2 | [0.20, 0.30) | 185 | 0.2786 | 0.2810 | 0.1868 | 0.0541 | 0.3484 | 0.960 | 0.0232 |
| B3 | [0.30, 0.40) | 493 | 0.3762 | 0.3874 | 0.2564 | 0.0406 | 0.4809 | 0.922 | 0.0223 |
| B4 | [0.40, 0.50) | 677 | 0.4743 | 0.4877 | 0.3282 | 0.0517 | 0.5538 | 0.873 | 0.0212 |
| B5 | [0.50, 0.60) | 705 | 0.4245 | 0.4543 | 0.2911 | 0.0411 | 0.5136 | 0.891 | 0.0223 |
| B6 | [0.60, 0.70) | 809 | 0.1840 | 0.2242 | 0.1256 | 0.0396 | 0.2209 | 0.975 | 0.0208 |
| B7 | [0.70, 0.80) | 67 | 0.3577 | 0.4777 | 0.2509 | 0.0448 | 0.4180 | 0.879 | 0.0153 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_⚠ marks bands with n < 30 — point estimates are noisy (CI widths exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale per Mohammadi 2025._

## KonJND-1k (full) (n=1008)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.1155 | 0.1469 | 0.0789 | 0.0437 | 0.2127 | 0.989 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for KonJND-1k (full) — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V_X bake | 0.8031 | 0.8166 | 0.6379 | 0.0567 | 0.8770 | 0.577 |

_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because saturation regions dominate the residual._

_Per-band breakdown skipped for AIC-3 CTC — the corpus uses a JND step grid (AIC-3) or a raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing read on this corpus._

---
Wall time: 3.57s (19025 pair rows scored across 5 corpora).
