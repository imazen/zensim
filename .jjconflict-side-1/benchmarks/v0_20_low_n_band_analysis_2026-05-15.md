# Low-sample-size band ceiling analysis (2026-05-15)

All (corpus, band) cells with **n < 100** across KADID / TID / CID22 evaluated against every V_X bake.

## What 'ceiling' means here

Two ceilings are relevant at low n:

1. **Sample-size ceiling** — the standard error of Spearman r is `SE(r) ≈ (1 − r²) / √(n − 1)`. The 95% CI is roughly `r ± 1.96·SE`. The CI upper bound is the **highest plausible SROCC at this n** given what we observed — anything above is sampling noise.

2. **Inter-observer ceiling** — bounded by human-MOS reliability. The CID22 paper (Sneyers / Ben Baruch / Vaxman 2023) reports inter-observer SROCC ≈ 0.93 on CID22 — no metric can exceed that on the population, but small-n samples can luck into higher r.

The per-band CI in our eval logs comes from a percentile bootstrap (BCa or quantile). Cells with n < 30 are marked ⚠ — CI widths exceed 0.3 SROCC and rankings between bakes are not statistically distinguishable.

## Per-band SROCC + CI across bakes

### TID2013 B0 ⚠ [0.00, 0.10), n = 29

| Bake | SROCC | 95% CI | SE est. |
|---|---:|---|---:|
| _fast-ssim2 (static)_ | 0.0835 | — | 0.188 |
| V_18 ship | 0.3897 | [0.050, 0.650] | 0.160 |
| V_18 base seed=1 | 0.0862 | [0.010, 0.460] | 0.188 |
| V_20 IS (98) | 0.4222 | [0.080, 0.710] | 0.155 |
| V_20b manifold | 0.3838 | [0.020, 0.690] | 0.161 |
| D1 3-way concat | 0.3124 | [0.040, 0.600] | 0.171 |
| D3 lift>=0.10 | 0.3604 | [0.050, 0.630] | 0.164 |
| V_20_4 multi-bake α=0.4 | 0.0000 | [0.000, 0.000] | 0.189 |

**Empirical sample ceiling** (max CI upper bound across bakes): 0.710
⚠ n < 30 — CI width is too large to discriminate between bakes.

### TID2013 B1 [0.10, 0.20), n = 34

| Bake | SROCC | 95% CI | SE est. |
|---|---:|---|---:|
| _fast-ssim2 (static)_ | 0.4399 | — | 0.140 |
| V_18 ship | 0.3339 | [0.020, 0.560] | 0.155 |
| V_18 base seed=1 | 0.5801 | [0.300, 0.730] | 0.115 |
| V_20 IS (98) | 0.6015 | [0.330, 0.760] | 0.111 |
| V_20b manifold | 0.5533 | [0.250, 0.710] | 0.121 |
| D1 3-way concat | 0.5336 | [0.240, 0.720] | 0.125 |
| D3 lift>=0.10 | 0.5130 | [0.220, 0.690] | 0.128 |
| V_20_4 multi-bake α=0.4 | 0.0000 | [0.000, 0.000] | 0.174 |

**Empirical sample ceiling** (max CI upper bound across bakes): 0.760

### TID2013 B7 [0.70, 0.80), n = 67

| Bake | SROCC | 95% CI | SE est. |
|---|---:|---|---:|
| _fast-ssim2 (static)_ | 0.4193 | — | 0.101 |
| V_18 ship | 0.0228 | [0.010, 0.290] | 0.123 |
| V_18 base seed=1 | 0.0028 | [0.000, 0.290] | 0.123 |
| V_20 IS (98) | 0.0154 | [0.000, 0.280] | 0.123 |
| V_20b manifold | 0.1959 | [0.020, 0.420] | 0.118 |
| D1 3-way concat | 0.0152 | [0.000, 0.300] | 0.123 |
| D3 lift>=0.10 | 0.0978 | [0.000, 0.360] | 0.122 |
| V_20_4 multi-bake α=0.4 | 0.0147 | [0.010, 0.300] | 0.123 |

**Empirical sample ceiling** (max CI upper bound across bakes): 0.420

### CID22 B3 [0.30, 0.40), n = 57

| Bake | SROCC | 95% CI | SE est. |
|---|---:|---|---:|
| _fast-ssim2 (static)_ | 0.1335 | — | 0.131 |
| V_18 ship | 0.0246 | [0.010, 0.340] | 0.134 |
| V_18 base seed=1 | 0.0471 | [0.010, 0.320] | 0.133 |
| V_20 IS (98) | 0.1534 | [0.010, 0.440] | 0.130 |
| V_20b manifold | 0.0270 | [0.010, 0.330] | 0.134 |
| D1 3-way concat | 0.1419 | [0.010, 0.450] | 0.131 |
| D3 lift>=0.10 | 0.0800 | [0.010, 0.360] | 0.133 |
| V_20_4 multi-bake α=0.4 | 0.1044 | [0.000, 0.380] | 0.132 |

**Empirical sample ceiling** (max CI upper bound across bakes): 0.450

### CID22 B9 [0.90, 1.00], n = 43

| Bake | SROCC | 95% CI | SE est. |
|---|---:|---|---:|
| _fast-ssim2 (static)_ | 0.1121 | — | 0.152 |
| V_18 ship | 0.1545 | [0.010, 0.490] | 0.151 |
| V_18 base seed=1 | 0.1694 | [0.010, 0.490] | 0.150 |
| V_20 IS (98) | 0.1146 | [0.010, 0.430] | 0.152 |
| V_20b manifold | 0.1181 | [0.010, 0.450] | 0.152 |
| D1 3-way concat | 0.1148 | [0.010, 0.430] | 0.152 |
| D3 lift>=0.10 | 0.1429 | [0.000, 0.460] | 0.151 |
| V_20_4 multi-bake α=0.4 | 0.1027 | [0.020, 0.440] | 0.153 |

**Empirical sample ceiling** (max CI upper bound across bakes): 0.490

## Least bad across all low-n bands

Aggregating across all `30 ≤ n < 100` bands, sorted by mean SROCC:

| Rank | Bake | Mean SROCC (low-n bands) | n_bands |
|---:|---|---:|---:|
| 1 | V_20b manifold | 0.2236 | 4 |
| 2 | V_20 IS (98) | 0.2212 | 4 |
| 3 | D3 lift>=0.10 | 0.2084 | 4 |
| 4 | D1 3-way concat | 0.2014 | 4 |
| 5 | V_18 base seed=1 | 0.1998 | 4 |
| 6 | V_18 ship | 0.1339 | 4 |
| 7 | V_20_4 multi-bake α=0.4 | 0.0554 | 4 |

**Least bad across low-n bands**: **V_20b manifold** (mean SROCC = 0.2236).

Caveat: 'mean of SROCC' is not a valid statistic for cross-band ranking — different bands have different n and noise floors. This is a heuristic for 'overall low-sample behavior'. For a rigorous ranking, test pairwise via MRR or Wilcoxon paired.
