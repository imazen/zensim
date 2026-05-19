# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/home/lilith/work/zen/zensim/zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` (label: `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18`)
- **B**: `/home/lilith/work/zen/zensim--v6-reship/zensim/weights/v_tuner_v6_2026-05-19.bin` (label: `v_tuner_v6_2026-05-19`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `500`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8324 | 0.8216 | 0.559 | 0.680 | 0.9006 | 0.8821 | 3.627 | 59.520 | +2.418 | A>>B |
| KonJND-1k (full) | 1008 | 0.8927 | 0.3734 | 0.376 | 0.960 | 0.9178 | 0.5297 | 17.869 | 22.964 | +14.891 | A>>B |
| TID2013 | 3000 | 0.9729 | 0.4447 | 0.236 | 0.809 | 0.9832 | 0.5451 | 35.568 | 64.279 | +29.640 | A>>B |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 9 cells
- **BDecisivelyBeatsA**: 0 cells
- **PromisingNotDecisive**: 2 cells
- **Tied**: 6 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `A`** (9 A wins vs 0 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 4292 | 0.8324 | 0.8289 | 0.6340 | 0.0443 | 0.9006 | 0.559 |
| B: v_tuner_v6_2026-05-19 | 4292 | 0.8216 | 0.7332 | 0.6249 | 0.0531 | 0.8821 | 0.680 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.8059 | 3.627 | 0.0003 | 59.520 | 0.0000 | +0.0186 | 4 | 0 | +2.418 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0487 | 0.1900 | 0.0276 | 0.0351 | 0.0703 | 0.982 |
| B4 | [0.40, 0.50) | 266 | 0.2510 | 0.2452 | 0.1658 | 0.0489 | 0.3169 | 0.969 |
| B5 | [0.50, 0.60) | 615 | 0.2466 | 0.2584 | 0.1655 | 0.0423 | 0.3049 | 0.966 |
| B6 | [0.60, 0.70) | 836 | 0.2220 | 0.2317 | 0.1484 | 0.0347 | 0.2580 | 0.973 |
| B7 | [0.70, 0.80) | 1092 | 0.3192 | 0.3244 | 0.2173 | 0.0513 | 0.3766 | 0.946 |
| B8 | [0.80, 0.90) | 1382 | 0.4958 | 0.5026 | 0.3354 | 0.0347 | 0.5846 | 0.865 |
| B9 | [0.90, 1.00] | 43 | 0.1056 | 0.1831 | 0.0698 | 0.0233 | 0.2725 | 0.983 |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0442 | 0.1398 | 0.0251 | 0.0351 | 0.1018 | 0.990 |
| B4 | [0.40, 0.50) | 266 | 0.2052 | 0.2253 | 0.1387 | 0.0414 | 0.2696 | 0.974 |
| B5 | [0.50, 0.60) | 615 | 0.2724 | 0.2644 | 0.1842 | 0.0504 | 0.3347 | 0.964 |
| B6 | [0.60, 0.70) | 836 | 0.3013 | 0.3177 | 0.2016 | 0.0395 | 0.3590 | 0.948 |
| B7 | [0.70, 0.80) | 1092 | 0.3289 | 0.3257 | 0.2227 | 0.0394 | 0.4053 | 0.945 |
| B8 | [0.80, 0.90) | 1382 | 0.4089 | 0.4129 | 0.2732 | 0.0246 | 0.4957 | 0.911 |
| B9 | [0.90, 1.00] | 43 | 0.0193 | 0.2439 | 0.0122 | 0.0465 | 0.0841 | 0.970 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B2 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B3 | 57 | 0.026 | 0.9795 | 0.048 | 0.9618 | -0.0315 | 0 | 0 | -0.000 | tied |
| B4 | 266 | 1.222 | 0.2218 | 0.132 | 0.8950 | +0.0474 | 0 | 0 | +0.000 | tied |
| B5 | 615 | -0.889 | 0.3740 | -0.058 | 0.9539 | -0.0297 | 0 | 0 | -0.000 | tied |
| B6 | 836 | -3.212 | 0.0013 | -1.025 | 0.3055 | -0.1009 | 0 | 5 | -0.000 | promising |
| B7 | 1092 | -0.657 | 0.5111 | -0.031 | 0.9751 | -0.0287 | 0 | 1 | -0.000 | promising |
| B8 | 1382 | 14.844 | 0.0000 | 8.686 | 0.0000 | +0.0889 | 5 | 1 | +12.370 | A>>B |
| B9 | 43 | 1.920 | 0.0549 | -0.301 | 0.7633 | +0.1884 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 1008 | 0.8927 | 0.9265 | 0.7070 | 0.0446 | 0.9178 | 0.376 |
| B: v_tuner_v6_2026-05-19 | 1008 | 0.3734 | 0.2813 | 0.2584 | 0.0347 | 0.5297 | 0.960 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.2267 | 17.869 | 0.0000 | 22.964 | 0.0000 | +0.3881 | 5 | 0 | +14.891 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 3000 | 0.9729 | 0.9717 | 0.8571 | 0.0357 | 0.9832 | 0.236 |
| B: v_tuner_v6_2026-05-19 | 3000 | 0.4447 | 0.5878 | 0.3055 | 0.0430 | 0.5451 | 0.809 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.4688 | 35.568 | 0.0000 | 64.279 | 0.0000 | +0.4381 | 5 | 0 | +29.640 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0621 | 0.3495 | 0.0666 | 0.0345 | 0.0778 | 0.937 |
| B1 | [0.10, 0.20) | 34 | 0.5319 | 0.5997 | 0.3411 | 0.0294 | 0.7017 | 0.800 |
| B2 | [0.20, 0.30) | 185 | 0.6479 | 0.6523 | 0.4607 | 0.0378 | 0.7180 | 0.758 |
| B3 | [0.30, 0.40) | 493 | 0.7352 | 0.7356 | 0.5396 | 0.0446 | 0.8173 | 0.677 |
| B4 | [0.40, 0.50) | 677 | 0.7625 | 0.7625 | 0.5598 | 0.0502 | 0.8370 | 0.647 |
| B5 | [0.50, 0.60) | 705 | 0.7077 | 0.7068 | 0.5096 | 0.0496 | 0.7907 | 0.707 |
| B6 | [0.60, 0.70) | 809 | 0.5860 | 0.5854 | 0.4099 | 0.0346 | 0.6825 | 0.811 |
| B7 | [0.70, 0.80) | 67 | 0.2842 | 0.4937 | 0.1928 | 0.0746 | 0.2801 | 0.870 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0825 | 0.4124 | 0.0321 | 0.0000 | 0.0071 | 0.911 |
| B1 | [0.10, 0.20) | 34 | 0.6252 | 0.7559 | 0.4125 | 0.0000 | 0.7246 | 0.655 |
| B2 | [0.20, 0.30) | 185 | 0.2158 | 0.2735 | 0.1467 | 0.0324 | 0.3043 | 0.962 |
| B3 | [0.30, 0.40) | 493 | 0.2468 | 0.2716 | 0.1696 | 0.0446 | 0.3020 | 0.962 |
| B4 | [0.40, 0.50) | 677 | 0.2092 | 0.2786 | 0.1389 | 0.0473 | 0.2635 | 0.960 |
| B5 | [0.50, 0.60) | 705 | 0.1067 | 0.1662 | 0.0705 | 0.0383 | 0.1421 | 0.986 |
| B6 | [0.60, 0.70) | 809 | 0.0190 | 0.0851 | 0.0124 | 0.0334 | 0.0263 | 0.996 |
| B7 | [0.70, 0.80) | 67 | 0.4070 | 0.4746 | 0.2745 | 0.0448 | 0.4777 | 0.880 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 ⚠ | 29 | -0.107 | 0.9149 | -0.145 | 0.8850 | +0.0707 | 0 | 0 | +0.000 | noisy |
| B1 | 34 | -1.148 | 0.2508 | -2.235 | 0.0254 | -0.0229 | 0 | 0 | -0.000 | tied |
| B2 | 185 | 10.939 | 0.0000 | 5.602 | 0.0000 | +0.4138 | 5 | 0 | +9.116 | A>>B |
| B3 | 493 | 15.365 | 0.0000 | 9.916 | 0.0000 | +0.5153 | 5 | 0 | +12.804 | A>>B |
| B4 | 677 | 17.644 | 0.0000 | 11.304 | 0.0000 | +0.5735 | 5 | 0 | +14.703 | A>>B |
| B5 | 705 | 16.614 | 0.0000 | 8.499 | 0.0000 | +0.6487 | 5 | 0 | +13.845 | A>>B |
| B6 | 809 | 18.131 | 0.0000 | 6.352 | 0.0000 | +0.6562 | 5 | 0 | +15.109 | A>>B |
| B7 | 67 | -1.125 | 0.2607 | 0.110 | 0.9122 | -0.1976 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

---
Wall time: 32.67s (8300 pair rows scored × 2 bakes across 3 corpora; 500 bootstrap resamples × bands).
