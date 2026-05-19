# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/home/lilith/work/zen/zensim--v6-reship/zensim/weights/v_tuner_v6_2026-05-19.bin` (label: `v_tuner_v6_2026-05-19`)
- **B**: `/home/lilith/work/zen/zensim--v9-ship/zensim/weights/v_tuner_v9_2026-05-20.bin` (label: `v_tuner_v9_2026-05-20`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `500`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8216 | 0.5274 | 0.680 | 0.847 | 0.8821 | 0.6150 | 32.177 | 22.593 | +26.814 | A>>B |
| KonJND-1k (full) | 1008 | 0.3734 | 0.0574 | 0.960 | 0.996 | 0.5297 | 0.0601 | 9.159 | 1.067 | +7.633 | promising |
| TID2013 | 3000 | 0.4447 | 0.3430 | 0.809 | 0.951 | 0.5451 | 0.3902 | 4.461 | 6.563 | +3.718 | A>>B |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 3 cells
- **BDecisivelyBeatsA**: 0 cells
- **PromisingNotDecisive**: 8 cells
- **Tied**: 6 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `A`** (3 A wins vs 0 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_tuner_v6_2026-05-19 | 4292 | 0.8216 | 0.7332 | 0.6249 | 0.0531 | 0.8821 | 0.680 |
| B: v_tuner_v9_2026-05-20 | 4292 | 0.5274 | 0.5319 | 0.3713 | 0.0485 | 0.6150 | 0.847 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.4330 | 32.177 | 0.0000 | 22.593 | 0.0000 | +0.2671 | 5 | 0 | +26.814 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

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

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.2084 | 0.2195 | 0.1479 | 0.0526 | 0.1469 | 0.976 |
| B4 | [0.40, 0.50) | 266 | 0.0646 | 0.1316 | 0.0415 | 0.0414 | 0.0936 | 0.991 |
| B5 | [0.50, 0.60) | 615 | 0.1488 | 0.1579 | 0.1004 | 0.0439 | 0.1859 | 0.987 |
| B6 | [0.60, 0.70) | 836 | 0.1693 | 0.1736 | 0.1119 | 0.0419 | 0.2030 | 0.985 |
| B7 | [0.70, 0.80) | 1092 | 0.1703 | 0.1696 | 0.1148 | 0.0357 | 0.2070 | 0.986 |
| B8 | [0.80, 0.90) | 1382 | 0.2224 | 0.2152 | 0.1492 | 0.0369 | 0.2497 | 0.977 |
| B9 | [0.90, 1.00] | 43 | 0.3953 | 0.4582 | 0.2580 | 0.0698 | 0.5123 | 0.889 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B2 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B3 | 57 | -1.122 | 0.2618 | -0.101 | 0.9196 | -0.0451 | 0 | 0 | -0.000 | tied |
| B4 | 266 | 2.674 | 0.0075 | 0.328 | 0.7430 | +0.1760 | 0 | 0 | +0.000 | tied |
| B5 | 615 | 3.429 | 0.0006 | 0.649 | 0.5162 | +0.1488 | 3 | 0 | +1.714 | promising |
| B6 | 836 | 4.152 | 0.0000 | 1.173 | 0.2408 | +0.1560 | 5 | 0 | +3.460 | promising |
| B7 | 1092 | 5.347 | 0.0000 | 1.376 | 0.1688 | +0.1983 | 5 | 0 | +4.456 | promising |
| B8 | 1382 | 7.729 | 0.0000 | 2.809 | 0.0050 | +0.2460 | 6 | 0 | +7.729 | A>>B |
| B9 | 43 | -1.958 | 0.0502 | -0.446 | 0.6556 | -0.4282 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_tuner_v6_2026-05-19 | 1008 | 0.3734 | 0.2813 | 0.2584 | 0.0347 | 0.5297 | 0.960 |
| B: v_tuner_v9_2026-05-20 | 1008 | 0.0574 | 0.0934 | 0.0393 | 0.0010 | 0.0601 | 0.996 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | -0.0737 | 9.159 | 0.0000 | 1.067 | 0.2858 | +0.4696 | 5 | 0 | +7.633 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_tuner_v6_2026-05-19 | 3000 | 0.4447 | 0.5878 | 0.3055 | 0.0430 | 0.5451 | 0.809 |
| B: v_tuner_v9_2026-05-20 | 3000 | 0.3430 | 0.3078 | 0.2365 | 0.0277 | 0.3902 | 0.951 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | -0.2442 | 4.461 | 0.0000 | 6.563 | 0.0000 | +0.1549 | 5 | 1 | +3.718 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

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

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0239 | 0.2398 | 0.0419 | 0.0345 | 0.0244 | 0.971 |
| B1 | [0.10, 0.20) | 34 | 0.4554 | 0.5633 | 0.3125 | 0.0882 | 0.5062 | 0.826 |
| B2 | [0.20, 0.30) | 185 | 0.1558 | 0.1731 | 0.1031 | 0.0216 | 0.2067 | 0.985 |
| B3 | [0.30, 0.40) | 493 | 0.0467 | 0.1604 | 0.0297 | 0.0284 | 0.0717 | 0.987 |
| B4 | [0.40, 0.50) | 677 | 0.1149 | 0.1569 | 0.0771 | 0.0162 | 0.1305 | 0.988 |
| B5 | [0.50, 0.60) | 705 | 0.0879 | 0.0571 | 0.0598 | 0.0199 | 0.1042 | 0.998 |
| B6 | [0.60, 0.70) | 809 | 0.0999 | 0.0999 | 0.0682 | 0.0124 | 0.1226 | 0.995 |
| B7 | [0.70, 0.80) | 67 | 0.2940 | 0.3602 | 0.1992 | 0.0448 | 0.3701 | 0.933 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 ⚠ | 29 | 0.334 | 0.7385 | 0.351 | 0.7257 | -0.0173 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | 2.752 | 0.0059 | 3.374 | 0.0007 | +0.2184 | 0 | 0 | +0.000 | tied |
| B2 | 185 | 1.923 | 0.0545 | 0.750 | 0.4535 | +0.0976 | 0 | 0 | +0.000 | tied |
| B3 | 493 | 6.961 | 0.0000 | 0.875 | 0.3818 | +0.2303 | 5 | 0 | +5.801 | promising |
| B4 | 677 | 1.732 | 0.0832 | 0.507 | 0.6124 | +0.1330 | 2 | 1 | +0.577 | promising |
| B5 | 705 | 0.308 | 0.7579 | 0.202 | 0.8403 | +0.0379 | 0 | 1 | +0.000 | promising |
| B6 | 809 | -1.504 | 0.1327 | -0.026 | 0.9795 | -0.0963 | 0 | 1 | -0.000 | promising |
| B7 | 67 | 2.241 | 0.0250 | 1.123 | 0.2616 | +0.1076 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

---
Wall time: 37.16s (8300 pair rows scored × 2 bakes across 3 corpora; 500 bootstrap resamples × bands).
