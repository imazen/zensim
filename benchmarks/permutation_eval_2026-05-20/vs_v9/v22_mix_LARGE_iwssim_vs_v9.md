# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/home/lilith/work/zen/zensim/zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` (label: `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18`)
- **B**: `/home/lilith/work/zen/zensim--v9-ship/zensim/weights/v_tuner_v9_2026-05-20.bin` (label: `v_tuner_v9_2026-05-20`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `500`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8324 | 0.5274 | 0.559 | 0.847 | 0.9006 | 0.6150 | 38.716 | 44.860 | +32.263 | A>>B |
| KonJND-1k (full) | 1008 | 0.8927 | 0.0574 | 0.376 | 0.996 | 0.9178 | 0.0601 | 20.653 | 19.001 | +17.211 | A>>B |
| TID2013 | 3000 | 0.9729 | 0.3430 | 0.236 | 0.951 | 0.9832 | 0.3902 | 32.495 | 53.939 | +27.079 | A>>B |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 9 cells
- **BDecisivelyBeatsA**: 0 cells
- **PromisingNotDecisive**: 3 cells
- **Tied**: 5 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `A`** (9 A wins vs 0 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 4292 | 0.8324 | 0.8289 | 0.6340 | 0.0443 | 0.9006 | 0.559 |
| B: v_tuner_v9_2026-05-20 | 4292 | 0.5274 | 0.5319 | 0.3713 | 0.0485 | 0.6150 | 0.847 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.5147 | 38.716 | 0.0000 | 44.860 | 0.0000 | +0.2856 | 5 | 0 | +32.263 | A>>B |

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
| B3 | 57 | -1.637 | 0.1016 | -0.065 | 0.9484 | -0.0765 | 0 | 0 | -0.000 | tied |
| B4 | 266 | 3.449 | 0.0006 | 0.410 | 0.6816 | +0.2233 | 3 | 0 | +1.725 | promising |
| B5 | 615 | 2.728 | 0.0064 | 0.606 | 0.5443 | +0.1190 | 1 | 0 | +0.455 | promising |
| B6 | 836 | 1.641 | 0.1009 | 0.381 | 0.7036 | +0.0551 | 0 | 0 | +0.000 | tied |
| B7 | 1092 | 5.827 | 0.0000 | 1.579 | 0.1144 | +0.1696 | 5 | 1 | +4.856 | promising |
| B8 | 1382 | 16.106 | 0.0000 | 6.856 | 0.0000 | +0.3350 | 5 | 0 | +13.421 | A>>B |
| B9 | 43 | -2.341 | 0.0193 | -0.786 | 0.4319 | -0.2398 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 1008 | 0.8927 | 0.9265 | 0.7070 | 0.0446 | 0.9178 | 0.376 |
| B: v_tuner_v9_2026-05-20 | 1008 | 0.0574 | 0.0934 | 0.0393 | 0.0010 | 0.0601 | 0.996 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.0486 | 20.653 | 0.0000 | 19.001 | 0.0000 | +0.8577 | 5 | 0 | +17.211 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 3000 | 0.9729 | 0.9717 | 0.8571 | 0.0357 | 0.9832 | 0.236 |
| B: v_tuner_v9_2026-05-20 | 3000 | 0.3430 | 0.3078 | 0.2365 | 0.0277 | 0.3902 | 0.951 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.3461 | 32.495 | 0.0000 | 53.939 | 0.0000 | +0.5930 | 5 | 1 | +27.079 | A>>B |

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
| B0 ⚠ | 29 | 0.208 | 0.8350 | 0.190 | 0.8489 | +0.0534 | 0 | 0 | +0.000 | noisy |
| B1 | 34 | 0.725 | 0.4685 | 0.299 | 0.7653 | +0.1955 | 0 | 0 | +0.000 | tied |
| B2 | 185 | 9.862 | 0.0000 | 4.870 | 0.0000 | +0.5113 | 5 | 0 | +8.219 | A>>B |
| B3 | 493 | 14.644 | 0.0000 | 7.464 | 0.0000 | +0.7456 | 5 | 0 | +12.204 | A>>B |
| B4 | 677 | 17.268 | 0.0000 | 10.227 | 0.0000 | +0.7064 | 5 | 1 | +14.390 | A>>B |
| B5 | 705 | 17.017 | 0.0000 | 8.737 | 0.0000 | +0.6865 | 5 | 1 | +14.181 | A>>B |
| B6 | 809 | 15.491 | 0.0000 | 6.191 | 0.0000 | +0.5599 | 5 | 1 | +12.909 | A>>B |
| B7 | 67 | -0.128 | 0.8983 | 0.877 | 0.3806 | -0.0900 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

---
Wall time: 22.68s (8300 pair rows scored × 2 bakes across 3 corpora; 500 bootstrap resamples × bands).
