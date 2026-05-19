# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/home/lilith/work/zen/zensim/zensim/weights/v_compression_2026-05-18.bin` (label: `v_compression_2026-05-18`)
- **B**: `/home/lilith/work/zen/zensim--v9-ship/zensim/weights/v_tuner_v9_2026-05-20.bin` (label: `v_tuner_v9_2026-05-20`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `500`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8580 | 0.5274 | 0.520 | 0.847 | 0.9126 | 0.6150 | 43.725 | 53.738 | +36.437 | A>>B |
| KonJND-1k (full) | 1008 | 0.8125 | 0.0574 | 0.498 | 0.996 | 0.8504 | 0.0601 | 21.171 | 16.065 | +17.643 | A>>B |
| TID2013 | 3000 | 0.8875 | 0.3430 | 0.436 | 0.951 | 0.9158 | 0.3902 | 33.496 | 37.502 | +27.913 | A>>B |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 8 cells
- **BDecisivelyBeatsA**: 0 cells
- **PromisingNotDecisive**: 4 cells
- **Tied**: 5 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `A`** (8 A wins vs 0 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_compression_2026-05-18 | 4292 | 0.8580 | 0.8539 | 0.6646 | 0.0480 | 0.9126 | 0.520 |
| B: v_tuner_v9_2026-05-20 | 4292 | 0.5274 | 0.5319 | 0.3713 | 0.0485 | 0.6150 | 0.847 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.5430 | 43.725 | 0.0000 | 53.738 | 0.0000 | +0.2976 | 5 | 0 | +36.437 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0788 | 0.1584 | 0.0476 | 0.0702 | 0.0305 | 0.987 |
| B4 | [0.40, 0.50) | 266 | 0.2470 | 0.2433 | 0.1674 | 0.0451 | 0.3100 | 0.970 |
| B5 | [0.50, 0.60) | 615 | 0.2502 | 0.2640 | 0.1691 | 0.0439 | 0.3037 | 0.965 |
| B6 | [0.60, 0.70) | 836 | 0.2520 | 0.2533 | 0.1689 | 0.0383 | 0.2922 | 0.967 |
| B7 | [0.70, 0.80) | 1092 | 0.3645 | 0.3744 | 0.2491 | 0.0421 | 0.4262 | 0.927 |
| B8 | [0.80, 0.90) | 1382 | 0.4861 | 0.4924 | 0.3303 | 0.0449 | 0.5693 | 0.870 |
| B9 | [0.90, 1.00] | 43 | 0.1796 | 0.4739 | 0.1096 | 0.0698 | 0.3397 | 0.881 |

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
| B3 | 57 | -0.929 | 0.3529 | -0.086 | 0.9318 | -0.1163 | 0 | 0 | -0.000 | tied |
| B4 | 266 | 2.928 | 0.0034 | 0.348 | 0.7278 | +0.2165 | 3 | 0 | +1.464 | promising |
| B5 | 615 | 2.632 | 0.0085 | 0.604 | 0.5458 | +0.1178 | 2 | 0 | +0.877 | promising |
| B6 | 836 | 2.632 | 0.0085 | 0.564 | 0.5730 | +0.0892 | 0 | 0 | +0.000 | tied |
| B7 | 1092 | 8.281 | 0.0000 | 2.537 | 0.0112 | +0.2192 | 5 | 0 | +6.901 | A>>B |
| B8 | 1382 | 16.983 | 0.0000 | 7.096 | 0.0000 | +0.3197 | 5 | 0 | +14.152 | A>>B |
| B9 | 43 | -2.738 | 0.0062 | 0.119 | 0.9054 | -0.1726 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_compression_2026-05-18 | 1008 | 0.8125 | 0.8672 | 0.6006 | 0.0417 | 0.8504 | 0.498 |
| B: v_tuner_v9_2026-05-20 | 1008 | 0.0574 | 0.0934 | 0.0393 | 0.0010 | 0.0601 | 0.996 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.0615 | 21.171 | 0.0000 | 16.065 | 0.0000 | +0.7904 | 5 | 0 | +17.643 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_compression_2026-05-18 | 3000 | 0.8875 | 0.9000 | 0.7117 | 0.0473 | 0.9158 | 0.436 |
| B: v_tuner_v9_2026-05-20 | 3000 | 0.3430 | 0.3078 | 0.2365 | 0.0277 | 0.3902 | 0.951 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.2554 | 33.496 | 0.0000 | 37.502 | 0.0000 | +0.5256 | 5 | 1 | +27.913 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.2276 | 0.4001 | 0.1998 | 0.0345 | 0.2199 | 0.916 |
| B1 | [0.10, 0.20) | 34 | 0.4740 | 0.5906 | 0.3375 | 0.0294 | 0.6222 | 0.807 |
| B2 | [0.20, 0.30) | 185 | 0.3655 | 0.3907 | 0.2487 | 0.0324 | 0.4532 | 0.921 |
| B3 | [0.30, 0.40) | 493 | 0.4785 | 0.4883 | 0.3309 | 0.0406 | 0.5820 | 0.873 |
| B4 | [0.40, 0.50) | 677 | 0.5437 | 0.5496 | 0.3803 | 0.0443 | 0.6345 | 0.835 |
| B5 | [0.50, 0.60) | 705 | 0.4628 | 0.4876 | 0.3191 | 0.0539 | 0.5492 | 0.873 |
| B6 | [0.60, 0.70) | 809 | 0.1837 | 0.2224 | 0.1248 | 0.0420 | 0.2260 | 0.975 |
| B7 | [0.70, 0.80) | 67 | 0.3464 | 0.4803 | 0.2346 | 0.0597 | 0.3899 | 0.877 |
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
| B0 ⚠ | 29 | 2.446 | 0.0145 | 0.676 | 0.4987 | +0.1956 | 0 | 0 | +0.000 | noisy |
| B1 | 34 | 0.200 | 0.8413 | 0.251 | 0.8017 | +0.1160 | 0 | 0 | +0.000 | tied |
| B2 | 185 | 4.073 | 0.0000 | 1.280 | 0.2007 | +0.2465 | 3 | 0 | +2.037 | promising |
| B3 | 493 | 11.149 | 0.0000 | 3.099 | 0.0019 | +0.5103 | 5 | 0 | +9.291 | A>>B |
| B4 | 677 | 10.016 | 0.0000 | 3.732 | 0.0002 | +0.5040 | 5 | 1 | +8.346 | A>>B |
| B5 | 705 | 9.328 | 0.0000 | 3.203 | 0.0014 | +0.4450 | 5 | 1 | +7.773 | A>>B |
| B6 | 809 | 2.516 | 0.0119 | 0.605 | 0.5449 | +0.1035 | 2 | 1 | +0.839 | promising |
| B7 | 67 | 0.779 | 0.4362 | 0.890 | 0.3736 | +0.0198 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

---
Wall time: 27.34s (8300 pair rows scored × 2 bakes across 3 corpora; 500 bootstrap resamples × bands).
