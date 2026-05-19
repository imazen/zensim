# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/home/lilith/work/zen/zensim/zensim/weights/v0_22_iw_v2_calibrated_2026-05-16.bin` (label: `v0_22_iw_v2_calibrated_2026-05-16`)
- **B**: `/home/lilith/work/zen/zensim--v9-ship/zensim/weights/v_tuner_v9_2026-05-20.bin` (label: `v_tuner_v9_2026-05-20`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `500`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8163 | 0.5274 | 0.569 | 0.847 | 0.8754 | 0.6150 | 31.125 | 36.549 | +25.938 | A>>B |
| KonJND-1k (full) | 1008 | 0.0303 | 0.0574 | 0.994 | 0.996 | 0.0883 | 0.0601 | -0.667 | 0.031 | -0.000 | tied |
| TID2013 | 3000 | 0.9617 | 0.3430 | 0.272 | 0.951 | 0.9766 | 0.3902 | 35.135 | 53.114 | +29.279 | A>>B |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 8 cells
- **BDecisivelyBeatsA**: 0 cells
- **PromisingNotDecisive**: 3 cells
- **Tied**: 6 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `A`** (8 A wins vs 0 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v0_22_iw_v2_calibrated_2026-05-16 | 4292 | 0.8163 | 0.8226 | 0.6317 | 0.0473 | 0.8754 | 0.569 |
| B: v_tuner_v9_2026-05-20 | 4292 | 0.5274 | 0.5319 | 0.3713 | 0.0485 | 0.6150 | 0.847 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.4227 | 31.125 | 0.0000 | 36.549 | 0.0000 | +0.2604 | 5 | 0 | +25.938 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1180 | 0.1640 | 0.0677 | 0.0526 | 0.0399 | 0.986 |
| B4 | [0.40, 0.50) | 266 | 0.2613 | 0.2989 | 0.1743 | 0.0489 | 0.3169 | 0.954 |
| B5 | [0.50, 0.60) | 615 | 0.2566 | 0.2738 | 0.1741 | 0.0520 | 0.3058 | 0.962 |
| B6 | [0.60, 0.70) | 836 | 0.2351 | 0.2549 | 0.1562 | 0.0431 | 0.2807 | 0.967 |
| B7 | [0.70, 0.80) | 1092 | 0.3415 | 0.3507 | 0.2364 | 0.0449 | 0.3824 | 0.936 |
| B8 | [0.80, 0.90) | 1382 | 0.4522 | 0.4541 | 0.3091 | 0.0449 | 0.5303 | 0.891 |
| B9 | [0.90, 1.00] | 43 | 0.0590 | 0.4033 | 0.0188 | 0.0465 | 0.1713 | 0.915 |

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
| B3 | 57 | -0.764 | 0.4450 | -0.093 | 0.9258 | -0.1069 | 0 | 0 | -0.000 | tied |
| B4 | 266 | 2.941 | 0.0033 | 0.562 | 0.5739 | +0.2234 | 5 | 0 | +2.451 | promising |
| B5 | 615 | 2.461 | 0.0139 | 0.595 | 0.5520 | +0.1199 | 4 | 0 | +1.641 | promising |
| B6 | 836 | 1.656 | 0.0977 | 0.456 | 0.6481 | +0.0777 | 0 | 0 | +0.000 | tied |
| B7 | 1092 | 5.569 | 0.0000 | 1.627 | 0.1037 | +0.1754 | 5 | 0 | +4.641 | promising |
| B8 | 1382 | 11.703 | 0.0000 | 4.510 | 0.0000 | +0.2806 | 5 | 0 | +9.753 | A>>B |
| B9 | 43 | -5.282 | 0.0000 | -0.460 | 0.6456 | -0.3410 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v0_22_iw_v2_calibrated_2026-05-16 | 1008 | 0.0303 | 0.1059 | 0.0229 | 0.0387 | 0.0883 | 0.994 |
| B: v_tuner_v9_2026-05-20 | 1008 | 0.0574 | 0.0934 | 0.0393 | 0.0010 | 0.0601 | 0.996 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | -0.2892 | -0.667 | 0.5048 | 0.031 | 0.9754 | +0.0282 | 0 | 0 | -0.000 | tied |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v0_22_iw_v2_calibrated_2026-05-16 | 3000 | 0.9617 | 0.9623 | 0.8280 | 0.0487 | 0.9766 | 0.272 |
| B: v_tuner_v9_2026-05-20 | 3000 | 0.3430 | 0.3078 | 0.2365 | 0.0277 | 0.3902 | 0.951 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.3538 | 35.135 | 0.0000 | 53.114 | 0.0000 | +0.5864 | 5 | 1 | +29.279 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.2035 | 0.3588 | 0.1603 | 0.0690 | 0.2138 | 0.933 |
| B1 | [0.10, 0.20) | 34 | 0.3807 | 0.3956 | 0.2375 | 0.0000 | 0.4635 | 0.918 |
| B2 | [0.20, 0.30) | 185 | 0.5630 | 0.5804 | 0.3902 | 0.0595 | 0.6570 | 0.814 |
| B3 | [0.30, 0.40) | 493 | 0.6408 | 0.6457 | 0.4623 | 0.0507 | 0.7113 | 0.764 |
| B4 | [0.40, 0.50) | 677 | 0.6732 | 0.6741 | 0.4859 | 0.0502 | 0.7479 | 0.739 |
| B5 | [0.50, 0.60) | 705 | 0.6229 | 0.6225 | 0.4415 | 0.0369 | 0.7122 | 0.783 |
| B6 | [0.60, 0.70) | 809 | 0.5505 | 0.5501 | 0.3813 | 0.0396 | 0.6543 | 0.835 |
| B7 | [0.70, 0.80) | 67 | 0.0348 | 0.1235 | 0.0195 | 0.0299 | 0.0172 | 0.992 |
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
| B0 ⚠ | 29 | 1.010 | 0.3127 | 0.218 | 0.8277 | +0.1895 | 0 | 0 | +0.000 | noisy |
| B1 | 34 | -0.461 | 0.6448 | -0.619 | 0.5359 | -0.0426 | 0 | 0 | -0.000 | tied |
| B2 | 185 | 7.281 | 0.0000 | 3.201 | 0.0014 | +0.4503 | 5 | 1 | +6.068 | A>>B |
| B3 | 493 | 12.860 | 0.0000 | 5.280 | 0.0000 | +0.6396 | 5 | 1 | +10.717 | A>>B |
| B4 | 677 | 16.243 | 0.0000 | 7.862 | 0.0000 | +0.6173 | 5 | 1 | +13.536 | A>>B |
| B5 | 705 | 16.550 | 0.0000 | 7.097 | 0.0000 | +0.6081 | 5 | 1 | +13.792 | A>>B |
| B6 | 809 | 14.713 | 0.0000 | 5.462 | 0.0000 | +0.5318 | 5 | 1 | +12.260 | A>>B |
| B7 | 67 | -1.806 | 0.0710 | -0.422 | 0.6729 | -0.3529 | 0 | 0 | -0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

---
Wall time: 25.19s (8300 pair rows scored × 2 bakes across 3 corpora; 500 bootstrap resamples × bands).
