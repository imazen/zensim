# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/mnt/v/output/zensim/synthetic-v2/runs/v06_film_20260505T212932.bin` (label: `v06_film_20260505T212932`)
- **B**: `zensim/weights/v_compression_persample_2026-05-18.bin` (label: `v_compression_persample_2026-05-18`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8755 | 0.8641 | 0.484 | 0.508 | 0.9267 | 0.9157 | 6.081 | 24.912 | +5.068 | A>>B |
| KADIK10k | 10125 | 0.8527 | 0.9316 | 0.526 | 0.362 | 0.9089 | 0.9602 | -114.819 | -458.490 | -0.000 | B>>A |
| TID2013 | 3000 | 0.8451 | 0.8893 | 0.502 | 0.432 | 0.8914 | 0.9173 | -22.747 | -71.752 | -0.000 | B>>A |
| KonJND-1k (full) | 1008 | 0.4971 | 0.8080 | 0.900 | 0.502 | 0.6386 | 0.8505 | -12.665 | -18.320 | -0.000 | B>>A |
| AIC-3 CTC | 600 | 0.7862 | 0.8183 | 0.607 | 0.565 | 0.8646 | 0.8856 | -13.294 | -28.143 | -0.000 | B>>A |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 2 cells
- **BDecisivelyBeatsA**: 11 cells
- **PromisingNotDecisive**: 5 cells
- **Tied**: 11 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (2 A wins vs 11 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v06_film_20260505T212932 | 4292 | 0.8755 | 0.8750 | 0.6867 | 0.0480 | 0.9267 | 0.484 |
| B: v_compression_persample_2026-05-18 | 4292 | 0.8641 | 0.8614 | 0.6742 | 0.0489 | 0.9157 | 0.508 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.8770 | 6.081 | 0.0000 | 24.912 | 0.0000 | +0.0110 | 5 | 0 | +5.068 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1790 | 0.3039 | 0.1178 | 0.0351 | 0.2423 | 0.953 |
| B4 | [0.40, 0.50) | 266 | 0.2957 | 0.3030 | 0.2007 | 0.0639 | 0.3586 | 0.953 |
| B5 | [0.50, 0.60) | 615 | 0.3719 | 0.3705 | 0.2536 | 0.0423 | 0.4461 | 0.929 |
| B6 | [0.60, 0.70) | 836 | 0.4005 | 0.4045 | 0.2704 | 0.0443 | 0.4759 | 0.915 |
| B7 | [0.70, 0.80) | 1092 | 0.3568 | 0.3685 | 0.2451 | 0.0495 | 0.4286 | 0.930 |
| B8 | [0.80, 0.90) | 1382 | 0.4669 | 0.4684 | 0.3140 | 0.0398 | 0.5517 | 0.883 |
| B9 | [0.90, 1.00] | 43 | 0.0246 | 0.1805 | 0.0122 | 0.0698 | 0.1216 | 0.984 |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0825 | 0.2911 | 0.0564 | 0.0526 | 0.0373 | 0.957 |
| B4 | [0.40, 0.50) | 266 | 0.2699 | 0.2646 | 0.1801 | 0.0489 | 0.3326 | 0.964 |
| B5 | [0.50, 0.60) | 615 | 0.2656 | 0.2734 | 0.1803 | 0.0504 | 0.3170 | 0.962 |
| B6 | [0.60, 0.70) | 836 | 0.2729 | 0.2751 | 0.1817 | 0.0455 | 0.3195 | 0.961 |
| B7 | [0.70, 0.80) | 1092 | 0.3792 | 0.3864 | 0.2602 | 0.0467 | 0.4420 | 0.922 |
| B8 | [0.80, 0.90) | 1382 | 0.4971 | 0.5011 | 0.3389 | 0.0441 | 0.5801 | 0.865 |
| B9 | [0.90, 1.00] | 43 | 0.1401 | 0.4358 | 0.1030 | 0.0465 | 0.2949 | 0.900 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B2 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B3 | 57 | 0.472 | 0.6369 | 0.021 | 0.9836 | +0.2050 | 0 | 0 | +0.000 | tied |
| B4 | 266 | 0.938 | 0.3481 | 0.430 | 0.6671 | +0.0260 | 0 | 0 | +0.000 | tied |
| B5 | 615 | 5.553 | 0.0000 | 1.799 | 0.0719 | +0.1292 | 5 | 0 | +4.627 | promising |
| B6 | 836 | 7.530 | 0.0000 | 2.888 | 0.0039 | +0.1564 | 5 | 0 | +6.275 | A>>B |
| B7 | 1092 | -1.586 | 0.1127 | -0.553 | 0.5802 | -0.0134 | 0 | 0 | -0.000 | tied |
| B8 | 1382 | -4.243 | 0.0000 | -2.881 | 0.0040 | -0.0283 | 0 | 1 | -0.000 | promising |
| B9 | 43 | -0.942 | 0.3462 | -0.693 | 0.4882 | -0.1733 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v06_film_20260505T212932 | 10125 | 0.8527 | 0.8503 | 0.6640 | 0.0493 | 0.9089 | 0.526 |
| B: v_compression_persample_2026-05-18 | 10125 | 0.9316 | 0.9321 | 0.7684 | 0.0462 | 0.9602 | 0.362 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9327 | -114.819 | 0.0000 | -458.490 | 0.0000 | -0.0513 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### KADIK10k 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2579 | 0.2639 | 0.1804 | 0.0496 | 0.3267 | 0.965 |
| B1 | [0.10, 0.20) | 910 | 0.2190 | 0.2280 | 0.1525 | 0.0484 | 0.2849 | 0.974 |
| B2 | [0.20, 0.30) | 1111 | 0.0672 | 0.1357 | 0.0471 | 0.0387 | 0.0859 | 0.991 |
| B3 | [0.30, 0.40) | 1291 | 0.1936 | 0.1901 | 0.1343 | 0.0473 | 0.2366 | 0.982 |
| B4 | [0.40, 0.50) | 1013 | 0.1684 | 0.1781 | 0.1180 | 0.0484 | 0.1984 | 0.984 |
| B5 | [0.50, 0.60) | 919 | 0.1104 | 0.1120 | 0.0769 | 0.0413 | 0.1451 | 0.994 |
| B6 | [0.60, 0.70) | 936 | 0.1015 | 0.1066 | 0.0701 | 0.0449 | 0.1270 | 0.994 |
| B7 | [0.70, 0.80) | 985 | 0.1997 | 0.1987 | 0.1404 | 0.0477 | 0.2280 | 0.980 |
| B8 | [0.80, 0.90) | 1699 | 0.3729 | 0.3770 | 0.2603 | 0.0400 | 0.4426 | 0.926 |
| B9 | [0.90, 1.00] | 486 | 0.1615 | 0.1622 | 0.1154 | 0.0350 | 0.1750 | 0.987 |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2461 | 0.2628 | 0.1728 | 0.0539 | 0.3139 | 0.965 |
| B1 | [0.10, 0.20) | 910 | 0.2599 | 0.2621 | 0.1803 | 0.0429 | 0.3219 | 0.965 |
| B2 | [0.20, 0.30) | 1111 | 0.1918 | 0.2016 | 0.1324 | 0.0342 | 0.2230 | 0.979 |
| B3 | [0.30, 0.40) | 1291 | 0.2201 | 0.2308 | 0.1539 | 0.0442 | 0.2601 | 0.973 |
| B4 | [0.40, 0.50) | 1013 | 0.2516 | 0.2523 | 0.1779 | 0.0395 | 0.3000 | 0.968 |
| B5 | [0.50, 0.60) | 919 | 0.2035 | 0.2324 | 0.1421 | 0.0424 | 0.2513 | 0.973 |
| B6 | [0.60, 0.70) | 936 | 0.2290 | 0.2447 | 0.1583 | 0.0459 | 0.2803 | 0.970 |
| B7 | [0.70, 0.80) | 985 | 0.2718 | 0.2751 | 0.1900 | 0.0406 | 0.3177 | 0.961 |
| B8 | [0.80, 0.90) | 1699 | 0.4514 | 0.4510 | 0.3169 | 0.0400 | 0.5343 | 0.893 |
| B9 | [0.90, 1.00] | 486 | 0.1708 | 0.1816 | 0.1168 | 0.0391 | 0.2037 | 0.983 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 705 | 1.797 | 0.0724 | 0.047 | 0.9628 | +0.0128 | 0 | 0 | +0.000 | tied |
| B1 | 910 | -4.769 | 0.0000 | -1.031 | 0.3028 | -0.0371 | 0 | 0 | -0.000 | tied |
| B2 | 1111 | -9.927 | 0.0000 | -0.910 | 0.3627 | -0.1371 | 0 | 5 | -0.000 | promising |
| B3 | 1291 | -2.648 | 0.0081 | -0.893 | 0.3719 | -0.0235 | 0 | 0 | -0.000 | tied |
| B4 | 1013 | -8.620 | 0.0000 | -1.726 | 0.0843 | -0.1015 | 0 | 5 | -0.000 | promising |
| B5 | 919 | -10.491 | 0.0000 | -2.392 | 0.0167 | -0.1062 | 0 | 5 | -0.000 | B>>A |
| B6 | 936 | -17.520 | 0.0000 | -3.423 | 0.0006 | -0.1532 | 0 | 5 | -0.000 | B>>A |
| B7 | 985 | -11.718 | 0.0000 | -3.097 | 0.0020 | -0.0896 | 0 | 5 | -0.000 | B>>A |
| B8 | 1699 | -29.040 | 0.0000 | -13.462 | 0.0000 | -0.0917 | 0 | 5 | -0.000 | B>>A |
| B9 | 486 | -1.747 | 0.0806 | -0.649 | 0.5165 | -0.0287 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v06_film_20260505T212932 | 3000 | 0.8451 | 0.8648 | 0.6630 | 0.0493 | 0.8914 | 0.502 |
| B: v_compression_persample_2026-05-18 | 3000 | 0.8893 | 0.9018 | 0.7130 | 0.0457 | 0.9173 | 0.432 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.8941 | -22.747 | 0.0000 | -71.752 | 0.0000 | -0.0259 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0305 | 0.1369 | 0.0271 | 0.0345 | 0.0144 | 0.991 |
| B1 | [0.10, 0.20) | 34 | 0.4377 | 0.5261 | 0.3089 | 0.0882 | 0.5835 | 0.850 |
| B2 | [0.20, 0.30) | 185 | 0.3102 | 0.3307 | 0.2067 | 0.0432 | 0.3935 | 0.944 |
| B3 | [0.30, 0.40) | 493 | 0.3648 | 0.3729 | 0.2497 | 0.0345 | 0.4442 | 0.928 |
| B4 | [0.40, 0.50) | 677 | 0.4031 | 0.4040 | 0.2753 | 0.0384 | 0.4804 | 0.915 |
| B5 | [0.50, 0.60) | 705 | 0.3132 | 0.3175 | 0.2148 | 0.0539 | 0.3740 | 0.948 |
| B6 | [0.60, 0.70) | 809 | 0.2652 | 0.2680 | 0.1828 | 0.0470 | 0.3150 | 0.963 |
| B7 | [0.70, 0.80) | 67 | 0.3848 | 0.4883 | 0.2672 | 0.0597 | 0.4557 | 0.873 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.1857 | 0.3541 | 0.1356 | 0.0345 | 0.2015 | 0.935 |
| B1 | [0.10, 0.20) | 34 | 0.4807 | 0.4927 | 0.3411 | 0.0294 | 0.6201 | 0.870 |
| B2 | [0.20, 0.30) | 185 | 0.3282 | 0.3478 | 0.2237 | 0.0378 | 0.4069 | 0.938 |
| B3 | [0.30, 0.40) | 493 | 0.4693 | 0.4819 | 0.3256 | 0.0365 | 0.5672 | 0.876 |
| B4 | [0.40, 0.50) | 677 | 0.5339 | 0.5398 | 0.3729 | 0.0443 | 0.6269 | 0.842 |
| B5 | [0.50, 0.60) | 705 | 0.4687 | 0.4924 | 0.3245 | 0.0525 | 0.5541 | 0.870 |
| B6 | [0.60, 0.70) | 809 | 0.1864 | 0.2186 | 0.1268 | 0.0420 | 0.2322 | 0.976 |
| B7 | [0.70, 0.80) | 67 | 0.3474 | 0.4519 | 0.2291 | 0.0746 | 0.4019 | 0.892 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 ⚠ | 29 | -11.939 | 0.0000 | -4.315 | 0.0000 | -0.1871 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | -3.354 | 0.0008 | 1.768 | 0.0770 | -0.0366 | 0 | 0 | +0.000 | tied |
| B2 | 185 | -1.489 | 0.1366 | -0.535 | 0.5926 | -0.0134 | 0 | 0 | -0.000 | tied |
| B3 | 493 | -6.344 | 0.0000 | -3.385 | 0.0007 | -0.1230 | 0 | 5 | -0.000 | B>>A |
| B4 | 677 | -7.221 | 0.0000 | -4.421 | 0.0000 | -0.1465 | 0 | 5 | -0.000 | B>>A |
| B5 | 705 | -8.929 | 0.0000 | -4.737 | 0.0000 | -0.1801 | 0 | 5 | -0.000 | B>>A |
| B6 | 809 | 7.645 | 0.0000 | 1.236 | 0.2166 | +0.0828 | 3 | 0 | +3.822 | promising |
| B7 | 67 | 2.850 | 0.0044 | 1.658 | 0.0973 | +0.0538 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v06_film_20260505T212932 | 1008 | 0.4971 | 0.4364 | 0.3476 | 0.0437 | 0.6386 | 0.900 |
| B: v_compression_persample_2026-05-18 | 1008 | 0.8080 | 0.8648 | 0.5935 | 0.0456 | 0.8505 | 0.502 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.2632 | -12.665 | 0.0000 | -18.320 | 0.0000 | -0.2119 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v06_film_20260505T212932 | 600 | 0.7862 | 0.7950 | 0.6152 | 0.0517 | 0.8646 | 0.607 |
| B: v_compression_persample_2026-05-18 | 600 | 0.8183 | 0.8248 | 0.6527 | 0.0550 | 0.8856 | 0.565 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9412 | -13.294 | 0.0000 | -28.143 | 0.0000 | -0.0210 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 233.04s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
