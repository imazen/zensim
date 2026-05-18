# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/tmp/v24_hybrid_nin_s2_packed_f16.bin` (label: `v24_hybrid_nin_s2_packed_f16`)
- **B**: `/home/lilith/work/zen/zensim--hybrid-runtime/zensim/weights/v_compression_persample_2026-05-18.bin` (label: `v_compression_persample_2026-05-18`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8727 | 0.8641 | 0.492 | 0.508 | 0.9239 | 0.9157 | 38.547 | 142.703 | +32.123 | A>>B |
| KADIK10k | 10125 | 0.9319 | 0.9316 | 0.362 | 0.362 | 0.9603 | 0.9602 | 10.065 | 24.261 | +0.000 | promising |
| TID2013 | 3000 | 0.8884 | 0.8893 | 0.436 | 0.432 | 0.9168 | 0.9173 | -28.801 | -236.059 | -0.000 | promising |
| KonJND-1k (full) | 1008 | 0.7906 | 0.8080 | 0.554 | 0.502 | 0.8309 | 0.8505 | -19.007 | -102.780 | -0.000 | B>>A |
| AIC-3 CTC | 600 | 0.8096 | 0.8183 | 0.575 | 0.565 | 0.8797 | 0.8856 | -53.644 | -98.804 | -0.000 | B>>A |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 2 cells
- **BDecisivelyBeatsA**: 3 cells
- **PromisingNotDecisive**: 4 cells
- **Tied**: 20 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (2 A wins vs 3 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_nin_s2_packed_f16 | 4292 | 0.8727 | 0.8707 | 0.6843 | 0.0485 | 0.9239 | 0.492 |
| B: v_compression_persample_2026-05-18 | 4292 | 0.8641 | 0.8614 | 0.6742 | 0.0489 | 0.9157 | 0.508 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.9854 | 38.547 | 0.0000 | 142.703 | 0.0000 | +0.0082 | 5 | 0 | +32.123 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0762 | 0.1584 | 0.0551 | 0.0526 | 0.0307 | 0.987 |
| B4 | [0.40, 0.50) | 266 | 0.2568 | 0.2580 | 0.1725 | 0.0451 | 0.3094 | 0.966 |
| B5 | [0.50, 0.60) | 615 | 0.2984 | 0.3036 | 0.2029 | 0.0439 | 0.3557 | 0.953 |
| B6 | [0.60, 0.70) | 836 | 0.2766 | 0.2770 | 0.1830 | 0.0383 | 0.3270 | 0.961 |
| B7 | [0.70, 0.80) | 1092 | 0.3807 | 0.3896 | 0.2611 | 0.0485 | 0.4472 | 0.921 |
| B8 | [0.80, 0.90) | 1382 | 0.5208 | 0.5261 | 0.3551 | 0.0463 | 0.6104 | 0.850 |
| B9 | [0.90, 1.00] | 43 | 0.1317 | 0.3594 | 0.0853 | 0.0233 | 0.2843 | 0.933 |

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
| B3 | 57 | -1.588 | 0.1124 | -7.843 | 0.0000 | -0.0066 | 0 | 0 | -0.000 | tied |
| B4 | 266 | -4.065 | 0.0000 | -0.569 | 0.5694 | -0.0232 | 0 | 0 | -0.000 | tied |
| B5 | 615 | 11.466 | 0.0000 | 3.309 | 0.0009 | +0.0387 | 3 | 0 | +5.733 | promising |
| B6 | 836 | 1.443 | 0.1491 | 0.229 | 0.8188 | +0.0075 | 0 | 0 | +0.000 | tied |
| B7 | 1092 | 0.850 | 0.3955 | 0.853 | 0.3934 | +0.0053 | 0 | 0 | +0.000 | tied |
| B8 | 1382 | 32.970 | 0.0000 | 24.098 | 0.0000 | +0.0304 | 5 | 0 | +27.475 | A>>B |
| B9 | 43 | -1.012 | 0.3116 | -4.249 | 0.0000 | -0.0106 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_nin_s2_packed_f16 | 10125 | 0.9319 | 0.9322 | 0.7686 | 0.0544 | 0.9603 | 0.362 |
| B: v_compression_persample_2026-05-18 | 10125 | 0.9316 | 0.9321 | 0.7684 | 0.0462 | 0.9602 | 0.362 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9967 | 10.065 | 0.0000 | 24.261 | 0.0000 | +0.0001 | 0 | 1 | +0.000 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### KADIK10k 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2463 | 0.2647 | 0.1726 | 0.0468 | 0.3109 | 0.964 |
| B1 | [0.10, 0.20) | 910 | 0.2570 | 0.2562 | 0.1785 | 0.0374 | 0.3209 | 0.967 |
| B2 | [0.20, 0.30) | 1111 | 0.1763 | 0.1883 | 0.1216 | 0.0396 | 0.2038 | 0.982 |
| B3 | [0.30, 0.40) | 1291 | 0.2151 | 0.2250 | 0.1502 | 0.0457 | 0.2540 | 0.974 |
| B4 | [0.40, 0.50) | 1013 | 0.2629 | 0.2621 | 0.1850 | 0.0405 | 0.3143 | 0.965 |
| B5 | [0.50, 0.60) | 919 | 0.1996 | 0.2216 | 0.1391 | 0.0381 | 0.2443 | 0.975 |
| B6 | [0.60, 0.70) | 936 | 0.2225 | 0.2396 | 0.1527 | 0.0449 | 0.2759 | 0.971 |
| B7 | [0.70, 0.80) | 985 | 0.2614 | 0.2644 | 0.1830 | 0.0386 | 0.3061 | 0.964 |
| B8 | [0.80, 0.90) | 1699 | 0.4501 | 0.4496 | 0.3164 | 0.0406 | 0.5315 | 0.893 |
| B9 | [0.90, 1.00] | 486 | 0.1690 | 0.1805 | 0.1162 | 0.0391 | 0.2005 | 0.984 |

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
| B0 | 705 | 0.148 | 0.8823 | 0.432 | 0.6655 | -0.0030 | 0 | 0 | -0.000 | tied |
| B1 | 910 | -3.293 | 0.0010 | -1.817 | 0.0691 | -0.0011 | 0 | 0 | -0.000 | tied |
| B2 | 1111 | -20.323 | 0.0000 | -3.543 | 0.0004 | -0.0192 | 0 | 3 | -0.000 | promising |
| B3 | 1291 | -7.670 | 0.0000 | -2.144 | 0.0320 | -0.0062 | 0 | 0 | -0.000 | tied |
| B4 | 1013 | 12.674 | 0.0000 | 3.048 | 0.0023 | +0.0144 | 0 | 0 | +0.000 | tied |
| B5 | 919 | -4.279 | 0.0000 | -2.801 | 0.0051 | -0.0070 | 0 | 0 | -0.000 | tied |
| B6 | 936 | -7.301 | 0.0000 | -1.474 | 0.1405 | -0.0044 | 0 | 0 | -0.000 | tied |
| B7 | 985 | -13.071 | 0.0000 | -3.891 | 0.0001 | -0.0115 | 0 | 0 | -0.000 | tied |
| B8 | 1699 | -4.028 | 0.0001 | -2.488 | 0.0128 | -0.0028 | 0 | 0 | -0.000 | tied |
| B9 | 486 | -3.535 | 0.0004 | -0.412 | 0.6803 | -0.0032 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_nin_s2_packed_f16 | 3000 | 0.8884 | 0.9001 | 0.7119 | 0.0453 | 0.9168 | 0.436 |
| B: v_compression_persample_2026-05-18 | 3000 | 0.8893 | 0.9018 | 0.7130 | 0.0457 | 0.9173 | 0.432 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.9982 | -28.801 | 0.0000 | -236.059 | 0.0000 | -0.0005 | 0 | 2 | -0.000 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.1419 | 0.2889 | 0.1011 | 0.0345 | 0.1285 | 0.957 |
| B1 | [0.10, 0.20) | 34 | 0.5096 | 0.5906 | 0.3625 | 0.0588 | 0.6466 | 0.807 |
| B2 | [0.20, 0.30) | 185 | 0.3236 | 0.3535 | 0.2181 | 0.0324 | 0.4007 | 0.935 |
| B3 | [0.30, 0.40) | 493 | 0.4773 | 0.4879 | 0.3307 | 0.0385 | 0.5780 | 0.873 |
| B4 | [0.40, 0.50) | 677 | 0.5377 | 0.5417 | 0.3755 | 0.0458 | 0.6276 | 0.841 |
| B5 | [0.50, 0.60) | 705 | 0.4595 | 0.4824 | 0.3169 | 0.0539 | 0.5464 | 0.876 |
| B6 | [0.60, 0.70) | 809 | 0.1876 | 0.2180 | 0.1272 | 0.0433 | 0.2333 | 0.976 |
| B7 | [0.70, 0.80) | 67 | 0.3555 | 0.4792 | 0.2382 | 0.0597 | 0.4098 | 0.878 |
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
| B0 ⚠ | 29 | -3.288 | 0.0010 | -1.734 | 0.0830 | -0.0730 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | 7.207 | 0.0000 | 18.123 | 0.0000 | +0.0265 | 0 | 0 | +0.000 | tied |
| B2 | 185 | -1.265 | 0.2058 | 0.618 | 0.5364 | -0.0061 | 0 | 0 | +0.000 | tied |
| B3 | 493 | 10.429 | 0.0000 | 4.971 | 0.0000 | +0.0108 | 0 | 0 | +0.000 | tied |
| B4 | 677 | 9.082 | 0.0000 | 3.419 | 0.0006 | +0.0006 | 0 | 0 | +0.000 | tied |
| B5 | 705 | -30.879 | 0.0000 | -21.391 | 0.0000 | -0.0077 | 0 | 4 | -0.000 | B>>A |
| B6 | 809 | 7.961 | 0.0000 | -0.802 | 0.4228 | +0.0011 | 0 | 0 | -0.000 | tied |
| B7 | 67 | 10.660 | 0.0000 | 21.253 | 0.0000 | +0.0079 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_nin_s2_packed_f16 | 1008 | 0.7906 | 0.8324 | 0.5784 | 0.0387 | 0.8309 | 0.554 |
| B: v_compression_persample_2026-05-18 | 1008 | 0.8080 | 0.8648 | 0.5935 | 0.0456 | 0.8505 | 0.502 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.9710 | -19.007 | 0.0000 | -102.780 | 0.0000 | -0.0195 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_nin_s2_packed_f16 | 600 | 0.8096 | 0.8184 | 0.6444 | 0.0567 | 0.8797 | 0.575 |
| B: v_compression_persample_2026-05-18 | 600 | 0.8183 | 0.8248 | 0.6527 | 0.0550 | 0.8856 | 0.565 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9961 | -53.644 | 0.0000 | -98.804 | 0.0000 | -0.0059 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 105.38s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
