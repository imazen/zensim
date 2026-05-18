# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/tmp/v24_hybrid_nin_s2_packed_f16.bin` (label: `v24_hybrid_nin_s2_packed_f16`)
- **B**: `/home/lilith/work/zen/zensim--hybrid-runtime/zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` (label: `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8727 | 0.8324 | 0.492 | 0.559 | 0.9239 | 0.9006 | 49.764 | 149.645 | +41.470 | A>>B |
| KADIK10k | 10125 | 0.9319 | 0.9677 | 0.362 | 0.249 | 0.9603 | 0.9804 | -90.195 | -794.012 | -0.000 | B>>A |
| TID2013 | 3000 | 0.8884 | 0.9729 | 0.436 | 0.236 | 0.9168 | 0.9832 | -54.117 | -307.034 | -0.000 | B>>A |
| KonJND-1k (full) | 1008 | 0.7906 | 0.8927 | 0.554 | 0.376 | 0.8309 | 0.9178 | -44.292 | -139.693 | -0.000 | B>>A |
| AIC-3 CTC | 600 | 0.8096 | 0.7845 | 0.575 | 0.606 | 0.8797 | 0.8630 | 18.342 | 38.134 | +15.285 | A>>B |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 4 cells
- **BDecisivelyBeatsA**: 17 cells
- **PromisingNotDecisive**: 1 cells
- **Tied**: 7 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (4 A wins vs 17 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_nin_s2_packed_f16 | 4292 | 0.8727 | 0.8707 | 0.6843 | 0.0485 | 0.9239 | 0.492 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 4292 | 0.8324 | 0.8289 | 0.6340 | 0.0443 | 0.9006 | 0.559 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.9472 | 49.764 | 0.0000 | 149.645 | 0.0000 | +0.0233 | 5 | 0 | +41.470 | A>>B |

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
| B3 | [0.30, 0.40) | 57 | 0.0487 | 0.1900 | 0.0276 | 0.0351 | 0.0703 | 0.982 |
| B4 | [0.40, 0.50) | 266 | 0.2510 | 0.2452 | 0.1658 | 0.0489 | 0.3169 | 0.969 |
| B5 | [0.50, 0.60) | 615 | 0.2466 | 0.2584 | 0.1655 | 0.0423 | 0.3049 | 0.966 |
| B6 | [0.60, 0.70) | 836 | 0.2220 | 0.2317 | 0.1484 | 0.0347 | 0.2580 | 0.973 |
| B7 | [0.70, 0.80) | 1092 | 0.3192 | 0.3244 | 0.2173 | 0.0513 | 0.3766 | 0.946 |
| B8 | [0.80, 0.90) | 1382 | 0.4958 | 0.5026 | 0.3354 | 0.0347 | 0.5846 | 0.865 |
| B9 | [0.90, 1.00] | 43 | 0.1056 | 0.1831 | 0.0698 | 0.0233 | 0.2725 | 0.983 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B2 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B3 | 57 | 0.112 | 0.9108 | -0.023 | 0.9816 | -0.0396 | 0 | 0 | +0.000 | tied |
| B4 | 266 | 0.390 | 0.6963 | 0.230 | 0.8179 | -0.0075 | 0 | 0 | -0.000 | tied |
| B5 | 615 | 5.165 | 0.0000 | 1.366 | 0.1718 | +0.0507 | 0 | 0 | +0.000 | tied |
| B6 | 836 | 7.424 | 0.0000 | 1.667 | 0.0954 | +0.0689 | 3 | 0 | +3.712 | promising |
| B7 | 1092 | 14.467 | 0.0000 | 6.209 | 0.0000 | +0.0707 | 5 | 0 | +12.056 | A>>B |
| B8 | 1382 | 11.969 | 0.0000 | 7.826 | 0.0000 | +0.0258 | 5 | 1 | +9.974 | A>>B |
| B9 | 43 | 1.443 | 0.1489 | 2.804 | 0.0051 | +0.0118 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_nin_s2_packed_f16 | 10125 | 0.9319 | 0.9322 | 0.7686 | 0.0544 | 0.9603 | 0.362 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 10125 | 0.9677 | 0.9686 | 0.8432 | 0.0433 | 0.9804 | 0.249 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9611 | -90.195 | 0.0000 | -794.012 | 0.0000 | -0.0202 | 0 | 6 | -0.000 | B>>A |

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
| B0 | [0.00, 0.10) | 705 | 0.4150 | 0.4235 | 0.2939 | 0.0241 | 0.5015 | 0.906 |
| B1 | [0.10, 0.20) | 910 | 0.4149 | 0.4191 | 0.2922 | 0.0374 | 0.4853 | 0.908 |
| B2 | [0.20, 0.30) | 1111 | 0.3995 | 0.4047 | 0.2807 | 0.0387 | 0.4741 | 0.914 |
| B3 | [0.30, 0.40) | 1291 | 0.3342 | 0.3325 | 0.2352 | 0.0418 | 0.3953 | 0.943 |
| B4 | [0.40, 0.50) | 1013 | 0.3754 | 0.3801 | 0.2666 | 0.0444 | 0.4405 | 0.925 |
| B5 | [0.50, 0.60) | 919 | 0.3454 | 0.3527 | 0.2442 | 0.0457 | 0.4190 | 0.936 |
| B6 | [0.60, 0.70) | 936 | 0.3649 | 0.3643 | 0.2549 | 0.0438 | 0.4434 | 0.931 |
| B7 | [0.70, 0.80) | 985 | 0.3603 | 0.3669 | 0.2552 | 0.0396 | 0.4420 | 0.930 |
| B8 | [0.80, 0.90) | 1699 | 0.5019 | 0.5025 | 0.3554 | 0.0383 | 0.5871 | 0.865 |
| B9 | [0.90, 1.00] | 486 | 0.1818 | 0.2299 | 0.1248 | 0.0370 | 0.2158 | 0.973 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 705 | -9.800 | 0.0000 | -3.537 | 0.0004 | -0.1906 | 0 | 5 | -0.000 | B>>A |
| B1 | 910 | -9.794 | 0.0000 | -3.784 | 0.0002 | -0.1645 | 0 | 5 | -0.000 | B>>A |
| B2 | 1111 | -15.030 | 0.0000 | -4.682 | 0.0000 | -0.2703 | 0 | 5 | -0.000 | B>>A |
| B3 | 1291 | -12.074 | 0.0000 | -3.260 | 0.0011 | -0.1413 | 0 | 5 | -0.000 | B>>A |
| B4 | 1013 | -10.106 | 0.0000 | -3.741 | 0.0002 | -0.1262 | 0 | 5 | -0.000 | B>>A |
| B5 | 919 | -11.180 | 0.0000 | -3.110 | 0.0019 | -0.1747 | 0 | 5 | -0.000 | B>>A |
| B6 | 936 | -10.635 | 0.0000 | -3.058 | 0.0022 | -0.1674 | 0 | 5 | -0.000 | B>>A |
| B7 | 985 | -11.574 | 0.0000 | -4.146 | 0.0000 | -0.1358 | 0 | 5 | -0.000 | B>>A |
| B8 | 1699 | -16.380 | 0.0000 | -10.153 | 0.0000 | -0.0556 | 0 | 5 | -0.000 | B>>A |
| B9 | 486 | -5.626 | 0.0000 | -4.612 | 0.0000 | -0.0153 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_nin_s2_packed_f16 | 3000 | 0.8884 | 0.9001 | 0.7119 | 0.0453 | 0.9168 | 0.436 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 3000 | 0.9729 | 0.9717 | 0.8571 | 0.0357 | 0.9832 | 0.236 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.9216 | -54.117 | 0.0000 | -307.034 | 0.0000 | -0.0664 | 0 | 6 | -0.000 | B>>A |

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

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 ⚠ | 29 | 0.938 | 0.3484 | -0.251 | 0.8018 | +0.0507 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | -0.607 | 0.5441 | -0.226 | 0.8215 | -0.0550 | 0 | 0 | -0.000 | tied |
| B2 | 185 | -10.904 | 0.0000 | -6.542 | 0.0000 | -0.3173 | 0 | 5 | -0.000 | B>>A |
| B3 | 493 | -15.044 | 0.0000 | -13.340 | 0.0000 | -0.2393 | 0 | 5 | -0.000 | B>>A |
| B4 | 677 | -19.761 | 0.0000 | -20.597 | 0.0000 | -0.2094 | 0 | 5 | -0.000 | B>>A |
| B5 | 705 | -18.293 | 0.0000 | -14.440 | 0.0000 | -0.2443 | 0 | 5 | -0.000 | B>>A |
| B6 | 809 | -23.962 | 0.0000 | -10.530 | 0.0000 | -0.4492 | 0 | 5 | -0.000 | B>>A |
| B7 | 67 | 0.584 | 0.5595 | -0.075 | 0.9399 | +0.1297 | 0 | 0 | -0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_nin_s2_packed_f16 | 1008 | 0.7906 | 0.8324 | 0.5784 | 0.0387 | 0.8309 | 0.554 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 1008 | 0.8927 | 0.9265 | 0.7070 | 0.0446 | 0.9178 | 0.376 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.9285 | -44.292 | 0.0000 | -139.693 | 0.0000 | -0.0868 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v24_hybrid_nin_s2_packed_f16 | 600 | 0.8096 | 0.8184 | 0.6444 | 0.0567 | 0.8797 | 0.575 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 600 | 0.7845 | 0.7953 | 0.6155 | 0.0433 | 0.8630 | 0.606 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9666 | 18.342 | 0.0000 | 38.134 | 0.0000 | +0.0168 | 5 | 0 | +15.285 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 93.33s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
