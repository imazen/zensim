# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/mnt/v/output/zensim/synthetic-v2/runs/v06_film_20260505T212932.bin` (label: `v06_film_20260505T212932`)
- **B**: `zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` (label: `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8755 | 0.8324 | 0.484 | 0.559 | 0.9267 | 0.9006 | 26.812 | 83.804 | +22.343 | A>>B |
| KADIK10k | 10125 | 0.8527 | 0.9677 | 0.526 | 0.249 | 0.9089 | 0.9804 | -89.993 | -431.777 | -0.000 | B>>A |
| TID2013 | 3000 | 0.8451 | 0.9729 | 0.502 | 0.236 | 0.8914 | 0.9832 | -48.851 | -218.273 | -0.000 | B>>A |
| KonJND-1k (full) | 1008 | 0.4971 | 0.8927 | 0.900 | 0.376 | 0.6386 | 0.9178 | -17.143 | -26.658 | -0.000 | B>>A |
| AIC-3 CTC | 600 | 0.7862 | 0.7845 | 0.607 | 0.606 | 0.8646 | 0.8630 | 1.032 | -0.303 | -0.000 | tied |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 3 cells
- **BDecisivelyBeatsA**: 18 cells
- **PromisingNotDecisive**: 0 cells
- **Tied**: 8 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (3 A wins vs 18 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v06_film_20260505T212932 | 4292 | 0.8755 | 0.8750 | 0.6867 | 0.0480 | 0.9267 | 0.484 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 4292 | 0.8324 | 0.8289 | 0.6340 | 0.0443 | 0.9006 | 0.559 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.8952 | 26.812 | 0.0000 | 83.804 | 0.0000 | +0.0261 | 5 | 0 | +22.343 | A>>B |

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
| B3 | 57 | 1.952 | 0.0510 | 0.445 | 0.6562 | +0.1719 | 0 | 0 | +0.000 | tied |
| B4 | 266 | 2.277 | 0.0228 | 0.867 | 0.3861 | +0.0417 | 0 | 0 | +0.000 | tied |
| B5 | 615 | 7.334 | 0.0000 | 2.260 | 0.0238 | +0.1412 | 5 | 0 | +6.112 | A>>B |
| B6 | 836 | 10.197 | 0.0000 | 3.441 | 0.0006 | +0.2178 | 5 | 0 | +8.498 | A>>B |
| B7 | 1092 | 3.755 | 0.0002 | 1.724 | 0.0846 | +0.0520 | 0 | 0 | +0.000 | tied |
| B8 | 1382 | -10.025 | 0.0000 | -7.448 | 0.0000 | -0.0329 | 0 | 5 | -0.000 | B>>A |
| B9 | 43 | -1.606 | 0.1084 | -0.010 | 0.9923 | -0.1509 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v06_film_20260505T212932 | 10125 | 0.8527 | 0.8503 | 0.6640 | 0.0493 | 0.9089 | 0.526 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 10125 | 0.9677 | 0.9686 | 0.8432 | 0.0433 | 0.9804 | 0.249 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.8839 | -89.993 | 0.0000 | -431.777 | 0.0000 | -0.0715 | 0 | 6 | -0.000 | B>>A |

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
| B0 | 705 | -8.235 | 0.0000 | -3.201 | 0.0014 | -0.1748 | 0 | 5 | -0.000 | B>>A |
| B1 | 910 | -10.216 | 0.0000 | -3.547 | 0.0004 | -0.2005 | 0 | 5 | -0.000 | B>>A |
| B2 | 1111 | -14.663 | 0.0000 | -3.466 | 0.0005 | -0.3882 | 0 | 5 | -0.000 | B>>A |
| B3 | 1291 | -11.172 | 0.0000 | -3.140 | 0.0017 | -0.1587 | 0 | 5 | -0.000 | B>>A |
| B4 | 1013 | -11.742 | 0.0000 | -3.434 | 0.0006 | -0.2420 | 0 | 5 | -0.000 | B>>A |
| B5 | 919 | -10.831 | 0.0000 | -2.714 | 0.0066 | -0.2739 | 0 | 5 | -0.000 | B>>A |
| B6 | 936 | -11.763 | 0.0000 | -2.866 | 0.0042 | -0.3163 | 0 | 5 | -0.000 | B>>A |
| B7 | 985 | -10.995 | 0.0000 | -3.494 | 0.0005 | -0.2139 | 0 | 5 | -0.000 | B>>A |
| B8 | 1699 | -23.068 | 0.0000 | -11.936 | 0.0000 | -0.1446 | 0 | 5 | -0.000 | B>>A |
| B9 | 486 | -3.068 | 0.0022 | -2.076 | 0.0379 | -0.0407 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v06_film_20260505T212932 | 3000 | 0.8451 | 0.8648 | 0.6630 | 0.0493 | 0.8914 | 0.502 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 3000 | 0.9729 | 0.9717 | 0.8571 | 0.0357 | 0.9832 | 0.236 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.8746 | -48.851 | 0.0000 | -218.273 | 0.0000 | -0.0918 | 0 | 6 | -0.000 | B>>A |

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
| B0 ⚠ | 29 | -0.264 | 0.7914 | -0.454 | 0.6497 | -0.0634 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | -3.693 | 0.0002 | -2.315 | 0.0206 | -0.1182 | 0 | 0 | -0.000 | tied |
| B2 | 185 | -11.161 | 0.0000 | -6.683 | 0.0000 | -0.3245 | 0 | 5 | -0.000 | B>>A |
| B3 | 493 | -15.280 | 0.0000 | -11.575 | 0.0000 | -0.3731 | 0 | 5 | -0.000 | B>>A |
| B4 | 677 | -18.294 | 0.0000 | -15.497 | 0.0000 | -0.3565 | 0 | 5 | -0.000 | B>>A |
| B5 | 705 | -20.287 | 0.0000 | -13.595 | 0.0000 | -0.4168 | 0 | 5 | -0.000 | B>>A |
| B6 | 809 | -20.024 | 0.0000 | -10.115 | 0.0000 | -0.3675 | 0 | 6 | -0.000 | B>>A |
| B7 | 67 | 0.905 | 0.3653 | -0.031 | 0.9749 | +0.1755 | 0 | 0 | -0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v06_film_20260505T212932 | 1008 | 0.4971 | 0.4364 | 0.3476 | 0.0437 | 0.6386 | 0.900 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 1008 | 0.8927 | 0.9265 | 0.7070 | 0.0446 | 0.9178 | 0.376 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.3567 | -17.143 | 0.0000 | -26.658 | 0.0000 | -0.2792 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v06_film_20260505T212932 | 600 | 0.7862 | 0.7950 | 0.6152 | 0.0517 | 0.8646 | 0.607 |
| B: v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18 | 600 | 0.7845 | 0.7953 | 0.6155 | 0.0433 | 0.8630 | 0.606 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9590 | 1.032 | 0.3019 | -0.303 | 0.7619 | +0.0017 | 0 | 0 | -0.000 | tied |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 224.67s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
