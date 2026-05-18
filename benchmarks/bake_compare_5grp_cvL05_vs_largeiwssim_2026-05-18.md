# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_5grp_s3_h128_packed.bin` (label: `v22_mix_cv40_konjnd_0_02_5grp_s3_h128_packed`)
- **B**: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128.bin` (label: `v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `1000`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8437 | 0.8323 | 0.557 | 0.559 | 0.9007 | 0.9005 | 7.900 | 3.335 | +1.317 | promising |
| KADIK10k | 10125 | 0.8989 | 0.9677 | 0.442 | 0.249 | 0.9402 | 0.9804 | -94.323 | -611.920 | -15.721 | B>>A |
| TID2013 | 3000 | 0.8845 | 0.9729 | 0.448 | 0.236 | 0.9123 | 0.9833 | -54.565 | -306.499 | -0.000 | B>>A |
| KonJND-1k (full) | 1008 | 0.7909 | 0.8928 | 0.434 | 0.375 | 0.8392 | 0.9181 | -31.526 | -42.610 | -0.000 | B>>A |
| AIC-3 CTC | 600 | 0.7980 | 0.7831 | 0.590 | 0.608 | 0.8661 | 0.8619 | 4.632 | 9.292 | +0.000 | promising |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 0 cells
- **BDecisivelyBeatsA**: 18 cells
- **PromisingNotDecisive**: 4 cells
- **Tied**: 7 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (0 A wins vs 18 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_0_02_5grp_s3_h128_packed | 4292 | 0.8437 | 0.8308 | 0.6412 | 0.0496 | 0.9007 | 0.557 |
| B: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128 | 4292 | 0.8323 | 0.8290 | 0.6339 | 0.0445 | 0.9005 | 0.559 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.9055 | 7.900 | 0.0000 | 3.335 | 0.0009 | +0.0002 | 1 | 0 | +1.317 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0398 | 0.3825 | 0.0125 | 0.0000 | 0.0295 | 0.924 |
| B4 | [0.40, 0.50) | 266 | 0.1396 | 0.2167 | 0.0939 | 0.0564 | 0.1728 | 0.976 |
| B5 | [0.50, 0.60) | 615 | 0.2098 | 0.2175 | 0.1429 | 0.0455 | 0.2482 | 0.976 |
| B6 | [0.60, 0.70) | 836 | 0.2471 | 0.2517 | 0.1640 | 0.0502 | 0.2875 | 0.968 |
| B7 | [0.70, 0.80) | 1092 | 0.3615 | 0.3729 | 0.2465 | 0.0449 | 0.4292 | 0.928 |
| B8 | [0.80, 0.90) | 1382 | 0.4216 | 0.4254 | 0.2840 | 0.0405 | 0.4993 | 0.905 |
| B9 | [0.90, 1.00] | 43 | 0.0667 | 0.2945 | 0.0432 | 0.0000 | 0.1870 | 0.956 |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.0635 | 0.1900 | 0.0363 | 0.0526 | 0.0885 | 0.982 |
| B4 | [0.40, 0.50) | 266 | 0.2520 | 0.2451 | 0.1666 | 0.0489 | 0.3181 | 0.970 |
| B5 | [0.50, 0.60) | 615 | 0.2508 | 0.2618 | 0.1684 | 0.0390 | 0.3097 | 0.965 |
| B6 | [0.60, 0.70) | 836 | 0.2238 | 0.2333 | 0.1495 | 0.0323 | 0.2602 | 0.972 |
| B7 | [0.70, 0.80) | 1092 | 0.3189 | 0.3242 | 0.2170 | 0.0513 | 0.3762 | 0.946 |
| B8 | [0.80, 0.90) | 1382 | 0.4955 | 0.5019 | 0.3350 | 0.0347 | 0.5843 | 0.865 |
| B9 | [0.90, 1.00] | 43 | 0.1045 | 0.1831 | 0.0653 | 0.0233 | 0.2696 | 0.983 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B2 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B3 | 57 | -0.099 | 0.9212 | 0.246 | 0.8060 | -0.0590 | 0 | 0 | +0.000 | tied |
| B4 | 266 | -5.074 | 0.0000 | -0.313 | 0.7546 | -0.1452 | 0 | 3 | -0.000 | promising |
| B5 | 615 | -2.829 | 0.0047 | -0.773 | 0.4395 | -0.0616 | 0 | 0 | -0.000 | tied |
| B6 | 836 | 1.858 | 0.0632 | 0.378 | 0.7051 | +0.0274 | 0 | 0 | +0.000 | tied |
| B7 | 1092 | 5.124 | 0.0000 | 2.305 | 0.0212 | +0.0529 | 2 | 0 | +1.708 | promising |
| B8 | 1382 | -20.943 | 0.0000 | -12.580 | 0.0000 | -0.0850 | 0 | 5 | -0.000 | B>>A |
| B9 | 43 | -1.106 | 0.2686 | 0.818 | 0.4132 | -0.0826 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KADIK10k (n=10125)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_0_02_5grp_s3_h128_packed | 10125 | 0.8989 | 0.8971 | 0.7184 | 0.0356 | 0.9402 | 0.442 |
| B: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128 | 10125 | 0.9677 | 0.9686 | 0.8432 | 0.0424 | 0.9804 | 0.249 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 10125 | 0.9307 | -94.323 | 0.0000 | -611.920 | 0.0000 | -0.0403 | 1 | 5 | -15.721 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### KADIK10k 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.2084 | 0.2149 | 0.1455 | 0.0468 | 0.2635 | 0.977 |
| B1 | [0.10, 0.20) | 910 | 0.1862 | 0.1949 | 0.1299 | 0.0440 | 0.2426 | 0.981 |
| B2 | [0.20, 0.30) | 1111 | 0.0992 | 0.1162 | 0.0685 | 0.0387 | 0.1119 | 0.993 |
| B3 | [0.30, 0.40) | 1291 | 0.1940 | 0.2006 | 0.1345 | 0.0387 | 0.2354 | 0.980 |
| B4 | [0.40, 0.50) | 1013 | 0.2149 | 0.2178 | 0.1512 | 0.0346 | 0.2528 | 0.976 |
| B5 | [0.50, 0.60) | 919 | 0.1572 | 0.1837 | 0.1092 | 0.0392 | 0.1968 | 0.983 |
| B6 | [0.60, 0.70) | 936 | 0.1974 | 0.2122 | 0.1361 | 0.0321 | 0.2473 | 0.977 |
| B7 | [0.70, 0.80) | 985 | 0.2247 | 0.2342 | 0.1567 | 0.0325 | 0.2669 | 0.972 |
| B8 | [0.80, 0.90) | 1699 | 0.4197 | 0.4212 | 0.2943 | 0.0394 | 0.4993 | 0.907 |
| B9 | [0.90, 1.00] | 486 | 0.1833 | 0.1871 | 0.1306 | 0.0391 | 0.2010 | 0.982 |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.4139 | 0.4237 | 0.2932 | 0.0255 | 0.5006 | 0.906 |
| B1 | [0.10, 0.20) | 910 | 0.4144 | 0.4186 | 0.2918 | 0.0407 | 0.4850 | 0.908 |
| B2 | [0.20, 0.30) | 1111 | 0.3964 | 0.4018 | 0.2784 | 0.0378 | 0.4714 | 0.916 |
| B3 | [0.30, 0.40) | 1291 | 0.3367 | 0.3355 | 0.2370 | 0.0434 | 0.3980 | 0.942 |
| B4 | [0.40, 0.50) | 1013 | 0.3746 | 0.3809 | 0.2662 | 0.0444 | 0.4402 | 0.925 |
| B5 | [0.50, 0.60) | 919 | 0.3460 | 0.3534 | 0.2448 | 0.0479 | 0.4194 | 0.935 |
| B6 | [0.60, 0.70) | 936 | 0.3652 | 0.3641 | 0.2550 | 0.0449 | 0.4431 | 0.931 |
| B7 | [0.70, 0.80) | 985 | 0.3613 | 0.3676 | 0.2554 | 0.0416 | 0.4434 | 0.930 |
| B8 | [0.80, 0.90) | 1699 | 0.5027 | 0.5035 | 0.3560 | 0.0388 | 0.5881 | 0.864 |
| B9 | [0.90, 1.00] | 486 | 0.1821 | 0.2235 | 0.1251 | 0.0370 | 0.2163 | 0.975 |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 | 705 | -10.312 | 0.0000 | -3.667 | 0.0002 | -0.2371 | 0 | 5 | -0.000 | B>>A |
| B1 | 910 | -11.646 | 0.0000 | -3.818 | 0.0001 | -0.2423 | 0 | 5 | -0.000 | B>>A |
| B2 | 1111 | -14.470 | 0.0000 | -3.861 | 0.0001 | -0.3596 | 0 | 5 | -0.000 | B>>A |
| B3 | 1291 | -11.824 | 0.0000 | -3.194 | 0.0014 | -0.1626 | 0 | 5 | -0.000 | B>>A |
| B4 | 1013 | -11.911 | 0.0000 | -3.946 | 0.0001 | -0.1874 | 0 | 5 | -0.000 | B>>A |
| B5 | 919 | -11.733 | 0.0000 | -3.022 | 0.0025 | -0.2226 | 0 | 5 | -0.000 | B>>A |
| B6 | 936 | -10.699 | 0.0000 | -3.008 | 0.0026 | -0.1959 | 1 | 5 | -1.783 | B>>A |
| B7 | 985 | -12.512 | 0.0000 | -3.993 | 0.0001 | -0.1765 | 0 | 5 | -0.000 | B>>A |
| B8 | 1699 | -23.111 | 0.0000 | -13.204 | 0.0000 | -0.0888 | 0 | 5 | -0.000 | B>>A |
| B9 | 486 | 0.254 | 0.7992 | -1.652 | 0.0986 | -0.0153 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_0_02_5grp_s3_h128_packed | 3000 | 0.8845 | 0.8941 | 0.7069 | 0.0350 | 0.9123 | 0.448 |
| B: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128 | 3000 | 0.9729 | 0.9718 | 0.8573 | 0.0380 | 0.9833 | 0.236 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.9191 | -54.565 | 0.0000 | -306.499 | 0.0000 | -0.0710 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0995 | 0.2233 | 0.0814 | 0.0345 | 0.1007 | 0.975 |
| B1 | [0.10, 0.20) | 34 | 0.4419 | 0.5581 | 0.3089 | 0.0588 | 0.5881 | 0.830 |
| B2 | [0.20, 0.30) | 185 | 0.2223 | 0.3182 | 0.1464 | 0.0378 | 0.2867 | 0.948 |
| B3 | [0.30, 0.40) | 493 | 0.4443 | 0.4486 | 0.3060 | 0.0385 | 0.5381 | 0.894 |
| B4 | [0.40, 0.50) | 677 | 0.5354 | 0.5381 | 0.3751 | 0.0502 | 0.6201 | 0.843 |
| B5 | [0.50, 0.60) | 705 | 0.4569 | 0.4767 | 0.3157 | 0.0397 | 0.5390 | 0.879 |
| B6 | [0.60, 0.70) | 809 | 0.1823 | 0.2106 | 0.1234 | 0.0383 | 0.2288 | 0.978 |
| B7 | [0.70, 0.80) | 67 | 0.3907 | 0.4988 | 0.2672 | 0.0746 | 0.4589 | 0.867 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**B's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0559 | 0.3495 | 0.0567 | 0.0345 | 0.0759 | 0.937 |
| B1 | [0.10, 0.20) | 34 | 0.5396 | 0.5997 | 0.3482 | 0.0294 | 0.7026 | 0.800 |
| B2 | [0.20, 0.30) | 185 | 0.6423 | 0.6464 | 0.4552 | 0.0432 | 0.7132 | 0.763 |
| B3 | [0.30, 0.40) | 493 | 0.7360 | 0.7367 | 0.5403 | 0.0467 | 0.8178 | 0.676 |
| B4 | [0.40, 0.50) | 677 | 0.7622 | 0.7623 | 0.5597 | 0.0458 | 0.8362 | 0.647 |
| B5 | [0.50, 0.60) | 705 | 0.7075 | 0.7070 | 0.5096 | 0.0496 | 0.7900 | 0.707 |
| B6 | [0.60, 0.70) | 809 | 0.5871 | 0.5864 | 0.4106 | 0.0334 | 0.6836 | 0.810 |
| B7 | [0.70, 0.80) | 67 | 0.2857 | 0.4937 | 0.1928 | 0.0746 | 0.2808 | 0.870 |
| B8 | [0.80, 0.90) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B9 | [0.90, 1.00] | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

**Per-band MRR + decisive rule (the ship-decision table):**

| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| B0 ⚠ | 29 | 0.424 | 0.6715 | -0.377 | 0.7059 | +0.0248 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | -3.151 | 0.0016 | -1.151 | 0.2496 | -0.1145 | 0 | 0 | -0.000 | tied |
| B2 | 185 | -11.882 | 0.0000 | -5.751 | 0.0000 | -0.4265 | 0 | 5 | -0.000 | B>>A |
| B3 | 493 | -16.558 | 0.0000 | -14.173 | 0.0000 | -0.2797 | 0 | 5 | -0.000 | B>>A |
| B4 | 677 | -20.019 | 0.0000 | -20.856 | 0.0000 | -0.2161 | 0 | 5 | -0.000 | B>>A |
| B5 | 705 | -19.351 | 0.0000 | -15.381 | 0.0000 | -0.2510 | 0 | 5 | -0.000 | B>>A |
| B6 | 809 | -23.992 | 0.0000 | -10.515 | 0.0000 | -0.4548 | 0 | 5 | -0.000 | B>>A |
| B7 | 67 | 0.867 | 0.3861 | 0.028 | 0.9779 | +0.1781 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_0_02_5grp_s3_h128_packed | 1008 | 0.7909 | 0.9010 | 0.5714 | 0.0446 | 0.8392 | 0.434 |
| B: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128 | 1008 | 0.8928 | 0.9270 | 0.7074 | 0.0446 | 0.9181 | 0.375 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.8997 | -31.526 | 0.0000 | -42.610 | 0.0000 | -0.0789 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## AIC-3 CTC (n=600)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v22_mix_cv40_konjnd_0_02_5grp_s3_h128_packed | 600 | 0.7980 | 0.8077 | 0.6332 | 0.0617 | 0.8661 | 0.590 |
| B: v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128 | 600 | 0.7831 | 0.7942 | 0.6139 | 0.0433 | 0.8619 | 0.608 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 600 | 0.9218 | 4.632 | 0.0000 | 9.292 | 0.0000 | +0.0042 | 0 | 1 | +0.000 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for AIC-3 CTC — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

---
Wall time: 150.94s (19025 pair rows scored × 2 bakes across 5 corpora; 1000 bootstrap resamples × bands).
