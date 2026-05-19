# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/home/lilith/work/zen/zensim/zensim/weights/v_compression_persample_2026-05-18.bin` (label: `v_compression_persample_2026-05-18`)
- **B**: `/home/lilith/work/zen/zensim--v6-reship/zensim/weights/v_tuner_v6_2026-05-19.bin` (label: `v_tuner_v6_2026-05-19`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `500`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.2332 | 0.8216 | 0.963 | 0.680 | 0.2791 | 0.8821 | -50.118 | -28.287 | -8.353 | B>>A |
| KonJND-1k (full) | 1008 | 0.4666 | 0.3734 | 0.893 | 0.960 | 0.5150 | 0.5297 | 3.142 | 2.350 | -2.095 | promising |
| TID2013 | 3000 | 0.1473 | 0.4447 | 0.984 | 0.809 | 0.1432 | 0.5451 | -36.938 | -22.413 | -0.000 | B>>A |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 0 cells
- **BDecisivelyBeatsA**: 4 cells
- **PromisingNotDecisive**: 7 cells
- **Tied**: 6 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `B`** (0 A wins vs 4 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_compression_persample_2026-05-18 | 4292 | 0.2332 | 0.2707 | 0.1618 | 0.0440 | 0.2791 | 0.963 |
| B: v_tuner_v6_2026-05-19 | 4292 | 0.8216 | 0.7332 | 0.6249 | 0.0531 | 0.8821 | 0.680 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.3306 | -50.118 | 0.0000 | -28.287 | 0.0000 | -0.6030 | 1 | 5 | -8.353 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### CID22 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | [0.10, 0.20) | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | [0.20, 0.30) | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| B3 | [0.30, 0.40) | 57 | 0.1503 | 0.1452 | 0.1053 | 0.0526 | 0.1326 | 0.989 |
| B4 | [0.40, 0.50) | 266 | 0.0480 | 0.0866 | 0.0330 | 0.0263 | 0.0559 | 0.996 |
| B5 | [0.50, 0.60) | 615 | 0.0743 | 0.0849 | 0.0483 | 0.0358 | 0.1019 | 0.996 |
| B6 | [0.60, 0.70) | 836 | 0.0704 | 0.0847 | 0.0476 | 0.0419 | 0.0908 | 0.996 |
| B7 | [0.70, 0.80) | 1092 | 0.0853 | 0.1126 | 0.0574 | 0.0375 | 0.1150 | 0.994 |
| B8 | [0.80, 0.90) | 1382 | 0.1231 | 0.1379 | 0.0806 | 0.0398 | 0.1375 | 0.990 |
| B9 | [0.90, 1.00] | 43 | 0.1712 | 0.2476 | 0.1185 | 0.0465 | 0.2548 | 0.969 |

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
| B3 | 57 | 0.525 | 0.5994 | 0.004 | 0.9969 | +0.0308 | 0 | 0 | +0.000 | tied |
| B4 | 266 | -3.277 | 0.0010 | -0.462 | 0.6444 | -0.2136 | 0 | 3 | -0.000 | promising |
| B5 | 615 | -5.074 | 0.0000 | -0.828 | 0.4079 | -0.2327 | 0 | 5 | -0.000 | promising |
| B6 | 836 | -7.197 | 0.0000 | -1.522 | 0.1280 | -0.2682 | 0 | 5 | -0.000 | promising |
| B7 | 1092 | -10.667 | 0.0000 | -2.145 | 0.0319 | -0.2903 | 0 | 5 | -0.000 | B>>A |
| B8 | 1382 | -20.733 | 0.0000 | -5.922 | 0.0000 | -0.3582 | 0 | 6 | -0.000 | B>>A |
| B9 | 43 | 6.595 | 0.0000 | 0.043 | 0.9656 | +0.1707 | 0 | 0 | +0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_compression_persample_2026-05-18 | 1008 | 0.4666 | 0.4504 | 0.3111 | 0.0238 | 0.5150 | 0.893 |
| B: v_tuner_v6_2026-05-19 | 1008 | 0.3734 | 0.2813 | 0.2584 | 0.0347 | 0.5297 | 0.960 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.0619 | 3.142 | 0.0017 | 2.350 | 0.0188 | -0.0147 | 4 | 0 | -2.095 | promising |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_compression_persample_2026-05-18 | 3000 | 0.1473 | 0.1758 | 0.1032 | 0.0363 | 0.1432 | 0.984 |
| B: v_tuner_v6_2026-05-19 | 3000 | 0.4447 | 0.5878 | 0.3055 | 0.0430 | 0.5451 | 0.809 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | 0.5671 | -36.938 | 0.0000 | -22.413 | 0.0000 | -0.4018 | 0 | 5 | -0.000 | B>>A |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

### TID2013 10-band per-band panel + decisive rule

**A's panel:**

| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---|--:|---:|---:|---:|---:|---:|---:|
| B0 ⚠ | [0.00, 0.10) | 29 | 0.0047 | 0.2064 | 0.0025 | 0.0345 | 0.0085 | 0.978 |
| B1 | [0.10, 0.20) | 34 | 0.2112 | 0.2418 | 0.1518 | 0.0294 | 0.3085 | 0.970 |
| B2 | [0.20, 0.30) | 185 | 0.2198 | 0.2861 | 0.1479 | 0.0270 | 0.2535 | 0.958 |
| B3 | [0.30, 0.40) | 493 | 0.0778 | 0.1056 | 0.0501 | 0.0284 | 0.1168 | 0.994 |
| B4 | [0.40, 0.50) | 677 | 0.1207 | 0.1647 | 0.0805 | 0.0399 | 0.1555 | 0.986 |
| B5 | [0.50, 0.60) | 705 | 0.0446 | 0.0804 | 0.0298 | 0.0369 | 0.0633 | 0.997 |
| B6 | [0.60, 0.70) | 809 | 0.0695 | 0.1316 | 0.0477 | 0.0445 | 0.0789 | 0.991 |
| B7 | [0.70, 0.80) | 67 | 0.3434 | 0.4483 | 0.2309 | 0.0448 | 0.3878 | 0.894 |
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
| B0 ⚠ | 29 | -0.342 | 0.7325 | -0.303 | 0.7620 | +0.0014 | 0 | 0 | +0.000 | noisy |
| B1 | 34 | -2.274 | 0.0230 | -1.833 | 0.0668 | -0.4161 | 0 | 2 | -0.000 | promising |
| B2 | 185 | 0.037 | 0.9708 | 0.035 | 0.9720 | -0.0508 | 0 | 0 | -0.000 | tied |
| B3 | 493 | -8.185 | 0.0000 | -1.567 | 0.1172 | -0.1852 | 0 | 5 | -0.000 | promising |
| B4 | 677 | -5.545 | 0.0000 | -1.649 | 0.0992 | -0.1079 | 0 | 5 | -0.000 | promising |
| B5 | 705 | -4.857 | 0.0000 | -0.838 | 0.4022 | -0.0788 | 0 | 0 | -0.000 | tied |
| B6 | 809 | 6.845 | 0.0000 | 0.689 | 0.4907 | +0.0526 | 0 | 0 | +0.000 | tied |
| B7 | 67 | -4.144 | 0.0000 | -1.000 | 0.3174 | -0.0899 | 0 | 0 | -0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

---
Wall time: 39.40s (8300 pair rows scored × 2 bakes across 3 corpora; 500 bootstrap resamples × bands).
