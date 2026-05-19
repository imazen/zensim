# bake_compare — decisive A vs B verdict (§ A.9)

- **A**: `/home/lilith/work/zen/zensim/zensim/weights/v_compression_persample_2026-05-18.bin` (label: `v_compression_persample_2026-05-18`)
- **B**: `/home/lilith/work/zen/zensim--v9-ship/zensim/weights/v_tuner_v9_2026-05-20.bin` (label: `v_tuner_v9_2026-05-20`)
- Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features`
- Bands: `10-band`  Bootstrap resamples: `500`  Seed: `42`

Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.

## Cross-corpus aggregate summary

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.2332 | 0.5274 | 0.963 | 0.847 | 0.2791 | 0.6150 | -28.837 | -12.011 | -0.000 | B>>A |
| KonJND-1k (full) | 1008 | 0.4666 | 0.0574 | 0.893 | 0.996 | 0.5150 | 0.0601 | 21.582 | 5.619 | +17.985 | A>>B |
| TID2013 | 3000 | 0.1473 | 0.3430 | 0.984 | 0.951 | 0.1432 | 0.3902 | -9.765 | -1.682 | -0.000 | promising |

## Decisive-band totals across all (corpus × band) cells

- **ADecisivelyBeatsB**: 1 cells
- **BDecisivelyBeatsA**: 1 cells
- **PromisingNotDecisive**: 7 cells
- **Tied**: 8 cells
- **Noisy** (n < 30, no decision): 1 cells

**Overall winner across decisive cells: `tie`** (1 A wins vs 1 B wins)

## CID22 (n=4292)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_compression_persample_2026-05-18 | 4292 | 0.2332 | 0.2707 | 0.1618 | 0.0440 | 0.2791 | 0.963 |
| B: v_tuner_v9_2026-05-20 | 4292 | 0.5274 | 0.5319 | 0.3713 | 0.0485 | 0.6150 | 0.847 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 4292 | 0.3452 | -28.837 | 0.0000 | -12.011 | 0.0000 | -0.3359 | 0 | 5 | -0.000 | B>>A |

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
| B3 | 57 | -0.650 | 0.5160 | -0.156 | 0.8761 | -0.0142 | 0 | 0 | -0.000 | tied |
| B4 | 266 | -0.448 | 0.6540 | -0.134 | 0.8938 | -0.0377 | 0 | 0 | -0.000 | tied |
| B5 | 615 | -2.679 | 0.0074 | -0.323 | 0.7470 | -0.0840 | 0 | 0 | -0.000 | tied |
| B6 | 836 | -4.217 | 0.0000 | -0.497 | 0.6195 | -0.1122 | 0 | 3 | -0.000 | promising |
| B7 | 1092 | -3.802 | 0.0001 | -0.367 | 0.7140 | -0.0920 | 0 | 3 | -0.000 | promising |
| B8 | 1382 | -4.821 | 0.0000 | -0.682 | 0.4955 | -0.1122 | 0 | 5 | -0.000 | promising |
| B9 | 43 | -1.231 | 0.2184 | -0.458 | 0.6472 | -0.2575 | 0 | 0 | -0.000 | tied |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

## KonJND-1k (full) (n=1008)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_compression_persample_2026-05-18 | 1008 | 0.4666 | 0.4504 | 0.3111 | 0.0238 | 0.5150 | 0.893 |
| B: v_tuner_v9_2026-05-20 | 1008 | 0.0574 | 0.0934 | 0.0393 | 0.0010 | 0.0601 | 0.996 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 1008 | 0.4186 | 21.582 | 0.0000 | 5.619 | 0.0000 | +0.4549 | 5 | 0 | +17.985 | A>>B |

_DecScore cutoff for decisive: |DecScore| > 7.84. h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B count panel stats (of 6) whose bootstrap CI excludes 0 in that bake's favor; the rule needs ≥4 in the winner's favor._

_Per-band breakdown skipped for KonJND-1k (full) — corpus uses a JND step grid (AIC-3) or raw threshold scale (KonJND) that doesn't partition cleanly into the CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above is the load-bearing read on this corpus._

## TID2013 (n=3000)

### Aggregate Mohammadi panel — A vs B

| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| A: v_compression_persample_2026-05-18 | 3000 | 0.1473 | 0.1758 | 0.1032 | 0.0363 | 0.1432 | 0.984 |
| B: v_tuner_v9_2026-05-20 | 3000 | 0.3430 | 0.3078 | 0.2365 | 0.0277 | 0.3902 | 0.951 |

### Aggregate MRR + decisive rule

| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |
|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|
| 3000 | -0.0895 | -9.765 | 0.0000 | -1.682 | 0.0926 | -0.2470 | 0 | 6 | -0.000 | promising |

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
| B0 ⚠ | 29 | -0.211 | 0.8331 | -0.086 | 0.9317 | -0.0159 | 0 | 0 | -0.000 | noisy |
| B1 | 34 | -1.668 | 0.0954 | -1.023 | 0.3063 | -0.1977 | 0 | 0 | -0.000 | tied |
| B2 | 185 | 0.747 | 0.4553 | 0.316 | 0.7516 | +0.0468 | 0 | 0 | +0.000 | tied |
| B3 | 493 | 0.674 | 0.5003 | -0.160 | 0.8729 | +0.0450 | 0 | 0 | -0.000 | tied |
| B4 | 677 | 0.159 | 0.8739 | 0.035 | 0.9717 | +0.0250 | 0 | 1 | +0.000 | promising |
| B5 | 705 | -0.815 | 0.4152 | 0.030 | 0.9759 | -0.0409 | 0 | 1 | +0.000 | promising |
| B6 | 809 | -0.574 | 0.5656 | 0.070 | 0.9442 | -0.0437 | 0 | 1 | +0.000 | promising |
| B7 | 67 | 1.403 | 0.1607 | 1.187 | 0.2352 | +0.0177 | 0 | 0 | +0.000 | tied |
| B8 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |
| B9 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |

_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._

---
Wall time: 30.96s (8300 pair rows scored × 2 bakes across 3 corpora; 500 bootstrap resamples × bands).
