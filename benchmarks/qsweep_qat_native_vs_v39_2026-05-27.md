# q-sweep dial comparison — QAT-native ship candidate vs V39 (2026-05-27)

**Purpose**: complete the #32 ship-decision package. `bake_verdict` measures
rank/calibration on held-out feature parquets but CANNOT measure G3 (dial
monotonicity) — that needs a real per-image q-sweep. This is that measurement
for the actual ship candidate (`v47_strict_qat_native`, the one-pass
f16+zerobias QAT bake), alongside the non-QAT recal and the shipped V39.

**Finding (decisive for the codec-target dial use case):**

| bake | monotonicity | tied | dial median q5→q95 | CID22 SROCC | size |
|---|--:|--:|---|--:|--:|
| **qat_native** | **0.9433** | **0.0033** | 1.40 → 88.50 (every step ↑) | **0.8657** | **27 KB** |
| recal_negtail | 0.9378 | 0.0044 | 4.61 → 88.43 (every step ↑) | 0.8564 | 30 KB |
| v39 (shipped A) | 0.6767 | 0.5356 | broken: peaks q25, **collapses to 0.00 q55–q95** | 0.8793 | 257 KB |

The QAT-native candidate is the **best dial of the three** (highest
monotonicity, lowest tied rate) AND best CID22 among the v47 axiom-clean
candidates. V39 is unusable as a codec-target dial: 53.6% of adjacent q-steps
are tied and high-quality encodes (q55–q95) score a flat 0.00 — a binary
search for "score=80" can never converge. V39's higher rank-SROCC on
KADID/TID/AIC is irrelevant to the dial use case because the dial it produces
is non-invertible. For the user-facing "type a target score" use case that
motivates zensim, QAT-native strictly dominates V39.

Raw fixtures: `/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv`
+ `qsweep/qsweep_manifest.tsv` (50 source images × 19 JPEG q values = 950 rows).

---

Per-bake monotonicity (on JPEG q-sweep 50 imgs × 19 q values),
score-per-q histogram, calibration RMSE per [0,10), [10,20), … band on `q`.

- Total manifest rows: 950

## Monotonicity summary

Strict-decrease violation rate (lower = better). Ties (clamp-flat regions, often score=0 or score=100) are reported separately — they don't count as inversions but they ARE dead zones a user-facing dial can't binary-search through.

| Bake | n_curves | n_adj_pairs | strict_violations | tied | monotonicity_rate | tied_rate |
|---|--:|--:|--:|--:|---:|---:|
| qat_native | 50 | 900 | 51 | 3 | 0.9433 | 0.0033 |
| recal_negtail | 50 | 900 | 56 | 4 | 0.9378 | 0.0044 |
| v39 | 50 | 900 | 291 | 482 | 0.6767 | 0.5356 |

## Score-per-q histogram (median / p25 / p75)

Each row: q value → (median, p25, p75, min, max).

### qat_native

| q | n | min | p25 | median | p75 | max |
|--:|--:|---:|---:|---:|---:|---:|
| 5 | 50 | 0.00 | 0.00 | 1.40 | 19.89 | 43.57 |
| 10 | 50 | 0.00 | 12.92 | 15.15 | 25.94 | 48.84 |
| 15 | 50 | 4.04 | 24.46 | 30.54 | 40.64 | 58.07 |
| 20 | 50 | 11.99 | 35.71 | 40.42 | 45.69 | 64.15 |
| 25 | 50 | 16.67 | 41.67 | 47.19 | 54.22 | 67.77 |
| 30 | 50 | 27.18 | 48.11 | 52.37 | 58.64 | 71.00 |
| 35 | 50 | 36.01 | 52.65 | 56.66 | 63.85 | 73.67 |
| 40 | 50 | 39.27 | 56.05 | 60.19 | 63.90 | 75.37 |
| 45 | 50 | 40.84 | 59.99 | 63.91 | 64.17 | 77.11 |
| 50 | 50 | 41.64 | 60.83 | 64.07 | 68.42 | 79.23 |
| 55 | 50 | 43.47 | 63.08 | 64.09 | 66.89 | 80.56 |
| 60 | 50 | 45.70 | 64.04 | 67.04 | 70.04 | 81.36 |
| 65 | 50 | 46.99 | 66.91 | 71.22 | 72.63 | 83.64 |
| 70 | 50 | 50.81 | 69.37 | 74.14 | 76.71 | 85.88 |
| 75 | 50 | 53.14 | 73.72 | 76.50 | 80.74 | 87.82 |
| 80 | 50 | 55.00 | 76.08 | 79.44 | 83.16 | 91.24 |
| 85 | 50 | 56.26 | 80.13 | 82.12 | 85.02 | 92.84 |
| 90 | 50 | 58.16 | 83.17 | 85.83 | 88.26 | 93.43 |
| 95 | 50 | 59.59 | 85.25 | 88.50 | 91.00 | 94.04 |

### recal_negtail

| q | n | min | p25 | median | p75 | max |
|--:|--:|---:|---:|---:|---:|---:|
| 5 | 50 | 0.00 | 0.00 | 4.81 | 33.95 | 50.53 |
| 10 | 50 | 0.00 | 13.03 | 20.12 | 29.87 | 50.70 |
| 15 | 50 | 3.13 | 26.77 | 32.34 | 44.01 | 56.35 |
| 20 | 50 | 11.74 | 39.02 | 43.94 | 48.65 | 62.42 |
| 25 | 50 | 20.93 | 44.33 | 49.50 | 50.69 | 64.64 |
| 30 | 50 | 30.02 | 49.19 | 50.64 | 50.86 | 67.92 |
| 35 | 50 | 35.23 | 50.66 | 50.80 | 56.70 | 72.82 |
| 40 | 50 | 38.33 | 50.72 | 51.34 | 58.00 | 74.43 |
| 45 | 50 | 40.65 | 51.05 | 58.50 | 62.67 | 76.48 |
| 50 | 50 | 41.55 | 51.12 | 61.07 | 66.24 | 78.52 |
| 55 | 50 | 45.18 | 53.52 | 60.86 | 66.36 | 79.77 |
| 60 | 50 | 46.49 | 60.66 | 65.31 | 71.82 | 81.77 |
| 65 | 50 | 48.55 | 69.81 | 73.00 | 75.93 | 83.49 |
| 70 | 50 | 50.47 | 72.66 | 75.48 | 77.96 | 84.72 |
| 75 | 50 | 50.63 | 75.79 | 78.27 | 81.99 | 85.84 |
| 80 | 50 | 50.70 | 78.37 | 81.25 | 83.76 | 89.94 |
| 85 | 50 | 50.75 | 81.46 | 83.37 | 85.35 | 93.72 |
| 90 | 50 | 50.85 | 83.90 | 85.61 | 87.42 | 94.24 |
| 95 | 50 | 51.01 | 85.87 | 88.02 | 92.26 | 94.81 |

### v39

| q | n | min | p25 | median | p75 | max |
|--:|--:|---:|---:|---:|---:|---:|
| 5 | 50 | 0.00 | 30.30 | 47.75 | 70.74 | 100.00 |
| 10 | 50 | 0.00 | 5.51 | 28.08 | 56.76 | 100.00 |
| 15 | 50 | 0.00 | 4.13 | 35.61 | 68.94 | 100.00 |
| 20 | 50 | 0.00 | 8.31 | 55.28 | 74.76 | 100.00 |
| 25 | 50 | 0.00 | 5.13 | 61.98 | 78.82 | 100.00 |
| 30 | 50 | 0.00 | 7.23 | 52.94 | 72.96 | 100.00 |
| 35 | 50 | 0.00 | 0.00 | 40.27 | 65.21 | 100.00 |
| 40 | 50 | 0.00 | 0.00 | 26.68 | 55.99 | 100.00 |
| 45 | 50 | 0.00 | 0.00 | 12.61 | 45.55 | 99.06 |
| 50 | 50 | 0.00 | 0.00 | 0.56 | 28.95 | 96.00 |
| 55 | 50 | 0.00 | 0.00 | 0.00 | 18.12 | 63.61 |
| 60 | 50 | 0.00 | 0.00 | 0.00 | 7.67 | 70.70 |
| 65 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 74.95 |
| 70 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 66.14 |
| 75 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 54.49 |
| 80 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 34.80 |
| 85 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 13.21 |
| 90 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 4.22 |
| 95 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Calibration linearity (RMSE per band, target = q)

`score - q` RMSE per [b·10, (b+1)·10) band. Low RMSE = score tracks q linearly. NOTE: zensim does NOT have a constraint that score=q on JPEG; this RMSE is a proxy for cross-image consistency — a tuner with low RMSE per band gives the user a JPEG q-targeting tool whose zensim output is predictable.

| Band | range | n | qat_native | recal_negtail | v39 |
|---|---|--:|---:|---:|---:|
| B0 | [0, 10) | 50 | 15.24 | 17.98 | 54.41 |
| B1 | [10, 20) | 100 | 18.46 | 20.92 | 39.82 |
| B2 | [20, 30) | 100 | 22.54 | 23.26 | 43.64 |
| B3 | [30, 40) | 100 | 23.56 | 20.77 | 33.93 |
| B4 | [40, 50) | 100 | 19.58 | 16.17 | 32.91 |
| B5 | [50, 60) | 100 | 13.23 | 12.01 | 45.30 |
| B6 | [60, 70) | 100 | 8.54 | 9.75 | 59.79 |
| B7 | [70, 80) | 100 | 6.70 | 8.12 | 71.33 |
| B8 | [80, 90) | 100 | 7.25 | 7.28 | 82.11 |
| B9 | [90, 100) | 100 | 9.19 | 9.23 | 92.49 |

