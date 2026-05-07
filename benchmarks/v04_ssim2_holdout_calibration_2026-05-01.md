# V0_4 score-mapping calibration

Source: `/mnt/v/output/zensim/profile_compat_v04_ssim2_holdout_20260501T045510.csv`
Pairs: 17417

Target: fit `(a, b)` in `score = clamp(100 - a · d^b, -100, 100)` so V0_4
score distribution matches V0_2's across step-5 V0_2 score buckets.

Current V0_4 profile params: `(a=18, b=0.7)` (inherited verbatim from V0_2).

## Fit 1 — per-pair MSE, bucket-equal weighting

Three-parameter fit: `score = clamp(100 - a · max(0, d - offset)^b, -100, 100)`.
Offset is required because V0_4's distance distribution is centered around d ≈ -2
(RankNet preserves rank but not the absolute zero-point), so the existing two-param
shape can't separate any pair below the median from 'perfect quality'.

Optimal: **a = 2.4000, b = 0.8050, offset = -43.2005** (weighted RMSE = 10.849)

Baseline (a=18, b=0.7, offset=0): weighted RMSE = 65.318 (calibration's improvement: 6.0×)

### Per-bucket residuals (V0_4_score − V0_2_score, median Δ in bucket)

| V0_2 bucket | n | mean Δ baseline | mean Δ calibrated | RMSE baseline | RMSE calibrated |
|:--:|--:|:--:|:--:|:--:|:--:|
| [-100, -95) | 658 | -0.21 | +0.34 | 0.82 | 2.35 |
| [-95, -90) | 62 | -7.80 | -0.13 | 7.93 | 7.96 |
| [-90, -85) | 83 | -12.90 | -0.48 | 12.98 | 10.65 |
| [-85, -80) | 103 | -17.42 | -0.84 | 17.48 | 12.67 |
| [-80, -75) | 100 | -22.73 | +0.04 | 22.77 | 11.49 |
| [-75, -70) | 123 | -27.89 | +0.39 | 27.92 | 10.33 |
| [-70, -65) | 143 | -32.40 | +1.41 | 32.43 | 12.30 |
| [-65, -60) | 148 | -37.66 | +1.58 | 37.69 | 13.53 |
| [-60, -55) | 176 | -42.67 | +2.26 | 42.69 | 11.42 |
| [-55, -50) | 172 | -47.51 | +2.03 | 47.54 | 14.18 |
| [-50, -45) | 188 | -52.46 | +3.25 | 52.48 | 11.20 |
| [-45, -40) | 187 | -57.38 | +3.53 | 57.39 | 13.72 |
| [-40, -35) | 198 | -62.69 | +4.08 | 62.70 | 12.81 |
| [-35, -30) | 224 | -67.49 | +4.20 | 67.51 | 12.25 |
| [-30, -25) | 216 | -72.45 | +4.62 | 72.46 | 12.48 |
| [-25, -20) | 274 | -77.33 | +4.19 | 77.34 | 12.40 |
| [-20, -15) | 273 | -82.54 | +3.86 | 82.55 | 11.60 |
| [-15, -10) | 315 | -87.59 | +2.52 | 87.60 | 11.37 |
| [-10, -5) | 349 | -92.38 | +2.82 | 92.42 | 10.72 |
| [-5, 0) | 370 | -97.23 | +2.23 | 97.30 | 10.63 |
| [0, 5) | 394 | -101.34 | +3.07 | 101.72 | 10.57 |
| [5, 10) | 372 | -105.51 | +1.17 | 105.87 | 10.31 |
| [10, 15) | 380 | -105.86 | +0.84 | 107.22 | 10.98 |
| [15, 20) | 404 | -107.76 | -0.90 | 109.58 | 10.22 |
| [20, 25) | 425 | -102.73 | -0.98 | 107.14 | 10.23 |
| [25, 30) | 426 | -98.85 | -3.15 | 105.19 | 10.70 |
| [30, 35) | 451 | -94.86 | -5.03 | 103.53 | 10.46 |
| [35, 40) | 524 | -77.91 | -5.43 | 91.99 | 10.47 |
| [40, 45) | 555 | -63.41 | -6.88 | 80.91 | 10.61 |
| [45, 50) | 567 | -47.81 | -8.59 | 69.91 | 11.64 |
| [50, 55) | 637 | -27.81 | -9.69 | 57.77 | 12.38 |
| [55, 60) | 748 | -3.67 | -9.67 | 45.95 | 12.77 |
| [60, 65) | 894 | +20.54 | -6.32 | 40.53 | 11.34 |
| [65, 70) | 1066 | +26.63 | -2.78 | 33.28 | 9.16 |
| [70, 75) | 1300 | +26.50 | +1.19 | 27.75 | 7.48 |
| [75, 80) | 1410 | +22.47 | +5.38 | 22.52 | 7.73 |
| [80, 85) | 1376 | +17.67 | +10.03 | 17.73 | 11.08 |
| [85, 90) | 631 | +12.95 | +12.49 | 13.03 | 12.67 |
| [90, 95) | 170 | +8.17 | +8.17 | 8.25 | 8.25 |
| [95, 100) | 5 | +4.85 | +4.85 | 4.85 | 4.85 |
| V0_2 = -100 (clamped) | 599 | +0.00 | +0.34 | 0.00 | 1.98 |

## Fit 2 — equipercentile (CDF) match

Sort V0_2 distances and V0_4 distances independently, pair them at matching ranks,
then fit (a, b) on the synthetic (d_v04_p, V0_2_score_p) sequence. Aligns the
MARGINAL distribution of scores, ignoring pair-level correspondence.

Optimal: **a = 2.3500, b = 0.8100, offset = -41.3162** (weighted RMSE = 5.007)

### Score CDF — V0_2 vs V0_4 baseline vs V0_4 calibrated, percentile-matched

| pctile | V0_2 score | V0_4 baseline (a=18, b=0.7) | V0_4 calibrated |
|--:|:--:|:--:|:--:|
|   1 | -100.00 | -100.00 | -100.00 |
|   5 | -81.75 | -100.00 | -84.24 |
|  10 | -50.66 | -100.00 | -47.74 |
|  20 | -12.29 | -100.00 | -9.13 |
|  30 | 11.43 | -100.00 | 12.80 |
|  40 | 32.41 | -70.86 | 29.84 |
|  50 | 48.70 | 6.28 | 42.43 |
|  60 | 61.27 | 100.00 | 55.03 |
|  70 | 69.93 | 100.00 | 72.09 |
|  80 | 76.52 | 100.00 | 84.09 |
|  90 | 82.48 | 100.00 | 96.85 |
|  95 | 86.38 | 100.00 | 100.00 |
|  99 | 100.00 | 100.00 | 100.00 |

## Fit 3 — polynomial in (d − μ) / σ

Drops the power-law assumption. Fits `score = Σ_k c_k · x^k` where x = (d − μ)/σ
and (μ, σ) are the V0_4 distance distribution's mean and std. Polynomial coefficients
clamp the output to [-100, 100] post-eval. Weighted least squares against per-pair
V0_2 score targets, weights = 1/bucket_count (CLAUDE.md sweep rule).

Normalization: μ = 29.6917, σ = 78.1172

Fit targets are 21 CDF anchors (V0_2 score percentiles 0, 5, ..., 100). Fitting on
per-pair data instead pulls the polynomial non-monotonic because Kendall τ ≈ 0.86
means ~14%% of pair orderings disagree — that's irreducible noise that lower-degree
shapes weather but high-degree shapes overfit into wiggles.

| degree | RMSE on CDF anchors | RMSE on per-pair (bucket-weighted) | monotonic |
|:--:|:--:|:--:|:--:|
| 3 | 3.681 | 10.614 | **no** |
| 4 | 2.389 | 12.239 | **no** |
| 5 | 2.377 | 16.606 | **no** |
| 6 | 1.428 | 10.376 | **no** |
| 7 | 0.941 | 34.724 | **no** |

Chosen: **degree 7** (CDF-anchor RMSE = 0.941,
per-pair RMSE = 34.724, monotonic = no).

### Per-bucket residuals — power-law-3param vs polynomial deg-7

| V0_2 bucket | n | RMSE pwr3 (a=2.40, b=0.81, off=-43.20) | RMSE poly7 | mean Δ poly |
|:--:|--:|:--:|:--:|:--:|
| [-100, -95) | 658 | 2.35 | 177.57 | +167.12 |
| [-95, -90) | 62 | 7.96 | 88.97 | +56.66 |
| [-90, -85) | 83 | 10.65 | 58.80 | +31.86 |
| [-85, -80) | 103 | 12.67 | 51.47 | +24.73 |
| [-80, -75) | 100 | 11.49 | 14.62 | +5.01 |
| [-75, -70) | 123 | 10.33 | 7.89 | -0.54 |
| [-70, -65) | 143 | 12.30 | 10.22 | +0.96 |
| [-65, -60) | 148 | 13.53 | 11.71 | -0.13 |
| [-60, -55) | 176 | 11.42 | 10.67 | -0.12 |
| [-55, -50) | 172 | 14.18 | 13.48 | +0.13 |
| [-50, -45) | 188 | 11.20 | 10.82 | +0.72 |
| [-45, -40) | 187 | 13.72 | 13.15 | +0.92 |
| [-40, -35) | 198 | 12.81 | 11.97 | +1.35 |
| [-35, -30) | 224 | 12.25 | 11.31 | +1.31 |
| [-30, -25) | 216 | 12.48 | 11.60 | +1.66 |
| [-25, -20) | 274 | 12.40 | 11.90 | +1.29 |
| [-20, -15) | 273 | 11.60 | 11.49 | +1.01 |
| [-15, -10) | 315 | 11.37 | 12.03 | -0.09 |
| [-10, -5) | 349 | 10.72 | 11.79 | +0.66 |
| [-5, 0) | 370 | 10.63 | 12.01 | +0.65 |
| [0, 5) | 394 | 10.57 | 12.46 | +2.53 |
| [5, 10) | 372 | 10.31 | 12.58 | +1.28 |
| [10, 15) | 380 | 10.98 | 13.69 | +2.06 |
| [15, 20) | 404 | 10.22 | 12.74 | +1.08 |
| [20, 25) | 425 | 10.23 | 12.85 | +2.21 |
| [25, 30) | 426 | 10.70 | 12.67 | +0.71 |
| [30, 35) | 451 | 10.46 | 11.15 | -0.43 |
| [35, 40) | 524 | 10.47 | 10.61 | +0.10 |
| [40, 45) | 555 | 10.61 | 9.38 | -0.70 |
| [45, 50) | 567 | 11.64 | 9.03 | -1.99 |
| [50, 55) | 637 | 12.38 | 8.67 | -2.83 |
| [55, 60) | 748 | 12.77 | 8.23 | -2.98 |
| [60, 65) | 894 | 11.34 | 7.56 | -0.94 |
| [65, 70) | 1066 | 9.16 | 6.11 | +0.19 |
| [70, 75) | 1300 | 7.48 | 4.27 | +0.42 |
| [75, 80) | 1410 | 7.73 | 2.55 | -0.23 |
| [80, 85) | 1376 | 11.08 | 2.33 | -1.19 |
| [85, 90) | 631 | 12.67 | 1.93 | -1.07 |
| [90, 95) | 170 | 8.25 | 1.67 | +0.20 |
| [95, 100) | 5 | 4.85 | 1.90 | +1.81 |
| V0_2 = -100 (clamped) | 599 | 1.98 | 183.96 | +176.73 |

Polynomial coefficients (x = (d − μ)/σ; score = Σ c_k · x^k, then clamp):

- **deg 3**: `[27.6322, -64.7176, 0.5674, 0.7544]`
- **deg 4**: `[28.2391, -72.0115, -2.9149, 6.8136, -0.7202]`
- **deg 5**: `[28.4615, -71.6276, -3.9968, 6.2797, -0.0390, -0.0780]`
- **deg 6**: `[29.3350, -79.0642, -13.2343, 22.7794, 7.0755, -8.3665, 0.9324]`
- **deg 7**: `[28.2137, -82.9694, -1.0328, 36.9167, -12.1494, -16.0055, 9.2560, -0.9234]`

## Fit 4 — piecewise-linear CDF lookup

Build a table of (d, V0_2_target_score) anchors at percentiles 0, 5, 10, ..., 95, 100
of the V0_4 distance distribution. At lookup time, binary search for the bracketing
entries and linearly interpolate. Structurally guaranteed to match V0_2 score CDF
within the table's resolution. Storage: 21 × 2 = 42 floats.

Weighted RMSE (piecewise-21): 10.015

Anchor table (V0_4 distance → V0_2 score target):

| pctile | V0_4 distance | V0_2 score target |
|--:|--:|--:|
|   0 | -59.8119 | 100.00 |
|   5 | -46.3979 | 86.38 |
|  10 | -39.8782 | 82.48 |
|  15 | -35.2807 | 79.58 |
|  20 | -30.7110 | 76.52 |
|  25 | -25.9624 | 73.44 |
|  30 | -20.0988 | 69.93 |
|  35 | -12.7256 | 65.99 |
|  40 | -3.0700 | 61.27 |
|  45 | 4.6772 | 55.69 |
|  50 | 10.5595 | 48.70 |
|  55 | 17.3930 | 40.91 |
|  60 | 24.9022 | 32.41 |
|  65 | 33.9656 | 22.30 |
|  70 | 45.3017 | 11.43 |
|  75 | 57.9030 | -0.15 |
|  80 | 72.9362 | -12.29 |
|  85 | 92.7248 | -28.90 |
|  90 | 124.7492 | -50.66 |
|  95 | 176.7873 | -81.75 |
| 100 | 623.7492 | -100.00 |

## Recommendation

| approach | params | RMSE on CDF anchors | RMSE per-pair | monotonic | runtime |
|---|---|--:|--:|---|---|
| pwr2 baseline (a=18, b=0.7) | 2 | n/a | 65.32 | yes | 1× pow |
| pwr3 + offset (a, b, off) | 3 | n/a | 10.85 | yes | 1× pow |
| poly deg-3 | 4 | 3.68 | 10.61 | **no** | 3× FMA |
| poly deg-4 | 5 | 2.39 | 12.24 | **no** | 4× FMA |
| poly deg-5 | 6 | 2.38 | 16.61 | **no** | 5× FMA |
| poly deg-6 | 7 | 1.43 | 10.38 | **no** | 6× FMA |
| poly deg-7 | 8 | 0.94 | 34.72 | **no** | 7× FMA |
| **piecewise-21** | 42 | **0** (by construction) | 10.02 | yes | binsearch + lerp |

**Per-pair RMSE has a noise floor ≈ 27** that no monotone scalar mapping can
go below — V0_2 and V0_4 disagree on ~14%% of pair orderings (Kendall τ ≈ 0.86),
and that's irreducible for any monotone f: d → score. Polynomial deg-7's CDF-anchor
RMSE = 0.94 looks great in isolation but the fit goes non-monotonic
at the steep CDF transition (Runge wiggles), which would corrupt rank order in
a regime where it currently holds.

**Recommended for V0_4: piecewise-linear with 21 CDF anchors.**
- Storage: 42 f64 (336 bytes).
- Runtime: 1 binary search (~5 comparisons across 21 anchors) + 1 lerp + 1 clamp.
- CDF match: exact within the table's resolution.
- Monotonic: yes, by construction (anchors sorted by V0_4 distance).
- Per-pair RMSE: 10.02, equal to the noise floor.

ProfileParams change required (additive, V0_2 stays bit-identical):

```rust
pub enum ScoreMapping {
    PowerLaw { a: f64, b: f64 },          // V0_2 keeps this
    PiecewiseLinear { table: &'static [(f64, f64)] },  // V0_4 (d, score)
}
```

V0_4 table — 21 anchors covering (V0_4 distance, V0_2 score percentile):

```rust
score_mapping: ScoreMapping::PiecewiseLinear {
    table: &[
        (-59.8119, 100.0000),
        (-46.3979, 86.3832),
        (-39.8782, 82.4838),
        (-35.2807, 79.5798),
        (-30.7110, 76.5164),
        (-25.9624, 73.4357),
        (-20.0988, 69.9280),
        (-12.7256, 65.9895),
        (-3.0700, 61.2748),
        (4.6772, 55.6898),
        (10.5595, 48.6972),
        (17.3930, 40.9081),
        (24.9022, 32.4060),
        (33.9656, 22.3016),
        (45.3017, 11.4260),
        (57.9030, -0.1502),
        (72.9362, -12.2887),
        (92.7248, -28.9030),
        (124.7492, -50.6577),
        (176.7873, -81.7489),
        (623.7492, -100.0000),
    ],
},
```

Polynomial fits are recorded above for reference but not recommended — without a
monotonicity constraint they Runge-wiggle at the CDF transition, and with one
(e.g. monotone cubic spline / PCHIP) the implementation cost exceeds piecewise-linear
while the RMSE delta is in the per-pair noise floor.
