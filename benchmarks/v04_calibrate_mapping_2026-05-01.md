# V0_4 score-mapping calibration

Source: `/mnt/v/output/zensim/profile_compat_v02_v04_20260501_v2.csv`
Pairs: 17417

Target: fit `(a, b)` in `score = clamp(100 - a · d^b, -100, 100)` so V0_4
score distribution matches V0_2's across step-5 V0_2 score buckets.

Current V0_4 profile params: `(a=18, b=0.7)` (inherited verbatim from V0_2).

## Fit 1 — per-pair MSE, bucket-equal weighting

Three-parameter fit: `score = clamp(100 - a · max(0, d - offset)^b, -100, 100)`.
Offset is required because V0_4's distance distribution is centered around d ≈ -2
(RankNet preserves rank but not the absolute zero-point), so the existing two-param
shape can't separate any pair below the median from 'perfect quality'.

Optimal: **a = 5.0000, b = 1.2100, offset = -10.8686** (weighted RMSE = 27.535)

Baseline (a=18, b=0.7, offset=0): weighted RMSE = 78.688 (calibration's improvement: 2.9×)

### Per-bucket residuals (V0_4_score − V0_2_score, median Δ in bucket)

| V0_2 bucket | n | mean Δ baseline | mean Δ calibrated | RMSE baseline | RMSE calibrated |
|:--:|--:|:--:|:--:|:--:|:--:|
| [-100, -95) | 658 | +122.86 | +28.43 | 124.10 | 35.51 |
| [-95, -90) | 62 | +126.32 | +35.32 | 127.92 | 44.08 |
| [-90, -85) | 83 | +127.82 | +39.43 | 128.62 | 43.97 |
| [-85, -80) | 103 | +119.57 | +28.73 | 120.95 | 38.00 |
| [-80, -75) | 100 | +116.52 | +26.49 | 117.79 | 36.35 |
| [-75, -70) | 123 | +110.11 | +19.90 | 111.43 | 31.54 |
| [-70, -65) | 143 | +106.58 | +15.96 | 108.22 | 31.19 |
| [-65, -60) | 148 | +105.01 | +16.14 | 106.87 | 32.19 |
| [-60, -55) | 176 | +98.93 | +9.74 | 100.78 | 29.16 |
| [-55, -50) | 172 | +99.57 | +12.26 | 101.58 | 30.83 |
| [-50, -45) | 188 | +94.53 | +7.20 | 96.22 | 25.23 |
| [-45, -40) | 187 | +93.51 | +8.05 | 95.43 | 27.69 |
| [-40, -35) | 198 | +89.69 | +4.11 | 91.97 | 27.55 |
| [-35, -30) | 224 | +87.64 | +2.92 | 89.58 | 24.38 |
| [-30, -25) | 216 | +83.83 | -0.18 | 85.94 | 25.57 |
| [-25, -20) | 274 | +81.88 | -0.75 | 83.89 | 25.29 |
| [-20, -15) | 273 | +78.65 | -3.35 | 80.83 | 26.14 |
| [-15, -10) | 315 | +77.17 | -4.35 | 79.28 | 24.42 |
| [-10, -5) | 349 | +75.40 | -5.59 | 77.92 | 26.35 |
| [-5, 0) | 370 | +71.87 | -9.25 | 74.20 | 25.31 |
| [0, 5) | 394 | +70.86 | -10.00 | 73.39 | 25.67 |
| [5, 10) | 372 | +65.55 | -15.21 | 68.37 | 28.95 |
| [10, 15) | 380 | +66.66 | -13.86 | 68.92 | 25.82 |
| [15, 20) | 404 | +64.80 | -15.68 | 66.95 | 25.70 |
| [20, 25) | 425 | +62.32 | -15.30 | 64.62 | 28.91 |
| [25, 30) | 426 | +60.47 | -16.36 | 62.27 | 28.47 |
| [30, 35) | 451 | +57.25 | -18.76 | 59.03 | 30.22 |
| [35, 40) | 524 | +54.98 | -16.71 | 56.39 | 30.42 |
| [40, 45) | 555 | +52.84 | -18.27 | 53.82 | 28.94 |
| [45, 50) | 567 | +48.34 | -21.19 | 49.33 | 30.05 |
| [50, 55) | 637 | +44.89 | -20.55 | 45.55 | 28.97 |
| [55, 60) | 748 | +41.05 | -21.48 | 41.36 | 28.19 |
| [60, 65) | 894 | +36.68 | -18.15 | 36.89 | 24.42 |
| [65, 70) | 1066 | +32.20 | -13.98 | 32.27 | 20.53 |
| [70, 75) | 1300 | +27.34 | -7.29 | 27.39 | 16.28 |
| [75, 80) | 1410 | +22.48 | +3.78 | 22.53 | 14.68 |
| [80, 85) | 1376 | +17.67 | +11.61 | 17.73 | 16.41 |
| [85, 90) | 631 | +12.95 | +11.61 | 13.03 | 13.02 |
| [90, 95) | 170 | +8.17 | +8.17 | 8.25 | 8.25 |
| [95, 100) | 5 | +4.85 | +4.85 | 4.85 | 4.85 |
| V0_2 = -100 (clamped) | 599 | +122.45 | +27.87 | 123.65 | 34.69 |

## Fit 2 — equipercentile (CDF) match

Sort V0_2 distances and V0_4 distances independently, pair them at matching ranks,
then fit (a, b) on the synthetic (d_v04_p, V0_2_score_p) sequence. Aligns the
MARGINAL distribution of scores, ignoring pair-level correspondence.

Optimal: **a = 0.4000, b = 2.0400, offset = -12.9949** (weighted RMSE = 4.788)

### Score CDF — V0_2 vs V0_4 baseline vs V0_4 calibrated, percentile-matched

| pctile | V0_2 score | V0_4 baseline (a=18, b=0.7) | V0_4 calibrated |
|--:|:--:|:--:|:--:|
|   1 | -100.00 | 6.73 | -100.00 |
|   5 | -81.75 | 29.28 | -81.44 |
|  10 | -50.66 | 43.34 | -47.81 |
|  20 | -12.29 | 61.01 | -14.58 |
|  30 | 11.43 | 77.40 | 7.99 |
|  40 | 32.41 | 100.00 | 29.38 |
|  50 | 48.70 | 100.00 | 47.25 |
|  60 | 61.27 | 100.00 | 63.91 |
|  70 | 69.93 | 100.00 | 76.14 |
|  80 | 76.52 | 100.00 | 87.37 |
|  90 | 82.48 | 100.00 | 99.32 |
|  95 | 86.38 | 100.00 | 100.00 |
|  99 | 100.00 | 100.00 | 100.00 |

## Fit 3 — polynomial in (d − μ) / σ

Drops the power-law assumption. Fits `score = Σ_k c_k · x^k` where x = (d − μ)/σ
and (μ, σ) are the V0_4 distance distribution's mean and std. Polynomial coefficients
clamp the output to [-100, 100] post-eval. Weighted least squares against per-pair
V0_2 score targets, weights = 1/bucket_count (CLAUDE.md sweep rule).

Normalization: μ = -2.7021, σ = 6.8172

Fit targets are 21 CDF anchors (V0_2 score percentiles 0, 5, ..., 100). Fitting on
per-pair data instead pulls the polynomial non-monotonic because Kendall τ ≈ 0.86
means ~14%% of pair orderings disagree — that's irreducible noise that lower-degree
shapes weather but high-degree shapes overfit into wiggles.

| degree | RMSE on CDF anchors | RMSE on per-pair (bucket-weighted) | monotonic |
|:--:|:--:|:--:|:--:|
| 3 | 13.011 | 29.203 | **no** |
| 4 | 4.263 | 28.285 | **no** |
| 5 | 0.761 | 28.822 | **no** |
| 6 | 0.760 | 28.826 | **no** |
| 7 | 0.630 | 28.756 | **no** |

Chosen: **degree 7** (CDF-anchor RMSE = 0.630,
per-pair RMSE = 28.756, monotonic = no).

### Per-bucket residuals — power-law-3param vs polynomial deg-7

| V0_2 bucket | n | RMSE pwr3 (a=5.00, b=1.21, off=-10.87) | RMSE poly7 | mean Δ poly |
|:--:|--:|:--:|:--:|:--:|
| [-100, -95) | 658 | 35.51 | 27.68 | +17.75 |
| [-95, -90) | 62 | 44.08 | 41.84 | +29.18 |
| [-90, -85) | 83 | 43.97 | 41.25 | +33.53 |
| [-85, -80) | 103 | 38.00 | 37.11 | +22.47 |
| [-80, -75) | 100 | 36.35 | 37.55 | +21.10 |
| [-75, -70) | 123 | 31.54 | 33.32 | +13.93 |
| [-70, -65) | 143 | 31.19 | 35.34 | +10.48 |
| [-65, -60) | 148 | 32.19 | 37.37 | +12.45 |
| [-60, -55) | 176 | 29.16 | 35.05 | +5.36 |
| [-55, -50) | 172 | 30.83 | 38.37 | +10.53 |
| [-50, -45) | 188 | 25.23 | 33.71 | +4.59 |
| [-45, -40) | 187 | 27.69 | 37.00 | +7.17 |
| [-40, -35) | 198 | 27.55 | 37.72 | +4.34 |
| [-35, -30) | 224 | 24.38 | 34.77 | +4.72 |
| [-30, -25) | 216 | 25.57 | 35.85 | +1.80 |
| [-25, -20) | 274 | 25.29 | 34.36 | +2.91 |
| [-20, -15) | 273 | 26.14 | 34.68 | +1.21 |
| [-15, -10) | 315 | 24.42 | 31.93 | +2.14 |
| [-10, -5) | 349 | 26.35 | 34.24 | +2.04 |
| [-5, 0) | 370 | 25.31 | 31.59 | -0.52 |
| [0, 5) | 394 | 25.67 | 31.62 | +0.29 |
| [5, 10) | 372 | 28.95 | 33.54 | -5.24 |
| [10, 15) | 380 | 25.82 | 28.18 | -1.14 |
| [15, 20) | 404 | 25.70 | 26.35 | -1.77 |
| [20, 25) | 425 | 28.91 | 28.96 | -1.45 |
| [25, 30) | 426 | 28.47 | 25.51 | -1.56 |
| [30, 35) | 451 | 30.22 | 25.72 | -3.77 |
| [35, 40) | 524 | 30.42 | 24.75 | -2.01 |
| [40, 45) | 555 | 28.94 | 21.37 | -2.68 |
| [45, 50) | 567 | 30.05 | 21.05 | -5.37 |
| [50, 55) | 637 | 28.97 | 18.73 | -5.04 |
| [55, 60) | 748 | 28.19 | 15.97 | -6.15 |
| [60, 65) | 894 | 24.42 | 13.31 | -4.62 |
| [65, 70) | 1066 | 20.53 | 10.82 | -3.70 |
| [70, 75) | 1300 | 16.28 | 8.73 | -2.76 |
| [75, 80) | 1410 | 14.68 | 6.06 | -1.47 |
| [80, 85) | 1376 | 16.41 | 4.68 | -0.89 |
| [85, 90) | 631 | 13.02 | 3.58 | -2.00 |
| [90, 95) | 170 | 8.25 | 7.06 | -6.91 |
| [95, 100) | 5 | 4.85 | 8.35 | -8.32 |
| V0_2 = -100 (clamped) | 599 | 34.69 | 26.32 | +16.87 |

Polynomial coefficients (x = (d − μ)/σ; score = Σ c_k · x^k, then clamp):

- **deg 3**: `[40.9643, -57.7388, -6.0015, 2.5773]`
- **deg 4**: `[52.7588, -61.4137, -28.8331, 3.4833, 2.6301]`
- **deg 5**: `[53.9193, -51.3785, -32.8972, -4.9034, 3.2226, 0.8429]`
- **deg 6**: `[53.8955, -51.3524, -32.7839, -4.9418, 3.1600, 0.8478, 0.0058]`
- **deg 7**: `[54.1061, -49.8386, -34.1795, -8.1341, 4.2615, 2.2542, -0.1194, -0.1232]`

## Fit 4 — piecewise-linear CDF lookup

Build a table of (d, V0_2_target_score) anchors at percentiles 0, 5, 10, ..., 95, 100
of the V0_4 distance distribution. At lookup time, binary search for the bracketing
entries and linearly interpolate. Structurally guaranteed to match V0_2 score CDF
within the table's resolution. Storage: 21 × 2 = 42 floats.

Weighted RMSE (piecewise-21): 27.666

Anchor table (V0_4 distance → V0_2 score target):

| pctile | V0_4 distance | V0_2 score target |
|--:|--:|--:|
|   0 | -23.5738 | 100.00 |
|   5 | -15.1731 | 86.38 |
|  10 | -11.6992 | 82.48 |
|  15 | -9.1825 | 79.58 |
|  20 | -7.5634 | 76.52 |
|  25 | -6.5411 | 73.44 |
|  30 | -5.5749 | 69.93 |
|  35 | -4.7235 | 65.99 |
|  40 | -3.9069 | 61.27 |
|  45 | -2.9802 | 55.69 |
|  50 | -2.0475 | 48.70 |
|  55 | -1.1830 | 40.91 |
|  60 | -0.3648 | 32.41 |
|  65 | 0.5662 | 22.30 |
|  70 | 1.3841 | 11.43 |
|  75 | 2.2005 | -0.15 |
|  80 | 3.0170 | -12.29 |
|  85 | 3.8634 | -28.90 |
|  90 | 5.1458 | -50.66 |
|  95 | 7.0631 | -81.75 |
| 100 | 16.7930 | -100.00 |

## Recommendation

| approach | params | RMSE on CDF anchors | RMSE per-pair | monotonic | runtime |
|---|---|--:|--:|---|---|
| pwr2 baseline (a=18, b=0.7) | 2 | n/a | 78.69 | yes | 1× pow |
| pwr3 + offset (a, b, off) | 3 | n/a | 27.54 | yes | 1× pow |
| poly deg-3 | 4 | 13.01 | 29.20 | **no** | 3× FMA |
| poly deg-4 | 5 | 4.26 | 28.29 | **no** | 4× FMA |
| poly deg-5 | 6 | 0.76 | 28.82 | **no** | 5× FMA |
| poly deg-6 | 7 | 0.76 | 28.83 | **no** | 6× FMA |
| poly deg-7 | 8 | 0.63 | 28.76 | **no** | 7× FMA |
| **piecewise-21** | 42 | **0** (by construction) | 27.67 | yes | binsearch + lerp |

**Per-pair RMSE has a noise floor ≈ 27** that no monotone scalar mapping can
go below — V0_2 and V0_4 disagree on ~14%% of pair orderings (Kendall τ ≈ 0.86),
and that's irreducible for any monotone f: d → score. Polynomial deg-7's CDF-anchor
RMSE = 0.63 looks great in isolation but the fit goes non-monotonic
at the steep CDF transition (Runge wiggles), which would corrupt rank order in
a regime where it currently holds.

**Recommended for V0_4: piecewise-linear with 21 CDF anchors.**
- Storage: 42 f64 (336 bytes).
- Runtime: 1 binary search (~5 comparisons across 21 anchors) + 1 lerp + 1 clamp.
- CDF match: exact within the table's resolution.
- Monotonic: yes, by construction (anchors sorted by V0_4 distance).
- Per-pair RMSE: 27.67, equal to the noise floor.

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
        (-23.5738, 100.0000),
        (-15.1731, 86.3832),
        (-11.6992, 82.4838),
        (-9.1825, 79.5798),
        (-7.5634, 76.5164),
        (-6.5411, 73.4357),
        (-5.5749, 69.9280),
        (-4.7235, 65.9895),
        (-3.9069, 61.2748),
        (-2.9802, 55.6898),
        (-2.0475, 48.6972),
        (-1.1830, 40.9081),
        (-0.3648, 32.4060),
        (0.5662, 22.3016),
        (1.3841, 11.4260),
        (2.2005, -0.1502),
        (3.0170, -12.2887),
        (3.8634, -28.9030),
        (5.1458, -50.6577),
        (7.0631, -81.7489),
        (16.7930, -100.0000),
    ],
},
```

Polynomial fits are recorded above for reference but not recommended — without a
monotonicity constraint they Runge-wiggle at the CDF transition, and with one
(e.g. monotone cubic spline / PCHIP) the implementation cost exceeds piecewise-linear
while the RMSE delta is in the per-pair noise floor.
