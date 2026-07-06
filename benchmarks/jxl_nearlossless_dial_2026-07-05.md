# zenjxl near-lossless sweep → B dial-top characterization (2026-07-05)

_AI-authored (Claude). Numbers are measured, not estimated. Raw data under
`/mnt/v/output/zensim-jxl-nearlossless/`._

## Motivation

B's dense-dial (`b_sdr_linear_cid80_dense_dial_2026-07-05.bin`, sha `b78adb15`)
had its **95–100 segment fit on a 600-row multiband-anchor extrapolation**, never
on real near-lossless codec output. The 5.7M picker sweeps stop at q90–95, so
pred_b topped at ~91–95 there with 0% of cells reaching 95. To calibrate the
dial-top on real data we swept genuine near-lossless zenjxl.

## Part 1 — zenjxl distance curve + a real encoder/decoder bug (issue #18)

`zenmetrics sweep --codec zenjxl --knob-grid '{"distance":[…]}'`, decoded via
`zenjxl-decoder`, scored with ssim2 (mean over 4 diverse sRGB refs; identical at
generic_quality q=90 and q=99):

| distance | ssim2 | file KB | note |
|--:|--:|--:|---|
| 0.01 | 33.94 | ~730 | **BROKEN** |
| 0.02 | 33.95 | 594 | **BROKEN** (largest file, worst quality) |
| 0.03 | 96.02 | 518 | lossy ceiling |
| 0.04 | 95.77 | 465 | |
| 0.05 | 95.65 | 427 | |
| 0.10 | 94.95 | 319 | |
| 0.20 | 94.06 | 235 | |
| 0.50 | 92.49 | 161 | |
| 1.00 | 87.50 | 125 | |

**zenjxl round-trip is broken at distance ≤ 0.02**: it spends the *most* bits yet
produces ssim2 ~34, a sharp cliff below 0.03. Filed as
[imazen/zenjxl#18](https://github.com/imazen/zenjxl/issues/18); fix in flight
(encoder-vs-decoder disambiguation via jxl-oxide reference decode).

**Consequence:** lossy zenjxl tops at ssim2 ~96 (distance 0.03); ssim2 97–100 is
reachable only via true lossless (=100 via `is_identical`) **until #18 is fixed**.

## Part 2 — B dial gap on real near-lossless (2200 cells: 200 refs × 11 distances)

Forwarded B and A over the stored 372-feature vectors (`rescore_parquet`,
bit-exact), joined to the sweep's ssim2:

| dist | n | ssim2 | pred_b | pred_a | B−ssim2 | A−ssim2 |
|--:|--:|--:|--:|--:|--:|--:|
| 0.03 | 200 | 95.58 | 91.39 | 95.37 | **−4.19** | −0.21 |
| 0.04 | 200 | 95.44 | 91.46 | 95.24 | −3.99 | −0.21 |
| 0.05 | 200 | 95.24 | 91.42 | 95.08 | −3.82 | −0.16 |
| 0.07 | 200 | 94.97 | 91.38 | 94.83 | −3.59 | −0.14 |
| 0.10 | 200 | 94.68 | 91.33 | 94.52 | −3.35 | −0.16 |
| 0.15 | 200 | 94.29 | 91.26 | 94.11 | −3.03 | −0.17 |
| 0.20 | 200 | 94.00 | 91.22 | 93.83 | −2.77 | −0.17 |
| 0.30 | 200 | 93.45 | 91.14 | 93.31 | −2.31 | −0.14 |
| 0.50 | 200 | 92.39 | 90.91 | 92.49 | −1.47 | +0.10 |
| 0.70 | 200 | 91.19 | 89.64 | 91.35 | −1.55 | +0.16 |
| 1.00 | 200 | 89.88 | 88.24 | 90.30 | −1.65 | +0.42 |

**Findings:**

1. **A is essentially ssim2-calibrated at near-lossless** (A−ssim2 within ±0.42
   across the whole range).
2. **B under-scores by 1.5–4.2 points, worst at the top** (distance 0.03:
   −4.19). In the ssim2≥93 zone (n=1488): pred_b mean **91.36** vs ssim2 **95.00**
   — B is **3.6 points low**. B reaches 95+ in only **5 of 2200 cells**
   (pred_b max 96.43).
3. This is a **dial/calibration defect, not a rank defect**: pred_b is monotonic
   in distance (91.39→88.24), just compressed into an 88–91 band instead of
   90–96. A rank-invariant output-spline refit stretches it back.

## Plan

1. **[in flight]** Fix zenjxl #18 → unlocks distance 0.01–0.02 (the true ssim2
   96–100 lossy top), or explicitly routes sub-floor distance to lossless.
2. **Re-sweep** including the now-working near-lossless distances → complete real
   curve to ssim2 ~99.
3. **Refit B's dial-top** (rank-invariant PCHIP/concave-saturation spline) on the
   real `(raw_b, ssim2)` near-lossless pairs, replacing the 600-row extrapolation,
   so near-lossless resolves toward 95–100 instead of piling at 91.
4. Re-validate via `bake_verdict` — rank panel unchanged (spline is monotone),
   dial-top corrected, G-RANGE tail gate still PASS.

## Data

- Features (2200×377): `/mnt/v/output/zensim-jxl-nearlossless/full/features.parquet`
- ssim2 + bytes: `/mnt/v/output/zensim-jxl-nearlossless/full/pareto.tsv`
- B/A forwards: `full/nl_b.parquet`, `full/nl_a.parquet`
- Distance-curve smoke: `/mnt/v/output/zensim-jxl-nearlossless/smoke/`
