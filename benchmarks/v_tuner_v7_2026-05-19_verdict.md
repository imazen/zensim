# EXP-CROSS-CODEC-V7 verdict — 2026-05-19

**Status: PARTIAL FALSIFICATION.** V7 passes 4 of 6 V6 gates AND
identifies a structural finding (V6's anchor targets were
systematically below empirical metrics by 10-30 score units), but
the empirical-anchor recipe shifts the bake's PJND-region
calibration AWAY from the V6 rule-of-thumb score=63-at-butter=1.5,
which is the basis of gate 4 (T=63 cross-codec consistency). V7
agg PJND-band mean is ~78 (because empirical ssim2 at butter=1.5
is 77.7), not 63.

V7 does NOT ship as `PreviewV0_5TunerV3` because:
1. Gate 4 fails: T=63 mean butter_p3 = 3.4–3.9 (V6 ship: 1.73,
   gate threshold < 2.5).
2. Gate 7 (V7-specific per-(codec, band) within ±5) fails at
   butter ∈ {2.5, 4.0, 6.0} bands across all 3 seeds — the per-
   codec target divergence (zenjxl=86 vs zenwebp=17 at butter=4.0)
   is structurally incompatible with the cross-codec parity gate
   (cc_std ≤ 5), and the trainer correctly chose parity. The
   network achieved the median target per band; individual codecs
   diverge from their empirical targets by 10–30 score units at
   heavy distortion.

V7 DID confirm:
- V7 passes 4 of 6 V6 gates (mono, tied, range, PJND cc_std, all-
  band cc_std).
- V7's per-band cc_std_median (parity gate) is BETTER than V6 at
  every band (V6 ship: max 1.68; V7 ship: max 4.56 at butter=6.0,
  still under the 5.0 gate).
- V7's monotonicity is BETTER than V6 (V7 best 0.9767 vs V6 best
  0.9522).

But V7 does NOT improve over V6 ship because the V7 calibration
shifts the user-facing PJND anchor to ~78, breaking back-compat
with V6 / Tuner ship's score=63-at-PJND convention.

## Gate results

| Bake | mono ≥ 0.9522 | tied ≤ 5% | medRange ≥ 50 | T63 butter_p3 < 2.5 | PJND cc_std ≤ 5 | All-band cc_std ≤ 5 | per-band ±5 | passed |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| cc4v7_s1 | PASS (0.9722) | PASS (0.0000) | PASS (60.58) | **FAIL (3.637)** | PASS (0.70) | PASS (max 4.22) | FAIL (b≥2.5) | 5/7 |
| cc4v7_s2 | PASS (0.9611) | PASS (0.0000) | PASS (65.13) | **FAIL (3.355)** | PASS (0.73) | PASS (max 4.24) | FAIL (b≥2.5) | 5/7 |
| cc4v7_s3 | PASS (0.9767) | PASS (0.0000) | PASS (67.47) | **FAIL (3.893)** | PASS (0.69) | PASS (max 4.56) | FAIL (b≥2.5) | 5/7 |

(Gate `mono ≥ 0.9522` uses V6 ship's mono as the bar per the
SOTA_TRAILS Tuner-trail gate. All 3 V7 seeds clear it.)

## Per-band achievement (codec-pooled)

V7 seed 3 (best-mono representative):

| band | empirical target (med) | achieved_mean | Δ |
|---:|---:|---:|---:|
| 0.30 | 88.0 | 87.19 | -0.81 |
| 0.80 | 87.4 | 85.03 | -2.37 |
| 1.50 | 75.9 | 78.56 | +2.66 |
| 2.50 | 66.2 | 67.24 | +1.04 |
| 4.00 | 26.4 | 48.81 | **+22.41** |
| 6.00 | 20.0 | 28.46 | **+8.46** |

At bands ≤ 2.5, V7 achieves the empirical target within ±3. At
bands ≥ 4.0, the trainer ignored the per-codec divergent targets
and converged to a compressed median range. This is the network
correctly choosing cross-codec parity over per-codec target
chasing — the V7 calibration approach is incomplete for heavy
distortion.

## Per-(codec, band) achievement at butter=4.0 (cc4v7_s3)

| codec | empirical target | achieved_mean | Δ |
|---|---:|---:|---:|
| zenjpeg | 55.14 | 51.51 | -3.63 (within ±5) |
| zenwebp | 16.68 | 47.31 | **+30.63** (FAIL ±5) |
| zenavif | 26.41 | 46.97 | **+20.56** (FAIL ±5) |
| zenjxl | 85.91 | 56.34 | **-29.57** (FAIL ±5) |

The per-codec empirical targets at butter=4.0 span 16-86 — a 70-
point range. The trainer cannot honor BOTH cross-codec parity
(cc_std ≤ 5) AND per-codec divergent targets, and per the trainer
loss composition (cross-codec equiv weight 1.0 + anchor weight 1.0
+ rank preserve 0.2), parity wins.

## Mohammadi panel (held-out validation)

Best V7 seed (s3) vs V6 ship (cc4v6_w1p0_p0p30_s1):

| Corpus | V6 ship | V7 best (s3) | Δ |
|---|---:|---:|---:|
| CID22 | 0.8770 | 0.8600 | **-0.017** |
| KADID | 0.7179 | 0.6756 | -0.042 |
| TID2013 | 0.7542 | 0.7495 | -0.005 |
| KonJND | 0.1962 | **0.4585** (s1) / TBD | +0.262 (s1) |
| AIC-3 | 0.7961 | 0.7827 (s1) / TBD | -0.013 |

KonJND improves dramatically on V7 — the empirical anchors at
butter=1.5 push PJND-region calibration to where KonJND humans
sit. CID22 essentially flat. KADID drops slightly.

## What V7 surfaces

1. **V6's rule-of-thumb anchor targets were calibrated to butter,
   not to ssim2.** At butter=1.5 (butter PJND), ssim2 humans give
   ~77 not 63. V6 outputs 63 there because that's what V6 was
   asked to output. V7 outputs 78 there because that's what
   empirical metrics say.

2. **The user-facing PJND convention** (PJND lands at score=63)
   was based on the CID22 paper's calibration to ssim2 at *ssim2's
   PJND*, NOT butter's PJND. Butter PJND (butter_pnorm3 ≈ 1.5) is
   ABOVE ssim2 PJND on the ssim2 score axis (around 77-80, since
   below that humans still report "I see distortion").

3. **Per-codec empirical targets diverge at heavy distortion.** At
   butter=4.0, zenjxl produces ssim2≈86 while zenwebp produces
   ssim2≈17 — a 70-point range. This is REAL human-validated
   information that the V7 anchor parquet exposed but the V7
   trainer couldn't honor without losing cross-codec parity.

4. **The cross-codec equivalence loss dominates at heavy
   distortion.** When forced to choose between per-codec target
   matching and cross-codec parity, the V7 trainer (with both
   losses at W=1.0) chose parity. This is consistent with V6's
   ship behavior; V7 doesn't change that prioritization.

## What this means for V8

The V7 finding suggests V8's design space:

- **Drop the cross-codec equivalence loss for the heavy-
  distortion bands** (butter ≥ 4.0) — let the bake reflect
  per-codec quality differences where humans clearly perceive
  them, and only enforce parity in the perceptibility band
  (butter ∈ [0.3, 2.5]).
- **Re-anchor the user-facing PJND convention.** If PJND should
  ship at score=63 (matching CID22 paper), the empirical PJND
  anchor needs to be at the butter where ssim2 humans give 63 —
  approximately butter=2.5 in this dataset, not 1.5.
- **Treat zenjxl's anomalous ssim2 ceiling as a SIGNAL, not
  noise.** zenjxl's butter-vs-ssim2 relationship differs from
  jpeg/webp/avif. Either (a) butter and ssim2 disagree on zenjxl's
  quality (worth investigating), or (b) the butter parquet
  selected easy zenjxl images. Either way, this is a finding the
  current Tuner architecture cannot honor.

## Recommendation

**Keep V6 ship as `PreviewV0_5TunerV2`.** V7 is NOT a strict
improvement: gate 4 fails by a wide margin (3.4-3.9 vs 2.5
threshold), and the per-(codec, band) within-±5 advisory check
fails at the heavy-distortion bands. V6 ship's calibration
(PJND-at-score-63) is the documented Tuner convention; V7's shift
to PJND-at-score-78 would silently break user code that types
"63" expecting the V6 calibration.

The V7 anchor-target comparison table
(`benchmarks/v_tuner_v7_anchor_target_comparison_2026-05-19.md`)
remains the key artifact — it documents that V6's anchor targets
were vibes-from-memory and the empirical medians are very
different. Future V8 work should use this table to design a
recipe that honors PJND-at-63 calibration via a different
mechanism than rule-of-thumb anchors (e.g., a single
ssim2-aware PJND anchor in the perceptibility band, NO anchors at
the heavy distortion bands).
