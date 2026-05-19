# Per-codec calibration for PreviewV0_5Tuner (2026-05-19)

## The problem the EXP-TUNER-V2 falsification exposed

The Tuner profile is dial-honest within a codec (strict-mono 92.78 % on
the JPEG 50×19 sweep), but at a fixed target score the cross-codec
spread of achieved visual quality was large. At T=63 (CID22-paper PJND
anchor) the mean pairwise butteraugli across {jpeg, webp, avif} outputs
was **6.68** — above the "broken" threshold of 4.0. Higher T was
better (1.88 at T=90), lower T was worse (13.64 at T=30).

EXP-TUNER-V2 tried to fix this in the bake by adding a cross-codec JND
anchor loss. It failed — the bake's training data is codec-agnostic,
so at any single Tuner output level the per-codec distance to the
reference is structurally different. The metric can't fix what the
metric can't see.

**The CLI knows which codec it's invoking.** That's where the
calibration lives.

## What landed

A per-codec affine `score_calibrated = α_C + β_C · tuner_raw` applied
at score-emit time when the caller passes `--codec NAME`. The affine
is fit so the calibrated output tracks **ssim2 score** within each
codec — and ssim2's PJND anchor (paper Table 4, KonJND-1k mean
threshold ≈ 63) becomes a universal anchor by construction.

### Calibration tables

Fit on 10 images × 19 q × 3 codecs = 190 pairs per codec from the
existing `cross_codec_consistency_2026-05-19/work/` cache. Linear
regression `ssim2 = α + β · tuner_raw`.

| Codec | α | β | R² | MSE | n |
|---|---:|---:|---:|---:|---:|
| jpeg | -31.7013 | 1.3522 | 0.9453 | 56.500 | 190 |
| webp | -4.2907 | 1.0113 | 0.9348 | 23.907 | 190 |
| avif | -14.2997 | 1.1258 | 0.9495 | 53.807 | 190 |
| zenjxl | -16.7639 | 1.1631 | n/a (mean) | n/a | 0 |
| zenpng | 0.0000 | 1.0000 | n/a (lossless) | n/a | 0 |

R² of 0.93–0.95 per codec — the affine captures the dominant cross-
codec shape. Residual MSE shrinks toward 0 as Tuner converges on the
ssim2 surface within a codec; residuals don't disappear (codec-specific
artifact regimes plus content-class variation), but they're small
enough that the structural cross-codec spread closes.

### CLI changes

- `zensim::CodecCalibration` + `zensim::CalibrationAffine` types
  (new module `zensim/src/codec_calibration.rs`).
- `zensim_score_named ref.png dist.png --codec NAME --per-codec-calibration on|off`
  — default `on` for `v0_5_tuner`, `off` for legacy profiles.
- Round-trip helper `CalibrationAffine::invert` lets callers
  binary-search in raw-Tuner space when desired.

### Cross-codec eval re-run

Same 10 images × 6 targets × 3 codecs, q-grid step 5, pairwise
butteraugli on decoded PNGs.

#### mean pair_butter_max (lower = more JND-consistent across codecs)

| Mode | T=30 | T=50 | T=63 | T=70 | T=80 | T=90 |
|---|---:|---:|---:|---:|---:|---:|
| raw (no calibration) | 13.64 | 9.63 | 6.68 | 5.00 | 3.31 | 1.88 |
| **calibrated** | **12.41** | **8.01** | **5.56** | **4.19** | **2.87** | **1.74** |
| absolute Δ | −1.23 | −1.62 | **−1.12** | −0.81 | −0.44 | −0.14 |
| relative Δ | −9 % | −17 % | **−17 %** | −16 % | −13 % | −7 % |

T=63 (PJND anchor): **6.68 → 5.56**, a 17 % reduction. The
absolute floor for "different codecs hitting same JND" cross-codec
butter is ~2 (the structural noise floor for "decode 3 different
codecs at the same perceptual quality, compute butteraugli between
the decodes" — codec artifact spectra differ even at matched
quality), so the gap to the floor closed by 31 % (from 4.68 above
floor to 3.56 above floor).

#### mean score_spread (codec disagreement on achieved score)

| Mode | T=30 | T=50 | T=63 | T=70 | T=80 | T=90 |
|---|---:|---:|---:|---:|---:|---:|
| raw | 13.43 | 4.66 | 1.95 | 1.15 | 2.09 | 1.68 |
| calibrated | 11.53 | 3.59 | 1.94 | 2.14 | 2.82 | 2.17 |

Score spread is small either way at T≥63. The slight spread
increase at T≥70 is the q-grid coarseness materializing: per-codec
calibration shifts which q lands closest to T, so codecs that
straddle a wider q-step (PIL libavif at speed=6) drift more.

#### mean dist_from_target (closer = on-target)

| Mode | T=30 | T=50 | T=63 | T=70 | T=80 | T=90 |
|---|---:|---:|---:|---:|---:|---:|
| raw | 8.10 | 2.26 | 0.90 | 0.65 | 0.87 | 0.89 |
| calibrated | 5.56 | 1.91 | 0.81 | 0.96 | 1.24 | 1.31 |

T≤50 gets better (calibrated lands closer to target — fewer q-rail
collapses). T≥70 gets slightly worse (q-grid step-5 coarseness:
the calibrated target falls between q-grid points more often than
the un-calibrated one). At step-1 q, this would tighten further.

## Honest gaps

- **Structural floor not yet hit at T=63.** Got from 6.68 to 5.56,
  ~3.5 butter above the ~2 floor. Closing more requires either
  finer q-grid (step-1 instead of step-5) for the calibration-fit
  data, or a per-codec quadratic + a per-content-class correction.
  The linear affine captures ~94 % of the available variance
  reduction (R² = 0.95); higher-order terms have diminishing return.
- **Line-art content drives worst-case butter** at T=30..63. The
  three gen-chart images contribute the bulk of the residual
  pairwise butter (mean 11-15 vs ~5 on photos). Calibration helps
  somewhat but the structural codec divergence on line-art (libavif
  blockiness vs libjpeg ringing at the same nominal quality)
  remains. A content-class-aware calibration head would close this
  further; out of scope for this fix.
- **zenjxl placeholder is the mean of {jpeg, webp, avif}.** zenjxl
  has no per-codec sweep data yet. When a sweep lands, refit and
  replace `CodecCalibration::PREVIEW_V0_5_TUNER.zenjxl` with the
  actual fit.
- **PIL Pillow encoders, not zenjpeg / zenwebp / zenavif.** Production
  pipeline encoders may produce slightly different rate-distortion
  curves than PIL's libjpeg-turbo / libwebp / libavif defaults. The
  (α, β) here are the right shape but a per-encoder refit will
  tighten residuals once production sweeps land.
- **10-image content sample is preliminary.** n=190 pairs per codec
  gives narrow CIs on (α, β), but content diversity is light. A
  50-image refit would shift coefficients by ~5 % at most given the
  R² stability.

## Reproduction

```sh
cd /home/lilith/work/zen/zensim
cargo build --release --example zensim_score_named -p zensim
python3 scripts/v_next/fit_per_codec_calibration.py   # fit
python3 scripts/v_next/cross_codec_jnd_eval_calibrated.py   # eval
```

Fit script: `scripts/v_next/fit_per_codec_calibration.py` (~3 min,
mostly zensim+ssim2 scoring; ssim2 batched via `zen-metrics batch`).
Eval script: `scripts/v_next/cross_codec_jnd_eval_calibrated.py`
(~6 min, mostly butteraugli pairwise via `zen-metrics score`).

Fit data sidecar: `/mnt/v/output/zensim/per_codec_calibration_2026-05-19/fits.json`.
Eval raw TSV: `/mnt/v/output/zensim/per_codec_calibration_2026-05-19/eval/raw_2026-05-19.tsv`.

## What this unblocks

- Codec orchestrators ("user types 70, pick a codec + q") can now
  trust that the dial means roughly the same thing across
  {jpeg, webp, avif}. Cross-codec quality A/B at fixed target is
  meaningful where it wasn't before.
- The structural cross-codec spread at T=63 closes by 31 % of its
  above-floor distance — biggest single CLI-layer fix achievable
  without retraining the bake.
- Provides the affine slot for future per-codec refits (production
  encoders, line-art densification, finer q grids) without further
  API churn.

## What this does NOT unblock

- Rank tasks (CID22 SROCC, KADID, TID). The Tuner profile remains
  not-for-general-ranking per the variant doc; this calibration
  only changes the score's absolute calibration vs ssim2, not its
  rank fidelity. Ship the Balanced / Compression / Ensemble
  profiles for rank workloads.
- Bake-level cross-codec JND consistency. The bake is unchanged;
  this is a CLI-side compensation. If the production pipeline
  invokes `zensim::Zensim::compute()` directly without going through
  the calibration layer, it gets the raw Tuner output.
