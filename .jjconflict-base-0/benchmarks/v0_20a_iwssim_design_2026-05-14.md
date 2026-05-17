# V0_20a IW-SSIM weighted pooling — design doc (2026-05-14)

**Goal**: lift B0-B5 CID22 SROCC via information-content-weighted
spatial pooling at the SSIMULACRA-2 feature extraction layer.

**Paper claim**: Wang & Li 2011 (IW-SSIM, IEEE TIP) — weighted spatial
pooling using GSM-on-wavelet info-content per region gives +0.006
weighted-avg SROCC over MS-SSIM across 6 IQA databases (LIVE / A57
/ IVC / Toyama / TID2008 / CSIQ).

**Adaptation magnitude per the paper's framework**: IW-pooled features
are most useful at high-distortion regimes where uniform spatial
pooling smears salient-region degradation with flat-region
degradation. That's exactly our B0-B5 weakness. Expected lift in our
setup: B0/B1 +0.01..0.03 SROCC vs V0_18.

## Current feature pipeline (V0_18)

Feature layout (228 = 4 scales × 3 channels × 19 features):

| Block | Indices | Features per channel | Pooling |
|---|---|---|---|
| Scored | [0, 156) | 13 | mean, L2, L4 (uniform spatial) |
| Peaks  | [156, 228) | 6 | max, p95 (no spatial pool needed) |

Extended profile (300 features at 4 scales × 3 channels × 25):
adds 6 **masked** features per channel per scale that DO use weighted
pooling — weight = `flatness mask` (texture-inverse). Currently used
in the extended 300-feature path but the V0_18 ship only uses the
basic 228.

So: zensim **already has weighted-pool infrastructure**. The
`apply_contrast_masking` path at `metric.rs` lines 352-490 does
texture-suppressed pooling. IW-SSIM points the OPPOSITE direction:
weight by **high** info-content (textured regions weighted UP, flat
regions weighted DOWN). The two are complementary, not redundant.

## Proposed integration

Add a third feature block — `iw` — alongside `scored` and `peaks` /
`masked`. Per channel per scale, emit 6 IW-pooled features:

| IW feature | Pooling weight |
|---|---|
| iw_ssim_mean   | weighted mean of SSIM(src, dst) |
| iw_ssim_4th    | weighted L4 of SSIM           |
| iw_ssim_2nd    | weighted L2 of SSIM           |
| iw_art_4th     | weighted L4 of edge_artifact  |
| iw_det_4th     | weighted L4 of detail_lost    |
| iw_mse         | weighted mean of (src-dst)²   |

Weight per pixel = local info-content of the **reference** image.
Two candidate definitions (will sweep):

1. **Wang 2011 GSM**: fit a Gaussian Scale Mixture to per-region
   wavelet coefficients; weight ∝ log-likelihood of the local
   scale parameter. Mathematically exact but expensive.
2. **Practical approximation**: weight ∝ local variance (or
   gradient L1) of the reference at the current scale. Same
   information-theoretic direction, ~10× cheaper.

Start with (2); if performance is encouraging, sweep against (1).

### Total feature count

228 (basic + peaks) + 72 (4 scales × 3 ch × 6 IW) = **300** features.

Coincidentally equal to the existing `FEATURES_PER_CHANNEL_EXTENDED = 25`
masked-block size. But the IW features are a DIFFERENT 72 features
(weight direction inverted from the masked block). Both can coexist:
228 + 72 masked + 72 IW = 372. For V0_20a start with 228 + 72 IW =
300; V0_20a.2 can add the masked block back for the full 372.

### MLP shape

228 → 384 → 1 (V0_18 ship) becomes either:
- **300 → 384 → 1** (replacement, smaller hidden ratio)
- **300 → 192 → 1** (smaller mid-layer to compensate)
- **300 → 384 → 1** trained from V0_18 weight init + zero-init for
  the new 72 inputs (warm-start)

Warm-start path is cleanest — preserves V0_18 behavior at initialization
and lets the MLP discover whether the IW features help.

## Implementation plan

| # | File | Change |
|---|---|---|
| 1 | `zensim/src/metric.rs` | Add `compute_iw_weights(&[f32]) -> Vec<f32>` (local variance / gradient form, configurable). |
| 2 | `zensim/src/streaming.rs` or new `pool_weighted.rs` | Add `pool_weighted_mean`, `pool_weighted_l2`, `pool_weighted_l4` helpers. |
| 3 | `zensim/src/metric.rs` | Add `FEATURES_PER_CHANNEL_IW = 6`; extend `ZensimConfig` with `compute_iw_features: bool` flag. |
| 4 | `zensim/src/metric.rs` | Emit IW block when flag set, immediately after peaks block (indices 228..300). |
| 5 | `zensim-validate/src/bin/zensim_mlp_train.rs` | Already supports `--max-features N`; will work with 300-input features. |
| 6 | New profile slot `PreviewV0_4` in `profile.rs` if shipping; or reuse `PreviewV0_3` with new bake bytes (228 → 300 expansion needs profile-shape change). |

## Reproducibility sweep

Per CLAUDE.md "push to paper-claimed benefit":

- **Weight kernel size**: 3×3, 5×5, 7×7 — coarser is more "regional", finer is more "per-pixel".
- **Weight type**: variance (Wang's GSM proxy), gradient L1, gradient L2.
- **Scale-pyramid depth**: same 4 scales as basic; the per-scale info weight differs from per-pixel because scale-3 weights are coarser.
- **Per-channel weighting**: Y-only (mimics IW-SSIM's grayscale), all 3 XYB, or per-channel separate weights.

That's 3 × 3 × 1 × 3 = 27 configurations. Triple seed (1, 42, 7) per config = 81 component bakes. Concat top-3 per the V0_18 recipe.

## Validation

Full-stat panel (per CLAUDE.md 2026-05-14): SROCC, PLCC, KROCC, OR,
PWRC against V0_18 baseline and ssim2 reference. Per-band 10-band
+ 4-band CID22 cuts. KADID, TID, CID22, KonJND, AIC-3 (when
acquired).

Specifically TRACK:
- **B0-B5 CID22 SROCC delta** vs V0_18 (priority bands per user directive)
- **PLCC delta** vs V0_18 (dial honesty)
- **Aggregate Z-RMSE** if AIC-3 acquired (HF-regime signal)

## Risk + fallback

- **Risk**: IW pool concentrates on textured regions; if our distortion
  set has more flat-region artifacts than textured-region artifacts,
  IW features are net negative. Compensation: the masked block (texture-
  inverse) is already in the 300-feature extended profile — train one
  bake that uses BOTH and let the MLP choose via learned weights.
- **Fallback if IW features don't help**: pivot to V0_20b
  (distortion-manifold pre-training, task #39) which addresses a
  different lever.

## Status

Design queued. Implementation deferred to next active work block — this
is half-day coding effort (metric.rs + streaming.rs + new profile slot
+ trainer flag), then 80-minute training × 27 configs = full sweep
fits in a single overnight run.
