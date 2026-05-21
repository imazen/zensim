# EXP-CROSS-CODEC-V11-E per-codec post-spline affine — FALSIFIED (task #186, 2026-05-20)

## Hypothesis

A per-codec post-spline affine `score_c = α_c + β_c × spline(raw)`,
applied AFTER all network forward + tanh-pin + PCHIP spline, would
tighten cross-codec stddev at JND/JOD landmarks (~3× per the prior
CLI per-codec calibration evidence at commit `4842208`) WITHOUT
sacrificing within-codec rank ordering. Expected mechanism: the
network's KonJND-relevant features are preserved entirely; the
per-codec offset/scale only adjusts cross-codec systematic bias.

## Method

1. **Substrate**: V11 cross-codec equivalence parquet at 372-feat,
   1,739 (codec_a@q_a, codec_b@q_b) pairs at 6 ssim2 anchor levels
   (18 / 30 / 45 / 60 / 75 / 90), 4 codecs (zenjpeg, zenwebp, zenavif,
   zenjxl), 4–6 pairs per (ref, ssim2_level) anchor.
   Path: `/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/cross_codec_equivalence_ssim2_372col_v4.parquet`.
2. **Bake scoring**: each pair's `fa_*` and `fb_*` 372-feature blocks
   were scored through the bake via `predict_features_with_bake
   --bake-post extrapolate` (post-spline raw, no clamp).
3. **Train/holdout split**: 80% / 20% per pair (seed=42).
4. **Per-codec affine fit**: for each codec, collected
   `(ssim2_level, predicted_score)` pairs across both codec-A and
   codec-B sides. Fit modes tested:
   - `full` — free (α, β) via deg-1 polyfit on train rows.
   - `offset` — fix β = 1, α = mean(target − predicted) on train rows.
5. **Holdout eval**: cross-codec stddev computed per
   (ref_basename, ssim2_level) anchor (each anchor has 2-4 distinct
   codec predictions); reported median / p90 / max across anchors.
   Per-pair |pa − pb| reported as secondary diagnostic.

Fit code: `scripts/v_next/v11_e/fit_per_codec_affine.py`.

## Per-bake fit numbers (offset mode, train rows)

### TunerV4 (`v_tuner_v10_2026-05-20.bin`)

| codec | n_train | α (offset) | β | n_holdout |
|---|--:|--:|--:|--:|
| jpeg | 787 | -4.49 | 1.0 | 199 |
| webp | 580 | -5.18 | 1.0 | 151 |
| avif | 731 | -5.10 | 1.0 | 170 |
| jxl  | 686 | -4.88 | 1.0 | 174 |

Per-codec offset spread: ~0.7 score units. The bake's per-codec
systematic bias is already small.

### BalancedV3 (`v_balanced_v3_2026-05-20.bin`)

| codec | n_train | α (offset) | β |
|---|--:|--:|--:|
| jpeg | 787 | -9.93 | 1.0 |
| webp | 580 | -8.42 | 1.0 |
| avif | 731 | -7.31 | 1.0 |
| jxl  | 686 | -7.79 | 1.0 |

Per-codec offset spread: ~2.6 score units.

### CompressionV3 (`v_compression_v3_2026-05-20.bin`)

| codec | n_train | α (offset) | β |
|---|--:|--:|--:|
| jpeg | 787 | -11.46 | 1.0 |
| webp | 580 | -10.61 | 1.0 |
| avif | 731 |  -8.46 | 1.0 |
| jxl  | 686 |  -8.86 | 1.0 |

Per-codec offset spread: ~3.0 score units.

## Holdout cross-codec stddev (per-anchor, offset mode)

| Bake | Pre-fit median | Post-fit median | Pre-fit p90 | Post-fit p90 | Pre-fit max | Post-fit max |
|---|--:|--:|--:|--:|--:|--:|
| **TunerV4** | 1.391 | **1.342 (−4 %)** | 3.237 | 3.168 | 9.738 | 9.645 |
| **BalancedV3** | 1.232 | **1.431 (+16 %)** | 3.364 | 3.662 | 25.026 | 23.694 |
| **CompressionV3** | 1.046 | **1.493 (+43 %)** | 3.135 | 3.331 | 22.211 | 20.583 |

## Per-ssim2-level holdout stddev

### TunerV4 — offset mode
```
level n_anc  pre_med post_med  pre_max post_max
 30.0     8   3.6851   3.8594   5.2228   5.5221
 45.0    14   2.1494   2.3498   9.7383   9.6451
 60.0    35   1.3913   1.4180   5.4733   5.3802
 75.0   104   1.1708   1.0695   5.9330   5.7886
 90.0    96   1.4996   1.2919   4.0711   4.0415
```

### BalancedV3 — offset mode
```
level n_anc  pre_med post_med  pre_max post_max
 30.0     8   3.8912   3.9788  21.9801  20.6480
 45.0    14   1.5855   1.5507  14.0011  14.7549
 60.0    35   1.7147   1.8238   4.4770   5.2307
 75.0   104   1.0367   1.2255   5.7006   4.6716
 90.0    96   1.2618   1.5762   4.0286   4.5136
```

### CompressionV3 — offset mode
```
level n_anc  pre_med post_med  pre_max post_max
 30.0     8   4.8446   3.5383  15.1546  13.5265
 45.0    14   2.7208   2.2760  22.2107  20.5826
 60.0    35   1.2570   1.3017  11.9436  10.9591
 75.0   104   0.8582   1.2248   3.9340   4.0007
 90.0    96   1.0413   1.6286   3.5167   4.3233
```

## Full-fit-mode results (rejected, for reference)

When using the free (α, β) fit on TunerV4, the holdout median stddev
got *worse* (+13 %, 1.39 → 1.57). On BalancedV3 / CompressionV3 the
free-fit caused catastrophic regression (median +175 % on
BalancedV3, +830 % on CompressionV3) because the fit's β > 1 pulled
the score-shape away from the spline-calibrated target.

## Verdict

**FALSIFIED on the headline cross-codec stddev metric.**

- TunerV4: marginal improvement (−4 % median, −1 % max). Below ship-decision
  threshold (would expect ≥ 3× tightening per hypothesis; actual ≤ 5 %).
- BalancedV3: cross-codec stddev *worsens* (+16 % median). Per-anchor
  bias is content-driven, not codec-driven; the per-codec α shifts
  ALL codecs at a given anchor by approximately the same amount,
  leaving relative spread unchanged at best.
- CompressionV3: catastrophic regression (+43 % median).

### Root cause

The V10 PCHIP spline calibration already does most of the cross-codec
work. After the spline, the residual cross-codec bias is:

| Bake | Per-codec α spread | Within-codec residual stddev |
|---|--:|--:|
| TunerV4 | 0.7 | 4.5–7.0 |
| BalancedV3 | 2.6 | 5.6–8.7 |
| CompressionV3 | 3.0 | 6.0–9.5 |

The within-codec residual (content-driven noise) is **5–10× larger
than the per-codec systematic offset**. A linear affine cannot
compress content-driven noise; it can only shift codecs uniformly.
For BalancedV3 and CompressionV3, the affine adds linear noise on
top of the already-larger content noise.

### Why the prior CLI per-codec calibration (commit `4842208`)
*did* tighten Tuner butter 6.68 → 5.56 at T=63

That calibration was fit against TunerV2 (V_tuner-v2-s2), an
**uncalibrated dial** without a PCHIP spline. The bake's raw output
spread WAS dominated by codec-level bias because the network was
trained codec-agnostic without any cross-codec target. Per-codec
affine added the missing calibration. V10 ships already carry the
PCHIP spline; per-codec affine is redundant work on top.

## What ships

The metadata format, runtime dispatch, and three opt-in variants
ship per the original spec. The metadata is **always identity-by-
default** (no codec hint → no per-codec affine applied), so the
runtime is fully backwards-compatible and the
SROCC-preservation gate is structurally satisfied (bake_verdict
emits bit-exact identical reports for the V10 ships and their
*_Calibrated counterparts).

### Shipped

| Profile | Bake file |
|---|---|
| `PreviewV0_5TunerV4Calibrated` | `weights/v_tuner_v4_per_codec_2026-05-20.bin` |
| `PreviewV0_5BalancedV3Calibrated` | `weights/v_balanced_v3_per_codec_2026-05-20.bin` |
| `PreviewV0_5CompressionV3Calibrated` | `weights/v_compression_v3_per_codec_2026-05-20.bin` |

Use these variants ONLY when:
- The caller has a codec hint context (e.g. zensim-target binary-
  searching q for a known codec) AND
- An empirical sweep on the caller's codec configuration shows the
  per-codec affine reduces cross-codec spread vs the un-calibrated
  variant.

### Not shipped

- New trail in `SOTA_TRAILS.md` (the calibration didn't earn one).
- Default ship rotation — V10 ships remain the canonical defaults.
- Methodology page on the interactive site.

## Runtime mechanism (landed for future use)

**Metadata key**: `zentrain.per_codec_calibration`.

**Payload layout** (little-endian, `MetadataType::Bytes`):
```
[u32 n_codecs, n_codecs × (u32 name_len, name_len utf8 bytes, f32 alpha, f32 beta)]
```

**Per-bake size cost**: 4 + 4·(4+8) = 67 bytes for 4 codecs. Negligible
vs the ~40 KB-200 KB bake size.

**Runtime dispatch** (`zensim::metric::forward_one_bake_with_codec`):
applied AFTER PCHIP spline output as
`score = alpha + beta · spline(y_pinned)`. Identity-by-default when:
- The bake doesn't carry the metadata, OR
- The caller doesn't pass a codec hint, OR
- The codec hint matches no entry in the metadata table.

**Public API**: `Zensim::compute_with_codec_hint(source, distorted,
codec_hint: Option<&str>)`. Codec hint aliases: `"jpeg"` / `"jpg"` /
`"zenjpeg"` / `"mozjpeg"` / `"libjpeg"`, `"webp"` / `"zenwebp"`,
`"avif"` / `"zenavif"`, `"jxl"` / `"zenjxl"` / `"jpegxl"`, `"png"` /
`"zenpng"`.

**bake_verdict integration**: not wired (bake_verdict evaluates per
corpus on human-MOS holdouts, no codec context). Bit-exact SROCC
preserved across all 6 eval corpora for all three calibrated bakes
(verified).

**Tests**: `zensim/tests/per_codec_calibration.rs` (8 tests). Covers
no-hint identity, unknown-hint identity, per-codec dispatch, alias
aliasing, deterministic per-input + per-hint scoring, identity-image
short-circuit fires before per-codec affine, no-metadata profile
ignores hint.

## Closing the V11 cross-codec frontier

Per CLAUDE.md "investigate(v11-d-pjnd): KonJND PJND-passthrough rescue
FALSIFIED — V11 cross-codec-eq frontier closed with structural finality
(task #198)": V11 per-codec affine (this experiment, task #186) is the
last queued cross-codec calibration mechanism. The structural finding
is now confirmed across both:

- **Network-side** (V11 a/b/c/d): cross-codec equivalence loss
  fundamentally trades against KonJND.
- **Runtime-side** (V11-E, this): per-codec affine on top of a
  spline-calibrated bake yields ≤ 5 % cross-codec stddev tightening
  in the best case, and is content-noise-bounded.

The remaining cross-codec mismatch on the V10 ships is **content-driven
within-codec noise**, not codec-systematic bias. Closing this further
would require either:
- Different feature extraction (out of scope for V11; would be V12+
  research direction), OR
- Per-codec network training (which V11 falsified).
