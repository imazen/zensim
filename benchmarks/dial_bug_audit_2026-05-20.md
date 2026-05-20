# Dial-bug audit — every bake in `zensim/weights/` (task #178)

**Date**: 2026-05-20

**Trigger**: BalancedV2 (#176, commit `5c5ca6b`) and CompressionV2 (#177, commit `ac7d156`) demonstrated that V_22-mix-LARGE+iwssim and V_24 per-sample-α s4 bakes both produce **distance-shaped raw output** (negative=high-quality, range typically `[-30, +30]`). The production runtime then applies either `clamp(0, 100)` (Balanced base, pinned 96.8 % of predictions to 0) or `100 / (1 + exp(-(raw - 50) / 20))` (Compression base, squashed to `[2.07, 18.24]`). Either way, the user-facing **dial is broken** — typing a target like "score 60" never matches any output because no input produces that score.

V9 PCHIP spline retrofit (`zentrain.output_calibration_spline` metadata) **fixes the dial without retraining**. Proven on Balanced + Compression in #176 / #177. This audit checks every other shipped bake for the same pathology.

## Procedure

1. Source: `/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet` (22 008 rows × 372 features).
2. Sampled 1000 random rows (seed `20260520`).
3. For each bake in `zensim/weights/*.bin`:
   a. Inspected via `zenpredict inspect` — if `zentrain.output_calibration_spline` metadata present, **SKIP** (V9-spline-fixed).
   b. Scored 1000 rows via `predict_features_with_bake --bake-post raw` (raw network output before any post-process, but **including** per-sample-α head and tanh-pin where present).
   c. Applied the per-profile production clamp/squash to obtain user-facing dial output:
      * `clamp` — `raw.clamp(0, 100)` (V0_3, V0_5 / V0_5Balanced, V0_5Tuner, V0_5TunerV2)
      * `soft_clamp` — `100 / (1 + exp(-(raw - 50) / 20))` (V0_4, V0_5Compression, V0_5CrossCodec, V0_5Ensemble)
   d. Computed p5, p50, p95, range = p95 − p5, fraction pinned at 0 / 100.
4. **Bug threshold**: range < 50 → **DIAL-BROKEN**. range ≥ 80 → **DIAL-OK**. 50 ≤ range < 80 → **DIAL-MARGINAL**.

`|SROCC_raw|` reported alongside as sign-tolerant Spearman against the V9 anchors' `human_score` column on the 1000-row sample. A high `|SROCC|` confirms the bake's **rank order** is fine — only the absolute output scale is wrong. Spline retrofit is a monotone reshaping; any high-`|SROCC|` bake is structurally fixable that way.

Reproduction:

```
python3 scripts/dial_bug_audit/run_dial_audit.py
```

## Results

| Bake | Profile | n_in | Post | range (p95−p5) | p5..p95 | %@0 | %@100 | \|SROCC_raw\| | Verdict |
|---|---|---:|---|---:|---|---:|---:|---:|---|
| `v0_18_zerobiased_lz4_2026-05-13.bin` | PreviewV0_3 | 228 | clamp | **84.11** | 15.89..100.00 | 0.4 % | 29.8 % | 0.950 | **DIAL-OK** |
| `v0_20_is_calibrated_2026-05-15.bin` | PreviewV0_4 (B3 secondary) | 228 | soft_clamp | **93.71** | 6.09..99.79 | 0.0 % | 0.0 % | 0.936 | **DIAL-OK** |
| `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` | PreviewV0_5 / V0_5Balanced | 300 | clamp | **13.61** | 0.00..13.61 | 84.6 % | 0.0 % | 0.957 | **DIAL-BROKEN** |
| `v_compression_persample_2026-05-18.bin` | PreviewV0_5Compression | 300 | soft_clamp | **13.38** | 2.08..15.46 | 0.0 % | 0.0 % | 0.951 | **DIAL-BROKEN** |
| `v_tuner_2026-05-18.bin` | PreviewV0_5Tuner | 372 | clamp | **87.48** | 12.03..99.51 | 0.4 % | 0.0 % | 0.946 | **DIAL-OK** |
| `v_cross_codec_2026-05-19.bin` | PreviewV0_5CrossCodec | 372 | soft_clamp | **0.08** | 65.63..65.71 | 0.0 % | 0.0 % | 0.934 | **DIAL-BROKEN** |
| `v_tuner_v6_2026-05-19.bin` | PreviewV0_5TunerV2 | 372 | clamp (tanh-pinned) | **77.26** | 12.87..90.12 | 0.0 % | 0.0 % | 0.973 | **DIAL-MARGINAL** |
| `v_tuner_v9_2026-05-20.bin` | PreviewV0_5TunerV3 | 372 | — | — | — | — | — | — | **SKIP — spline fixed** |
| `v_balanced_v2_2026-05-20.bin` | PreviewV0_5BalancedV2 | 300 | — | — | — | — | — | — | **SKIP — spline fixed** |
| `v_compression_v2_2026-05-20.bin` | PreviewV0_5CompressionV2 | 300 | — | — | — | — | — | — | **SKIP — spline fixed** |
| `v05_ensemble_classifier_2026-05-18.bin` | classifier (logit only) | 300 | — | — | — | — | — | — | **SKIP — routes to other bakes, not a dial itself** |
| `v_compression_2026-05-18.bin` | (archived prior compression ship, 372feat) | 372 | clamp | **12.47** | 0.00..12.47 | 78.1 % | 0.0 % | 0.947 | **DIAL-BROKEN (archived)** |
| `v0_22_iw_v2_2026-05-16.bin` | (research; not shipped) | 372 | clamp | **7.78** | 0.00..7.78 | 90.4 % | 0.0 % | 0.890 | **DIAL-BROKEN (research)** |
| `v0_22_iw_v2_calibrated_2026-05-16.bin` | (research; not shipped) | 372 | clamp | **72.88** | 27.12..100.00 | 0.0 % | 23.7 % | 0.890 | **DIAL-MARGINAL (research)** |
| `v0_18_2026-05-13.bin` | (archived uncompressed V0_3 source) | 228 | clamp | **84.11** | 15.89..100.00 | 0.4 % | 29.8 % | 0.950 | **DIAL-OK (archived; same bytes as `v0_18_zerobiased_lz4`)** |

## Verdict — currently shipped profiles

**Shipped V0_5* base trail bakes that still need V9 spline retrofit (3):**

| Profile | Bake | Production clamp | Why broken |
|---|---|---|---|
| **`PreviewV0_5` / `PreviewV0_5Balanced`** | `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` | hard `clamp(0, 100)` | Bake raw distribution `[-23.7, 38.4]`, p95=13.6. Hard clamp pins 84.6 % of predictions at 0. Already fixed in **BalancedV2** (#176) via `v_balanced_v2_2026-05-20.bin`. |
| **`PreviewV0_5Compression`** | `v_compression_persample_2026-05-18.bin` | `soft_clamp` | Bake raw distribution `[-27.2, 34.0]`, p95=16.0. After soft-clamp: `[2.08, 15.46]` — flat dial. Already fixed in **CompressionV2** (#177) via `v_compression_v2_2026-05-20.bin`. |
| **`PreviewV0_5CrossCodec`** | `v_cross_codec_2026-05-19.bin` | `soft_clamp` | Bake raw distribution `[60.71, 63.01]` (range 2.3 across 1000 random rows). After soft-clamp: `[65.63, 65.71]` (range 0.08). The per-sample-α head's pool branch dominates with a nearly-flat reducer (layer-2 is identity-passthrough, layer-1 abs_p99 is 0.025). `|SROCC|=0.934` → rank info IS present; spline retrofit will fix the dial. **NOT YET FIXED**. |

**Shipped DIAL-MARGINAL bakes (1):**

| Profile | Bake | Production clamp | Range | Note |
|---|---|---|---|---|
| **`PreviewV0_5TunerV2`** | `v_tuner_v6_2026-05-19.bin` | hard clamp after tanh pin | 77.26 (p5..p95 = 12.87..90.12) | The `zentrain.tanh_output_head` (scale 15.0) pins the per-sample-α raw output into [0, 100]; the hard clamp is a no-op safety net. Range is below the 80 threshold but the dial is still usable: 0 % at floor/ceiling, `|SROCC|=0.973` is the highest of any bake. Spline retrofit would improve range; not strictly necessary. |

**Shipped DIAL-OK bakes (3):**

| Profile | Bake | Range | Note |
|---|---|---:|---|
| `PreviewV0_3` | `v0_18_zerobiased_lz4_2026-05-13.bin` | 84.11 | Affine-calibrated (α=28.04, β=-5.07). 29.8 % pinned at 100 — visually lossless saturation, expected; not a dial-bug. |
| `PreviewV0_4` (B3 secondary, mixed with V0_3) | `v0_20_is_calibrated_2026-05-15.bin` | 93.71 | Soft-clamp + affine calibration produces well-spread output. |
| `PreviewV0_5Tuner` | `v_tuner_2026-05-18.bin` | 87.48 | Affine-calibrated (α=−1590.55, β=52.02). 0 % pinned. Healthy dial. |

## Non-shipped bakes

* `v_compression_2026-05-18.bin` (archived prior compression ship, 372feat) — DIAL-BROKEN. No retrofit needed (archived).
* `v0_22_iw_v2_2026-05-16.bin` (research, never shipped) — DIAL-BROKEN. No retrofit needed.
* `v0_22_iw_v2_calibrated_2026-05-16.bin` (research, never shipped) — DIAL-MARGINAL. No retrofit needed.
* `v0_18_2026-05-13.bin` (archived uncompressed source of v0_18_zerobiased_lz4) — DIAL-OK; identical numerics.

## Recommended next actions

1. **Retrofit `PreviewV0_5CrossCodec`** with V9 PCHIP spline calibration (~30 min following the BalancedV2 / CompressionV2 recipe). This is the only currently-shipped variant that's structurally DIAL-BROKEN and not yet fixed.
2. **Consider retrofitting `PreviewV0_5TunerV2`** for completeness — the dial is usable (range 77, no boundary pinning) but a spline would lift to ≥ 80.
3. **`PreviewV0_5` / `PreviewV0_5Balanced` and `PreviewV0_5Compression` already have V9-spline replacements** in `PreviewV0_5BalancedV2` and `PreviewV0_5CompressionV2`. Callers wanting a working dial should select the V2 variants explicitly; the base variants remain DIAL-BROKEN for backward compatibility.
4. **No action needed** for V0_3 / V0_4 / V0_5Tuner / V0_5TunerV3 — all DIAL-OK or spline-fixed.

Sign-tolerant SROCC stays in `[0.89, 0.98]` for every broken bake — the rank order is preserved across all of them, confirming that PCHIP spline retrofit (which is monotone-by-construction) will restore the dial without retraining for any of them.

## Bake-to-profile mapping reference

Source: `zensim/src/profile.rs` (commit `ac7d1562`, main as of audit time).

```
v0_18_zerobiased_lz4_2026-05-13.bin       -> PreviewV0_3                hard clamp
v0_20_is_calibrated_2026-05-15.bin        -> PreviewV0_4 (B3 secondary, mixed w/ V0_3 raw before soft clamp)
v22_mix_cv40_konjnd_002_LARGE_iwssim_*    -> PreviewV0_5 / PreviewV0_5Balanced  hard clamp
v_compression_persample_2026-05-18.bin    -> PreviewV0_5Compression     soft clamp
v_tuner_2026-05-18.bin                    -> PreviewV0_5Tuner           hard clamp
v_cross_codec_2026-05-19.bin              -> PreviewV0_5CrossCodec      soft clamp
v_tuner_v6_2026-05-19.bin                 -> PreviewV0_5TunerV2         hard clamp after tanh pin
v_tuner_v9_2026-05-20.bin                 -> PreviewV0_5TunerV3         V9 spline (FIXED)
v_balanced_v2_2026-05-20.bin              -> PreviewV0_5BalancedV2      V9 spline (FIXED)
v_compression_v2_2026-05-20.bin           -> PreviewV0_5CompressionV2   V9 spline (FIXED)
v05_ensemble_classifier_2026-05-18.bin    -> classifier (routes, not a dial)
v_compression_2026-05-18.bin              -> ARCHIVED prior compression ship (372feat)
v0_22_iw_v2_2026-05-16.bin                -> never shipped (research)
v0_22_iw_v2_calibrated_2026-05-16.bin     -> never shipped (research)
v0_18_2026-05-13.bin                      -> ARCHIVED V0_3 uncompressed
```

Raw structured data: `benchmarks/dial_bug_audit_2026-05-20.json`.
