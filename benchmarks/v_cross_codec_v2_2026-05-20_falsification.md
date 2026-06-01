# CrossCodecV2 PCHIP spline calibration — FALSIFIED (2026-05-20, task #179)

**Outcome:** `PreviewV0_5CrossCodec` cannot be salvaged with PCHIP spline
calibration. The cross-codec-equivalence training objective compresses
the network's raw output dynamic range to ~0.18 score units across the
entire training quality range — too tight for the per-band median
extraction the spline calibrator depends on. **Recommendation:
deprecate `PreviewV0_5CrossCodec` from `SOTA_TRAILS.md` and the
public-facing profile catalog.** The dial-broken behavior is structural
to the training loss, not a fixable calibration artifact.

## Context

Per dial-bug audit #178 (commit `28ed2552`), `PreviewV0_5CrossCodec`
ships at `zensim/weights/v_cross_codec_2026-05-19.bin` with production
raw output range 0.08 across 1000 random anchor pairs — meaning every
input gets approximately the same score after `soft_clamp_score`.
`|SROCC|=0.934` against MOS confirms the rank information is preserved
inside that compressed band, but no user-facing dial can hit a
specific target ("score 60", "score 30") when the dynamic range is
sub-unit.

Sibling V9 PCHIP spline calibrations succeeded on:

- **TunerV3** (commit `5386d55`): tanh-pinned distance-shaped bake →
  7-knot spline lands JND at 60, JOD at 30, full [0, 100] dial.
- **BalancedV2** (commit `5c5ca6b`): per-sample-α distance-shaped
  bake → 7-knot spline same anchors, SROCC preserved bit-exact.
- **CompressionV2** (commit `ac7d156`): per-sample-α distance-shaped
  bake → 7-knot spline same anchors, SROCC preserved bit-exact.

All three of those bakes had raw output range >40 score units across
the V9 anchor parquet's 8 target bands. CrossCodec's range is **0.18**
— two orders of magnitude tighter.

## Method

Reused the existing PCHIP spline calibrator script
(`scripts/v_next/calibrate_balanced_v9_spline.py`, proven
architecture-agnostic for 300- and 372-input bakes). The calibrator:

1. Scores the bake on every row of
   `/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet`
   (22,008 rows × 372 features, 8 target bands) via
   `predict_features_with_bake --bake-post raw` (per-sample-α dispatch
   active inside the binary).
2. Computes the median raw prediction per target band.
3. Sorts bands by median raw value ascending.
4. Drops bands whose raw median is within 1e-4 of the prior kept band
   (x-collision) or violates the monotone direction set by the first
   two non-collapsed bands.
5. Fits Fritsch-Carlson PCHIP derivatives.
6. Verifies monotonicity on a 1000-point grid spanning [knot_min - 5,
   knot_max + 5].
7. Injects the surviving knots as `zentrain.output_calibration_spline`
   metadata via the JSON pipeline through `zenpredict-bake`.

Command:

```sh
python3 scripts/v_next/calibrate_balanced_v9_spline.py \
    --bake zensim/weights/v_cross_codec_2026-05-19.bin \
    --out zensim-experimental/weights/v_cross_codec_v2_2026-05-20.bin \
    --n-features 372 \
    --predict-bin ./target/release/predict_features_with_bake \
    --zenpredict-bin /home/lilith/work/zen/zenanalyze/target/release/zenpredict \
    --spline-csv benchmarks/v_cross_codec_v2_2026-05-20_spline.csv
```

Full log: `benchmarks/v_cross_codec_v2_2026-05-20_calibration.log`.

## Findings

### Per-band median raw output (CrossCodec base bake)

| target | n | median_raw | p25 | p75 | min | max |
|---:|---:|---:|---:|---:|---:|---:|
| 0   | 1159 | 62.961 | 62.729 | 62.987 | 60.071 | 63.001 |
| 10  |  296 | **62.829** | 62.569 | 62.984 | 60.071 | 62.999 |
| 30  | 2055 | 62.985 | 62.978 | 62.992 | 61.899 | 63.003 |
| 50  | 3934 | 62.995 | 62.992 | 62.998 | 62.749 | 63.006 |
| 60  | 3933 | 63.002 | 63.000 | 63.003 | 62.914 | 63.009 |
| 80  | 3714 | 63.006 | 63.006 | 63.006 | 62.999 | 63.008 |
| 90  | 3406 | 63.007 | 63.006 | 63.007 | 63.003 | 63.008 |
| 100 | 3511 | 63.007 | 63.007 | 63.007 | 63.004 | 63.008 |

The raw output spans **0.178 score units across the 8 training bands**
(62.829 at target=10 → 63.007 at target=100). For comparison, the
Compression bake spans 40.77 score units (-26.89 at target=100 →
+13.89 at target=0) and the Tuner bake spans 38.9 score units.

The shape is also **non-monotone at the low end**: target=10
(raw=62.829) sits *below* target=0 (raw=62.961). This is not noise —
296 vs 1159 sample counts are both well above the noise floor — it's
a structural property of the cross-codec-equivalence loss pulling
adjacent quality levels toward a common butter-anchor value
regardless of MOS direction.

### Spline knots that survived monotonicity filtering

| keep? | x (raw_med) | y (target) | reason |
|---|---:|---:|---|
| KEEP   | 62.829 | 10  | first knot |
| KEEP   | 62.961 |  0  | x > prev, y direction set (descending) |
| DROP   | 62.985 | 30  | direction violation (y=30 > y=0) |
| DROP   | 62.995 | 50  | direction violation |
| DROP   | 63.002 | 60  | direction violation |
| DROP   | 63.006 | 80  | direction violation |
| DROP   | 63.007 | 90  | direction violation |
| DROP   | 63.007 | 100 | direction violation |

**6 of 8 bands dropped.** Only the structurally-anomalous target=10 → 0
inversion survives, leaving a 2-knot spline that does not contain the
information needed to land the dial anchors.

### Dial range (production runtime, `--bake-post clamp` on 1000 random anchor pairs)

| Metric | CrossCodec base | CrossCodecV2 (spline) |
|---|---:|---:|
| min | 61.907 | 0.000 |
| max | 63.008 | 79.827 |
| range | **1.101** | 79.827 |
| p5  | 62.958 | 0.000 |
| p95 | 63.007 | 0.210 |

Headline range jumps from 1.10 → 79.83 — but the post-spline
distribution is bimodal-degenerate: **p95 = 0.21**, meaning 95% of
in-distribution inputs collapse to ~0 after the spline + hard clamp.
The 79.83 max comes from a small tail of pairs whose raw value falls
*below* the y=10 knot (raw < 62.829), which the linear extrapolation
shoots upward off the left edge of the spline. The "fixed" dial is
not a dial — it's a near-binary classifier between "almost zero" and
"OOD extrapolation tail".

### JND landing (target_score=60 band, n=3933)

| Metric | value |
|---|---:|
| Base raw output median | 63.0016 |
| V2 raw (spline applied) median | -3.0781 |
| V2 clamp output median | 0.0000 |
| |V2_clamp_median − target| | **60.0000** |

JND lands at score 0, not 60. The spline maps all `raw > 62.961`
inputs into y < 0, which the hard clamp pins to 0. Since 6 of 8
training bands (50% of the corpus) have median raw > 62.961, the
majority of pairs collapse to score 0.

### Cross-corpus SROCC (held-out Mohammadi panel via bake_verdict)

| Corpus | n | Base SROCC | V2 SROCC | Δ |
|---|---:|---:|---:|---:|
| CID22  | 4292 | 0.8797 | 0.8797 | 0.0000 |
| KADIK10k | 10125 | 0.8003 | 0.8003 | 0.0000 |
| TID2013 |  3000 | 0.8215 | 0.8215 | 0.0000 |
| KonJND-1k |  1008 | 0.3269 | 0.3269 | 0.0000 |
| AIC-3 CTC |   600 | 0.8060 | 0.8060 | 0.0000 |

SROCC bit-exact preserved across all five corpora — confirms PCHIP
is rank-invariant by construction. But the rank invariance is
**meaningless** when the user-facing output is bimodal-degenerate;
the rank information that bake_verdict's sign-tolerant SROCC sees
through is the same information that `soft_clamp_score` was
squashing on the base bake. The dial cannot be recovered without
expanding the network's raw output range, which is a retraining
problem, not a calibration problem.

## Root cause: cross-codec-equivalence loss compresses dynamic range BY DESIGN

The CrossCodec bake's training loss includes a cross-codec
equivalence term:

```text
L_cc = mean((y_codec_a - y_codec_b)^2)  for (a, b) ∈ equivalence_pairs
```

where equivalence pairs are `(a, b)` with `|butter_pnorm3_a -
butter_pnorm3_b| ≤ 0.5` across `zenjpeg`, `zenwebp`, `zenavif`,
`zenjxl`. Minimizing this loss pulls the network's prediction
*toward a common value at each butter level* regardless of which
codec produced the artifact.

The objective is structurally at odds with dial calibration:

1. The cross-codec loss minimizes inter-codec variance at each
   butter level.
2. Butter levels in the training corpus span the full quality range.
3. To minimize inter-codec variance everywhere, the network learns
   a near-constant function of the features (because the only way
   to predict the same value across 4 different codecs' feature
   distributions at the same butter level is to ignore most of the
   features and output a value close to the corpus mean).
4. The resulting raw output is approximately constant (= corpus
   mean) plus a small residual that preserves rank within each
   (image, codec) pair.

This is the same mechanism that delivers the cross-codec consistency
that motivated the bake — but it sacrifices the dial as a
side-effect. **Spline calibration cannot recover what the loss
discarded.** A monotone spline can stretch a tight band onto a
wider range, but only if the per-band medians are distinct (and
correctly ordered). Here they are neither (6 bands collapse to
[62.985, 63.007], and target=10 is anomalously below target=0).

## What would unfreeze CrossCodec

Two paths, neither is a calibration step:

1. **Retrain with a balance term against the cross-codec loss.**
   E.g., add a `--rank-preserve-weight ≥ 0.5` or
   `--dynamic-range-floor 30` to the trainer, so the loss penalizes
   raw-output range collapse. This would trade some cross-codec
   parity for dial-honest range. The trainer flags exist; the
   experimental cost is one retrain on the cross-codec corpus
   (~hours on the in-flight gpu-trainer).

2. **Replace the cross-codec-eq pair term with a cross-codec-eq
   anchor term.** Instead of `(y_a - y_b)^2`, anchor each codec's
   prediction to the multi-band target (`y_codec_a → target_at_band(a)`,
   etc.), so the training signal still enforces "different codecs
   at the same butter level should hit the same target" but the
   target is the band's MCOS, not the inter-codec mean. This
   preserves dial calibration as a first-class objective. Similar
   to the V9 TunerV3 multi-band anchor recipe.

Both paths are retraining experiments, both are tracked elsewhere
(see EXP-CROSS-CODEC-V8 / V9 follow-ups). Neither is a calibration
step that can be done on the existing bake bytes.

## Ship decision

**Do not ship `PreviewV0_5CrossCodecV2`.** The candidate bake at
`zensim-experimental/weights/v_cross_codec_v2_2026-05-20.bin` is preserved for
provenance but not wired into any `ZensimProfile` variant.

**Deprecate `PreviewV0_5CrossCodec` from public-facing surfaces:**

1. Mark the variant `#[deprecated(note = "dial-broken — raw output
   compressed to ~0.18 score units; cannot be calibrated. See
   benchmarks/v_cross_codec_v2_2026-05-20_falsification.md")]` in
   `zensim/src/profile.rs`. Keep the variant alive so existing
   callers don't break, but flag the dial pathology at use site.
2. Remove the "CrossCodec" trail row from `zensim/SOTA_TRAILS.md`'s
   ship-table — it is not a ship, it is a measurement-only profile.
3. Add a `CHANGELOG.md` entry under `[Unreleased]` documenting the
   dial-broken structural finding.

These follow-ups are tracked under task #179 follow-on; this
falsification doc is the prerequisite artifact.

## Files

- Spline calibration log:
  `benchmarks/v_cross_codec_v2_2026-05-20_calibration.log`
- Per-band spline CSV:
  `benchmarks/v_cross_codec_v2_2026-05-20_spline.csv`
- bake_verdict on CrossCodec base:
  `benchmarks/v_cross_codec_base_2026-05-20_verdict.md`
- bake_verdict on CrossCodecV2 candidate:
  `benchmarks/v_cross_codec_v2_2026-05-20_verdict.md`
- Candidate bake (preserved for provenance):
  `zensim-experimental/weights/v_cross_codec_v2_2026-05-20.bin` (197,073 bytes)

## Provenance

- Worktree: `/home/lilith/work/zen/zensim--cross-codec-v8/`
- Base commit: `28ed2552` (dial-bug audit, task #178)
- Calibrator: `scripts/v_next/calibrate_balanced_v9_spline.py` (unchanged
  from BalancedV2 / CompressionV2 ship)
- Anchor parquet:
  `/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet`
  (22,008 rows × 372 features, 8 target bands)
- `predict_features_with_bake`:
  `target/release/predict_features_with_bake` (built fresh in this
  worktree)
- `zenpredict`:
  `/home/lilith/work/zen/zenanalyze/target/release/zenpredict`
- Task: #179
