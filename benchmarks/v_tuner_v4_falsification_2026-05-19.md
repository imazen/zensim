# `PreviewV0_5TunerV2` from V4 candidates — FALSIFIED on monotonicity + range (2026-05-19)

## Hypothesis (parent task brief)

V4 closes the V3 mono gap (best V3 strict mono 0.9100 vs gate 0.9378)
with two architectural counterweights:

1. **tanh-pinned [0, 100] output head** (V4-C from V3 falsification doc).
   Wraps the per-sample-α head output as `y_score = 100·σ(y_pre/10)`,
   eliminating the post-affine β=5–10× amplification path V3 identified.

2. **Multi-codec PJND anchor** (user directive 2026-05-19 14:55).
   1000 sources × 4 codecs at each codec's PJND-q, all targeting
   score=63. Binds score=63 ↔ PJND across codecs during training.

Plus V4-B: `--monotonicity-reg 1.0` (NOT 5.0; V3's σ-floor already
prevents collapse).

Sweep: 3 seeds × W ∈ {0.5, 1.0} = 6 bakes.

## Verdict: **falsified.**

**0 of 6 V4 candidates pass the Tuner-trail gate.** The architectural
combination achieves the cross-codec calibration goal SPECTACULARLY
(cross-codec PJND score std median 0.10–0.14, **35–50× tighter than
gate ≤ 5.0**) but COMPRESSES the dynamic range so tightly around 63
that q-sweep range collapses to 8–16 score units (gate ≥ 50). Strict
monotonicity correspondingly drops to 0.78–0.84 (gate ≥ 0.9378),
WORSE than V3's best of 0.9100.

## Per-bake gate scorecard

Gates: `strict_mono ≥ 0.9378`, `tied ≤ 5 %`, `range ≥ 50`,
`T=63 butter_max < 2.5 OR butter_p3 < 2.5`,
`cross-codec PJND score std median ≤ 5.0`.

| Bake | mono | tied | q5_med | q95_med | range | T=63 b_max | T=63 b_p3 | PJND std | mono | tied | range | xc | pjnd | ALL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|
| baseline_tuner (existing ship) | 0.9278 | 0.0044 | 4.96 | 94.64 | 89.68 | 8.07 | 2.11 | n/a | ✗ | ✓ | ✓ | ✓ | n/a | (current) |
| cc4v4_s1_w0_5 | 0.7756 | 0.0000 | 56.26 | 64.48 | 8.22 | 4.97 | **1.95** | **0.110** | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |
| cc4v4_s1_w1_0 | 0.8122 | 0.0000 | 54.53 | 64.39 | 9.86 | 5.98 | 2.27 | **0.100** | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |
| cc4v4_s2_w0_5 | 0.8111 | 0.0000 | 49.41 | 65.10 | 15.69 | 5.18 | **2.00** | **0.140** | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |
| cc4v4_s2_w1_0 | 0.8433 | 0.0000 | 55.26 | 64.39 | 9.13 | 4.37 | **1.76** | **0.110** | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |
| cc4v4_s3_w0_5 | 0.8178 | 0.0000 | 53.21 | 65.10 | 11.89 | 5.89 | 2.26 | **0.130** | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |
| cc4v4_s3_w1_0 | 0.8322 | 0.0000 | 54.49 | 65.00 | 10.51 | 4.87 | **1.97** | **0.110** | ✗ | ✓ | ✗ | ✓ | ✓ | FAIL |

(Bold values pass their respective gates.)

## What V4 solved

| Goal | V3 result | V4 result | Verdict |
|---|---|---|---|
| Cross-codec PJND score parity | 5.52 butter_max (single-codec anchor) | std 0.10–0.14 (multi-codec anchor) | **SOLVED dramatically** |
| Tied-rate | 2.8–9.9 % (V3 was borderline) | 0.0 % across all 6 V4 bakes | **SOLVED — tanh pin has no boundary ties** |
| Output range | 89.94–90.02 (V3 after affine) | **8.22–15.69** | **REGRESSED** |
| Strict monotonicity | 0.82–0.91 (V3 best 0.9100) | **0.78–0.84** | **REGRESSED** |
| Cross-codec T=63 butter | 2/6 passed | 5/6 passed (butter_p3) | Improved |

## Why V4 broke range + monotonicity

The multi-codec anchor at weight=1.0 + step_p=0.15 pulls so hard
toward score=63 that the output collapses to a narrow band:

### 1. Anchor MSE dominates the per-pair gradient

Per pair-step:
- 1 RankNet/MSE pair step contributes `2 · mse_weight · (y - target) / (2 · K)`
  where `K = pairs_per_epoch = 50000`. With `mse_weight=1.0`, gradient
  magnitude per step is `~5e-5` × (y − target).
- With probability 0.15, an anchor step ALSO fires, contributing
  `2 · anchor_loss_weight · row_w · (y − 63)` directly (no `1/K` scaling).
  With `anchor_loss_weight=1.0` and `row_w=1.0`, gradient magnitude is
  `~2.0` × (y − 63).

The anchor's gradient is ~40,000× stronger per fired step. Even at
15 % firing rate vs 100 % pair-step rate, the anchor contributes
~6000× more total gradient signal toward score=63 than the per-pair
MSE contributes toward spreading the range.

### 2. σ-floor probe can't overcome anchor pull

The σ-floor probe (target σ=15 across 40 random equiv-pool rows)
fires at 5 % rate with weight 0.2 — total contribution ~`0.05 · 0.2 ·
40 · gradient ≈ 0.4 gradient per pair-step on average`. The anchor
contribution per step (anchor_loss_weight 1.0, step_p 0.15) is
`0.15 · 2.0 · err ≈ 0.3 · err`. With `err` typically 30 score units
(network predicts 30-95, anchor wants 63), anchor dominates 9:1.

### 3. tanh pin compresses extremes back toward the mean

`y_score = 100 · σ(y_pre / 10)` saturates at the extremes — to push
y_score from 63 to 95, y_pre needs to move from 0 to +20 (sigmoid
saturation region). The combined pressure: anchor pulling toward
y_pre=0, σ-floor pushing toward y_pre spread, MSE pulling toward
target. The anchor wins.

Result: q-sweep range 8-16 score units AT BEST (vs V3's 90).

## Multi-codec PJND check passed dramatically

Despite the range collapse, the multi-codec PJND check is the
brightest result of the experiment:

| Bake | agg_mean | per-codec mean spread | cc_std median | cc_std p95 |
|---|---:|---:|---:|---:|
| cc4v4_s1_w0_5 | 62.97 | 62.93–63.00 (0.07) | 0.11 | 0.40 |
| cc4v4_s1_w1_0 | 62.99 | 62.93–63.00 (0.06) | 0.10 | 0.44 |
| cc4v4_s2_w0_5 | 62.94 | 62.94–62.95 (0.01) | 0.14 | 0.41 |
| cc4v4_s2_w1_0 | 62.81 | 62.78–62.84 (0.06) | 0.11 | 0.46 |
| cc4v4_s3_w0_5 | 63.18 | 63.16–63.19 (0.03) | 0.13 | 0.42 |
| cc4v4_s3_w1_0 | 62.87 | 62.84–62.92 (0.08) | 0.11 | 0.42 |

All 6 V4 bakes predict score≈63 for each (source, codec) PJND pair,
with std≈0.5 across all 4000 anchor rows AND cross-codec std per
source <0.5 — the cross-codec calibration mechanism works
perfectly. The cost is the dynamic range.

## What V4 reveals about the architecture

The multi-codec anchor + tanh pin combination MAKES cross-codec
parity at a single anchor point achievable. But the loss landscape
created by 4000 anchor rows at the same target overpowers the
range-spreading mechanisms (per-pair MSE + σ-floor). To recover
both range AND cross-codec parity, the anchor must be:

1. **Weaker per-step** (smaller `anchor_loss_weight`), OR
2. **Lower firing rate** (smaller `anchor_step_p`), OR
3. **Distributed across score targets** (not all rows at score=63
   — some at near-lossless 80, some at noticeable 50, etc.), OR
4. **Score-relative** rather than absolute (e.g., enforce equal
   scores across the 4 codecs WITHOUT specifying what that score
   should be).

V4b is now in flight (dispatched 2026-05-19) testing option 1+2:
anchor_loss_weight ∈ {0.05, 0.10} AND anchor_step_p=0.05 (instead
of V4's 1.0 + 0.15). Same tanh pin + multi-codec anchor data.

## V5 direction proposals (if V4b also falsifies)

### V5-A: anchor-as-rank-preserve (not absolute MSE)

Replace `(y_anchor − 63)²` MSE with a RankNet-style sigmoid loss
that penalizes only when two SAME-source DIFFERENT-codec anchor
rows produce different scores:

```
L_anchor_rank = w · |Δ| · −log(sigmoid(s · (y_a − y_b)))
```

where (a, b) are two codecs at the same source's PJND, and `s` is
the sign of their (closer-to-63) preference. Doesn't pull either
toward an absolute value — just toward each other. The
score-spread comes entirely from the per-pair MSE.

### V5-B: smaller multi-codec anchor with target distribution

Build the anchor parquet at NOT just PJND-q but ALSO at other q
levels (q=20, 40, 60, 80) per codec, with target scores aligned to
the butter quality curve (e.g., butter_pnorm3 = 0.5 → target 90,
1.5 → target 63, 3.0 → target 30). The anchor enforces calibration
across a RANGE of scores, not just at 63 — which keeps range
expressed.

### V5-C: schedule the anchor weight

Start with `anchor_loss_weight = 0.0` for the first 100 epochs
(per-pair MSE establishes the score-shaped output range), THEN
ramp up the anchor weight from 0.0 to 1.0 over epochs 100–200
(cross-codec calibration aligned to the established range). The
gradient balance shifts over training, not all-at-once.

### V5-D: explicit range floor regularizer

Replace the σ-floor probe (heterogeneous A-side equiv pool) with
a per-image q-sweep substrate: for each of N training images,
forward at q ∈ {5, 25, 50, 75, 95}, enforce `σ ≥ 25 raw units`
(within-image, NOT cross-image). Aligned with V3 falsification
proposal V4-A but compatible with multi-codec anchor.

## Decision

**No PreviewV0_5TunerV2 ship from V4.** PreviewV0_5Tuner (baseline
tuner, V_tuner-v2-s2 calibrated, range 89.68, strict mono 0.9278)
remains the dial profile. PreviewV0_5CrossCodec (V2 W=1.0 seed=1,
T=63 butter 5.52) remains the cross-codec profile.

The V4 architectural changes (tanh-pinned [0, 100] output head +
multi-codec PJND anchor) ARE shipped to main as
infrastructure-on-shelf — they work cleanly and the cross-codec
calibration component is independently valuable. Future V5
experiments will reuse this infrastructure.

V4b is in flight at the time of this writing (relaxed-anchor sweep,
anchor_loss_weight ∈ {0.05, 0.10}, anchor_step_p=0.05). Verdict
will be appended below when complete.

## Files produced this session

- `benchmarks/v_tuner_v4_methodology_2026-05-19.md` — hypothesis +
  recipe + post-hoc training observations.
- `benchmarks/v_tuner_v4_falsification_2026-05-19.md` — this doc.
- `scripts/v_next/build_multi_codec_pjnd_anchors.py` — anchor parquet builder.
- `scripts/v_next/run_cross_codec_v4_seed.sh` — V4 trainer driver.
- `scripts/v_next/run_cross_codec_v4_consistency.sh` — V4 T=63 driver.
- `scripts/v_next/eval_cross_codec_v4.sh` — V4 eval pipeline (no affine).
- `scripts/v_next/eval_v4_pjnd_check.py` — multi-codec PJND analyzer.
- `scripts/v_next/summarize_v4.py` — combined-table renderer.
- `scripts/v_next/run_cross_codec_v4b_seed.sh` — V4b (relaxed anchor) driver.
- `/mnt/v/zen/zensim-training/2026-05-19-multi-codec-jnd-anchors/anchors_multi_codec_372col.parquet`
  — 4000-row multi-codec PJND anchor.
- `/mnt/v/zen/zensim-eval/exp_cross_codec_v4_2026-05-19/` — bakes + eval.

## Trainer + runtime commits

- `2934d0c6` (main, 2026-05-19): trainer tanh-output-head + multi-codec
  anchor + runtime dispatch.
- `7a45eb67` (main, 2026-05-19): eval pipeline + runtime tanh-pin
  dispatch in every bake-aware tool (bake_verdict, qsweep_eval,
  ensemble_score_rows, score_pair_with_bake).

## Reproduction

```bash
cd /home/lilith/work/zen/zensim--cross-codec-metric

# 0. Build trainer + tools.
cargo build --release --bins -p zensim-validate
cargo build --release --bin score_pair_with_bake -p zensim-validate

# 1. Build multi-codec anchor parquet.
python3 scripts/v_next/build_multi_codec_pjnd_anchors.py

# 2. Train 6 V4 bakes in parallel (~45 min on 7950X).
for seed in 1 2 3; do
  for w in 0.5 1.0; do
    bash scripts/v_next/run_cross_codec_v4_seed.sh $seed $w &
  done
done; wait

# 3. Eval pipeline.
bash scripts/v_next/eval_cross_codec_v4.sh
bash scripts/v_next/run_cross_codec_v4_consistency.sh
python3 scripts/v_next/eval_v4_pjnd_check.py /mnt/v/zen/zensim-eval/exp_cross_codec_v4_2026-05-19

# 4. Summary.
python3 scripts/v_next/summarize_v4.py /mnt/v/zen/zensim-eval/exp_cross_codec_v4_2026-05-19
```

Total compute: ~50 min on 7950X. Total data: ~1.6 MB bakes + ~15 MB
eval artifacts + 7 MB anchor parquet.
