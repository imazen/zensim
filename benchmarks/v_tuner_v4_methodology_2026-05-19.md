# `PreviewV0_5TunerV2` from V4 candidates — methodology (2026-05-19)

EXP-CROSS-CODEC-V4 builds on V3 (commits `de097f1`, `579229e`,
`11d8af3`) with two architectural counterweights to close the
mono-violation gap that falsified all 6 V3 bakes (best V3 strict mono
0.9100 vs gate 0.9378). The V3 falsification doc
(`benchmarks/v_tuner_v3_falsification_2026-05-19.md`) identified the
post-affine β-amplification path as the dominant mono-violation cause,
and single-codec anchor as the cross-codec calibration weak point.

## Hypothesis (V4 brief)

> V3 candidates have raw output range 10–13 score units (q5 median ~ 52,
> q95 median ~ 67). Post-affine β = 5–10× maps that to 0–100, which
> amplifies per-pair raw jitter (~0.05–0.2 raw) into 0.5–2 score-unit
> mono violations. Pinning the output natively in [0, 100] AT TRAINING
> TIME eliminates the β-amplification path entirely.
>
> Replacing V3's single-codec (zenjpeg-only) anchor with a multi-codec
> anchor (4 codecs × 1000 sources at score=63 each) binds the PJND
> calibration directly across codecs during training, rather than
> relying on the equivalence-pair MSE alone to align them.

## V4 architectural changes

### 1. tanh-pinned [0, 100] output head (V4-C from falsification doc)

Wraps the per-sample-α head output as:

```
y_pre   = α(x) · y_rank + (1 − α(x)) · y_pool       (current V3 output)
y_score = 100 · σ(y_pre / scale)                     (V4 sigmoid pin)
```

with `scale = 10.0` (active linear region `y_pre ∈ [−30, 30]` mapping to
roughly `[5, 95]` score units, saturating beyond that).

Backprop: `dL/dy_pre = dL/dy_score · (100/scale) · σ · (1 − σ)`. Every
upstream loss site (RankNet pair, MSE pair, monotonicity hinge, anchor
MSE, equivalence MSE, rank-preserve, range-floor probe) computes its
gradient in score-space (`dL/dy_score`), then multiplies by
`dy_score/dy_pre` before passing to `backprop_step_per_sample_alpha_head`.

Bake-side: a `zentrain.tanh_output_head` metadata entry with payload
`[scale: f32 LE]` (4 bytes). The runtime in `zensim::metric::apply_mlp_scoring`,
`bake_verdict::score_row`, `qsweep_eval::score_row`,
`ensemble_score_rows::score_row`, and `score_pair_with_bake::score_with_bake`
all check for this key and apply the matching sigmoid pin
bit-exactly.

### 2. Multi-codec PJND anchor (user directive 2026-05-19 14:55)

V3 used a single-codec (zenjpeg-only) anchor at
`/mnt/v/zen/zensim-training/2026-05-19-jnd-anchors/anchors_372col.parquet`
(9373 rows from KonJND-1k+zenjpeg PJND samples). The single-codec anchor
binds score=63 ↔ JPEG PJND, but provides no signal to align WebP/AVIF/JXL
at the same score.

V4 anchor at
`/mnt/v/zen/zensim-training/2026-05-19-multi-codec-jnd-anchors/anchors_multi_codec_372col.parquet`
is built by `scripts/v_next/build_multi_codec_pjnd_anchors.py`:

  - For each codec in `{zenjpeg, zenwebp, zenavif, zenjxl}`:
    - For each source image (1000 per codec from
      `/mnt/v/zen/picker-training/2026-05-19/butter/<codec>.parquet`):
      - Pick the q where `|butter_pnorm3 − 1.5|` is smallest (PJND
        per `CORRECTION_FROM_PARENT.md`).
      - Emit one row: `(ref_basename, anchor_source=<codec>_pjnd,
        anchor_weight=1.0, target_score=63.0, codec, pjnd_q, ...features f0..f371)`.

  - Final size: 4000 rows × 379 cols.

  - Per-codec median PJND-q observed:
      - zenjpeg: q=60 (median butter_pnorm3=1.5017)
      - zenwebp: q=55 (median butter_pnorm3=1.5003)
      - zenavif: q=55 (median butter_pnorm3=1.5071)
      - zenjxl:  q=70 (median butter_pnorm3=1.4975)

The trainer's anchor-step samples (with prob `--anchor-step-p`) one row
at random, forwards through the per-sample-α + tanh-pin head, and
applies `w · row_w · (y_score − 63)²` MSE. Since rows span 4 codecs at
their per-codec PJND-q, the network must learn to predict score=63 for
ALL of them — directly binding cross-codec calibration during training.

### 3. Other recipe knobs (V4-B + V3 carryover)

| Knob | V3 | V4 | Why |
|---|---|---|---|
| `--monotonicity-reg` | 5.0 | **1.0** | V3 falsification finding: σ-floor already prevents collapse structurally; high mono-reg was counterproductive (interferes with per-curve smoothness) |
| `--anchor-step-p` | 0.10 | **0.15** | Multi-codec anchor has 4× signal density at same epoch budget; slightly higher step-p balances vs the 50000 pair-steps |
| `--dynamic-range-floor-weight` | 0.2 | 0.2 (unchanged) | σ-floor still needed to prevent collapse in the relaxed-mono regime |
| `--cross-codec-eq-weight` | {0.5, 1.0} | {0.5, 1.0} (matrix unchanged) | Sweep both weights |
| `--cross-codec-rank-preserve-weight` | 0.2 | 0.2 (unchanged) | Rank-preserve still useful for equiv-pair ordering |
| `--ranknet-weight` | 0.0 | 0.0 | Pure-MSE training on score-shaped target |
| `--mse-weight` | 1.0 | 1.0 | Per-pair MSE against `mix_cv40_iw60` (already in 0..100 range) |
| `--tanh-output-head-scale` | n/a | **10.0** | V4 new |

## V4 sweep matrix

3 seeds × 2 cross-codec-eq weights = 6 bakes:

| Bake | Seed | W (cross-codec-eq) | Output |
|---|---|---|---|
| cc4v4_s1_w0_5 | 1 | 0.5 | bake + log under /mnt/v/zen/zensim-eval/exp_cross_codec_v4_2026-05-19/ |
| cc4v4_s1_w1_0 | 1 | 1.0 | " |
| cc4v4_s2_w0_5 | 2 | 0.5 | " |
| cc4v4_s2_w1_0 | 2 | 1.0 | " |
| cc4v4_s3_w0_5 | 3 | 0.5 | " |
| cc4v4_s3_w1_0 | 3 | 1.0 | " |

Driver: `scripts/v_next/run_cross_codec_v4_seed.sh <seed> <W>`.
Trainer: `target/release/zensim_mlp_train` (built from `feat/cross-codec-v4`).
Total compute: ~30–45 min on 7950X at 6-way parallel.

## Ship gate (combined)

A V4 candidate passes if ALL of the following hold:

| Phase | Gate | Source |
|---|---|---|
| qsweep mono | strict_mono ≥ 0.9378 | qsweep_eval on JPEG 50img × 19q |
| qsweep tied | tied ≤ 5 % | qsweep_eval |
| qsweep range | range ≥ 50 | qsweep_eval (calibrated clamp(0, 100)) |
| T=63 cross-codec | butter_max < 2.5 OR butter_p3 < 2.5 | cross_codec_consistency.py n=20 |
| Multi-codec PJND | cross-codec score std per source median ≤ 5.0 | eval_v4_pjnd_check.py |

If any candidate passes ALL: ship as `PreviewV0_5TunerV2`. Wire into
`zensim/src/profile.rs` (PreviewV0_5CrossCodec pattern from commit
`6bef807` but add tanh-pin metadata via the runtime applied at score
time — already plumbed in `apply_mlp_scoring`). Add bake at
`zensim/weights/v_tuner_v4_2026-05-19.bin`. Update
`zensim/SOTA_TRAILS.md` Tuner trail.

If none pass: write `benchmarks/v_tuner_v4_falsification_2026-05-19.md`
with the per-bake table + proposed V5 direction.

## Training-time observations (preliminary)

Across all 6 V4 bakes, α(x) saturates to ~0 within 10 epochs (μ < 0.001
log-line). With α=0, the per-sample-α head degenerates to pure pool-head:

```
y_pre = 0 · y_rank + 1 · y_pool = y_pool
y_score = 100 · σ(y_pool / 10)
```

This is expected behavior for pure-MSE training on a score-shaped target:
the rank-head's gradient signal vanishes (RankNet weight = 0), and the
pool-head's 4 stat features ([μ, σ, max, p_6](h)) + bias are sufficient
to predict the mix_cv40_iw60 score directly. The 128-wide rank head's
extra capacity isn't needed when the target is already in score space.

The tanh pin's gradient `(100/scale)·σ·(1−σ)` stays in [0.5, 2.5]
across the interior [5, 95] score region — gradient never vanishes, so
the pool-head + sigmoid pin trains stably. SROCC climbs from 0.97 → 0.98+
within 50 epochs.

## Implementation provenance

- Trainer changes: `zensim-validate/src/mlp_train.rs` (added
  `tanh_output_head_scale` to `MlpHyperparams`, wired pin into every
  per-sample-α loss site).
- New baker: `zensim-train-core/src/per_sample_alpha_head.rs` adds
  `bake_per_sample_alpha_head_v3_with_tanh` (emits both
  `per_sample_alpha_head` + `tanh_output_head` metadata).
- Runtime: `zensim/src/metric.rs` adds `apply_tanh_output_pin` +
  dispatch in `forward_one_bake`; `bake_verdict.rs`,
  `qsweep_eval.rs`, `ensemble_score_rows.rs`,
  `score_pair_with_bake.rs` all extract + apply the pin.
- Tests: `zensim-train-core/per_sample_alpha_head.rs` has 2 new tests
  (`bake_per_sample_v3_with_tanh_has_both_metadata_keys`,
  `bake_per_sample_v3_with_tanh_rejects_zero_scale`).
- Anchor builder: `scripts/v_next/build_multi_codec_pjnd_anchors.py`.
- Driver: `scripts/v_next/run_cross_codec_v4_seed.sh`.
- Eval driver: `scripts/v_next/eval_cross_codec_v4.sh` (no affine
  phase — V4 outputs natively in [0, 100]).
- PJND check: `scripts/v_next/eval_v4_pjnd_check.py`.

All file paths are absolute under `/home/lilith/work/zen/zensim--cross-codec-metric/`.
