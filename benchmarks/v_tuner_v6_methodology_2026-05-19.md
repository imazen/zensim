# PreviewV0_5TunerV2 (V6 anchor-pressure-restored) methodology — 2026-05-19

**Status:** SHIPPED 2026-05-19 as `PreviewV0_5TunerV2`.
**Bake:** `/home/lilith/work/zen/zensim--cross-codec-metric/zensim/weights/v_tuner_v6_2026-05-19.bin`
**md5:** `c5c32659b15b47e8a569464749cf7019`
**Size:** 261,351 bytes (F32 uncompressed).
**Ship gate:** ALL 6 Tuner-trail gates PASS (3 of 3 evaluated seeds at anchor_w=1.0).

## Hypothesis

V5 (commit e0d869f4) introduced the piecewise multi-band anchor — 6
butter bands × 4 codecs × ~1000 sources with per-row `target_score`
∈ {90, 75, 63, 45, 25, 10}. V5 passed 5 of 6 Tuner gates spectacularly
(cross-codec cc_std_median ≤ 1.04 at every band, monotonicity 0.9767)
but **failed `median_range ≥ 50`** (V5 outputs clustered in [37, 70]
regardless of which band the anchor row targeted).

**Root cause (V5 falsification doc, 2026-05-19):** anchor pressure
too weak. At `--anchor-loss-weight 0.05 --anchor-step-p 0.15`, only
~7.5% of pair-steps trigger an anchor step with weight 0.05. The
per-pair MSE on `mix_cv40_iw60` + the `--cross-codec-eq-weight 1.0`
equivalence loss dominate, compressing the output range to the
training-target distribution centre. Anchor band targets never
materialized as actual output values.

**V6 fix:** restore anchor pressure by 20× weight × 2× step
probability while keeping every other V5 lever identical.

```diff
-  --anchor-loss-weight 0.05
-  --anchor-step-p 0.15
+  --anchor-loss-weight 1.0   (or 0.5 in the parallel arm)
+  --anchor-step-p 0.30
```

## Trainer matrix

3 seeds × 2 anchor weights = 6 bakes. Anchor step probability fixed
at 0.30 throughout.

| Bake | seed | anchor_w | anchor_step_p |
|---|---:|---:|---:|
| cc4v6_w0p5_p0p30_s1 | 1 | 0.5 | 0.30 |
| cc4v6_w0p5_p0p30_s2 | 2 | 0.5 | 0.30 |
| cc4v6_w0p5_p0p30_s3 | 3 | 0.5 | 0.30 |
| cc4v6_w1p0_p0p30_s1 | 1 | **1.0** | 0.30 |
| cc4v6_w1p0_p0p30_s2 | 2 | 1.0 | 0.30 |
| cc4v6_w1p0_p0p30_s3 | 3 | 1.0 | 0.30 |

Out dir: `/mnt/v/zen/zensim-eval/exp_cross_codec_v6_2026-05-19/`.

Driver: `scripts/v_next/run_cross_codec_v6_seed.sh <seed> <anchor_w> <anchor_step_p>`.

## Full trainer command (per bake, only anchor_w differs)

```sh
target/release/zensim_mlp_train \
    --group safesyn:/mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet:1.0:0.0 \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 1 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale 15.0 \
    --ranknet-weight 0.0 \
    --mse-weight 1.0 \
    --monotonicity-reg 1.0 \
    --monotonicity-margin 0.0 \
    --anchor-parquet /mnt/v/zen/zensim-training/2026-05-19-multi-band-anchors/anchors_multi_band_372col.parquet \
    --anchor-loss-weight ${ANCHOR_W} \
    --anchor-target-score 63.0 \
    --anchor-step-p 0.30 \
    --cross-codec-eq-parquet /mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet \
    --cross-codec-eq-weight 1.0 \
    --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.2 \
    --dynamic-range-sigma-threshold 15.0 \
    --dynamic-range-step-p 0.05 \
    --dynamic-range-probe-n 40 \
    --seed ${SEED} --out ${OUT_BAKE} --log-path ${LOG}
```

## Inputs (md5 / row count where computable)

- Train corpus: `/mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet`
  (196,086 rows × 372 features × multiple target columns; canonical
  store per CLAUDE.md "Canonical training/validation corpora 2026-05-18").
- Multi-band anchor:
  `/mnt/v/zen/zensim-training/2026-05-19-multi-band-anchors/anchors_multi_band_372col.parquet`
  (18,459 rows × 381 cols, 6 butter bands × 4 codecs × ~1000 sources).
  Target_score per row ∈ {90, 75, 63, 45, 25, 10}.
- Cross-codec equivalence pool:
  `/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet`
  (zenjpeg/zenwebp/zenavif/zenjxl pairs at matched butter levels).

## All 6 ship gates — final verdict

The Tuner trail has 6 gates total. V5 best (cc4v5_s1) failed gate 3
(medRange = 30.73, needed ≥ 50). **V6 passes all 6 gates across all
3 anchor_w=1.0 seeds AND all 3 anchor_w=0.5 seeds:**

| Bake | mono ≥ 0.9378 | tied ≤ 5% | medRange ≥ 50 | T63 butter_p3 < 2.5 | PJND cc_std ≤ 5 | All-band cc_std ≤ 5 | passed |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| cc4v6_w0p5_p0p30_s1 | PASS (0.9422) | PASS (0.0000) | PASS (75.67) | PASS (1.685) | PASS (1.09) | PASS (max 2.33) | **6/6** |
| cc4v6_w0p5_p0p30_s2 | PASS (0.9389) | PASS (0.0000) | PASS (74.33) | PASS (1.745) | PASS (1.20) | PASS (max 2.58) | **6/6** |
| cc4v6_w0p5_p0p30_s3 | PASS (0.9489) | PASS (0.0000) | PASS (71.82) | PASS (1.712) | PASS (0.99) | PASS (max 1.94) | **6/6** |
| cc4v6_w1p0_p0p30_s1 | PASS (0.9522) | PASS (0.0000) | PASS (78.17) | PASS (1.731) | PASS (0.91) | PASS (max 1.68) | **6/6** SHIP |
| cc4v6_w1p0_p0p30_s2 | PASS (0.9433) | PASS (0.0000) | PASS (75.48) | PASS (1.685) | PASS (0.93) | PASS (max 1.73) | **6/6** |
| cc4v6_w1p0_p0p30_s3 | PASS (0.9378) | PASS (0.0000) | PASS (76.17) | PASS (1.707) | PASS (0.89) | PASS (max 1.67) | **6/6** |

**Ship selection: `cc4v6_w1p0_p0p30_s1`** — highest monotonicity
(0.9522), highest CID22 SROCC (0.8770), best PJND cross-codec parity
(cc_std_median 0.91).

Comparison vs V5 best (cc4v5_s1):

| Metric | V5 best (cc4v5_s1) | V6 ship (cc4v6_w1p0_p0p30_s1) | Δ |
|---|---:|---:|---:|
| strict monotonicity | 0.9767 | 0.9522 | −0.025 |
| tied rate | 0.0000 | 0.0000 | 0.000 |
| median range | 30.73 (FAIL) | **78.17** (PASS) | **+47.4** |
| T=63 butter_p3 | 1.53 | 1.73 | +0.20 |
| PJND cc_std_median | 1.04 | 0.91 | −0.13 |
| all-band cc_std max | 1.04 | 1.68 | +0.64 |

The mono drop of 0.025 and all-band cc_std bump of 0.64 are the
trade-offs to gain the +47-point range that V5 failed. All 6
absolute gate thresholds remain satisfied with margin.

## Per-band anchor-target achievement

V6 outputs span the anchor band targets in a way V5 could not.
Achieved means (per-band aggregate across all 4 codecs × ~1000
sources, V6 anchor_w=1.0 seed=1):

| butter | target_score | V5 best (cc4v5_s1) | **V6 ship (cc4v6_w1p0_p0p30_s1)** |
|---:|---:|---:|---:|
| 0.30 | 90.0 | 70.61 | **86.54** |
| 0.80 | 75.0 | 68.26 | **76.90** |
| 1.50 | 63.0 | 61.14 | **62.40** |
| 2.50 | 45.0 | 52.73 | **45.09** |
| 4.00 | 25.0 | 45.34 | **28.11** |
| 6.00 | 10.0 | 40.50 | **14.48** |

V5's outputs compressed into [40, 70] regardless of band — anchor
targets at score=10/90 were ignored. V6 reaches all band targets
within ±5 score units while preserving cross-codec parity at
cc_std_median ≤ 1.68 at every band.

## Mohammadi panel (held-out validation, ship bake cc4v6_w1p0_p0p30_s1)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8770 | 0.8704 | 0.6944 | 0.0424 | 0.9205 | 0.492 |
| KADIK10k | 10125 | 0.7179 | 0.7132 | 0.5328 | 0.0431 | 0.7994 | 0.701 |
| TID2013 | 3000 | 0.7542 | 0.7507 | 0.5660 | 0.0600 | 0.8099 | 0.661 |
| KonJND-1k (full) | 1008 | 0.1962 | 0.1716 | 0.1348 | 0.0387 | 0.3077 | 0.985 |
| AIC-3 CTC | 600 | 0.7961 | 0.8058 | 0.6291 | 0.0650 | 0.8722 | 0.592 |

**CID22 0.8770 essentially ties PreviewV0_5Tuner (0.8786)** while
adding the multi-band cross-codec parity property. Per Tuner-trail
gate criteria in `SOTA_TRAILS.md`: no SROCC gate ("rank-honest
cross-corpus performance is explicitly secondary for this trail").

## Honest gaps — what V6 does worse than predecessors

- **Mono drops 0.025 vs V5 best** (0.9767 → 0.9522). Still passes
  the 0.9378 gate with margin. Trade made for range.
- **All-band cc_std max climbs from 1.04 to 1.68.** Still well
  under the 5.0 gate at every band.
- **Synthetic-corpus ranking** (KADID 0.72, TID 0.75, KonJND 0.20)
  remains the safesyn-only-training limitation inherited from
  PreviewV0_5Tuner. This profile is opt-in for codec auto-targeting,
  NOT general ranking — same caveat documented on the variant.
- **B0..B5 lift not measured here** — V6 is a Tuner-trail ship
  defending a different objective (anchor-band coverage + cross-codec
  parity), not the rank-trail compression frontier.

## Calibration script + α/β

**None.** The bake's `zentrain.tanh_output_head` metadata payload
(scale=15.0) is applied by the runtime via the sigmoid pin in
`zensim::metric::forward_one_bake` (commits 6bef807). The raw
per-sample-α head output `y_pre` maps to
`100 / (1 + exp(-y_pre/15))`, producing values in (0, 100) by
construction. No external affine calibration needed; the active
linear region `y_pre ∈ [-30, 30]` covers `[5, 95]` score units.

## Runtime wiring

- Profile variant: `ZensimProfile::PreviewV0_5TunerV2` (added 2026-05-19).
- Profile params: `PROFILE_PREVIEW_V0_5_TUNER_V2` (same shape as
  `PROFILE_PREVIEW_V0_5_CROSS_CODEC`, with `skip_score_mapping: true`,
  `extended_features: true`, `compute_iw_features: true`,
  `soft_clamp_score: false` — the tanh pin makes hard clamp a no-op).
- Bake bytes loader: `mlp_bake_preview_v0_5_tuner_v2()` →
  `include_bytes!("../weights/v_tuner_v6_2026-05-19.bin")`.
- Runtime forward dispatches per-sample-α head (via the
  `zentrain.per_sample_alpha_head` metadata key on the bake) AND the
  tanh-output-head pin (via the `zentrain.tanh_output_head` key).
  Both keys parsed in `metric.rs` and applied in `forward_one_bake`
  — same path used by `PreviewV0_5CrossCodec`.

## Smoke tests

`zensim/tests/tuner_v2_profile.rs` — 4 tests:
1. `tuner_v2_profile_name` — variant name matches.
2. `tuner_v2_score_in_range` — single-pair score is finite + in [0, 100].
3. `tuner_v2_score_in_range_across_distortion_levels` — score
   remains finite + in-range across delta 0..50.
4. `tuner_v2_differs_from_tuner_and_cross_codec_on_typical_pair` —
   bake bytes are distinct from PreviewV0_5Tuner AND
   PreviewV0_5CrossCodec (different weights + different metadata).

All 4 tests pass on release build.

## Data lineage

| File | Path | Rows | sha256 (prefix) | CID22-contam |
|---|---|--:|---|---|
| safesyn.parquet | `canonical-2026-05-18/train/` | 196,086 | `1ee0565fb6cb` | clean (per CLAUDE.md canonical section) |
| anchors_multi_band_372col.parquet | `2026-05-19-multi-band-anchors/` | 18,459 | (not recorded) | derived from safesyn-disjoint pool, picker-side build |
| cross_codec_equivalence_tight_v3.parquet | `picker-training/2026-05-19-v2/` | ~58k pairs | (not recorded) | picker-side build, distinct from CID22 ref pool |

No CID22 human MOS used in training (per CLAUDE.md "CID22 is
VALIDATION-ONLY" rule). CID22 SROCC reported above is held-out only.

## Eval scripts (reproducible)

- Driver: `scripts/v_next/eval_cross_codec_v6.sh` — runs Phases 1-5.
- Phase 1 (qsweep): `target/release/qsweep_eval` against the 50-image
  × 19-q JPEG sweep manifest, mode=clamp.
- Phase 2 (Mohammadi panel): `target/release/bake_verdict` against
  canonical val parquets.
- Phase 3 (T=63 cross-codec consistency): `scripts/v_next/cross_codec_consistency.py`.
- Phase 4 (single-band PJND): `scripts/v_next/eval_v6_pjnd_check.py`.
- Phase 5 (multi-band): `scripts/v_next/eval_v6_multi_band_check.py`.

Outputs in `/mnt/v/zen/zensim-eval/exp_cross_codec_v6_2026-05-19/`:
- `qsweep_v6.md` — per-bake monotonicity, tied rate, score-per-q histogram.
- `verdicts/*.md` — per-bake Mohammadi panel (5 corpora).
- `cross_codec_t63/*.tsv` — per-bake T=63 cross-codec butter measurements.
- `v6_pjnd_check.md` — single-band PJND cc_std table.
- `v6_multi_band_check.md` — per-bake × per-band achievement + cc_std.

## Predecessor docs

- V5 falsification: `benchmarks/v_tuner_v5_falsification_2026-05-19.md`
- V5 methodology: `benchmarks/v_tuner_v5_methodology_2026-05-19.md`
- V4 / V4b: `benchmarks/v_tuner_v4_methodology_2026-05-19.md`
- V3: `benchmarks/v_tuner_v3_methodology_2026-05-19.md`
- V2 (the original PreviewV0_5Tuner): `benchmarks/v_tuner_v2_methodology_2026-05-19.md`
- Original Tuner: `benchmarks/v_tuner_2026-05-18_methodology.md`
