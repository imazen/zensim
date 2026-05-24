# zensim — context handoff (2026-05-24)

Written for the next session reset. Read this first, then
[SESSION-RESUME.md](SESSION-RESUME.md), [RESEARCH.md](RESEARCH.md),
and [CLAUDE.md](CLAUDE.md).

## TL;DR — codec-target metric is shipped + canonical

`ZensimProfile::codec_target()` is the **stable alias every zen
codec uses** for the quality dial + picker training. Currently
routes to `PreviewV0_5TunerV4` (`v_tuner_v10_2026-05-20.bin`).
See [`docs/CODEC_TARGET_METRIC.md`](docs/CODEC_TARGET_METRIC.md)
for the integration guide.

**Measured cross-trail advantage (2026-05-24)**: Tuner v10 is
**3-6× tighter cross-codec** than the other two trail ships
(Balanced/Compression) — measured on 68,788 matched-anchor pairs.
Tuner is **also the only trail with zero clamp-flat tied regions**
on the JPEG q-sweep, vs 7-13 dead zones in Balanced/Compression.
Picking either of those as the codec dial would have given users
±20-50 score-unit precision swings; Tuner's ±3-8 is the production
floor. See
[`benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md`](benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md).

## What happened on 2026-05-24 (today's session)

### A. Three-trail SOTA framework — codec_target() alias landed

- `ZensimProfile::codec_target()` (commit `5ca977c`) — stable alias
  pointing at the current Tuner ship. Rotates via single-line edit
  when next Tuner variant ships.
- `docs/CODEC_TARGET_METRIC.md` — integration guide for Patterns
  A (quality-target dial, e.g. zenwebp's `target_zensim`),
  B (picker training), C (in-encoder RDO — DEFERRED per
  `docs/RDO_LOSS_FEASIBILITY_2026-05-24.md`; output-only zensim is
  the current SOTA pattern for all production codecs).

### B. Per-source aggregation head for konjnd-dense — TRAINER LANDED

Architectural fix for the V11-D zero-gradient pathology. New trainer
flags: `--konjnd-aggregation-{parquet,weight,step-p,samples-per-ref,refs-per-step}`.

Mechanism: sample K refs × S rows per fire, forward K·S times,
aggregate per-ref mean, MSE against per-ref pjnd_target, backprop
`(2w/S)·residual` per row. RUNTIME UNCHANGED.

- Phase 1 (commit `d1ac861`): hyperparams + struct + parquet loader.
- Phase 2 (commit `a08151d`): wire aggregation step.
- Phase 3 (commit `ebf5f2e`): synthetic gradient-flow tests (2 pass).

### C. CVVDP + IW-SSIM backfill on cid22_train.parquet — DONE

17,611 pairs / 201 non-validation CID22 refs now have cvvdp + iwssim
+ mix_cv40_iw60 populated alongside the existing ssim2_gpu anchor.
Canonical-2026-05-21 manifest sha256 updated (54.04 MB, sha256
`523a96c0cd93…`). Enables Tuner v11 retrain with mix_cv40_iw60 target
on the new corpus.

### D. Tuner v11 retrain — IN FLIGHT (attempt 2 of N)

The 5-criterion ship gate (per
`benchmarks/v_tuner_v11_methodology_2026-05-24.md`):
1. KonJND val SROCC ≥ 0.85 (v10 floor 0.2317)
2. CID22 SROCC ≥ 0.864
3. Monotonicity ≥ 92.78%
4. Cross-codec p50 |Δ| ≤ 1.0 in score 60-90
5. Score 0-55 dial recovers from v10 floor pathology

**Attempt 1 (`konjnd_aggregation_weight=0.3`) FALSIFIED 03:30 UTC.**
Seed 1 result: KonJND +0.53 (0.23→0.76, aggregation head WORKS) but
CID22 -0.35 (0.85→0.51). Weight too high — aggregation gradient
overwhelms rank signal, α gate collapses to 0 (pool-only). Evidence
preserved at `tuner_v11_w0.3_s1_evidence.bin`.

**Attempt 2 (`w=0.05 step_p=0.10`, ~5.5% of original effective rate)
in flight.** Pipeline running in background as task `bch7n3mzp`.
Early signal: α(x) dynamic [0, 1] with μ=0.84 — network finds
per-sample blend. Final verdict landing ~04:00 UTC.

**Attempt 3 plan** (if attempt 2 also fails): konjnd-dense as BOTH
training group (--target-column mix_cv40_iw60, train_w=0.3) AND
aggregation pool (--konjnd-aggregation-weight 0.1). Script at
`scripts/v_next/run_tuner_v11_attempt3_seed.sh`. Hypothesis: the
per-pair MSE on konjnd-dense (which v10 never trained on) is the
missing signal — aggregation alone gives only per-ref hints.

### E. Hparam sweep tool

`scripts/v_next/tuner_v11_hparam_sweep.sh` maps the
(weight, step_p) surface across 6 cells in ~90 min. Use this if
attempts 2 and 3 both fail to find the right balance.

## Outstanding tasks (per TaskList)

- **#6 in progress**: Tuner v11 retrain. Attempt 2 in flight; attempt
  3 queued if needed.
- All other tasks completed (#1–#5, #7).

## What's PROHIBITED (per CLAUDE.md sharpened 2026-05-24)

The "pausing is prohibited" rule got sharpened today after I drafted
a "natural milestone" pause framing. Rule body now at the bottom of
`NEVER PAUSE LAZILY` in `~/work/claudehints/CLAUDE.md`. Re-read
before drafting any end-of-chunk message.

## Pointers (canonical)

| What | Where |
|---|---|
| Codec-target integration guide | `docs/CODEC_TARGET_METRIC.md` |
| Cross-codec baseline (v10 + cross-trail) | `benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md` |
| Tuner v11 methodology + iteration history | `benchmarks/v_tuner_v11_methodology_2026-05-24.md` |
| Aggregation head design | `docs/KONJND_AGGREGATION_HEAD_DESIGN_2026-05-24.md` |
| RDO loss scoping | `docs/RDO_LOSS_FEASIBILITY_2026-05-24.md` |
| 5-seed CI pipeline driver | `scripts/v_next/tuner_v11_full_pipeline.sh` |
| Per-seed runner | `scripts/v_next/run_tuner_v11_seed.sh` |
| Attempt 3 runner (konjnd as group + agg) | `scripts/v_next/run_tuner_v11_attempt3_seed.sh` |
| Hparam sweep driver | `scripts/v_next/tuner_v11_hparam_sweep.sh` |
| Canonical training data | `/mnt/v/zen/zensim-training/canonical-2026-05-21/` |
| Eval out dir | `/mnt/v/zen/zensim-eval/exp_tuner_v11_2026-05-24/` |
