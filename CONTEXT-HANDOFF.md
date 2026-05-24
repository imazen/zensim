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

### D. Tuner v11 retrain — COMPLETE (no ship; v10 stays canonical)

5 attempts run, all FALSIFIED on the 5-criterion ship gate.
Architectural breakthrough but trade-off doesn't clear gate.

| Attempt | Recipe | CID22 | KonJND | Verdict |
|---|---|--:|--:|---|
| v10 baseline | (canonical ship) | 0.854 | **0.232** | reference |
| a1 (w=0.3, 2 groups) | agg w=0.3 step_p=0.30, no konjnd training group | 0.508 | **0.758** | α=0 collapse — rank corpora destroyed |
| a2 (w=0.05) | 5.5 % effective rate | 0.742 | 0.066 | too weak — KonJND worse than v10 |
| a3 (3 groups, w=0.1) | konjnd_dense as training group + light agg | **0.814** | 0.113 | rank-stable but no KonJND |
| **a4 (3 groups, w=0.3)** | konjnd_dense as training group + a1 agg | 0.769 | **0.615** | BEST balance; spline-calibrated still fails gate |
| a5 (3 groups, w=0.5) | strong agg pressure | 0.707 | 0.568 | over-aggregation — non-monotonic |

**Architectural breakthrough proven**: per-source aggregation head
DOES escape V11-D zero-gradient pathology (a4 KonJND +0.38 over
v10). The mechanism works mechanically. But the trade-off curve
peaks at a4 — KonJND 0.62 vs ship-gate 0.85, CID22 0.77 vs gate
0.86. 3 of 5 criteria fail.

**v10 (PreviewV0_5TunerV4) remains the canonical codec-target.**
`ZensimProfile::codec_target()` unchanged.

What's preserved for the next iteration:
- a4 best bake: `tuner_v11_a4_s1.bin` + spline-calibrated variant
- All 5 attempt-evidence bakes + verdicts + qsweep reports at
  `/mnt/v/zen/zensim-eval/exp_tuner_v11_2026-05-24/`
- CID22-train substrate (17,611 pairs) in canonical-2026-05-21
- All scripts: run_tuner_v11_{seed,attempt3_seed}.sh,
  tuner_v11_hparam_sweep.sh, calibrate_v9_spline.py, etc.

Recovery phase 4 hypotheses (next session):
1. **Per-pair PJND-anchored data** (the missing per-pair signal).
   Either derive PJND-per-pair from konjnd-dense or train on
   KonJND-1k val (compromises holdout).
2. **Pool-head-only architecture** (drop --per-sample-α). Tests
   whether the α gate is the noise source.
3. **Multi-target supervision** (pjnd_target + mix_cv40_iw60
   simultaneously, with trainer flag `--per-row-multi-target`).
   Closest to what attempt 4 was trying to do via separate
   training group + aggregation; might be more efficient as a
   single multi-target loss.
4. **vast.ai trainer infra build** — every iteration is ~14 min
   CPU on the local 7950X. vast.ai parallel-across-seeds would
   make the inner-loop ~14 min wall instead of 70 min.

### E. Hparam sweep tool

`scripts/v_next/tuner_v11_hparam_sweep.sh` maps the
(weight, step_p) surface across 6 cells in ~90 min. Not run in
this session — superseded by the iterative attempts.

## Outstanding tasks (per TaskList)

ALL TASKS COMPLETE: #1, #2, #3, #4, #5, #6, #7.

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
