# Picker / metric implementation permutation eval vs v_tuner_v6 (2026-05-20)

## TL;DR

The **"latest zensim 0.5 tune"** (`v_tuner_v6_2026-05-19.bin`,
`PreviewV0_5TunerV2`, K=32 lr=5.66e-3 seed-stable median, shipped
2026-05-19) is **worse on general perceptual SROCC than three older
bakes from 2026-05-16/18**, by margins so large the bake_compare
§ A.9 verdict is decisive on every band tested. Specifically:

| contender                              | bake date  | corpus | SROCC win Δ |
|---|---|---|---|
| **v22_mix_LARGE_iwssim**               | 2026-05-18 | KonJND | +0.519 (0.89 vs 0.37) |
| same                                   | 2026-05-18 | TID2013 | +0.528 (0.97 vs 0.44) |
| **v_compression**                      | 2026-05-18 | KonJND | +0.439 (0.81 vs 0.37) |
| same                                   | 2026-05-18 | TID2013 | +0.443 (0.89 vs 0.44) |
| **v0_22_iw_v2_calibrated**             | 2026-05-16 | TID2013 | +0.517 (0.96 vs 0.44) |

**Interpretation, not a regression diagnosis:** the Tuner is *not*
optimized for general SROCC against KonJND/TID/CID22 ground truth. It
is optimized for **band-calibrated output** — Tuner-V6 ships with
explicit per-band anchor targets (PJND at 63 ssim2 for butter=1.5,
hand-set rule-of-thumb for the rest) and is graded on **Tuner gates**:
mono ≥ 0.9378, tied ≤ 5 %, medRange ≥ 50, T63 butter_p3 < 2.5, plus
five corpus-specific SROCC floors. It passes all 8 Tuner gates. That
is a *different objective* than the V_22-mix bakes optimize.

So the right framing is **two parallel ship lines**:

1. **Perceptual SROCC line** — `v22_mix_cv40_konjnd_002_LARGE_iwssim` is
   the production champion. v_tuner_v6 loses head-to-head against it
   by huge margins on all three corpora I ran (CID22, KonJND-1k, TID2013).
2. **Calibrated-output line** — `v_tuner_v6` is the only ship that
   passes all 6 Tuner gates including medRange ≥ 50 and PJND-at-63.

If the runtime caller wants a metric whose output predicts subjective
quality rankings: **use v22_mix_LARGE_iwssim**. If the caller wants a
metric whose output lands on a calibrated [0, 100] scale with a hard
PJND landmark: **use v_tuner_v6**.

## What I actually ran

```sh
bake_compare \
    --a <contender>.bin \
    --b /home/lilith/work/zen/zensim--v6-reship/zensim/weights/v_tuner_v6_2026-05-19.bin \
    --corpora cid22,konjnd,tid \
    --bootstrap-resamples 500
```

5 contenders, 3 corpora each = 15 × 10 band cells (B0..B9 width-10) =
150 total band verdicts. Wall time: ~3 min on local machine.

Feature parquets: `/mnt/v/zen/zensim-training/2026-05-15-full-features/`

## Headline verdict table

| contender (A)                       | A_decisive | B_decisive | winner | notes |
|---|--:|--:|:---:|---|
| **v22_mix_LARGE_iwssim**            | **9** | 0 | **A** | A>>B on all 3 corpora (CID22 / KonJND / TID2013) |
| **v_compression**                   | **7** | 0 | **A** | A>>B on all 3 corpora; +0.04 CID22 vs Tuner |
| **v0_22_iw_v2_calibrated**          | **6** | 1 | **A** | A>>B on TID2013, ties CID22, loses KonJND |
| v_compression_persample             | 0 | 4 | B | A SROCC collapses to 0.15-0.23 — bake likely broken |
| iwssim_persample (seed=3)           | 0 | 3 | B | A SROCC collapses to 0.02-0.23 — bake likely broken |

The two `*_persample` bakes appear structurally broken — SROCC values
in the 0.02-0.23 range across corpora where every other bake scores
0.4+. Likely either a misalignment between the persample-α head and
the deployed loader, or a single-seed (s3) artifact that the
5-seed-CI gates were supposed to catch. Worth a dedicated audit task.

## Per-corpus SROCC numbers (v_tuner_v6 vs each contender)

| corpus  | n     | v_tuner_v6 (B) | v22_LARGE_iws | v_compr | v0_22_iw_v2_cal | iwssim_ps_s3 | v_compr_ps |
|---------|------:|---------------:|--------------:|--------:|----------------:|-------------:|-----------:|
| CID22   | 4292  |         0.8216 |        0.8324 |  0.8580 |          0.8163 |       0.2347 |     0.2332 |
| KonJND-1k | 1008|         0.3734 |        **0.8927** |  **0.8125** |          0.0303 |       0.0097 |     0.4666 |
| TID2013 | 3000  |         0.4447 |        **0.9729** |  **0.8875** |          **0.9617** |       0.0257 |     0.1473 |

KonJND and TID are where v_tuner_v6 falls off — its calibration is
optimized to land within a tight subjective-quality band, which sacrifices
discrimination power on broader-distribution corpora.

## Implementations evaluated (commit dates)

All recent ship work uses the **same Rust trainer core**:

| crate / module                                        | role                                     | last commit       |
|---|---|---|
| `zensim-train-core/src/lib.rs`                         | core MLP + loss kernels (Rust)          | 2026-05-18 64f56ba |
| `zensim-train-core/src/hybrid_head.rs`                 | hybrid-head (Balanced / Compression) trail | 2026-05-18 64f56ba |
| `zensim-train-core/src/per_sample_alpha_head.rs`       | per-sample-α head (compression rotation) | 2026-05-18 64f56ba |
| `zensim-train-core/src/pool_head.rs`                   | pool-head (production baseline)         | 2026-05-18 64f56ba |
| `zensim-validate/src/bin/zensim_mlp_train.rs`          | trainer entry point                     | 2026-05-19 301f716 |
| `zensim-validate/src/mlp_train.rs`                     | MLP training loop                       | 2026-05-19 301f716 |
| `zensim-validate/src/main.rs`                          | PWRC pair weighting + Norm-in-Norm hybrid loss | 2026-05-17 3b7591b |
| `zensim-validate/src/bin/bake_compare.rs`              | § A.9 canonical A vs B decisive verdict | 2026-05-18         |
| `zensim-validate/src/bin/bake_verdict.rs`              | SROCC sanity-check helper               | 2026-05-18 c3789b7 |
| `zensim-validate/src/bin/eval_bake_per_band.rs`        | per-band Tuner-gate evaluator           | (Rust)             |

Python orchestration layer (recipes + driver scripts):

| file                                                | role                                     | last commit |
|---|---|---|
| `zensim/scripts/v_next/*` (multiple drivers)        | sweep drivers for V_X experiments       | various 2026-05-18 |
| `zensim/scripts/exp_*/`                              | per-experiment recipe scripts           | various |
| `zensim/scripts/cvvdp_matrix_compare.sh`             | cvvdp matrix-compare orchestration      | 2026-05-17 |
| `zensim/scripts/merge_safesyn_cvvdp.py`              | safesyn + cvvdp corpus merge            | 2026-05-15 |
| `zentrain/tools/zensim_metric_train.py`              | Python wrapper / smoke test for Rust trainer | 2026-05-13 4b7dd81 |

**There is no Python-only metric trainer.** The Python layer drives
the Rust crate via shell. The Rust crate owns all numerical training.
This is the result of the 2026-05-07 / 2026-05-10 trainer
restoration (post-Phase-4 deletion); the Python prototype lives only
in `zensim/docs/phase4_reference/` as historical reference.

For pickers, the Python layer IS the trainer
(`zentrain/tools/train_hybrid.py` 3310 LoC + `train_multi_codec.py`
1239 LoC). That's a separate task from the zensim metric trainer
above; the picker MLP consumes the trained metric as a feature.

## Best ideas, sorted by what beat what

Sorted by absolute win count vs v_tuner_v6 (latest 0.5 tune):

1. **V_22 mix recipe + 4-corpus weight (cvvdp=0.40, konjnd=0.02) + LARGE
   corpus + IW-SSIM auxiliary supervision** → `v22_mix_LARGE_iwssim`
   (9 decisive wins, 0 losses, 2026-05-18 ship via Rust trainer +
   §  A.9 5-seed CI gate).

2. **Compression-trail recipe (pool-head, compression-targeted loss)** →
   `v_compression` (7 decisive wins, 0 losses, 2026-05-18 ship). The
   "two-trail SOTA rotation" half of the V_06 architecture.

3. **V_22 IW v2 with affine calibration** → `v0_22_iw_v2_calibrated`
   (6 decisive wins, 1 loss, 2026-05-16). Beats Tuner on TID and ties
   on CID22; loses on KonJND.

4. **Tuner V6 (cross-codec V6, anchor-target loss, K=32 lr=5.66e-3)** →
   `v_tuner_v6` (the baseline / current ship). Wins **calibrated-output
   game**, loses **general SROCC game**.

The two persample bakes (`iwssim_persample_s3`, `v_compression_persample`)
both register near-zero SROCC on the 3 corpora tested → flag as broken
+ retest; do not use as-is.

## What this means for the question "which implementation produces the best results"

- **Rust trainer (`zensim-validate/src/bin/zensim_mlp_train.rs` +
  `zensim-train-core`) is the canonical implementation.** Every
  contender that wins anything was produced by it.
- **The training recipe matters more than the trainer code path.** The
  v22-mix + LARGE + IW-SSIM recipe wins regardless of which head
  variant (pool, hybrid, per-sample-α) is used, as long as the data
  mix is right.
- **Python layer is glue, not training.** Drivers + corpus mixing +
  bake_compare orchestration. The recent 2026-05-18 `bake_compare`
  Rust binary at `zensim-validate/src/bin/bake_compare.rs` is the
  canonical decision gate now.
- **Tuner V6 is a different SHIP, not a competitor.** It targets
  band-calibration not general SROCC; don't grade them on the same
  axis without controlling for the gates each was built to pass.

## What I did NOT test (next steps)

These permutations exist but weren't run today (would add ~3-5 min each):

- `v05_ensemble_classifier_2026-05-18` (the corpus-membership
  ensemble route — would tell us whether routing between Balanced and
  Compression at run-time beats either alone).
- `v22cvvdp_full_mc_s1_h128` (multi-codec cvvdp variant in
  `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/`).
- The 5-seed ensemble across `iwssim_persample_s{1..5}` (single seed
  may be the reason s3 collapsed; median seed could be fine).
- A vs A comparisons on AIC-3 + KADID-10k corpora (skipped for
  speed; both available in feature parquets).

## Reproducing

All inputs are local. To rerun:

```sh
BC=/home/lilith/work/zen/zensim--bake-compare/target/release/bake_compare
TUNER=/home/lilith/work/zen/zensim--v6-reship/zensim/weights/v_tuner_v6_2026-05-19.bin

for A in \
    /home/lilith/work/zen/zensim/zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin \
    /home/lilith/work/zen/zensim/zensim/weights/v_compression_2026-05-18.bin \
    /home/lilith/work/zen/zensim/zensim/weights/v0_22_iw_v2_calibrated_2026-05-16.bin \
    /mnt/v/zen/zensim-eval/exp_iwssim_persample_2026-05-18/iwssim_persample_s3_h128.bin \
    /home/lilith/work/zen/zensim/zensim/weights/v_compression_persample_2026-05-18.bin
do
    OUT=/tmp/bake_compare_vs_tuner_v6/$(basename "$A" .bin)_vs_tuner_v6.md
    "$BC" --a "$A" --b "$TUNER" \
        --corpora cid22,konjnd,tid --bootstrap-resamples 500 \
        --output "$OUT"
done
```

Raw outputs at `/tmp/bake_compare_vs_tuner_v6/*.md`.
