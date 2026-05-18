# V_22-mix-LARGE-372 — 72 IW-pool features added (f300..f371) — FALSIFIED

**Date:** 2026-05-18
**Status:** Pareto gate FAILS 3/5 corpora on both 5-group and noLARGE
variants. 300-feature V_22-mix-LARGE remains the ship.

## Hypothesis (pre-run, per task brief)

The 72 IW-pool features (f300..f371: info-content-weighted SSIM/edge/MSE
pool stats — `iw_ssim_mean / iw_ssim_4th / iw_ssim_2nd / iw_art_4th /
iw_det_4th / iw_mse` per channel × scale) are computed but unused by
V_22-mix-LARGE-300. Adding them as inputs (NOT as the target — anti-
pattern #6 cleared by inspection of `zensim/src/iw_pool.rs`) should
improve perceptual rank correlation, possibly clearing the +0.005
CID22 SROCC + AIC-3 SROCC ship gate.

## Falsification criteria

- CID22 ≥ baseline + 0.005 (0.8389)
- KADID ≥ baseline − 0.005 (0.9623, anchor — no large regression)
- TID   ≥ baseline − 0.005 (0.9676)
- KonJND ≥ baseline + 0.000 (0.8869, anchor — no regression)
- AIC-3 ≥ baseline + 0.005 (0.7922)

Any FAIL → 300-feature input is not the bottleneck.

## Setup

| | |
|---|---|
| Workspace | `/home/lilith/work/zen/zensim--372feat` (jj workspace add from `feat/ex2-stdpool-head`) |
| Trainer | `target/release/zensim_mlp_train` (zensim-validate) |
| Recipe baseline | V_22-mix-LARGE-iwssim (`v24_hybrid_nin_train.sh` minus `--hybrid-head`) |
| Architecture | 372 → 128 → 1 MLP, LeakyReLU 0.01, no pool-head, no hybrid-head, norm-in-norm 0.1 |
| Loss | RankNet + PWRC pair-weighting (--pwrc-pair-weight --pwrc-sensory-threshold 5.0) |
| Epochs | 300 full (early-stop-patience 0, val-policy=min) |
| Hyperparams | h=128, lr=1e-3 cosine to 0, l2=1e-5, leaky=0.01, minibatch=256, pairs/epoch=50000 |
| Target column | `mix_cv40_iw60` (0.4·cvvdp_log_norm + 0.6·iwssim_log_norm), scale 100.0 |
| Seeds | 1, 2, 3, 4, 5 |

## Group weights (5-group main variant)

| Group | Rows | Path | Train_w | Val_w |
|---|--:|---|--:|--:|
| safesyn | 196,086 | 2026-05-17-cvvdp/safesyn_features_mix_targets_372col.parquet | 1.0 | 0.0 |
| kadid   | 10,125  | 2026-05-17-cvvdp/kadid_features_mix_targets_372col.parquet   | 0.3 | 1.0 |
| tid     | 3,000   | 2026-05-17-cvvdp/tid_features_mix_targets_372col.parquet     | 0.3 | 1.0 |
| konjnd  | 1,008   | 2026-05-17-cvvdp/konjnd_features_mix_targets_372col.parquet  | 0.02 | 1.0 |
| cvvdp_iwssim_large | **73,300** | 2026-05-18-372feat/cvvdp_iwssim_large_372col_**padded**.parquet | 0.5 | 0.0 |

The noLARGE variant drops the LARGE group entirely (4 anchor groups,
same hyperparams).

## Padded LARGE corpus

`cvvdp_iwssim_large_372col_padded.parquet` is built by
`scripts/v_next/pad_large_to_372col.py` — appends 72 zero columns
(`f300..f371`) to the existing 300-col v2 LARGE parquet.

**Why padded, not re-extracted:** The LARGE corpus's distorted images
live on vast.ai workers from the v15 sweep, long since terminated.
Source images exist at `/mnt/v/input/zensim/sources/gen-chart__*.png`
but the distorted variants do not exist locally — they would need to
be re-encoded across 6 codecs × 200 sources × ~370 (codec, q,
knob_tuple) combinations, requiring a multi-hour reproduction of the
sweep. **The 4 anchor groups already carry real f300..f371 values**
(safesyn 196k, kadid 10k, tid 3k, konjnd 1k = 210,219 rows of real
IW signal). The padded LARGE acts as "IW-signal-absent" baseline; the
noLARGE ablation isolates the contamination effect.

## Headline 5-seed SROCC (mean ± std)

| Corpus | V_22-300 baseline (ship) | V_22-372feat 5-group | V_22-372feat noLARGE |
|---|---:|---:|---:|
| CID22            | 0.8339±0.0071 | **0.8493±0.0069** | 0.8425±0.0110 |
| KADIK10k         | **0.9673±0.0002** | 0.9306±0.0014 | 0.9311±0.0022 |
| TID2013          | **0.9726±0.0004** | 0.8878±0.0014 | 0.8897±0.0015 |
| KonJND-1k (full) | **0.8869±0.0034** | 0.8173±0.0071 | 0.8371±0.0066 |
| AIC-3 CTC        | 0.7872±0.0078 | **0.8057±0.0041** | 0.8059±0.0057 |

## Pareto gate verdict

| Corpus | Gate target | 5-group | noLARGE | Verdict |
|---|---:|---:|---:|---|
| CID22 | ≥ 0.8389 | 0.8493 | 0.8425 | **PASS / PASS** |
| KADID | ≥ 0.9623 | 0.9306 | 0.9311 | **FAIL** (−0.0317 / −0.0312) |
| TID   | ≥ 0.9676 | 0.8878 | 0.8897 | **FAIL** (−0.0798 / −0.0779) |
| KonJND | ≥ 0.8869 | 0.8173 | 0.8371 | **FAIL** (−0.0696 / −0.0498) |
| AIC-3 | ≥ 0.7922 | 0.8057 | 0.8059 | **PASS / PASS** |

**3 of 5 corpora fail decisively. NOT a ship candidate.** The CID22 +
AIC-3 lift comes at a catastrophic cost on KADID/TID/KonJND.

## A.9 decisive comparison (best 372feat seed vs V_22-300 ship)

`bake_compare --a v22_372feat_s5_h128.bin --b v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128.bin`:

| Corpus | SROCC A | SROCC B | Z-RMSE A | Z-RMSE B | PWRC A | PWRC B | dec_rule |
|---|---:|---:|---:|---:|---:|---:|:---:|
| CID22    | 0.8582 | 0.8323 | 0.520 | 0.559 | 0.9127 | 0.9005 | **A>>B** |
| KADIK10k | 0.9314 | 0.9677 | 0.363 | 0.249 | 0.9599 | 0.9804 | **B>>A** |
| TID2013  | 0.8880 | 0.9729 | 0.436 | 0.236 | 0.9161 | 0.9833 | **B>>A** |
| KonJND-1k| 0.8121 | 0.8928 | 0.498 | 0.375 | 0.8499 | 0.9181 | **B>>A** |
| AIC-3 CTC| 0.8089 | 0.7831 | 0.577 | 0.608 | 0.8805 | 0.8619 | **A>>B** |

372feat wins 2/5 corpora decisively, loses 3/5 decisively. Per § A.9,
ship the bake that wins on the majority of corpora — that's the
**V_22-300 baseline.**

## Why 372 features hurts

1. **The IW-pool features (f300..f371) and the LARGE corpus are in
   tension.** The LARGE rows carry IW=0 (padded), the 4 anchor groups
   carry real IW signal. The MLP's first-layer scaler standardizes
   per-feature across the union — so the IW columns get a global
   distribution dominated by the 73,300 zeros from LARGE. The model
   learns the IW columns are noise (mostly zero, sometimes non-zero)
   rather than informative.

2. **noLARGE confirms it's not just the padding.** Without LARGE,
   KADID/TID still regress vs V_22-300 — even with 4 anchor groups
   carrying real IW signal end-to-end. This means the IW features
   ADD NOISE relative to the canonical 300-feature input on
   compression-style distortions (which dominate KADID/TID/KonJND).
   The MLP overweights the new IW columns and underfits the original
   300.

3. **AIC-3 + CID22 lift is consistent across both variants
   (+0.018–0.019 SROCC).** Those corpora cover human-MOS-scored
   compression distortions on natural content (CID22) and human-JND-
   scored low-q encodes (AIC-3) — exactly the regimes where IW pool
   (info-content weighting) is paper-claimed to help. The KADID/TID
   regression is on synthetic blur/noise/geometric distortions where
   IW-weighting deemphasizes the affected flat regions; that's the
   OPPOSITE of what those benchmarks measure.

4. **The trade fails the user dial.** Users typing "zensim 70" land
   in B7. The KonJND/KADID/TID regression breaks calibration in B5–B8
   where most product decisions live. AIC-3 + CID22 lift in the
   compression-quality regime doesn't compensate for breaking the
   visually-lossless and near-PJND bands.

## Load-bearing finding

**The 300-feature input is not corpus-build-inertia; it's
empirically the right size for the V_22 5-group recipe.** Adding 72
IW-pool features (with or without the padded LARGE) actively harms
the model on 3 of 5 corpora.

This pushes the next-direction priority toward:
- **Re-extract IW on the LARGE corpus** (long-pole, requires
  reproducing 73,300 vast.ai distortions locally or on a fresh
  cluster). The noLARGE variant ALSO regresses, so re-extraction
  alone may not fix the KADID/TID regression — but it would settle
  whether the LARGE corpus's IW-zero contamination is the dominant
  failure mode, or whether IW-on-anchors-only just doesn't
  generalize.
- **Architecture levers** (the V_24 series — pool-head, hybrid-head,
  per-sample-α, NiN composition, PJND pair-weighting) — already
  explored this week, mostly falsified. The architecture space
  appears saturated around V_22-mix-LARGE-300's behavior.
- **Multi-target supervision** (per ssim2-target-training-bias rule)
  — train against ssim2 AND iwssim AND cvvdp together rather than
  the mix_cv40_iw60 weighted target. May break the bias that locks
  the model to ssim2-shape.

## Wall time

- LARGE pad (300→372col): 1.4s
- 10 trainer runs in parallel (5 × 5grp + 5 × noLARGE), 300 epochs
  each, ~8-9 min wall per seed (~520s)
- All 10 bake_verdict evals: 30s total
- bake_compare (A.9 decisive, 1000-bootstrap): 95s
- Total wall: ~12 min from `cargo build` to verdict

## Artifacts

- Trainer bakes: `/mnt/v/zen/zensim-eval/v22_372feat_2026-05-18/v22_372feat_s{1..5}_h128.bin` (5-group) and `/mnt/v/zen/zensim-eval/v22_372feat_2026-05-18/v22_372feat_noLARGE_s{1..5}_h128.bin` (noLARGE)
- Per-bake verdicts: `verdict_v22_372feat_*.md` (full Mohammadi panel, 10-band)
- Baseline verdicts: `baselines/verdict_v22_baseline_s{1..5}.md`
- Aggregate summary: `SUMMARY_5seed.md`
- A.9 decisive report: `bake_compare_372feat_s5_vs_v22_300_s3.md`
- Padded LARGE parquet: `/mnt/v/zen/zensim-training/2026-05-18-372feat/cvvdp_iwssim_large_372col_padded.parquet`

## Verdict summary

| | Result |
|---|---|
| CID22 +0.005 gate | **PASS** (+0.0154 5grp, +0.0086 noLARGE) |
| KADID −0.005 tolerance | **FAIL** (−0.0367 5grp, −0.0362 noLARGE) |
| TID −0.005 tolerance | **FAIL** (−0.0848 5grp, −0.0829 noLARGE) |
| KonJND ≥0 | **FAIL** (−0.0696 5grp, −0.0498 noLARGE) |
| AIC-3 +0.005 gate | **PASS** (+0.0185 5grp, +0.0187 noLARGE) |
| **Ship?** | **No.** V_22-mix-LARGE-300 stays the ship. |

## No packed bake produced

Per the rule "don't pack failed bakes" — the 372feat 5-group and
noLARGE bakes are kept as evidence at f32 dtype but NOT i8-packed.
If a future session wants to inspect them, they're at
`/mnt/v/zen/zensim-eval/v22_372feat_2026-05-18/`.
