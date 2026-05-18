# EXP-DISTILL-ENSEMBLE — single-bake distillation of PreviewV0_5Ensemble (FALSIFIED)

_Date: 2026-05-18. Hypothesis: train a single per-sample-α (h=128) MLP
to predict the PreviewV0_5Ensemble's outputs (knowledge distillation).
Expected: match or near-match Ensemble on the panel without the
classifier-routing runtime overhead._

**Verdict: FALSIFIED.** The distilled student catastrophically loses
on KonJND-1k (SROCC 0.064 vs Ensemble 0.879) and underperforms vs
Ensemble on every other corpus by 0.02–0.05 SROCC. Fails both
balanced and compression trail gates per § A.10. The hypothesis that
distillation transfers the routing benefit to a single bake is
falsified — but informatively: the failure mode reveals that the
Ensemble's panel performance depends on the routed bake's _raw_
output range, not the post-soft-clamp range that the student
necessarily sees as its training target.

## Methodology

**Architecture** Per-sample-α head (V_24-per-sample-α), h=128, n_inputs=300, 2-layer MLP (300→128→128), with metadata payload
`zentrain.per_sample_alpha_head` carrying (w_α, b_α, rank_w, rank_b,
reducer_w, reducer_b, p_norm). Identical to the V0_5Compression ship
architecture; only the training target differs.

**Training data** Five canonical training parquets at `/mnt/v/zen/zensim-training/canonical-2026-05-18/train/` (safesyn, kadid, tid, konjnd-dense, cvvdp_iwssim_LARGE). New parquet set built at `/mnt/v/zen/zensim-training/2026-05-18-distill-ensemble/`, identical features per row, with one new target column `ensemble_teacher`.

**Teacher labels** For each (ref, dist) pair in each training parquet:

1. Score the Balanced bake (`v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin`) → `raw_bal`.
2. Score the Compression bake (`v_compression_persample_2026-05-18.bin`) → `raw_cmp`.
3. Score the classifier (`v05_ensemble_classifier_2026-05-18.bin`) → `clf_logit`.
4. Routed raw: `raw_routed = (clf_logit > 0) ? raw_cmp : raw_bal`
5. Soft-clamp (matches `zensim::metric::soft_clamp_score`): `score = 100 / (1 + exp(-(raw_routed - 50) / 20))`
6. Store as `ensemble_teacher = score / 100` in [0,1] so `--target-scale 100.0` recovers the soft-clamped score.

All scoring done via `target/release/ensemble_score_rows`, which dispatches per-sample-α and hybrid-head paths bit-exactly via the same code as `bake_verdict::score_row`.

Routing statistics:

| Corpus | n | fraction routed → compression |
|---|--:|---:|
| safesyn | 196,086 | 0.3680 |
| kadid | 10,125 | 0.0024 |
| tid | 3,000 | 0.0067 |
| konjnd-dense | 20,160 | 0.3794 |
| cvvdp_iwssim_LARGE | 73,300 | 0.7453 |

Teacher score distributions (after soft-clamp, in score units [0,100]):

| Corpus | min | mean | max | std |
|---|---:|---:|---:|---:|
| safesyn | 2.14 | 6.75 | 100.00 | 4.11 |
| kadid | 2.45 | 9.64 | 74.85 | 4.25 |
| tid | 3.12 | 9.71 | 41.51 | 5.01 |
| konjnd-dense | 2.31 | 7.07 | 35.61 | 4.80 |
| cvvdp_iwssim_LARGE | 2.07 | 4.55 | 19.40 | 2.44 |

Note: post-soft-clamp ranges are heavily compressed because the underlying bakes typically emit raw values much less than 50 (the soft-clamp midpoint) on degraded pairs. This is the root of the failure (see "What went wrong" below).

**Trainer command** Mirror V_24-per-sample-α s4 (Compression ship) exactly except `--target-column ensemble_teacher`. Recipe in `scripts/exp_distill_ensemble/run_distill_seed.sh`. Hyperparams:

```
--hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 300
--target-column ensemble_teacher --target-scale 100.0
--val-policy min --minibatch-size 256
--pwrc-pair-weight --pwrc-sensory-threshold 5.0
--norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0
--per-sample-alpha-head
--group safesyn:safesyn.parquet:1.0:0.0
--group kadid:kadid.parquet:0.3:1.0
--group tid:tid.parquet:0.3:1.0
--group konjnd:konjnd-dense.parquet:0.02:1.0
--group cvvdp_iwssim_large:cvvdp_iwssim_LARGE.parquet:0.5:0.0
```

Group weights match the Compression ship recipe (train_w:val_w).

**Seeds** 5 seeds (1..5). Wall time ~6 minutes parallel.

## Results — 5-seed CI

| Seed | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| s1 | 0.8078 | 0.9225 | 0.9228 | 0.0421 | 0.8039 |
| s2 | 0.8443 | 0.9295 | 0.9245 | 0.1273 | 0.8034 |
| s3 (median CID22) | **0.8412** | 0.9228 | 0.9180 | 0.0646 | 0.8099 |
| s4 | 0.8430 | 0.9213 | 0.9190 | 0.0450 | 0.8020 |
| s5 | 0.8368 | 0.9225 | 0.9207 | 0.2110 | 0.7851 |
| **mean** | 0.8346 | 0.9237 | 0.9210 | 0.0980 | 0.8009 |
| **std** | 0.0148 | 0.0030 | 0.0026 | 0.0688 | 0.0090 |

Median-CID22 seed = **s3**.

## Pack drift verification

s3 packed via `zenpredict repack --dtype i8 --zerobias 0.005 --compress --optimize`:

- Input: 223,876 bytes
- Packed: 43,896 bytes (19.6% of input)
- Zerobias τ=0.005: 19,124 of 54,784 weights zeroed (34.9%)

SROCC drift per corpus (orig → packed):

| Corpus | orig SROCC | packed SROCC | drift |
|---|---:|---:|---:|
| CID22 | 0.8412 | 0.8409 | -0.0003 |
| KADID | 0.9228 | 0.9228 | -0.0000 |
| TID | 0.9180 | 0.9179 | -0.0001 |
| KonJND | 0.0646 | 0.0641 | -0.0005 |
| AIC-3 | 0.8099 | 0.8099 | +0.0000 |

All within the ≤0.0005 packing-drift policy.

## Headline panel — Distill vs ships vs Ensemble (full corpus, SROCC)

| Corpus | Ensemble | Balanced | Compression | Distill (s3 packed) | Δ vs Ensemble |
|---|---:|---:|---:|---:|---:|
| CID22 | 0.8632 | 0.8324 | 0.8641 | 0.8409 | -0.0223 |
| KADID | 0.9676 | 0.9677 | 0.9316 | 0.9228 | -0.0448 |
| TID | 0.9719 | 0.9729 | 0.8893 | 0.9179 | -0.0540 |
| KonJND | 0.8792 | 0.8927 | 0.8080 | **0.0641** | **-0.8151** |
| AIC-3 | 0.8131 | 0.7845 | 0.8183 | 0.8099 | -0.0032 |

Distilled student is dominated by Ensemble on every corpus.

## Controls (from `benchmarks/baseline_panels_2026-05-18.md`)

| Corpus | fast-ssim2 SROCC | cvvdp SROCC | iwssim SROCC |
|---|---:|---:|---:|
| CID22 | 0.8895 | 0.8214 | 0.7836 |
| KADID (corpus-strict CSV) | — | — | — |
| TID | — | — | — |
| KonJND | — | — | — |
| AIC-3 | — | — | — |

(Distilled student loses to fast-ssim2 control on CID22: 0.8409 vs 0.8895.)

## § A.9 verdicts

### Distill (A) vs Balanced (B) — full bake_compare report at `compare/distill_vs_balanced.md`

| Corpus | SROCC_A | SROCC_B | h_SROCC | DecScore | Verdict |
|---|---:|---:|---:|---:|---|
| CID22 | 0.8409 | 0.8324 | +4.568 | +0.000 | **tied** |
| KADID | 0.9228 | 0.9677 | -103.719 | -0.000 | **B>>A** |
| TID | 0.9179 | 0.9729 | -61.691 | -0.000 | **B>>A** |
| KonJND | 0.0641 | 0.8927 | -17.999 | -0.000 | **B>>A** |
| AIC-3 | 0.8099 | 0.7845 | +15.005 | +12.505 | **A>>B** |

Overall across decisive (corpus × band) cells: **B wins** (1 A wins vs 16 B wins).

### Distill (A) vs Compression (B) — full bake_compare report at `compare/distill_vs_compression.md`

| Corpus | SROCC_A | SROCC_B | h_SROCC | DecScore | Verdict |
|---|---:|---:|---:|---:|---|
| CID22 | 0.8409 | 0.8641 | -15.047 | -0.000 | **B>>A** |
| KADID | 0.9228 | 0.9316 | -19.845 | -0.000 | **B>>A** |
| TID | 0.9179 | 0.8893 | +26.888 | +22.407 | **A>>B** |
| KonJND | 0.0641 | 0.8080 | -16.697 | -0.000 | **B>>A** |
| AIC-3 | 0.8099 | 0.8183 | -5.468 | -0.000 | **tied** |

Overall across decisive (corpus × band) cells: **tie** (7 A wins vs 7 B wins) but headline corpora are decisive in favor of Compression on CID22, KADID, KonJND.

## § A.10 trail-gate verdicts

### Balanced trail gate — FAIL

> A>>B on CID22 decisively per § A.9 AND not decisively B>>A on any of {KADID, TID, KonJND, AIC-3}.

- CID22 verdict: tied (not A>>B) → fails clause 1.
- KADID, TID, KonJND all B>>A → fails clause 2.

### Compression trail gate — FAIL

> A>>B on ≥1 of {CID22, AIC-3} decisively per § A.9 AND not decisively B>>A on the other compression corpus AND mean SROCC regression on {KADID, TID, KonJND} no worse than −0.10 on any single corpus.

- CID22 verdict: B>>A (Distill loses) → fails clause 1.
- AIC-3 verdict: tied (not A>>B) → fails clause 1.
- KonJND regression vs Compression: -0.744 (0.0641 vs 0.8080) → fails clause 3 (much worse than -0.10).

## Decision

**No ship, no rotation, no new variant.** Distilled student fails both
trail gates and is dominated by the Ensemble on every corpus. Adding
it as PreviewV0_5Distilled is unacceptable because:

1. KonJND SROCC 0.064 breaks the visually-lossless calibration anchor — the user-facing dial would mis-predict "at-PJND quality" by orders of magnitude.
2. It does not Pareto-match Ensemble — it loses ≥0.02 on every corpus, so there's no compression-ship-grade single-bake "distillation alternative" to recommend.

## What went wrong (mechanism)

Distillation transferred the score _magnitudes_ but not the _ranking_, particularly on KonJND. The mechanism:

1. **Soft-clamp compression**. The Ensemble's post-soft-clamp output range on KonJND-1k val is **[4.66, 8.17]** (only 3.5 score units out of 100). The teacher labels for konjnd-dense training had similar tight ranges. The student is trying to learn a near-constant function with tiny per-pair variation — but the variation IS the rank signal.

2. **Loss-function mismatch**. The Compression ship recipe uses RankNet pair-loss (PWRC-pair-weight) which rewards rank concordance. With teacher labels that are nearly identical across "harder vs easier" pairs (because soft-clamp squashes everything below 50 into the [4, 10] range), the RankNet gradient is dominated by noise rather than the genuine rank signal. The Balanced bake's raw output range on KonJND val is [-10.39, 0.76] — 11 score units pre-clamp, which is what makes the Balanced bake achieve SROCC 0.89. The post-clamp targets lose 70%+ of the dynamic range.

3. **Teacher-distillation paradox**. Knowledge distillation works when the teacher's _output_ carries the soft information you want to transfer. Here, the Ensemble's final output (post-soft-clamp) carries less ranking information than its internal raw values would have. The student would need to learn from the _raw_ routed values, not the soft-clamped ones. But that breaks calibration — raw distance values trained as direct supervision give scores outside [0, 100] without the soft-clamp.

4. **No PJND-anchor signal**. The teacher labels don't preserve the PJND anchor (the konjnd-dense parquet's `pjnd_target` was dropped in favor of `ensemble_teacher`). The Balanced ship gets KonJND 0.89 because its training recipe weights konjnd-dense with `--target-column pjnd_target`-equivalent signal via the mix targets. The Ensemble inherits this from Balanced for KonJND routing (91% of val pairs route to Balanced per `frac_to_compression = 0.091`). The student, training against post-clamp scores from a 91%-Balanced-routed corpus, gets the score magnitudes near-right but loses the rank discriminability.

## Future work (NOT in this experiment, NOT a follow-up TODO)

A revised distillation might:

1. Train against the routed _raw_ teacher (before soft-clamp), and apply soft-clamp only at runtime. This gives the student >10× more dynamic range for the gradient signal.
2. Combine teacher distillation with original target column (mix_cv40_iw60 + ensemble_teacher) via dual-target trainer.
3. Use the Balanced bake alone (no ensemble routing) for KonJND-heavy corpora, only routing CID22-like corpora to the compression bake.

These are NOT pursued in this experiment per the brief's hypothesis scope. The
falsification stands.

## Artifacts

- Teacher parquets: `/mnt/v/zen/zensim-training/2026-05-18-distill-ensemble/{safesyn,kadid,tid,konjnd-dense,cvvdp_iwssim_LARGE}.parquet`
- Teacher build script: `scripts/exp_distill_ensemble/build_teacher_parquets.py`
- Training script: `scripts/exp_distill_ensemble/run_distill_seed.sh`
- Bakes: `/mnt/v/zen/zensim-eval/exp_distill_ensemble_2026-05-18/distill_s{1..5}_h128.bin`
- Packed bake (median seed): `/mnt/v/zen/zensim-eval/exp_distill_ensemble_2026-05-18/distill_s3_h128_packed.bin`
- Per-seed verdicts: `/mnt/v/zen/zensim-eval/exp_distill_ensemble_2026-05-18/verdicts/distill_s{1..5}_verdict.md`
- bake_compare reports: `/mnt/v/zen/zensim-eval/exp_distill_ensemble_2026-05-18/compare/distill_vs_{balanced,compression}.md`
- Scoring logs: `/tmp/exp_distill_ensemble_seed{1..5}.log`, `/tmp/exp_distill_ensemble_scoring.log`
