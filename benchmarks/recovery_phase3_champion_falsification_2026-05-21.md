# Recovery Phase 3 champion train — falsification

**Task:** #202 (RECOVERY-PHASE-3-CHAMPION)
**Date:** 2026-05-21
**Verdict:** FAIL ship gate (KonJND collapse)

## TL;DR

The recovery-phase-3 trainer foundation (zenanalyze commit `ce7e41b`, task #201,
ports 1–4 of `RECOVERY_PLAN_2026-05-08.md` Phase 3) runs end-to-end against the
canonical-2026-05-21 corpus and produces bakes with **strong CID22 / KADID /
TID / AIC-3 performance, but a structural KonJND collapse**. Median CID22 SROCC
0.8498 (vs Balanced ship 0.8324, +0.0174 absolute margin) and AIC-3 0.8046 (vs
Balanced 0.7845, +0.0201 margin) **both clear the Balanced-trail gate**, but
**KonJND 0.3689 falls −0.4238 below the 0.7927 gate**. The recovery-plan recipe
as specified does not produce a shippable Balanced-trail variant against the
modern corpus.

The trainer foundation itself ships as research infrastructure (already on
zenanalyze main at `ce7e41b`). No zensim variant ships from this cycle.

## 5-seed CI table (variant B — port 1 only, h=64, epochs=100, val-policy=mean)

Recipe: `safesyn (w=1.0) + kadid (w=0.3) + tid (w=0.3)` → `human_score`, ranknet+mse
loss, no magnitude-matching, no low-band oversample, `--val-policy mean`,
372 features. Validation: cid22 + kadid + tid (KonJND val held out because
KonJND's `human_score` carries [-65, 96] signed deltas on the `dense` corpus
vs [28, 70] positive PJND thresholds on val, breaking SROCC sign).

| seed | CID22 | KADID | TID | KonJND | AIC-3 | best_epoch |
|------|-------|-------|-----|--------|-------|-----------|
| 1 | 0.8634 | 0.9082 | 0.8879 | 0.3510 | 0.7936 | 78 |
| 2 | 0.8363 | 0.9077 | 0.8760 | 0.3446 | 0.7973 | 66 |
| 3 | 0.8459 | 0.9045 | 0.8785 | 0.3419 | 0.8000 | 82 |
| 4 | 0.8585 | 0.9125 | 0.8922 | 0.4343 | 0.7983 | 93 |
| **5 (median CID22)** | **0.8498** | **0.9101** | **0.8845** | **0.3689** | **0.8046** | **81** |

Wall: ~13 s × 5 seeds = ~65 s GPU (RTX 4090, 209k rows, 100 epochs).

## Mohammadi panel (median bake, seed 5, CID22 4-param-logistic rescale)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|--------|---|-------|------|-------|----|----|--------|
| CID22 | 4292 | 0.8498 | 0.8526 | 0.6540 | 0.0487 | 0.9065 | 0.523 |
| KADID-10k | 10125 | 0.9101 | 0.9127 | 0.7356 | 0.0463 | 0.9438 | 0.409 |
| TID2013 | 3000 | 0.8845 | 0.8991 | 0.7010 | 0.0440 | 0.9277 | 0.438 |
| KonJND-1k | 1008 | 0.3689 | 0.3156 | 0.2553 | 0.0387 | 0.5318 | 0.949 |
| AIC-3 | 600 | 0.8046 | 0.8118 | 0.6324 | 0.0567 | 0.8717 | 0.586 |

Full report: `recovery_champion_s5_bake_verdict_2026-05-21.txt`. All five
per-seed verdicts archived at `recovery_champion_s{1..5}_bake_verdict_2026-05-21.txt`.

## Ship gate vs Balanced trail (V_22-mix-LARGE+iwssim s3, current ship)

Per `zensim/SOTA_TRAILS.md` baseline + task #202 spec margins:

| Gate | V10 BalancedV3 baseline | Gate threshold | Champion (median) | Result |
|------|------------------------|----------------|-------------------|--------|
| CID22 SROCC | 0.8324 | ≥ 0.8374 | 0.8498 | **PASS** (+0.0124) |
| CID22 Z-RMSE | 0.564 | ≤ 0.530 | 0.523 | **PASS** (−0.041) |
| KADID SROCC | 0.9677 | ≥ 0.8677 (V10 − 0.10) | 0.9101 | **PASS** (+0.0424) |
| TID SROCC | 0.9729 | ≥ 0.8729 (V10 − 0.10) | 0.8845 | **PASS** (+0.0116) |
| KonJND SROCC | 0.8927 | ≥ 0.7927 (V10 − 0.10) | 0.3689 | **FAIL** (−0.4238) |
| AIC-3 SROCC | 0.7845 | ≥ 0.7795 (V10 − 0.005) | 0.8046 | **PASS** (+0.0251) |

Five gates clear (4 with comfortable margin); KonJND fails decisively. **Overall
ship verdict: FAIL.**

## Root cause: KonJND has no shared training signal in the (safesyn+kadid+tid)
recipe

`KonJND-1k` val (the gate corpus) measures **how well a metric ranks the PJND
threshold across content classes**. All 1008 pairs are at the just-noticeable
difference for their respective reference image — `human_score` clusters around
22–70, KonJND's job is to rank these PJND values across diverse content.

The (safesyn + kadid + tid) training corpora carry no PJND-anchored signal.
They train the model to predict `1 − ssim2/100`-style scores against general
distortions. The model becomes very good at ranking distortion strength within
a single ref (CID22's strong suit) but **cannot rank PJND thresholds across
content classes** because that ordering is not present in the training data.

The four variants I attempted to address this all failed:

### Variant A — `konjnd-dense` in train at native scale (`weight=0.3`)

Konjnd-dense's `human_score` is on scale [−65, 96] (signed SSIM2 deltas at
multiple distortion levels per ref). Concatenating with safesyn/kadid/tid
(all on [0, 1]) the MSE loss is dominated by konjnd's ~100²-magnitude term →
all corpora collapse:

```
epoch   0  train_loss=36183  cid22=-0.71 kadid=-0.78 tid=-0.77 konjnd=+0.42
epoch   4  train_loss=35953  cid22=-0.78 kadid=-0.66 tid=-0.75 konjnd=+0.50
```

### Variant C — `konjnd-dense` in train with `--corpus-target-scale konjnd:0.01`

(Added a `--corpus-target-scale NAME:FACTOR` flag to the trainer to multiply
the named corpus's target by FACTOR before loss.) Scaling konjnd-dense to
[−0.66, 0.96] brought training loss back to ~0.3 range. CID22 still hits 0.81
range, but **KonJND val SROCC stays negative across all 5 seeds** (best −0.14,
worst −0.34). Root cause: konjnd-dense's `human_score` is signed-SSIM2-delta
across a 7-step distortion ladder per ref; konjnd val's `human_score` is the
PJND threshold (single positive scalar per ref). Training on (signed delta)
teaches the model to predict distortion magnitude, which is anti-correlated
with the PJND threshold (high-distortion-tolerance content has a HIGH PJND
threshold = high human_score, but a LOW model prediction because the content
also makes high SSIM2 deltas indistinguishable from low). The sign flips.

### Variant D — Ports 3 + 4 active (magnitude-match λ=0.1 α=30, low-band-oversample 4.0 cutoff=0.6, h=128)

Worse than port-1-only on all axes — CID22 ranges −0.54 to 0.82, mean ~0.55.
The magnitude-matching loss with α=30 on the [0, 1]-scale target produces
gradient instability when combined with the low-band sampler bias on the
synth-heavy 209k-row dataset. Per RECOVERY_PHASE3 doc the V0_7 recipe was
hand-tuned on the 228+3=231 V0_7 schema, and the canonical-2026-05-21's
372-feature schema + recoded `human_score` scale don't reproduce the conditions
in which those ports were validated.

### Variant E — `--target-col mix_target` (V_22 production target)

`mix_target` is canonical-safesyn's CVVDP+IWSSIM+SSIM2 mix on [15, 72] — the
target the V_22 production ship trains against. **canonical val parquets do
not carry `mix_target`** (cid22.parquet `mix_target` is entirely null), so the
trainer's safety check stops the run. Other mix variants (`mix_cv40_iw60`,
`mix_cv55_iw45`) likewise live only in the train parquets. The val schema needs
to be expanded with the mix targets — or the trainer needs a per-corpus
`--target-col-override` to use `mix_target` for train and `human_score` for val.
Either is a 1–2 hour expansion of the trainer/corpus, beyond the wall budget
for this falsification pass.

## What works (and ships as research infrastructure)

1. **The recovery-phase-3 trainer port itself.** `zentrain/tools/zensim_metric_train.py`
   on zenanalyze main at `ce7e41b` runs end-to-end against canonical-2026-05-21
   with all four ports (train_loop + zenanalyze sidecar appender +
   magnitude-match loss + low-band sampler bias). ZNPR v3 bake (header byte 4
   = 0x03) verified on every seed.

2. **`--corpus-target-scale NAME:FACTOR` flag** (added in this cycle). Useful
   for any future experiment mixing corpora whose `--target-col` carries
   incompatible scales. Trivial 30-line addition to the trainer's `main()`.
   Lives on zenanalyze main.

3. **5-seed CI baseline for "vanilla MLP on human_score" against
   canonical-2026-05-21.** Reproducible numbers + bake artifacts + train logs
   under `zensim/benchmarks/recovery_champion_*_2026-05-21.{log,txt}`. Future
   trainer ablations can compare against these.

## Why the V0_7 0.8893 baseline number from RECOVERY_PLAN doesn't survive

Per `zentrain/RECOVERY_PHASE3_2026-05-21.md` §"Structural conflict with the
recovery plan's ship gate":

1. The V0_7 CSV (`training_safe_synthetic_perceptual_clean.csv`) was produced
   by a d≤16 dHash purge that was reverted on 2026-05-14. Bakes against that
   CSV carry over-aggressive contamination flags.
2. canonical-2026-05-21 is a 372-feature corpus; V0_7 was 228+3=231.
3. The `bake_verdict` harness applies a 4-parameter logistic rescale per
   Mohammadi 2025; the V0_7 0.8893 was measured by an older `dataset_metric_baseline`.

The 0.8893 ceiling is not a comparable target. The current ships' published
numbers (Balanced 0.8324, Compression 0.8641, both per
`bake_verdict`) are the only honest reference.

## What WOULD likely beat KonJND gate (not executed — future work)

1. **Train on `mix_cv55_iw45` target** (which is what V_22-mix-LARGE+iwssim s3
   uses). This requires either (a) injecting `mix_*` targets into canonical
   val parquets, or (b) a `--target-col-val human_score --target-col-train
   mix_cv55_iw45` split that the current trainer doesn't support.

2. **Augment training with a per-ref pjnd-anchor batch.** KonJND signal is
   structurally different from KADID/TID supervised learning — it's a calibration
   task. Adding a small (~1k) per-ref pjnd-anchor side-loss might recover the
   KonJND axis without sacrificing CID22.

3. **Hybrid head architecture (the Rust trainer's `--per-sample-α-head` path).**
   The V_24-per-sample-α s4 Compression ship uses a per-sample alpha head that
   models per-source-image distortion-sensitivity calibration. This is one of
   the Phase-3 ports that was deferred to the Rust trainer (port 5/6 in the
   recovery plan). The Python trainer lacks this architecture.

4. **Don't ship a Balanced-trail update from this trainer.** The current
   Balanced ship (V_22-mix-LARGE+iwssim s3, KADID 0.9677, TID 0.9729, KonJND
   0.8927) is significantly stronger on the supervised-MOS / PJND-ranking axes
   than what this Python trainer can produce on canonical-2026-05-21 with the
   `human_score` target. The Compression ship has comparable CID22 but
   different trade-offs; not a swap target either.

## Files touched

- `zenanalyze/zentrain/tools/zensim_metric_train.py` — added
  `--corpus-target-scale NAME:FACTOR` CLI flag (no behavioural change for
  existing callers; defaults to no rescaling).
- `zensim/benchmarks/recovery_champion_h64_meanpol_s{1..5}_train_2026-05-21.log`
  — five per-seed training logs (raw stdout including per-epoch SROCC, val
  losses, best-epoch selection).
- `zensim/benchmarks/recovery_champion_s{1..5}_bake_verdict_2026-05-21.txt`
  — five per-seed full Mohammadi panels (5 corpora × 6 metrics × 10 bands).
- `zensim/benchmarks/recovery_phase3_champion_falsification_2026-05-21.md`
  — this doc.

## Bake artifacts (NOT in repo)

Under `/mnt/v/output/zensim/exp_recovery_champion_2026-05-21/seeds/`:

- `cc4_recovery_meanpol_h64_s{1..5}.bin` (5 × 99,215 bytes, ZNPR v3, 372→64→1)
- Plus three additional 5-seed variants (`p1only_h64_s*`, `konjnd01_h64_s*`,
  `full4ports_h128_s*`) for the failed-variant log.

## Commits

zenanalyze: `--corpus-target-scale` flag + this falsification cycle.
zensim: benchmarks + this doc (no shipped weights changed).
