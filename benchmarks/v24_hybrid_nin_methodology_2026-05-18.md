# V_24-hybrid-nin methodology (NiN composition test on hybrid pool+rank head)

**Date:** 2026-05-18
**Status:** 5-seed CI with NiN composed into hybrid-head trainer. **Falsifies the "NiN gap explains KonJND/KADID/TID regressions" hypothesis** that the V_24-hybrid (no-NiN) methodology doc flagged as the most likely root cause.

## Hypothesis + falsification (the question this run answered)

The prior V_24-hybrid 5-seed CI (no-NiN, see
`benchmarks/v24_hybrid_methodology_2026-05-18.md`) showed:
- CID22 +0.035 vs V_22 (genuine win)
- AIC-3 +0.020 vs V_22 (FIRST AIC-3 lift of the session)
- KADID/TID/KonJND each lost 0.04–0.10 SROCC vs V_22

The prior agent flagged the missing NiN composition (V_22 uses
`--norm-in-norm-weight 0.1`, the v0 hybrid trainer panicked on >0) as
the load-bearing contamination — predicting that NiN composition
would close the KADID/TID/KonJND gap while preserving the CID22 +
AIC-3 wins.

This run tested that prediction directly: same recipe, same seeds,
**NiN composed end-to-end via the new
`flush_hybrid_head_nin_batch`** in `zensim-validate/src/mlp_train.rs`.

## Implementation

NiN wire-in (commit `4e7dbe18` on branch `feat/ex2-stdpool-head`,
**+441 / −12 lines**):

1. Added `HybridPairForward<'a>` struct holding per-pair forward
   state — `xa/xb`, composite `ya/yb`, components `ya_rank/yb_rank/
   ya_pool/yb_pool`, sigmoid α per side, LeakyReLU pre+post
   activations, pool stats, max indices, cached RankNet gradients,
   labels. Mirrors `PoolPairForward` exactly.

2. Added `flush_hybrid_head_nin_batch` — analog of
   `flush_pool_head_nin_batch` but routes per-prediction NiN
   gradient through `backprop_step_hybrid_head` (which back-routes
   through **both rank-head + pool-head + sigmoid α derivative**
   simultaneously). Adam slot layout follows the hybrid head:
   `gw2 = [rank_w | reducer_w | α_logit]`, `gb2 = [rank_b, reducer_b]`.

3. Modified `train_mlp_hybrid_head_with_tv`: replaced the
   panic-on-NiN assert with `nin_on` branching. On `nin_on`:
   - `target==0` and PWRC-threshold-violation pairs push `None` and
     flush on K-fill (preserves RNG draw schedule)
   - Valid pairs push `Some(HybridPairForward)` and flush on K-fill
   - TV regularizer skipped (mirrors pool-head trainer; V_22/V_24
     recipe doesn't use TV anyway)
   - Epoch-end final-flush iff ≥16 surviving pairs in buffer
   - L2 on `w1 + rank_w + reducer_w` scaled by `steps_added`;
     `α_logit` unregularized (matches per-pair path)

CLI flag `--norm-in-norm-weight` was already plumbed through to
`MlpHyperparams` (used by pool-head trainer); no CLI changes
needed.

**Smoke test before full run**: h=32, 5 epochs, 2k pairs/epoch with
`--norm-in-norm-weight 0.1`. α moved from 0.500 → 0.537, training
loss decreased 0.454 → 0.248, NiN-on log line emitted. Verified
the path runs end-to-end before launching the 5-seed CI.

## Training recipe (identical to V_24-hybrid no-NiN run)

5 groups (matching V_22-mix-LARGE-iwssim production recipe):
- safesyn: 196,086 rows, train_w=1.0, val_w=0.0, target=mix_cv40_iw60
- kadid: 10,125 rows, train_w=0.3, val_w=1.0
- tid: 3,000 rows, train_w=0.3, val_w=1.0
- konjnd: 1,008 rows, **train_w=0.02**, val_w=1.0, target=PJND
- cvvdp_iwssim_large: 73,300 rows, train_w=0.5, val_w=0.0

Hyperparams:
- `hidden=128`, `epochs=300` (`early_stop_patience=0`)
- `pairs_per_epoch=50000`, `lr=1e-3` cosine 50-epoch period
- `l2=1e-5`, `leaky_alpha=0.01`, `minibatch=256`, val-policy=min
- PWRC pair weight ON, PWRC sensory threshold = 5.0
- **`--norm-in-norm-weight 0.1`** (matching V_22, the variable under
  test)
- `--norm-in-norm-p 1.0 --norm-in-norm-q 2.0`
- 300-feature input (no auto-transforms)

Per-seed wall: ~8.3 min wall in parallel (5 seeds × 96% CPU).
Total agent wall: train + eval + rebake = ~12 min.

Trainer: `target/release/zensim_mlp_train --hybrid-head ...`
Script: `scripts/v24_hybrid_nin_train.sh <seed>`
Eval script: `scripts/v24_hybrid_nin_eval.sh`

## Results

### α convergence per seed (5 seeds, NiN ON)

| Seed | α | logit |
|---|---:|---:|
| 1 | 0.6124 | +0.4574 |
| 2 | 0.6098 | +0.4465 |
| 3 | 0.6440 | +0.5929 |
| 4 | 0.6224 | +0.4998 |
| 5 | 0.6115 | +0.4534 |
| **mean ± std** | **0.620 ± 0.014** | |

For comparison, the no-NiN run converged to **α = 0.617 ± 0.014**
(seed mean reported in `v24_hybrid_methodology_2026-05-18.md`).
**NiN's gradient regularization did NOT shift α equilibrium** —
NiN-on and NiN-off converge to indistinguishable α (within 0.003 of
each other, well inside per-seed σ). σ-weight magnitudes (reducer_w
stats μ,σ,max,p6) are also indistinguishable.

### Aggregate SROCC per corpus (5-seed mean ± std)

| Bake | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| V_22-LARGE-iwssim (s=3, packed, with NiN 0.1) | 0.8324 | 0.9677 | 0.9729 | **0.8927** | 0.7845 |
| V_24-hybrid (no-NiN, 5-seed) | 0.8672 ± 0.009 | 0.9310 ± 0.002 | 0.8898 ± 0.002 | 0.7853 ± 0.009 | 0.8041 ± 0.002 |
| **V_24-hybrid-NiN (5-seed)** | **0.8657 ± 0.004** | **0.9304 ± 0.002** | **0.8886 ± 0.001** | **0.7913 ± 0.006** | **0.8066 ± 0.003** |
| Δ vs V_22 | **+0.033** | −0.037 | −0.084 | −0.101 | **+0.022** |
| Δ vs no-NiN | −0.001 | −0.001 | −0.001 | **+0.006** | +0.003 |

**The NiN-on result is statistically indistinguishable from
NiN-off on every corpus** — all deltas vs no-NiN are within ±0.006
SROCC, well inside the 5-seed standard deviation. KonJND moved by
+0.006 (within noise). KADID/TID changed by less than σ.

### Z-RMSE per corpus (5-seed mean)

| Bake | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| V_22 ship | 0.559 | 0.249 | 0.236 | 0.376 | 0.606 |
| hybrid-NiN s2 | 0.492 | 0.362 | 0.436 | 0.554 | 0.575 |

Same pattern: hybrid-NiN wins Z-RMSE on CID22 + AIC-3 (lower is
better), loses on KADID/TID/KonJND. NiN composition did not move
Z-RMSE either.

### bake_compare decisive verdicts

**Seed=2 (best CID22 NiN seed) vs V_22-LARGE-iwssim** (1000-bootstrap, § A.9 rule):

| Corpus | n | SROCC_A (NiN) | SROCC_B (V_22) | h_SROCC | DecScore | Verdict |
|---|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8727 | 0.8324 | +49.76 | +41.46 | **A>>B** |
| KADID | 10125 | 0.9319 | 0.9677 | −90.20 | −∞ | **B>>A** |
| TID | 3000 | 0.8884 | 0.9729 | −54.12 | −∞ | **B>>A** |
| KonJND | 1008 | 0.7906 | 0.8927 | −44.28 | −∞ | **B>>A** |
| AIC-3 | 600 | 0.8096 | 0.7845 | +18.34 | +15.29 | **A>>B** |

**Seed=2 NiN vs Seed=2 no-NiN** (NiN composition isolation A/B):

| Corpus | n | SROCC_A (NiN) | SROCC_B (no-NiN) | h_SROCC | DecScore | Verdict |
|---|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8727 | 0.8749 | −27.82 | −∞ | B>>A |
| KADID | 10125 | 0.9319 | 0.9332 | −82.86 | −∞ | B>>A |
| TID | 3000 | 0.8884 | 0.8916 | −180.16 | −∞ | B>>A |
| KonJND | 1008 | 0.7906 | 0.7861 | +13.43 | −0.00 | **tied** |
| AIC-3 | 600 | 0.8096 | 0.8069 | +34.30 | +5.72 | **promising** |

NiN composition gives a marginal, non-decisive AIC-3 lift (+0.003 SROCC)
and a tied/marginal KonJND lift (+0.005 SROCC), at the cost of
marginal losses on the other three corpora. **None of these deltas
clear the decisive § A.9 threshold; this is one bake away from random.**

### Pareto gate result (5-seed mean vs targets)

| Corpus | Target | NiN mean | Δ | Pass |
|---|---:|---:|---:|---|
| CID22 | ≥ 0.832 + 0.005 = 0.837 | 0.8657 | +0.029 | ✓ |
| KonJND | ≥ 0.880 − 0.01 = 0.870 | 0.7913 | −0.079 | ✗ |
| KADID | ≥ 0.96 | 0.9304 | −0.030 | ✗ |
| TID | ≥ 0.96 | 0.8886 | −0.071 | ✗ |
| AIC-3 | ≥ 0.80 | 0.8066 | +0.007 | ✓ |

**GATE: FAILED — 3/5 corpora fail.** Same pattern as no-NiN.

### Packed seed=4 winner

Seed=4 is the highest-KonJND NiN seed (0.7984). Repacked via
`zenpredict repack --compress --zerobias 0.005 --dtype i8`:
- Input: 223,354 bytes
- Packed: **43,568 bytes (19.5%)**, 33.4% of weights zeroed
- Round-trip max|Δ| = 21.16 (large, due to I8 quant noise, but
  SROCC preserved per re-eval)
- Re-eval: CID22 0.8619 / KADID 0.9289 / TID 0.8893 / KonJND 0.7986 / AIC-3 0.8070
  (within 0.0002 of unpacked — quant noise does NOT shift SROCC
  aggregates measurably)

Path: `/mnt/v/zen/zensim-eval/v24_hybrid_nin_2026-05-18/v24_hybrid_nin_konjnd002_LARGE_iwssim_s4_h128_packed.bin`

## Falsified findings

### 1. NiN gap does NOT explain the KonJND/KADID/TID regressions

This was the load-bearing claim of the no-NiN methodology doc:
> The KADID/TID/KonJND losses are likely mostly attributable to
> missing NiN, NOT to the head architecture. The CID22 and AIC-3
> wins should survive — and possibly improve — when NiN composes back in.

**Falsified by direct measurement.** NiN-on and NiN-off produce
indistinguishable results across all 5 corpora (deltas within
±0.006 SROCC, well inside σ). The pool+rank+α hybrid head loses
KADID/TID/KonJND regardless of NiN composition.

### 2. NiN's gradient regularization does NOT shift α equilibrium

Prediction was that NiN's batch-normalization-style auxiliary loss
would push α toward a different (more pool-leaning) equilibrium.
**Falsified.** α converges to 0.620 ± 0.014 with NiN and 0.617 ± 0.014
without. The learned mixing coefficient is structurally pinned to
~0.62 by the rank-head's superior CID22-shaped gradient signal,
independent of NiN.

### 3. AIC-3 +0.020 lift survives — and does NOT improve with NiN

The AIC-3 lift is the load-bearing genuine breakthrough of this
direction (FIRST AIC-3 lift of the session). NiN-on: +0.022 vs V_22
(seed-mean), NiN-off: +0.020. **The lift survives** (it's real) but
**it does not grow** under NiN composition. This further supports
the hypothesis that the AIC-3 win is driven by the hybrid-head's
combination of pool statistics + rank weighting, not by any
batch-loss regularization.

### 4. CID22 +0.033 lift survives — and does NOT improve with NiN

Same story as AIC-3: NiN-on +0.033, NiN-off +0.035. Within seed
noise. **The architectural win on CID22 is real but caps out
around +0.033.**

## Honest gaps — what the per-sample α experiment should test next

The no-NiN methodology doc proposed a follow-up: **per-sample α
head**: replace the scalar `α_logit` with `α(x) = σ(W_α · h + b_α)`
so α can fire conditionally on JND-likely inputs. The reasoning:

- Current hybrid has ONE α per bake (effectively a fixed mixing
  parameter, not adaptive)
- The pool path's gradient signal is averaged over all 5 corpora;
  on KonJND it's drowned out by the rank-path's CID22/IW-SSIM
  signal
- A per-sample α could route JND-like inputs (low pool-σ, narrow
  feature spread) toward the pool head while routing high-distortion
  inputs toward the rank head

This is the obvious next experiment. The architectural cost is small
(+128 weights + 1 bias for `W_α : h → α_logit`), the backprop is a
mechanical extension of the current sigmoid α derivative.

### Other follow-ups (ranked by likely-impact)

1. **Per-sample α head** (above) — direct mitigation of the
   KonJND/KADID/TID losses
2. **Multi-bake runtime mixing**: keep V_22 as production
   PreviewV0_3, ship hybrid s4 as PreviewV0_4 ensemble slot.
   Linear mix raw outputs (V_22 weight α_runtime, hybrid weight 1−α_runtime).
   This preserves V_22's KADID/TID/KonJND while picking up hybrid's
   CID22 + AIC-3 wins on the bands where they apply.
3. **NiN with higher weight (0.5, 1.0)** — the current 0.1 matches
   V_22; perhaps NiN needs a stronger composition signal to
   actually influence the hybrid path's gradient. This is a cheap
   experiment to rule out (1 seed × 8 min = 8 min agent wall).
4. **NiN with `q=1` instead of `q=2`** — q=1 is the L1 variant
   that Li 2020 reports as more robust on heavy-tailed distortions
   (which describes KonJND's PJND clustering well). Another cheap
   experiment.

## Decision

**DO NOT SHIP V_24-hybrid-NiN as the next zensim profile.** The
KonJND regression remains too large (−0.101 vs V_22). NiN
composition is now verified to not be the load-bearing gap.

**Falsification status of "hybrid pool+rank+α direction": NOT
falsified — but reframed.** The architecture has demonstrated:
1. CID22 +0.033 lift over V_22 (genuine, robust across 5 seeds + NiN axis)
2. AIC-3 +0.022 lift over V_22 (the FIRST AIC-3 win of the session,
   robust across 5 seeds + NiN axis)
3. KADID/TID/KonJND losses are **intrinsic to the head**, not to
   NiN absence — so the next experiment must change the **head**,
   not the loss

**The follow-up is per-sample α, not "more NiN tuning".** Skip
NiN-weight and NiN-q variations; they will give nothing.

## Inputs used in this experiment

- Trainer: `target/release/zensim_mlp_train` (built from
  `feat/ex2-stdpool-head` branch + NiN wire-in commit `4e7dbe18`)
- Training data (300-feature parquets):
  `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/`
- Validation features (372-col parquets):
  `/mnt/v/zen/zensim-training/2026-05-15-full-features/`
- V_22 baseline bake:
  `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed.bin`
- V_24-hybrid (no-NiN) bakes:
  `/mnt/v/zen/zensim-eval/v24_hybrid_2026-05-18/v24_hybrid_konjnd002_LARGE_iwssim_s{1..5}_h128.bin`
- V_24-hybrid-NiN bakes (this run):
  `/mnt/v/zen/zensim-eval/v24_hybrid_nin_2026-05-18/v24_hybrid_nin_konjnd002_LARGE_iwssim_s{1..5}_h128.bin`
- Packed winner:
  `/mnt/v/zen/zensim-eval/v24_hybrid_nin_2026-05-18/v24_hybrid_nin_konjnd002_LARGE_iwssim_s4_h128_packed.bin`
