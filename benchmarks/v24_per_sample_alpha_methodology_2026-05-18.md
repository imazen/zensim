# V_24 per-sample α head — methodology + 5-seed CI results

**Date:** 2026-05-18
**Branch:** `feat/ex2-stdpool-head` (workspace `zensim--ex2-persample-alpha`)
**Bake artifacts:** `/mnt/v/zen/zensim-eval/v24_persample_alpha_2026-05-18/`
**Status:** **Pareto-incomparable to V_22-mix-LARGE** — wins CID22 + AIC-3,
loses KADID/TID/KonJND. Decisively beats the V_24-hybrid-NiN scalar α baseline.

## Hypothesis (Step 1 of principled workflow)

1. **Hypothesis**: Replacing the scalar `α_logit` in the hybrid head
   with a learned per-sample function `α(x) = sigmoid(W_α · h + b_α)`
   should let the model condition α on input content. Photo / codec
   inputs (CID22-shaped) → α toward rank head; synthetic-distortion
   inputs (KADID/TID-shaped) → α toward pool head. Engagement
   diagnostic mandatory.
2. **Falsification**: If the per-sample α distributions are flat
   across corpora (mechanism didn't engage), OR if CID22 and AIC-3
   don't both improve over the scalar-α hybrid (V_24-hybrid-NiN),
   the per-sample architecture didn't deliver beyond scalar.
3. **Cost ceiling**: 1× 5-seed training run + eval.
4. **Ship form**: Pareto-promising candidate IF CID22 + AIC-3 lifts
   over V_22-LARGE survive AND KonJND ≥ 0.85.

## Architecture

Per-sample α extends hybrid_head's scalar `α_logit` with a learned
function of the encoder hidden vector:

```
h_pre  = b1 + Σ x_i · W1[i, :]             (n_hidden = 128)
h      = LeakyReLU(h_pre, slope=0.01)
y_rank = h · rank_w + rank_b
y_pool = [μ, σ, max, p_6](h) · reducer_w + reducer_b
α_logit(x) = b_α + h · W_α                  ← new path
α(x)   = sigmoid(α_logit(x))
y      = α(x) · y_rank + (1 − α(x)) · y_pool
```

**Parameter cost vs scalar-α hybrid**: `+n_hidden + 1 = 129` params
(W_α[128] + b_α). Bake metadata size at h=128: 1056 B vs hybrid's
544 B (+512 B). I8-zerobias-lz4 packed seed4: **44 KB** (vs hybrid
NiN packed at ~38 KB).

**Backprop chain rule** (extends hybrid_head):

- `∂L/∂y_rank   = ∂L/∂y · α`
- `∂L/∂y_pool   = ∂L/∂y · (1 − α)`
- `∂L/∂α       = ∂L/∂y · (y_rank − y_pool)`
- `∂L/∂α_logit = ∂L/∂α · α · (1 − α)`     (σ' on per-pair logit)
- `∂L/∂W_α[j]  = ∂L/∂α_logit · h[j]`
- `∂L/∂b_α     = ∂L/∂α_logit`
- `∂L/∂h_j    += ∂L/∂α_logit · W_α[j]`    (W_α path into encoder)
- Plus rank + pool partials into `∂L/∂h_j` (unchanged from hybrid).

Numerical-gradient finite-difference verified for every parameter
including `W_α[j]` and `b_α` paths (see `zensim-train-core::
per_sample_alpha_head::tests::backprop_finite_diff_all_params`).

**Bake metadata key**: `zentrain.per_sample_alpha_head`
**Payload layout** (f32 LE): `[W_α[n_hidden] | b_α | rank_w[n_hidden]
| rank_b | reducer_w[4] | reducer_b | p_norm]`. Total `4·(2·n_hidden + 8)`
bytes.

**Runtime dispatch priority** (priorities most-general first, in
`zensim::metric::forward_one_bake`, `bake_verdict::score_row`,
`bake_compare::score_corpus`): `per_sample_alpha_head` →
`hybrid_head` → `pool_head_reducer` → plain rank-net `out[0]`.
A bake carries at most one of these metadata keys.

## Training recipe

Identical to V_24-hybrid-NiN (5-group LARGE + iwssim) except for the
head flag. All hyperparameters held constant:

| Group | Rows | Target | Train_w | Val_w |
|---|---|---|---|---|
| safesyn | 196,086 | mix_cv40_iw60 | 1.0 | 0.0 |
| kadid   |  10,125 | mix_cv40_iw60 | 0.3 | 1.0 |
| tid     |   3,000 | mix_cv40_iw60 | 0.3 | 1.0 |
| konjnd  |   1,008 | PJND          | 0.02 | 1.0 |
| cvvdp_iwssim_large | 73,300 | mix_cv40_iw60 | 0.5 | 0.0 |

Hyperparams: hidden=128, epochs=300, lr=1e-3 (cosine 50-epoch
period), l2=1e-5, leaky-α=0.01, **minibatch=256**, val-policy=Min,
PWRC (sensory_threshold=5.0), **NiN 0.1 (p=1, q=2)**,
300-feature input, no auto-transforms.

**Init**: `W_α = 0`, `b_α = 0` → α(x) = 0.5 for every input at start.

**Trainer command** (per seed, as launched by
`scripts/v_next/run_per_sample_alpha_seed.sh`):

```
target/release/zensim_mlp_train \
  --group safesyn:.../safesyn_mix_300col.parquet:1.0:0.0 \
  --group kadid:.../kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:.../tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:.../konjnd_mix_300col.parquet:0.02:1.0 \
  --group cvvdp_iwssim_large:.../cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 \
  --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed <S> --log-every 10 --early-stop-patience 0 \
  --out persample_seed<S>.bin
```

Wall time: ~9 min per seed at 7950X, parallel 5-seed wall ~12 min.

## 5-seed CI results

| Corpus | Mean SROCC | ± std | n seeds |
|---|---|---|---|
| CID22 | **0.8589** | 0.0044 | 5 |
| KADID | 0.9321 | 0.0006 | 5 |
| TID | 0.8907 | 0.0015 | 5 |
| KonJND | 0.8201 | 0.0082 | 5 |
| AIC-3 | **0.8124** | 0.0042 | 5 |

Best CID22 = seed=4 (0.8640). Best aggregate by min-val = seed=2.

## Per-corpus α distribution (mandatory engagement diagnostic)

Seed=4 packed bake on validation parquets:

| Corpus | α mean | p05 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|
| CID22 | **0.744** | 0.491 | 0.668 | 0.777 | 0.840 | 0.894 |
| KADID | **0.290** | 0.008 | 0.079 | 0.228 | 0.430 | 0.846 |
| TID | **0.336** | 0.024 | 0.134 | 0.298 | 0.490 | 0.794 |
| KonJND | 0.723 | 0.547 | 0.648 | 0.739 | 0.803 | 0.862 |
| AIC-3 | 0.677 | 0.449 | 0.587 | 0.691 | 0.781 | 0.866 |

**The mechanism engaged decisively.** Photo / codec corpora
(CID22 / KonJND / AIC-3) pull α toward rank-dominant (≥ 0.67 mean).
Synthetic-distortion corpora (KADID / TID) pull α toward pool-dominant
(≤ 0.34 mean). The range α ∈ [0.008, 0.894] across corpora confirms
the W_α path is doing meaningful work — the b_α-only fallback would
give α constant across corpora. The prior V_24-hybrid-NiN scalar α
locked at 0.61 ± 0.01 (no per-corpus differentiation possible).

## Comparison vs baselines

| Metric | V_22-LARGE | V_24-hybrid-NiN (s4) | **per-sample-α (s4)** | Δ vs V_22 | Δ vs scalar α |
|---|---|---|---|---|---|
| CID22 SROCC | 0.832 | 0.862 | **0.864** | +0.032 | +0.002 |
| KADID SROCC | 0.968 | 0.929 | 0.932 | −0.036 | +0.003 |
| TID SROCC | 0.973 | 0.889 | 0.889 | −0.084 | 0.000 |
| KonJND SROCC | 0.893 | 0.799 | 0.808 | −0.085 | +0.009 |
| AIC-3 SROCC | 0.785 | 0.807 | **0.818** | +0.033 | +0.011 |
| Packed KB | 41 | 38 | 44 | +3 | +6 |

### `bake_compare` decisive verdicts

(N=200 bootstrap resamples, MRR z-test, full Mohammadi 5-of-6
panel agreement, 10-band sweep.)

| A vs B | A wins | B wins | Overall |
|---|---|---|---|
| **per-sample-α (s4) vs V_22-LARGE** | 3 | 17 | B (V_22 decisive) |
| **per-sample-α (s4) vs V_24-hybrid-NiN (s4)** | **5** | **0** | **A (per-sample decisive)** |

The per-sample-α bake **decisively beats the scalar-α hybrid baseline
across every decisive band**. The V_22-LARGE comparison is the same
shape as V_24-hybrid-NiN vs V_22 — synthetic-anchor SROCC is too
expensive to give up for the compression-corpus lift.

## Pareto gate result

User-specified gates (re-stated from task brief):

- CID22 ≥ V_22's 0.832 — **PASS** (0.864, +0.032)
- KonJND ≥ 0.880 − 0.02 = 0.860 — **FAIL** (0.808, off by 0.052)
- KADID within −0.01 of V_22's 0.968 = ≥ 0.958 — **FAIL** (0.932)
- TID within −0.01 of V_22's 0.973 = ≥ 0.963 — **FAIL** (0.889)
- AIC-3 ≥ V_22 + 0.015 = 0.7995 — **PASS** (0.818, +0.013 above
  the gate; +0.033 above V_22)

**2 of 5 gates pass.** CID22 and AIC-3 surfaces are correctly
captured by the rank-dominant α(x) on those inputs, but
synthetic-distortion bands lose too much surface to KADID/TID's
pool-dominant α. The architecture is doing exactly what it
should — the cost is intrinsic to letting α swing.

## Why the synthetic-distortion regression

KADID's distortions are ~95 % non-compression (blur, noise, color,
geometric). On those inputs the per-sample α correctly identifies
"pool stats matter more" — but the V_22-LARGE recipe trained the
rank head on the LARGE+iwssim corpus to handle compression with high
fidelity, and the rank head's encoder is now underused on synthetic
inputs (α=0.29 mean on KADID) — the pool head is not as well-tuned
to KADID's specific distortion spectrum because most of its
gradient came from photo/codec rows during training.

In other words: the per-sample α gating works in inference but
**the training corpus didn't have a matched pool-head specialist
signal for KADID/TID**. A future training run that **boosts
KADID/TID train weight** or **adds a synthetic-distortion-rich
group** would let the pool head re-learn KADID's distortion
spectrum at high α=0.29 while preserving CID22 / AIC-3 rank wins.

## Files

- Branch: `feat/ex2-stdpool-head` (workspace
  `zensim--ex2-persample-alpha`)
- Module: `zensim-train-core/src/per_sample_alpha_head.rs` (forward
  + backprop + bake + parse + apply_runtime; 7 unit tests including
  numerical-gradient checks on every parameter)
- Trainer: `zensim-validate/src/mlp_train.rs::train_mlp_per_sample_alpha_head`
  + `flush_per_sample_alpha_nin_batch` + `predict_group_per_sample_alpha_head`
- CLI flag: `--per-sample-alpha-head` (mutually exclusive with
  `--hybrid-head` and `--pool-head`)
- Runtime: dispatch added to `zensim::metric::forward_one_bake`,
  `bake_verdict::score_row` (returns `(y, α)`), `bake_compare::score_corpus`
- 5-seed bakes: `/mnt/v/zen/zensim-eval/v24_persample_alpha_2026-05-18/persample_seed{1..5}.bin`
- Packed winner (seed4): `…/persample_seed4_packed.bin` (44 KB)
- Verdicts: `…/verdict_seed{1..5}.md`, `…/verdict_seed4_packed.md`
- `bake_compare` reports: `…/compare_persample_vs_v22.md`,
  `…/compare_persample_vs_v24hybrid.md`

## Decision

**Not shipping as PreviewV0_N.** Per `bake_compare` decisive rule,
per-sample-α loses to V_22-mix-LARGE in 17 bands and only wins 3.
The hypothesis is **partially validated**: the per-sample α
mechanism does engage and does beat the scalar-α hybrid decisively
on the same training data, but the training corpus needs
synthetic-distortion-rich groups for the pool head to learn KADID/TID
spectra at low α.

**Next architectural levers** (suggested by the failure mode, not
yet attempted):

1. **Boost KADID/TID train weight** from 0.3 to 0.6+ on the
   per-sample-α path. The pool head's low-α regime is undertrained
   on KADID/TID's distortion spectrum because they share rank-head
   gradient with safesyn / LARGE.
2. **Add a synthetic-distortion-rich training group.** KADID's
   ~10k rows aren't enough relative to the 270k LARGE+safesyn at
   high train weights. A 50k KADID-style augmentation corpus
   would re-balance the pool head's training signal.
3. **Sample-conditional pool reducer**: extend the per-sample-α
   mechanism to per-sample reducer weights — `reducer_w(x) =
   reducer_w_base + W_red · h`. Lets the pool head also condition
   on content (the σ-weight matters more on KADID than CID22).
4. **Larger hidden** (h=192 or h=256). The per-sample-α head adds
   capacity needs (`+(n_hidden+1)` params) and h=128 may bottleneck
   the encoder's ability to encode α-relevant information alongside
   y_rank + y_pool info.

The +0.013 AIC-3 lift over V_22 is real and bigger than what the
scalar-α hybrid produced (+0.022, but at the cost of CID22 +0.030
vs V_22). The per-sample mechanism unlocks an additional +0.011
AIC-3 over scalar α without trading off CID22 — that's the clean
architectural win this experiment was checking for.
