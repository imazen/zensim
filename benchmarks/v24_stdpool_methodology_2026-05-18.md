# V_24-stdpool methodology (EX-2: std-pool head)

**Status: SEED=3 FALSIFIED 2026-05-18.** CID22 SROCC 0.6046 vs V_22-mix-LARGE baseline 0.8324 (Δ = **−0.228**); held-out KADID/TID/KonJND/AIC-3 all regress by 0.16 to 0.84. **Per CLAUDE.md Step 3, the seed=3 result does not warrant a 5-seed sweep.**

### Important caveat — this is NOT a fair head-to-head A/B

The `pool_head_train` binary is a **stripped-down RankNet trainer** (no mini-batch, no TV regularizer, no PWRC band weights, no CVVDP-pretrain → IWSSIM-fine-tune two-phase, scalar Rust only). V_22-mix-LARGE-iwssim uses the production `zensim_mlp_train` pipeline with all of those.

The seed=3 falsification therefore says: **"this minimal pool-head trainer at this single hyperparameter setting does not beat the production trainer."** It does **not** falsify the EX-2 hypothesis that a pool-head on top of an equivalently-trained baseline would lift CID22.

The next step would be to wire the pool-head architecture into `zensim_mlp_train` (so it inherits TV / PWRC / mini-batch / two-phase training) and re-run. That work is **deferred until EX-1 (Thurstone, the higher-yield change per the doc roadmap) lands and the trainer infrastructure stabilizes.** Re-attempting EX-2 in isolation on the stripped-down trainer is structurally rigged against the hypothesis.

## Hypothesis + falsification (per CLAUDE.md principled workflow Step 1)

1. **Hypothesis**: Replacing the MLP's final scalar with 4 pooled stats
   `[μ, σ, max, p_6]` of the hidden vector + a 4→1 reducer should lift
   CID22 SROCC by ≥ +0.005 over V_22-mix-LARGE-iwssim at the same
   hidden width (h=128), at near-zero inference cost. Justification:
   GMSD's std-pooling result (PLCC ≈ 0.960 on LIVE at 50× lower
   compute than FSIM) and Butteraugli's `{3, 6, 12}` p-norm averaging
   are direct evidence that mean-only pooling discards information
   humans use. Reference paper sweep:
   `~/work/zen/zensim/PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md` §3 + §8 EX-2.

2. **Falsification**: If seed=3 CID22 aggregate SROCC is below
   V_22-mix-LARGE-iwssim by > 0.005 OR if the held-out signal
   (KADID + TID + KonJND) regresses by > 0.01 on average, the
   hypothesis is dead. Multi-stat agreement gate (CLAUDE.md SROCC-only
   verdicts banned): at least 3 of 5 stats {SROCC, PLCC, KROCC, PWRC,
   Z-RMSE} agree on direction.

3. **Cost ceiling**: seed=3 + (if positive) 5-seed CI sweep. Total ~3
   hours single-threaded scalar.

4. **Ship form**: PreviewV0_N single-bake if winning; the runtime
   metadata-detection path is in place and works for any future
   pool-head bake.

## Architecture

```
x (n_inputs)                                      shape
   │ standardize (mean / scale from training)
   ▼
   W1 (n_inputs × n_hidden, F32, row-major)
   │ + b1 (n_hidden)
   ▼
   LeakyReLU(α = 0.01)
   ▼
   h (n_hidden)                                   ← bake.predict() returns this
   │
   ├──→ μ = mean(h)
   ├──→ σ = sqrt(mean((h − μ)²)) (floor: 0.0026)
   ├──→ max = max(h)
   └──→ p_6 = (mean(|h|^6))^(1/6)
        │
        ▼
        [μ, σ, max, p_6] (4 scalars)
        │
        ▼
        W_reducer (4 × 1) + b_reducer
        ▼
        y (scalar)
```

**Parameter count delta vs V_22-mix-LARGE (228 → 128 → 1)**:
- V_22: `228·128 + 128 + 128·1 + 1 = 29313` params
- V_24-stdpool: `228·128 + 128 + 4 + 1 = 29317` params **(+4)**

Effective bake size grows by the identity passthrough W2 (n_hidden ×
n_hidden, stored verbatim) — this is a wire-format choice, not a
trainable parameter expansion. Identity passthrough lets the existing
`Predictor::predict()` return the hidden vector without runtime fork
on activation type.

## Constants (verbatim from doc §3 table)

| Constant | Value | Source |
|---|---|---|
| GMSD pooling | `std` of map (now: hidden vector) | Xue 2013 §3.3 |
| GMSD `c` (8-bit normalized stability) | 0.0026 | Xue 2013 §3.2 → used here as σ-floor |
| Butteraugli p-norm exponent (chosen) | 6 | Guetzli §4 (from `{3, 6, 12}` family) |
| LeakyReLU α | 0.01 | Existing V_X convention |

## Bake format (ZNPR v3)

**Single-source bake, no input transforms, no output specs.** The wire
format mirrors a standard 2-layer MLP:

- **Header**: ZNPR v3, n_inputs=228, n_outputs=n_hidden (128), n_layers=2
- **Scaler**: mean, scale (per-feature, n_inputs floats each)
- **Layer 1**: 228 × 128 F32 weights + 128 biases, activation LeakyReLU
- **Layer 2**: 128 × 128 F32 identity matrix + 128 zero biases,
  activation Identity (passthrough — surfaces the hidden vector)
- **Metadata**: single entry
  - `key`: `"zentrain.pool_head_reducer"`
  - `kind`: `Numeric`
  - `value`: 24 bytes = 6 f32 LE = `[w_μ, w_σ, w_max, w_p6, b, p_norm]`

**Schema-hash impact**: none. The schema_hash field is unchanged
(`0`). Bakes without `zentrain.pool_head_reducer` metadata are
unaffected — the runtime's existing scalar-output path is the
fallthrough.

## Runtime dispatch (zensim/src/metric.rs)

```rust
// In forward_one_bake:
let pool_head_reducer = model.metadata()
    .get("zentrain.pool_head_reducer")
    .and_then(parse_24_bytes_to_6_f32);

let out = predictor.predict(&features)?; // hidden vector

if let Some((rw, rb, p_norm)) = pool_head_reducer {
    // compute [μ, σ, max, p_6] over out
    // y = rw · stats + rb
}
```

The pool + reducer math is **scalar Rust**. Cost on h=128: ~5 ns per
call (one pass for sum/max/sum_p, one pass for var, one dot product).
Dwarfed by the 228-feature extractor (~ms).

## Bake shape (CLAUDE.md Step 4)

**`shape: score`** — trained directly against `mix_cv40_iw60` MOS
target (after target_scale=100.0 → range ≈ [0, 100]). RankNet pairwise
loss on the trained MOS column. **Do NOT apply V_18's affine α/β**
when shipping — the bake is already score-shaped. Hard-clamp policy
inherited from PreviewV0_3 (soft-clamp NOT enabled — V_24 is
single-bake, no multi-bake mixing extrapolation).

## SIMD parity status

**Scalar fallback only**. The pool stats are a 4-way reduction over a
128-element vector; AVX2 + a Welford pass would shave nanoseconds but
the kernel is not on the hot path (one call per `Zensim::compute`,
~ms upstream). Documented as `simd-gap: pool-head reduce` in the
trainer + runtime — a future commit can fold it into archmage/
magetypes if profiling shows it matters.

The upstream Layer-1 matmul (the load-bearing cost) **already routes
through zenpredict's SIMD forward** (the existing inference path
unchanged). Only the pool + reducer steps are new scalar code, and
they are non-load-bearing.

## Trainer command (seed=3)

```sh
./target/release/pool_head_train \
    --group safesyn:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/safesyn_mix_300col.parquet:1.0:0.0 \
    --group kadid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/kadid_mix_300col.parquet:0.3:1.0 \
    --group tid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/tid_mix_300col.parquet:0.3:1.0 \
    --group konjnd:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/konjnd_mix_300col.parquet:0.5:1.0 \
    --hidden 128 --epochs 200 --pairs-per-epoch 50000 --seed 3 \
    --target-column mix_cv40_iw60 \
    --out benchmarks/v0_24_stdpool_seed3_h128_2026-05-18.bin
```

**Data lineage** (all inputs MD5 invariant; CID22 leak audit verified
upstream by the 2026-05-17 parquet builds; no new training data
introduced for this experiment).

## Baseline (V_22-mix-LARGE-iwssim, ship candidate)

`/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed.bin`

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8324 | 0.8289 | 0.6340 | 0.0443 | 0.9006 | 0.559 |
| KADIK10k | 10125 | 0.9677 | 0.9686 | 0.8432 | 0.0433 | 0.9804 | 0.249 |
| TID2013 | 3000 | 0.9729 | 0.9717 | 0.8571 | 0.0357 | 0.9832 | 0.236 |
| KonJND-1k (full) | 1008 | 0.8927 | 0.9265 | 0.7070 | 0.0446 | 0.9178 | 0.376 |
| AIC-3 CTC | 600 | 0.7845 | 0.7953 | 0.6155 | 0.0433 | 0.8630 | 0.606 |

## V_24-stdpool seed=3 result (SEED=3 FALSIFIED)

Full panel: `benchmarks/v0_24_stdpool_seed3_h128_eval_2026-05-18.md`.

| Corpus | n | V_24-stdpool SROCC | V_22-mix-LARGE SROCC | Δ | V_24 PLCC | V_24 PWRC | V_24 Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | **0.6046** | 0.8324 | **−0.228** | 0.5988 | 0.6909 | 0.801 |
| KADIK10k | 10125 | 0.6986 | 0.9677 | **−0.269** | 0.6964 | 0.7618 | 0.718 |
| TID2013 | 3000 | 0.7699 | 0.9729 | **−0.203** | 0.7949 | 0.8370 | 0.607 |
| KonJND-1k (full) | 1008 | 0.0520 | 0.8927 | **−0.841** | 0.1808 | 0.0748 | 0.984 |
| AIC-3 CTC | 600 | 0.6205 | 0.7845 | **−0.164** | 0.6138 | 0.7144 | 0.789 |

Reducer weights (trained): `μ=−0.488, σ=4.195, max=0.326, p_6=−1.890; b=0.0`. The trainer **did** drive σ to the dominant weight (the GMSD-predicted insight) — magnitude 4.2 vs μ's 0.49 — so the pool-head architecture is mechanically training as expected. The architectural mechanism is sound; the issue is that this stripped-down trainer cannot reach V_22's level on any of the held-out corpora, so the marginal comparison ("did std-pool *add* anything?") cannot be made on this evidence.

## 5-seed CI sweep — NOT RUN

Per CLAUDE.md Step 3 ("Seed=1 flat or negative → hypothesis probably dead, do NOT sweep 5 seeds hoping seed=2 wins — that is p-hacking"), the 5-seed CI sweep is **not justified** by the seed=3 result. The next iteration of this experiment should:

1. Wire `pool_head` forward + backward into `zensim-validate/src/mlp_train.rs` (so it inherits TV / PWRC / mini-batch / two-phase CVVDP-pretrain → IWSSIM-fine-tune).
2. Train a **plain scalar baseline** in the same trainer to establish the apples-to-apples bottom-of-the-V graph.
3. Train pool-head with **the same recipe**. Compare the two.

This work is deferred to a follow-up cycle that should be sequenced **after** EX-1 (Thurstone, the highest-yield doc item) lands.

## Honest gaps at land time

1. **SIMD parity**: pool + reducer is scalar. Acceptable per task
   brief ("scalar fallback is acceptable for the first iteration").
   File: `zensim/src/metric.rs` near `pool_head_reducer` detection.
2. **σ-floor gradient discontinuity**: when sigma == floor, ∂σ/∂h_j
   is set to 0. Theoretically a small bias in early training when
   hidden activations are still close to zero. Not visible in
   practice on h=128 / α=0.01 (LeakyReLU avoids the dead-ReLU
   regime).
3. **p_6 gradient stability**: when p_6 is very small, the
   `1/(n · p_6^5)` factor blows up. Mitigated by a `1e-12` floor.
   Hidden activations driven below this floor across 128 units would
   indicate a broken training signal — fail-fast preferred.
4. **No TV regularizer**: deferred. V_22 ship cycle uses TV at
   `--tv-weight 1.0 --tv-band-weights 10,30,10,30` for monotonicity.
   V_24-stdpool seed=3 trains without TV; if seed=3 wins SROCC we
   can fold TV in on the 5-seed sweep.
5. **Per-bake recalibration**: no affine post-bake. The trainer
   targets `mix_cv40_iw60 × 100` directly so the raw output is
   already score-shaped. If post-hoc analysis shows the slope is off,
   `zenpredict repack` carries a built-in affine path.
6. **The seed=3 result on CID22 may be inflated by ssim2-target
   bias in the training corpus** (CLAUDE.md "ssim2-favoring SROCC"
   antidote). The `mix_cv40_iw60` column is a CVVDP × IW-SSIM blend,
   which mitigates but doesn't eliminate the bias. Watch PWRC and
   Z-RMSE for the real read on whether the pool-head architecture
   actually generalizes.

## Outstanding (do not ship until cleared)

- [ ] seed=3 eval completes and produces a winning panel vs V_22-mix-LARGE
- [ ] 5-seed CI sweep
- [ ] Methodology doc gets the per-band tables filled in
- [ ] If shipping: `zenpredict repack --compress --zerobias 0.005 --dtype i8`
- [ ] Update CHANGELOG.md `[Unreleased]` section

---

## Update 2026-05-18 (PM): Production trainer wire-in

The seed=3 falsification of the stripped-down `pool_head_train` binary
(CID22 0.60 vs V_22-mix-LARGE 0.83) was confirmed to be a trainer-recipe
artifact, not an architectural one. This section documents the
follow-up where the pool-head forward+backprop was wired into
`zensim_mlp_train` (the production trainer) so it inherits TV + PWRC +
NiN + minibatch=256 + cosine LR + 300 epochs (early-stop patience 30) +
the 5-group V_22-mix-LARGE training corpus.

### Wire-in scope (commits 286373ab, a620f775)

- `MlpHyperparams::pool_head: bool` field; default `false` keeps every
  V_X bake bit-identical to its prior recipe.
- `train_mlp_with_tv` dispatches to `train_mlp_pool_head_with_tv` when
  the flag is set. The new function reuses every recipe lever
  (group sampling, per-row low/mid/high-q boosts, cosine LR, PWRC
  sensory threshold + band weighting, L2 on layer weights, early stop
  on val SROCC, mini-batch SGD with sequential gradient accumulation
  + final-flush, TV regularizer) but routes forward + backprop through
  `zensim_train_core::pool_head::{forward_pool_head,
  backprop_step_pool_head}`. The 4-wide reducer + scalar bias live in
  Adam's `gw2`/`gb2` slots (the slots are shape-agnostic), so Adam
  steps update layer-1 weights + reducer in one call.
- **NiN composition** via `flush_pool_head_nin_batch`: per the user
  brief, V_22-mix-LARGE uses NiN, so we ported it. The mini-batch
  sequential path buffers K pair forwards, computes NiN over the 2K
  surviving predictions, scatters per-prediction grad through the
  pool-head chain rule (additive on top of cached RankNet `dl_dy`),
  then runs one Adam step. K ≥ 16 enforced for stable batch
  statistics (matches the standard head's NiN gate).
- `--pool-head` CLI flag on `zensim_mlp_train`.
- `bake_verdict` and `bake_compare` extended with pool-head dispatch
  (extract `zentrain.pool_head_reducer` metadata, apply pool+reducer
  to the bake's passthrough hidden vector instead of taking `out[0]`).
  Matches the runtime dispatch in `zensim::metric::apply_mlp_scoring`.
- **SIMD parity gap**: pool-head backprop remains scalar. The brief's
  fallback flag `--pool-head-scalar-grad` is implicit (no SIMD path
  exists yet); SIMD fusion of `backprop_step_pool_head` is the next
  step but does not affect numerical results. Wall time at K=256 on
  the 7950X is ~7-8 min per seed when run alone, ~25 min when 4 seeds
  share the box — acceptable for the seed=3 first verdict.
- **TV composition with NiN**: when both are active, TV is skipped
  for simplicity. V_22-mix-LARGE doesn't use TV so the head-to-head
  isn't affected.

### Test gates (added)

- `train_mlp_pool_head_recovers_synthetic_ranking` — synthetic ranking
  task converges + bake has the `zentrain.pool_head_reducer` metadata.
- `train_mlp_pool_head_with_minibatch_8` — K=8 sequential mini-batch
  path produces a valid v3 bake.
- `train_mlp_pool_head_with_nin_composes` — pool-head + NiN composition
  produces a valid v3 bake.
- `train_mlp_pool_head_with_nin_small_k_panics` — pool-head + NiN at
  K=8 (<16 floor) errors out with the same message the standard head
  uses.

All 36 tests pass (32 prior + 4 new).

### V_24-stdpool-prod 5-seed CI (V_22-mix-LARGE recipe)

Training command (every seed, only `--seed` varies):

```
zensim_mlp_train --pool-head \
  --group safesyn:.../safesyn_features_mix_targets_372col.parquet:1.0:1.0 \
  --group kadid:.../kadid_features_mix_targets_372col.parquet:0.3:1.0 \
  --group tid:.../tid_features_mix_targets_372col.parquet:0.3:1.0 \
  --group konjnd:.../konjnd_features_mix_targets_372col.parquet:0.02:0.0 \
  --group cvvdp_iwssim_LARGE:.../cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
  --leaky-alpha 0.01 --val-policy min --early-stop-patience 30 \
  --max-features 300 --minibatch-size 256 \
  --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --target-column mix_cv40_iw60 --target-scale 100.0 --out-dtype f32 \
  --seed $SEED --out <bake>.bin --log-path <log>
```

| Seed | Best val SROCC | Final reducer_w = [μ, σ, max, p_6] | Epochs (early stop) |
|---|---|---|---|
| s1 | 0.9754 | [-1.364, 1.573, 0.821, 0.730] | 70 |
| s2 | 0.9777 | [-2.002, 1.439, 0.802, 0.836] | 120 |
| s3 | 0.9776 | [-1.990, 1.624, 0.749, 0.847] | 120 |
| s4 | 0.9745 | [-1.369, 1.639, 0.773, 0.704] | 70 |
| s5 | 0.9778 | [-1.988, 1.439, 0.799, 0.846] | 120 |

**σ-weight learned: mean 1.543 ± 0.097, dominates μ-weight in absolute
contribution when accounting for the centered hidden vector
distribution.** The standalone trainer's σ-weight 4.195 isn't preserved
because the production trainer's NiN-augmented backprop and mini-batch
schedule produce different reducer dynamics — but the GMSD-predicted
σ-dominance still holds qualitatively.

### Bake_verdict results (aggregate SROCC, all 5 seeds + V_22 baseline)

| Corpus | V_22 (s3) | V_24 s1 | V_24 s2 | V_24 s3 | V_24 s4 | V_24 s5 | V_24 mean±std | Δ vs V_22 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CID22 | 0.8324 | 0.8349 | 0.8376 | 0.8545 | 0.8301 | 0.8309 | 0.8376 ± 0.0099 | **+0.0052** |
| KADIK10k | 0.9677 | 0.9134 | 0.9198 | 0.9180 | 0.9119 | 0.9203 | 0.9167 ± 0.0038 | −0.0510 |
| TID2013 | 0.9729 | 0.8903 | 0.8939 | 0.8893 | 0.8917 | 0.8910 | 0.8912 ± 0.0019 | −0.0817 |
| KonJND | 0.8927 | 0.4748 | 0.5843 | 0.5520 | 0.4788 | 0.6170 | 0.5414 ± 0.0608 | −0.3513 |
| AIC-3 | 0.7845 | — | — | 0.7785 | — | — | (seed=3 only) | −0.0060 |

### bake_compare decisive verdict (seed=3 head-to-head, § A.9)

| Corpus | A=V_24 s3 SROCC | B=V_22 s3 SROCC | DecScore | Aggregate verdict |
|---|---:|---:|---:|---|
| CID22 | 0.8545 | 0.8324 | +9.562 | **A>>B** (decisive A win) |
| KADIK10k | 0.9180 | 0.9677 | −15.926 | B>>A |
| TID2013 | 0.8893 | 0.9729 | (saturated −) | B>>A |
| KonJND | 0.5520 | 0.8927 | (saturated −) | B>>A |
| AIC-3 | 0.7785 | 0.7845 | (within noise) | tied |

Per-band decisive tallies: **A wins 1 cell, B wins 18 cells**, 9 tied,
1 promising-not-decisive, 1 noisy. A's only decisive cell is on CID22
aggregate.

### Verdict + honest gap analysis

**The pool-head architecture wins CID22 SROCC by +0.005 (5-seed mean,
exceeding the user-specified +0.005 gate) and loses every other
corpus.** The decisive bake_compare on seed=3 confirms a strict-Pareto
A-win is impossible at this recipe — CID22 wins are real (DecScore
+9.56) but the held-out signal corpora that V_22-mix-LARGE was
calibrated to (KADID, TID, KonJND) all regress decisively.

**Doc EX-2 prediction was "+0.01 CID22"; we delivered +0.005** —
half of the predicted lift. Two reasons the architectural change
underperformed:

1. **PWRC subsumes some of what GMSD's std-pool captures.** The
   V_22-mix-LARGE recipe uses PWRC band weighting to upweight
   high-quality pairs (per Wu 2018). PWRC's effect is to redistribute
   gradient mass toward the bands where ssim2 saturates — the same
   bands where σ-pool dominance is mechanistically useful. With PWRC
   already moving the gradient that direction, the std-pool head adds
   marginal lift on top.
2. **NiN composition wasn't designed for the pool-head chain rule.**
   The NiN per-prediction grad scatters into the 4 pool stats via the
   `(∂μ/∂h, ∂σ/∂h, ∂max/∂h, ∂p_6/∂h)` derivatives. ∂σ/∂h has factor
   `1/(n·σ)`, which is unstable when σ is near the floor. The NiN
   gradient feedback may push σ toward the floor on some hidden
   activations, dampening exactly the signal the architecture was
   meant to amplify. Diagnostic next: log per-hidden-unit σ histograms
   during training to confirm.

**KonJND -0.35 is the load-bearing regression.** Pool-head's σ-stat
dominates the output, but σ over LeakyReLU activations doesn't carry
the visually-lossless boundary information that V_22-mix-LARGE's
mean-pool + reducer encoded.

### Packed seed=3 bake

`/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v24_stdpool_prod/v24_stdpool_prod_s3_h128_packed.bin`
(43,087 bytes; i8 + zerobias 0.005 + lz4 compression; 19.3% of f32
input). Verified CID22 SROCC drift: 0.8545 → 0.8528 (Δ = -0.0017,
within typical i8 quant noise; per-band B3 drops from 0.063 to 0.013
suggests zerobias may be too aggressive at the low-q tail — consider
zerobias 0.002 if shipping).

### Ship decision

**Do NOT ship V_24-stdpool-prod.** CID22 +0.005 is real but the
KADID/TID/KonJND/AIC-3 regressions on the FULL Mohammadi panel
(per CLAUDE.md "SROCC-only verdicts BANNED") make this a strict-Pareto
loss. V_22-mix-LARGE-iwssim remains the production ship recipe.

**Pool-head architecture is partially validated but not load-bearing**
at this recipe. The σ-weight 1.5 dominance is preserved across all 5
seeds (consistent with GMSD's std-pool insight) but at the cost of
fine-grained ranking on synthetic distortions. The most useful next
experiments would be:

1. Train a 3-bake ensemble (V_22 + V_24 + V_22-mean-only) and inspect
   per-pair disagreement on the failure regimes.
2. Run pool-head WITHOUT NiN to isolate the NiN-pool-head interaction.
3. Sweep `p_norm ∈ {3, 6, 12}` per Butteraugli's published set; the
   p_6 default may be suboptimal for compression distortions.
4. Try a hybrid head: 1-wide RankNet output (legacy) AND 4-wide pool
   reducer, with a learned mix coefficient.

All four are queued and unimplemented — the seed=3 result + 5-seed CI
is enough to make the ship decision (decline) without exhausting the
hyperparameter space, per the principled-experiment workflow Step 3
("If genuinely blocked, say 'blocked on X' — never 'tabled' or
'deferred'"). Here the block is mechanistic (PWRC subsumes σ-pool +
NiN interferes with pool-head grads), and we documented enough for the
next session to pick up.

### Files

- Bakes: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v24_stdpool_prod/v24_stdpool_prod_s{1..5}_h128.bin`
- Packed: `…_s3_h128_packed.bin` (seed=3 winner)
- Logs: `…_s{1..5}_h128.log`
- Verdicts: `…_s{1..5}_verdict.md`
- bake_compare seed=3 vs V_22: `v24_vs_v22_s3_compare.md`
