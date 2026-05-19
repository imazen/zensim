# V_tuner-v3 methodology — range floor + rank-preserve (2026-05-19)

## Hypothesis

EXP-CROSS-CODEC-V2 (`benchmarks/v_tuner_v2_cross_codec_2026-05-19_falsification.md`)
established that the V2 `cc4v2_s1_w2_0` candidate hit T=63 cross-codec
butter_pnorm3 = 0.536 / butter_max = 1.152 — well under the 2.5 strict
gate — but FAILED the Tuner trail's dynamic-range floor by 500× (range
0.10 score units vs the ≥ 50 floor). The mechanism: the
`--cross-codec-eq-weight 2.0` gradient drove the network into a
constant-output basin (~63 for every input) that simultaneously
minimizes the cross-codec-eq MSE AND the JND anchor MSE.

V3 hypothesis: the V2 mechanism (cross-codec-eq + anchor losses) is
ARCHITECTURALLY incompatible with high cross-codec consistency on a
score-spreading network — the equivalence-MSE term has a stable
attractor at "predict the same value for both sides," which is exactly
the collapse failure mode. Adding **two architectural counterweights**
should rescue the recipe:

1. **Cross-codec rank-preserve regularizer.** When the equiv pair
   carries `butter_diff = butter_a − butter_b`, add a RankNet-style
   sigmoid loss weighted by `|butter_diff|` that pushes `(y_a, y_b)`
   ordering to match the pivot-metric quality ordering. Collapse to
   `y_a = y_b` violates rank for every pair with `|butter_diff| > 0`,
   so the rank-preserve gradient grows with collapse severity.
2. **Dynamic-range floor probe.** With probability `step_p` (default
   5 %), sample N rows from the equiv pool, forward each, compute σ
   across the outputs, penalize the model when σ < threshold:
   `L = w · max(0, σ_threshold − σ_obs)²`. Directly forbids the
   constant-output failure mode at the σ level.

Plus a stronger monotonicity regularizer (`--monotonicity-reg 5.0`
vs V2's 0.5) to give the per-curve mono hinge more bandwidth to fight
collapse.

## Hypothesis → falsification gates

V3 ships as `PreviewV0_5TunerV2` if at least one candidate passes ALL
of:

- **Strict monotonicity ≥ 0.9378** (1 pp over current Tuner's 0.9278
  on the 50-image × 19-q JPEG sweep).
- **Tied rate ≤ 5 %.**
- **Dynamic range ≥ 50 score units** between q=5 median and q=95
  median across the sweep (a user-facing dial must span useful
  quality).
- **T=63 cross-codec butter_max < 2.5 OR butter_p3 < 2.5** on the
  20-image n=20 binary-search test.

If multiple candidates pass: pick the one with best (strict mono +
inverse butter_p3) joint score.

## V3 trainer additions

`zensim-validate/src/mlp_train.rs` and `bin/zensim_mlp_train.rs` —
commit `de097f1c`:

```rust
// EquivPairs (existing struct) gains:
pub butter_diff: &'a [f64],   // butter_a − butter_b per pair (empty = off)

// MlpHyperparams (new fields):
cross_codec_rank_preserve_weight: f64,  // default 0.0
dynamic_range_floor_weight: f64,        // default 0.0
dynamic_range_sigma_threshold: f64,     // default 15.0
dynamic_range_step_p: f64,              // default 0.05
dynamic_range_probe_n: usize,           // default 40
```

CLI flags: `--cross-codec-rank-preserve-weight`,
`--dynamic-range-floor-weight`, `--dynamic-range-sigma-threshold`,
`--dynamic-range-step-p`, `--dynamic-range-probe-n`.

### Rank-preserve loss math

For each equiv step that fires (probability `--cross-codec-eq-step-p`),
the trainer samples one pair `(features_a, features_b)`, forwards both
through the per-sample-α head to get `(y_a, y_b)`, computes the
equivalence MSE as before, AND (if `--cross-codec-rank-preserve-weight
> 0` AND `butter_diff[ei] != 0`):

```
s    = sign(butter_diff[ei])        // +1 if A is quality-WORSE than B
w_rp = cross_codec_rank_preserve_weight · |butter_diff[ei]|
u    = s · (y_b − y_a)              // logit: positive iff rank correct
L_rp = w_rp · softplus(−u)          // = w_rp · −log(sigmoid(u))
dL_rp/dy_b = −w_rp · s · (1 − sigmoid(u))
dL_rp/dy_a = +w_rp · s · (1 − sigmoid(u))
```

Sign convention: the trainer's target is `mix_cv40_iw60` (HIGHER =
better quality, 0..100 scale). LOWER butter = HIGHER quality. So
`butter_a > butter_b` means A is quality-WORSE → we want `y_a < y_b`
→ `(y_b − y_a) > 0` → softplus loss is small. Backprop adds
`(dL_rp/dy_a, dL_rp/dy_b)` to the existing equiv MSE gradients
`(dL_eq/dy_a, dL_eq/dy_b)` before the single Adam step.

### Dynamic-range floor probe math

With probability `--dynamic-range-step-p` per pair-step (independent
of equiv-step probability), sample `probe_n` random indices from the
equiv-pool A-side, forward each:

```
y_i = forward(std_equiv_a[idx_i])   for i in 0..probe_n
μ   = mean(y_i)
σ   = sqrt(mean((y_i − μ)²))
viol = max(0, sigma_threshold − σ)
L_dr = w · viol²
dL_dr/dy_i = −2 · w · viol · (y_i − μ) / (σ · N)
```

When `viol > 0`, each per-row gradient pushes `y_i` away from `μ` —
shrinking outputs that are below mean further down and pushing those
above further up. Accumulated into a single Adam step (one probe →
one Adam step). When `viol = 0` (σ ≥ threshold), penalty + gradient
are both 0, so the probe is cheap (just a forward pass).

Using equiv-pool A-side as the probe substrate avoids loading a
separate q-sweep parquet — the equiv pool already spans the full
q range across codecs and refs, which is exactly the σ-substrate
the probe needs.

## Recipe (`scripts/v_next/run_cross_codec_v3_seed.sh`)

```bash
"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 1 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --ranknet-weight 0.0 \
    --mse-weight 1.0 \
    --monotonicity-reg 5.0 \
    --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 1.0 \
    --anchor-target-score 63.0 \
    --anchor-step-p 0.10 \
    --cross-codec-eq-parquet "${EQUIV}" \
    --cross-codec-eq-weight "${WEIGHT}" \
    --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.2 \
    --dynamic-range-sigma-threshold 15.0 \
    --dynamic-range-step-p 0.05 \
    --dynamic-range-probe-n 40 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}"
```

Sweep: `W ∈ {0.5, 1.0}` × `seeds ∈ {1, 2, 3}` = 6 bakes.

`--monotonicity-reg 5.0` is 10× the V2 value (0.5) and 5× the
original Tuner (1.0). The mono hinge fires on pairs from the same
ref_basename with target ordering, so a higher reg gives the trainer
stronger "preserve per-image curve shape" signal — the
counter-mechanism to anchor + cross-codec-eq collapse.

`--dynamic-range-step-p 0.05` is 5 % of pair-steps. At 50k pairs/epoch
that's 2500 probes/epoch. Combined with `--dynamic-range-probe-n 40`,
each probe costs ~40 forward passes — adds roughly 10 % to per-epoch
compute.

`--cross-codec-rank-preserve-weight 0.2` is held fixed across the
sweep (V3 sweeps `W` for the underlying equiv MSE, not the rank-preserve
weight). The 0.2 default was chosen so that for a typical
`|butter_diff| = 1.0`, the rank-preserve term has gradient magnitude
~0.2 — comparable to the equiv MSE gradient `2·w·diff` for a typical
`diff = 1` score unit at `W = 0.5..1.0`.

## Data lineage

| Path | Rows | Status |
|---|--:|---|
| `canonical-2026-05-18/train/safesyn.parquet` | 196,086 | CID22-leak-purged |
| `2026-05-19-jnd-anchors/anchors_372col.parquet` | 9,373 | KonJND+safesyn PJND anchors |
| `picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet` | 68,788 | 4-codec equiv pool (gap ≤ 0.3 butter) |

NO CID22 human MOS in training (validation-only per CLAUDE.md).

## Results

(See `benchmarks/v_tuner_v3_eval_2026-05-19.md` for per-bake mono /
tied / range / cross-codec table and ship decision.)
