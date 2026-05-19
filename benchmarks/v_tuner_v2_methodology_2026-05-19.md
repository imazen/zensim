# V_tuner-v2 methodology — cross-codec JND anchor loss (2026-05-19)

## Hypothesis

User directive: *"don't we want jnd anchoring at a certain number cross codec"*

Today's Tuner (PreviewV0_5Tuner, V_tuner-v2-s2 calibrated) trains MSE on
`mix_cv40_iw60` from safesyn — a CID22-paper-consistent target. This is
internally consistent within a codec curve but doesn't ENFORCE that PJND
lands at the same score across codecs. The 2026-05-19 cross-codec
consistency eval measured **mean pairwise butter_max at T=63 = 6.68**
across JPEG/WebP/AVIF (above the 4.0 "broken" threshold).

V_tuner-v2 adds an explicit cross-codec JND anchor loss: every K
training pair-steps, sample one anchor row (features extracted from
a real or synthetic at-PJND distorted pair), forward through the
per-sample-α head, and add `w · (y - 63)²` MSE. This forces score=63
to mean PJND across all codecs represented in the anchor pool.

## Hypothesis → falsification gates

- **Cross-codec butter at T=63 < 3.0** (vs baseline 6.68) — *primary*.
- **Cross-codec butter at T=70 < 2.5** (vs baseline 5.00).
- **CID22 SROCC ≥ 0.85** (preserves Tuner's compression-trail value).
- **KonJND SROCC ≥ 0.80** (anchor data injects PJND signal → should
  lift KonJND from Tuner's catastrophic 0.235).
- **strict_mono ≥ 0.92** (matches Tuner's 92.78 %).

3-of-5 panel agreement is the ship rule per CLAUDE.md "Multi-stat
agreement" gate.

## Anchor dataset

`/mnt/v/zen/zensim-training/2026-05-19-jnd-anchors/anchors_372col.parquet`

9373 rows × 378 cols total:

| Anchor source | n | anchor_weight | Provenance |
|---|---:|---:|---|
| `konjnd_jpeg` | 504 | 1.5 | KonJND-1k SRC0001..SRC0504, JPEG, mean PJND q (real human data) |
| `konjnd_bpg` | 504 | 1.5 | KonJND-1k SRC0505..SRC1008, BPG, mean PJND q (real human data) |
| `safesyn_synth` | 8365 | 1.0 | safesyn pairs with ssim2_gpu ∈ [61, 65], stratified-sampled |

All rows are pre-extracted 372-feature vectors (from the canonical
`val/konjnd.parquet` for KonJND, and `train/safesyn.parquet` for
synthetic). Builder script:
`scripts/v_next/build_jnd_anchors.py` (commit on this branch).

Each row regresses to the global anchor target `--anchor-target-score
63.0` (the CID22 paper Table 4 PJND calibration point on ssim2 scale).
Per-row `anchor_weight` multiplies the MSE term; KonJND rows pull
1.5× harder than synthetic ssim2-derived rows because real human PJND
is the gold standard.

## Trainer additions

`zensim-validate/src/mlp_train.rs` and `bin/zensim_mlp_train.rs`:

```rust
// Public:
pub struct AnchorRows<'a> { name, features, row_weights }
pub fn train_mlp_with_tv_anchored(groups, n_features, hp, log, tv, anchor)

// MlpHyperparams (new fields):
anchor_loss_weight: f64,   // default 0.0 (off)
anchor_target_score: f64,  // default 63.0
anchor_step_p: f64,        // default 0.10
```

Anchor stepping is wired ONLY on the per-sample-α head with
`minibatch-size 1` (NiN incompatible — asserted). Each anchor step:

1. Sample one anchor row by `row_weight`-proportional CDF.
2. Forward through the per-sample-α head (same standardization as
   training features — re-uses the training-group scaler).
3. Compute `L = anchor_loss_weight · row_w · (y - target)²`.
4. Backprop `dL/dy = 2 · anchor_loss_weight · row_w · (y - target)`
   through `psah::backprop_step_per_sample_alpha_head`.
5. Apply Adam step independently of the pair-step gradient (K=1 so
   each step is its own Adam batch).

## Recipe (`scripts/v_next/run_tuner_v2_seed.sh`)

Identical to today's Tuner (`run_tuner_seed_v2.sh`) plus the anchor flags:

```bash
"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 1 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --ranknet-weight 0.0 --mse-weight 1.0 \
    --monotonicity-reg 0.5 --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 1.0 \
    --anchor-target-score 63.0 \
    --anchor-step-p 0.10 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}"
```

`anchor_step_p=0.10` means roughly 5000 anchor steps per epoch (10%
of 50k pair-steps). Total training compute ~10% higher than today's
Tuner per epoch but identical wall time on the 7950X.

## Data lineage

| Path | sha256 (prefix) | Rows | Status |
|---|---|--:|---|
| `canonical-2026-05-18/train/safesyn.parquet` | `1ee0565fb6cb` | 196,086 | CID22-leak-purged (per CLAUDE.md canonical) |
| `canonical-2026-05-18/val/konjnd.parquet` | `3e999a372577` | 1,008 | KonJND-1k 504 JPEG + 504 BPG anchors |
| `2026-05-19-jnd-anchors/anchors_372col.parquet` | computed at run | 9,373 | this experiment's anchor pool |

NO CID22 human MOS or KADID-train cross-overlap involvement —
training continues on safesyn + KonJND-PJND-anchors only. KADID/TID/
CID22/AIC-3 remain validation-only per CLAUDE.md.

## Results

(See `aggregate_3seed.md` and `ship_decision.json`.)
