# V_24 EX-DUAL — dual-target multi-task head methodology

**Date**: 2026-05-18
**Branch**: `feat/ex-dual-target-head` (rebased on `ex2-persample-alpha`
which itself rebases on `main` at `kpoptomy 972201dd`).
**Commit lineage**:
- `feat(ex-dual): dual_target_head module — forward + backward + bake + 8 tests pass` (`npklrxvr`)
- `feat(ex-dual): wire dual_target_head into mlp_train + CLI flags` (`xomvmmpo`)

## Hypothesis

Previous V_24 experiments (per-sample-α, KonJND-densify ssim2-per-pair,
PJND-broadcast densify) hit the same wall — **a single scalar MLP
output cannot encode per-pair quality AND per-source PJND
simultaneously**. The KonJND-densify session ended with the
load-bearing finding:

> Single-target scalar regression cannot encode per-pair quality AND
> per-source PJND simultaneously. Densification alone is not the right
> lever — the supervision SIGNAL needs richer structure (genuine
> per-pair PJND labels via KonJND++, or a dual-target MLP head).

KonJND++ is unavailable. This experiment tests the dual-target MLP
head: a final-layer that produces (y_quality, y_pjnd) where y_quality
is the shipping output (RankNet-trained on the 5-group mix) and y_pjnd
is an **auxiliary task** trained on the KonJND-PJND-broadcast group
via MSE, dropped at bake time. The encoder is shared, so y_pjnd
gradient forces PJND-relevant representations into the hidden layer
WITHOUT polluting y_quality.

### Falsification

If no `λ_pjnd ∈ {0.01, 0.05, 0.1, 0.3, 1.0}` produces a bake that is
Pareto-better than the λ=0.0 single-head control AND than the
V_22-mix-LARGE+iwssim ship baseline on the full Mohammadi panel
(SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE) aggregate of
CID22+KADID+TID+KonJND+AIC-3, the dual-target hypothesis is
falsified at this trainer configuration.

### Cost ceiling

5 λ values × 1 seed each at 100 epochs / 10k pairs/epoch (≈ 2.3 min
per run on a 7950X) = 15 min for the sweep. If a candidate wins,
extend to 5 seeds = +15 min. Total compute budget: ≤ 30 min.

### Ship form

If a candidate Pareto-wins, pack via the existing repack tool
(F32→I8 + zerobias + lz4) and propose as PreviewV0_6 (preserving the
production runtime path — bake structure is identical to V_22 since
y_pjnd is training-only).

## Architecture

```text
shared encoder:
  h_pre = W1 · x + b1                              (300 → 128)
  h     = LeakyReLU(h_pre, α=0.01)
output heads (both Identity-activated):
  y_quality = h · w_qual + b_qual                  (128 → 1, INFERENCE)
  y_pjnd    = h · w_pjnd + b_pjnd                  (128 → 1, TRAINING-ONLY)

per-step loss:
  L = ranknet(y_quality on non-PJND pair)
    + λ_pjnd · (y_pjnd − target_pjnd)²  (when PJND sample present)
```

### Bake structure (ZNPR v3, inference-compatible)

Standard two-layer single-output v3 bake:
- Layer 1: 300 → 128, LeakyReLU, F32, weights = encoder W1, b1.
- Layer 2: 128 → 1, Identity, F32, weights = w_qual, b_qual.

**y_pjnd weights are NOT in the inference path** — they're stored
as optional `zentrain.dual_target_pjnd_head` numeric metadata for
offline analysis (did the PJND head learn anything useful?). Runtime
ignores this entry.

Metadata flags:
- `zentrain.dual_target_head` = `"true"` (utf8 provenance flag)
- `zentrain.dual_target_pjnd_loss_weight` = λ_pjnd as 1×f32
- `zentrain.dual_target_pjnd_head` = `[w_pjnd[128]] [b_pjnd]` as f32 LE

### Backprop verification

`zensim-train-core/src/dual_target_head.rs` ships 8 unit tests
including 3 finite-difference backprop checks:
- `backprop_finite_diff_quality_path`: λ_pjnd=0 → exact match with
  numeric grad on w1, w_qual, b_qual paths; g_w_pjnd[*] = 0 (no
  PJND signal).
- `backprop_finite_diff_pjnd_path`: only PJND target → exact match
  on w1, w_pjnd, b_pjnd paths; g_w_qual[*] = 0 (no rank signal).
- `backprop_finite_diff_combined_paths`: both targets active →
  encoder gradient correctly sums quality + pjnd contributions.

All three pass with `|grad_analytic − grad_numeric| < 1e-4`.

## Trainer scope (deliberate minimal-baseline)

This experiment uses the **simplest training recipe** that exercises
the dual-target hypothesis cleanly:

| Knob | EX-DUAL trainer | V_22-LARGE+iwssim ship |
|---|---|---|
| Hidden | 128 | 128 |
| Epochs | 100 | 300 |
| Pairs/epoch | 10 000 | 50 000 |
| Mini-batch K | 1 (sequential) | 256 |
| NiN | OFF | ON (w=0.1, p=1, q=2) |
| TV | OFF | OFF |
| PWRC weights | OFF | ON |
| Low/Mid/Hi-Q boosts | OFF | OFF |
| LR schedule | Cosine, 50-epoch period | Cosine, 50-epoch period |
| Validation policy | Min | Min |

**Why minimal**: V_22-LARGE+iwssim composes NiN + K=256 + PWRC for a
multiplicative ~10-15× compute cost vs the minimal-baseline. Layering
dual-target on top of NiN + K=256 + PWRC is the integration step IF
dual-target proves to help at minimal. Testing at minimal first
isolates "does the auxiliary signal help" from "does it compose with
V_22's other knobs." A win at minimal earns the right to integrate;
a loss at minimal falsifies the mechanism regardless of composition.

## Data

| Group | Path | Rows | Train w | Val w | PJND target? |
|---|---|---:|---:|---:|---:|
| safesyn | `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/safesyn_mix_300col.parquet` | 196 086 | 1.0 | 0.0 | no |
| kadid | `…/kadid_mix_300col.parquet` | 10 125 | 0.3 | 1.0 | no |
| tid | `…/tid_mix_300col.parquet` | 3 000 | 0.3 | 1.0 | no |
| konjnd | `…/konjnd_mix_300col.parquet` | 1 008 | 0.02 | 1.0 | no |
| cvvdp_iwssim_large | `…/cvvdp_iwssim_large_300col_v2.parquet` | 73 300 | 0.5 | 0.0 | no |
| konjnd_pjnd | `/mnt/v/zen/zensim-training/2026-05-18-konjnd-dense/konjnd_dense_pjndtarget_300col.parquet` | 20 160 | 0.0 | 0.0 | **yes** |

The PJND group is referenced as a training group with `train_w=0.0`
(so it doesn't participate in y_quality sampling) but its `human_score`
column carries the per-source PJND broadcast value (range 22..70) that
gets read by the PJND-MSE auxiliary head.

`target-column` for non-PJND groups: `mix_cv40_iw60` (= 0.4·cvvdp_log_norm
+ 0.6·iwssim_log_norm), matching the V_22-mix-LARGE+iwssim recipe.

CID22 is **VALIDATION-ONLY** per the project sanctity rule — never
trained on.

## Sweep

`scripts/v_next/run_ex_dual_sweep.sh /mnt/v/output/zensim/ex-dual-2026-05-18`

Single seed=3, λ ∈ {0.0, 0.01, 0.05, 0.1, 0.3, 1.0}. λ=0.0 is the
single-head control (architecture identical but auxiliary loss
disabled), used to verify that any λ>0 improvement is attributable
to the PJND signal and not to the extra parameter count.

## Validation

Each bake scored via `bake_verdict` on cid22+kadid+tid+konjnd+aic3
using pre-extracted 372-feature parquet sidecars (only the first 300
features are used by the dual_target bake; the extras are ignored).

For Pareto-comparison vs V_22-mix-LARGE+iwssim, `bake_compare` is run
pairwise with bootstrap n=500 and the §A.9 decisive rule
(n_band≥30 ∧ |h_SROCC|>1.96 ∧ |h_Z-RMSE|>1.96 ∧ PWRC_A>PWRC_B ∧
≥4/6 panel stats agree).

## Results

(filled in after sweep + validation complete — see
`benchmarks/v24_dual_target_RESULTS_2026-05-18.md` for the final
verdict table)

## Lineage notes

This experiment builds on per_sample_alpha_head's infrastructure
(CLI flags, `Predictor::predict` runtime path, bake_verdict +
bake_compare). The dual_target_head module is a parallel module
under `zensim-train-core/src/` matching the file structure of
`pool_head.rs` / `hybrid_head.rs` / `per_sample_alpha_head.rs`.

The trainer dispatcher in `zensim-validate/src/mlp_train.rs` adds
`dual_target_head` to the head_flags mutual-exclusion set
(`pool_head | hybrid_head | per_sample_alpha_head | dual_target_head`
≤ 1).

The bake produces a single-output ZNPR v3 wire that loads through
the production `apply_mlp_scoring` path unchanged — the
`zentrain.dual_target_head=true` metadata is descriptive only.
