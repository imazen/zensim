# Phase 4 reference: snapshot of Rust mlp_train.rs at deletion (since RESTORED)

> **CURRENT STATUS (2026-05-13)**: The Rust trainer
> `zensim-validate/src/mlp_train.rs` is **LIVE in main** and has been
> since 2026-05-10 commit `ec40ec8` ("tick 41 — restore Rust mlp_train").
> The binary `target/release/zensim_mlp_train` is the canonical
> trainer that produced V0_15 and V0_16 (current ship). See
> `../../CONTEXT-HANDOFF.md` for the V0_16 recipe.
>
> **This file is a historical snapshot**, NOT a description of current
> state. It preserves the source as it existed when the trainer had
> just been deleted (commit `e613224`, 2026-05-07) so the restore-
> and-modernize work at tick 41 had something to start from. **If you
> are reading this and considering "restoring the deleted Rust
> trainer" — stop and run `ls zensim-validate/src/mlp_train.rs`. It
> exists.** Three separate sessions have now hallucinated a
> not-deleted file as deleted by reading the old framing in this doc;
> do not be the fourth.

---

This is the source of `zensim-validate/src/mlp_train.rs` recovered from
git commit `3ffc74a` (the commit before its deletion in PR #29 commit
`e613224`). It produced the V0_5 SSIM2-proxy bake at
`/mnt/v/output/zensim/synthetic-v2/runs/v04_mlp_ssim2_holdout_20260501T045510.bin`
which gives CID22 SROCC 0.8893 — a number that Python `train_v_next_mlp.py`
could not reach even with the literal recipe ingredients (h=32, pure
RankNet, cosine LR, no TV) — best Python yielded CID22 0.8472 (-0.042).
This Python-vs-Rust gap is why the Rust trainer was restored on 2026-05-10.

## Identified ingredient differences

The Python trainer matches the Rust trainer on:
- ✅ Architecture (228 → n_hidden → 1, LeakyReLU, Identity)
- ✅ Loss (pure RankNet, sigmoid CE on signed distance deltas)
- ✅ LR schedule (cosine annealing — added in commit `c583bb81`)
- ✅ Feature standardization (z-score on train split)
- ✅ Mixed-supervision via `--human-csv NAME:PATH:WEIGHT`

The Python trainer DIFFERS on (suspected gap source):
- ❌ **Pair sampling**: Rust does `pairs_per_epoch=50000` per-step
  group-weighted pair-by-pair sampling. Python does batched RankNet
  (all in-batch pairs, weighted by group via per-row `train_weight`).
  This produces structurally different gradient distributions.
- ❌ **Weight init**: Rust uses `std = sqrt(2/(n_features+n_hidden))`
  (Glorot/Xavier normal). Python defaults to PyTorch Kaiming-uniform
  via `torch.nn.Linear`. Different init basin.
- ❌ **Adam variant**: Rust hand-rolls Adam. Python uses AdamW
  (decoupled weight decay). Subtle gradient differences at the
  L2-regularizer step.
- ❌ **Validation aggregation**: Rust `ValidationPolicy::Min` (worst
  per-group). Python uses min-over-datasets via val_min computation,
  but selects best by global val_srocc. Different selection criterion.
- ❌ **Pairs-per-epoch budget**: Rust draws 50,000 pairs/epoch
  regardless of dataset size. Python iterates whole dataset per epoch
  in batches. ~3-5× different total pair-views over training.

## Phase 4 future work (estimated 1 day)

To close the -0.042 CID22 gap, port these to Python:

1. **Group-weighted per-step pair sampling**: replace batched RankNet
   with explicit `(group_idx, pair_a_idx, pair_b_idx)` sampling per
   step. ~30 LOC.

2. **Glorot-normal init**: override PyTorch's Linear.reset_parameters.
   ~5 LOC.

3. **Hand-rolled Adam (or `Adam` not `AdamW`)**: avoid decoupled weight
   decay. ~3 LOC (just swap optimizer constructor).

4. **`ValidationPolicy::Min` for best-checkpoint selection**: select
   epoch by min-over-groups SROCC, not val_srocc. ~10 LOC.

5. Validate end-to-end on the V0_5-faithful recipe (h=32, ep=200,
   pure RankNet, cosine LR, no TV, lr=1e-3, l2_lambda=1e-5) and
   compare CID22 SROCC. Target: ≥ 0.8800 (within 0.01 of V0_5's 0.8893).

## Until Phase 4 lands

The current loop's CHAMPION (`benchmarks/h192x128_ep300_safesyn218k_kt_2026-05-10.bin`)
achieves +0.042 aggregate SROCC vs V0_5 (CID22 -0.009, KADID +0.088,
TID +0.046, smoothness 4.56% beats V0_2 floor). It's the right ship
even without closing the Rust trainer gap.
