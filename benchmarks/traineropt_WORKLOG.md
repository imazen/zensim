# TRAINEROPT worklog (lane, 2026-10-01; times MT)

Goal: cut `zensim_mlp_train` v2-cell time, byte-identical to the v8 trainer (`/var/tmp/fitv2/bin-v3`, zensim cafed5ca).
Baseline profile (coordinator, r0-class cell, 1 rayon thread): 66.8% `adam_pair_fused_inner_v3`, 10.4% `forward_avx2`,
~14.5% `zenstats::panel` (per-epoch dev PLCC fit), 3.7% `StdFeatures::row`.

## Task 1 — fused w1 Adam over kept rows only (dfafec94)
`--keep-features` zeroes dropped inputs' raw columns, so their standardized value is +0.0 and `zero_masked_w1_rows` pins
their layer-1 rows at 0. For such a row the full fused update is the identity, bit for bit: `g = 0` (x = 0 on both sides, L2 term
`sm * 0 = 0`), `m' = (1-b1)*0 + b1*0 = +0`, `v' = +0`, `w' = 0 - (lr*(+0*inv_bc1))/(sqrt(+0*inv_bc2) + eps) = 0 - 0 = +0`, and
`g` is stored +0. Coarse decay and group-L1 are off by default (`rate <= 0` / `lambda <= 0` return early) and both keep
zero rows at zero anyway. Adam is per parameter, so skipping those rows cannot move any other element.
Change: `AdamW1FusedArgs.active_rows: Option<&[(u32,u32)]>` (ascending half-open kept-row runs, `None` = all rows);
`kept_row_ranges(mask, n_rows)` in `mlp_train/mod.rs` builds the runs once per training run from `INPUT_KEEP_MASK`
(`None` when nothing is dropped, so full-width runs take exactly the old path). The off-grid composition fallback ignores the
ranges and updates every row (same fixed point). Test: `adam_simd::tests::fused_w1_active_rows_bit_identical`
(6 steps, with/without L2 and per-row multipliers, w/m/v/g bit-compared after every step against the full walk).

## Task 2 — forward over kept inputs only: NO CHANGE NEEDED
`simd_mlp::forward_avx2` (and the avx512 / scalar twins) already accumulate over the ascending **nonzero-x index set** `nz`
(register-blocked, input-sequential per hidden lane). Dropped inputs have x == 0, so they are already skipped, and the
accumulation order over the surviving inputs is unchanged by construction. The per-call cost left is the `nz` scan over
n_features (a comparison per input, no weight traffic) and the useful FMAs. The 10.4% is real work on kept inputs; skipping
more would require dropping genuine standardized zeros of kept columns, which changes nothing numerically but is not
bit-obvious to gate (it is already what the code does: `s != 0.0`).

## Task 3 — dev evaluation every 17th epoch under EPOCH_RULE == "last" (e1077ddd)
Proof the trajectory does not depend on `--log-every` (`mlp_train/mod.rs`, the `train_mlp_with_tv_*` epoch loop):
* `lr` is a function of the epoch index only (`0.5*initial_lr*(1+cos(pi*(epoch%50)/50))`), set before the evaluation block.
* The evaluation block (`if epoch % log_every == 0 || epoch == n_epochs-1`) reads `w1/b1/w2/b2` immutably:
  `std_features.predict` (pure forward, rayon only splits rows) and `compute_light_panel_subsampled` (deterministic stride
  decimation, no RNG), then `log_line`, the H-TRAJ checkpoint dump (read-only), and best-val bookkeeping.
* The sampler RNG (`rng`) is not touched inside it; the only other use of `log_every` is `stale_epochs += log_every`, which
  matters only when `--early-stop-patience > 0` (the recipe passes 0).
* `--dump-checkpoints-every 119` fires inside the same block at `epoch % 119 == 0`, i.e. at 0 and 119; 17 divides 119 so both
  are evaluated epochs, and the final epoch always evaluates.
Recipe: `v2_lodo_mlp.LOG_EVERY = 17 if EPOCH_RULE == "last" else 1` (guarded: must divide EPOCHS-1); `read_curve` expects
epochs `{0, 17, ..., 119}` (union the final epoch). `best.bin` (best-dev bake) differs under sparser evaluation by design and
is unused under the `last` rule; `last.bin` is the dumped checkpoint.

## Gate driver
`scripts/rev4_featpot/traineropt_gate.py` trains one cell per variant through the exact v2 `train_command`, hashes the final
checkpoint with `harvest_fit_cells.weights_sha` semantics, predicts the held-out table with the same `bake_dial_refit`, and
compares weights hash, prediction hash and dev values at shared epochs. `ZENSIM_MAX_TIER=v3`, `RAYON_NUM_THREADS=1`.

## Gate (all byte-identical to the v8 trainer `/var/tmp/fitv2/bin-v3/zensim_mlp_train`, cafed5ca)
Root `/var/tmp/rev4-featpot/v2c`, `ZENSIM_MAX_TIER=v3`, `EPOCH_RULE = "last"`, 1 rayon thread. Compared: final-epoch weights
hash (`weights_sha`), held-out prediction file hash, dev value at every epoch both runs logged. Variants: base (log-every 1)
vs new (task 1, log-every 1) vs new17 (tasks 1+3, log-every 17).
* 8 cells x 4 epochs (final binary a5f40576...): r0 N/F, oracle_hi N/F, minus_basic N/F, rall N/F; seeds 0/1; held-out kadid,
  tid2013, konfig, aic3, cid22_a25 -- all IDENTICAL (base vs new vs new17).
* 4 cells x 120 epochs (base vs new17; binary = same non-test code): r0 N kadid s0, oracle_hi F aic3 s1, rall N tid2013 s0,
  minus_basic F cid22_a25 s0 -- all IDENTICAL, 8 shared dev epochs each.
* Paired timing cells (full 120 epochs, base/new/new17 concurrently): r0 N tid2013 s1, oracle_hi F konfig s1 -- IDENTICAL
  weights + predictions; screen_main N aic3 s0 -- IDENTICAL weights + dev curve (predict not run: the v8 predictor refuses a
  1853-wide bake, "bake reads features unavailable to the extraction plan", unrelated to this change).

## Speed (single thread, base/new/new17 run concurrently on three pinned cores, full 120 epochs, training wall)
| arm | kept | base s | task 1 s | tasks 1+3 s | speedup |
|---|---|---|---|---|---|
| r0 | 944 | 665.4 | 460.7 | 317.6 | 2.10x |
| oracle_hi (aux) | 945 | 671.6 | 464.5 | 292.2 | 2.30x |
| screen_main (canon 1853) | 1853 | 787.3 | 789.6 | 628.4 | 1.25x |
screen_main drops no input, so task 1 is a no-op there (`kept_row_ranges` returns None); its gain is task 3 only.
Box was busy (load 10-20); concurrent paired runs share the load, absolute seconds are not quiet-box numbers.
