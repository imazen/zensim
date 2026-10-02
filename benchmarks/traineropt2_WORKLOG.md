# TRAINEROPT2 worklog (lane, 2026-10-02)

Base: main@origin 34c41507 (zensim-validate source identical to 538d3549, the TRAINEROPT tree; reference trainer
`/var/tmp/fitv2/bin-v5/zensim_mlp_train`, sha a5f40576). Workspace `zensim--traineropt2`, bookmark `quarantine/claude/traineropt2`.

## Changes
1. **Group-lasso prox fused into the fused w1 Adam pass** (`adam_simd.rs`: `AdamW1FusedArgs.group_l1_tau`, `group_l1_row`,
   `group_l1_rows`; `mlp_train/mod.rs`: `group_l1_step_tau`, `step_w1_fused(.., group_l1_tau)`; `apply_group_l1` now calls
   `group_l1_row`, the single owner of the arithmetic). The kernel proxes each 8-row block right after that block's Adam
   update (rows still in L1). Rows are independent; per-row arithmetic is the unchanged sequential `iter().map(w*w).sum()`,
   `norm <= tau` zeroing and `1 - tau/norm` scale. The 8 rows of a block run their sum-of-squares chains side by side
   (independent accumulators, each in element order): the old pass was bound by one 4-cycle-latency chain per row, not memory.
   Coarse decay: when `coarse_decay_rate() > 0` the call site keeps the separate `apply_post_adam_penalties` pass (decay must
   precede the prox over the whole array); fusing is done only when coarse decay is off. `active_rows`: skipped rows are
   exactly zero and the prox maps zero to zero. The non-v3 composition fallback applies the prox to every row after
   `adam_update`.
2. **`g_zero_in`** (skip the `g` loads and the `g = 0` stores in the v3/v4 kernel). Writers of `AdamState.gw1`: `AdamState::new`
   (zeros); the unfused `backprop_step`+`add_l2_grad_layer1` (only when `fuse_w1` is false); the TV regularizer's
   `backprop_step` (gw1 written, then `adam.step` always runs in the same iteration at K=1, which stores `g = +0.0`);
   K>1 / parallel / NiN accumulators (never combine with `fuse_w1`, which requires `k == 1`). So at entry to
   `step_w1_fused`, `gw1` is all `+0.0` under every recipe, including the v2 recipe. Proof artifact: the kernel's
   per-row `debug_assert!` (bit-pattern zero) plus test `fused_w1_group_l1_and_g_zero_bit_identical` (g compared at every step).
3. Gate driver: `traineropt_gate.py` now passes the spec's recipe tokens (`H<n>`, `gl<λ>`) and a `--group-l1` override
   (`recipe_of` caps the spec token at λ ≤ 1; the brief's λ = 3 is passed as `--group-l1 3`).
4. Bench `benches/gl_step_timing.rs`: interleaved OLD/NEW layer-1 step timing (not a gate).

## Gate
(see TRAINEROPT2_DONE.md for the final table; raw records `/var/tmp/traineropt2/{g1,g120}/*/gate.json`)
(The final-source gate below ran on `/var/tmp/traineropt2/binFinal/zensim_mlp_train`, sha256
f9d076c6d68374404a28370d1447f9697958b3589244a673a55ae9d19ecc8451, rustc 1.98.1.)

Reference: `/var/tmp/fitv2/bin-v5/zensim_mlp_train` (a5f40576), `ZENSIM_MAX_TIER=v3`, 1 rayon thread, v2 recipe via
`v2_lodo_mlp.train_command`. Compared: final-epoch weights hash (`weights_sha`), held-out prediction file hash
(`bake_dial_refit predict`), dev values at every epoch both logged. 12 cells, all IDENTICAL:

| spec | head | held-out | seed | H | group-L1 | kept | epochs | weights sha12 | result |
|---|---|---|---|---|---|---|---|---|---|
| r0 | N | konfig | 1 | 128 | off | 944 | 4 | 6734e3d39fac | IDENTICAL |
| r0 | F | tid2013 | 0 | 128 | 3 | 944 | 4 | a690d63c631f | IDENTICAL |
| r0 | F | aic3 | 0 | 64 | off | 944 | 4 | 59cc7eeaa664 | IDENTICAL |
| r0 | N | kadid | 1 | 64 | 3 | 944 | 4 | 62ce301e362b | IDENTICAL |
| screen_main | N | tid2013 | 1 | 128 | off | 1853 | 4 | ac08cd6c4d6f | IDENTICAL |
| screen_main | F | aic3 | 1 | 128 | 0.001 | 1853 | 4 | d98aa13b513a | IDENTICAL |
| screen_main | N | kadid | 0 | 128 | 3 | 1853 | 4 | 656506b77603 | IDENTICAL |
| screen_main | F | cid22_a25 | 0 | 64 | 0.001 | 1853 | 4 | e3e069258d88 | IDENTICAL |
| r0 | F | aic3 | 1 | 128 | 0.001 | 944 | **120** (log-every 17) | f62ec1771200 | IDENTICAL |
| r0 | F | cid22_a25 | 0 | 64 | 3 | 944 | **120** | fd134d3e244c | IDENTICAL |
| screen_main | N | kadid | 0 | 128 | 3 | 1853 | **120** | 4572c606a37a | IDENTICAL |
| screen_main | N | tid2013 | 1 | 64 | off | 1853 | **120** | 9cb775f6ac54 | IDENTICAL |

(Earlier builds of the same change, before the inline(never)/conditional-block tuning below, also passed 8 + 3 + 4 cells:
`/var/tmp/traineropt2/{g1,gF,g120}`.) Unit: `fused_w1_group_l1_and_g_zero_bit_identical` (v3, fallback, with/without
`active_rows`, L2 on/off, per-row mult, both prox branches hit, prox must change w). `cargo test -p zensim-validate --release`
all pass (273 lib tests); mlp_train tests also pass in a debug profile (89), i.e. with the `g_zero_in` debug assertion live.

## Timing (single thread, base and new run CONCURRENTLY on two pinned cores, cores swapped each rep, same host)
Method: per-epoch seconds = (wall(6 epochs) − wall(2 epochs)) / 4 with `--log-every 100` (no eval), 3 reps, cores swapped
each rep, base = this toolchain's build of 538d3549 (`/var/tmp/traineropt2/binB`), new = binFinal. Box load 1.7–5.8 (other
lanes' processes on the dense fleet host; SMT siblings of the pinned cores were not idle), so read ratios, not absolutes.
`epoch_time.py` is the driver (`/var/tmp/traineropt2/epoch_time.py`).

| arm | base s/epoch (3 reps) | new s/epoch | speedup |
|---|---|---|---|
| H128 screen_main + group-L1 3 | 15.1 / 15.7 / 16.9 | 13.1 / 12.9 / 13.0 | **1.22x** |
| H128 screen_main (no group-L1) | 12.3 / 11.1 / 12.2 | 12.1 / 10.7 / 12.0 | 1.02x |
| H128 r0 (944 kept) | 5.49 / 5.42 / 5.48 | 5.33 / 5.36 / 5.55 | 1.00x |
| H64 screen_main | 6.32 / 5.44 / 5.68 | 5.38 / 5.57 / 5.40 | ~1.0x (rep 1 base outlier) |

Isolated layer-1 step (`benches/gl_step_timing.rs`, 1853 inputs, interleaved, quiet cores): H128 + group-L1 1.50x
(0.245 → 0.165 ms), H128 plain 1.03–1.10x, H64 + group-L1 1.27x, H64 plain 1.00x.

## What did not work / lessons
* The first build had `group_l1_rows` `#[inline(always)]` and the 8-row block loop always on. Its whole-cell wall time showed
  no gain on group-lasso cells (224 s vs 225 s for 12 epochs) although the isolated step bench said 1.5x. Making
  `group_l1_rows` `#[inline(never)]` (keeps the Adam loop body small) and using blocks only when a prox is requested fixed it
  (1.22x). Cause not profiled (`perf` is blocked by `perf_event_paranoid` for this user).
* An early epoch-time driver measured the second process's time as max(both), which made the faster arm look equal. Fixed
  (threads record each end time) before the numbers above.
* The brief's expected prox saving (38% of time) is mostly not realized: the fused kernel still pays the sequential norm
  chains; with 8 interleaved chains it costs about 0.035 ms/step at H128 vs 0.115 ms before.

## Task 3
`forward_avx2` reads w1 once per sample (hidden-axis blocks over the ascending nonzero-x set), so a pair reads it twice. A
shared-pass pair forward would need per-side nonzero filtering inside one union walk; not attempted (see DONE report).
