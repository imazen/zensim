# V42 — konjnd-aggregation head wired for 2-layer (G5 lever) + weight sweep

**Date:** 2026-05-26
**Goal:** unblock CODEC_TARGET_GOALS.md **G5 (KonJND HF rank ≥ 0.70)**
by wiring the purpose-built konjnd-aggregation head for the 2-layer
architecture (shipped V39 is 2-layer; the head was previously gated
off for 2-layer/skip with a panic).

## Wiring change

`zensim-validate/src/mlp_train/mod.rs`:
- Removed the 2-layer/skip guard panic on `konjnd_aggregation_weight`
  (commit a923383).
- Rewrote the aggregation step (the `if konjnd_agg_active &&
  steps_since_adam == 0` block) to dispatch through
  `arch_forward`/`arch_backward` (the architecture-aware path, matching
  the anchor step) instead of the 1-layer-only
  `psah::forward_per_sample_alpha_head` /
  `backprop_step_per_sample_alpha_head`. The two-pass aggregation
  structure is preserved: forward all K×S rows caching
  `(ri_global, ya, dya_dpre, ArchForward)`; compute each ref's mean
  aggregate + residual; backprop each cached row with
  `dl_dy = (2w/S)·err·dya_dpre` via `arch_backward`.
- Fixed a latent `do_adam_step` slot-dim bug (`n_hidden` →
  `n_hidden_final`) that would mis-unpack head weights in 2-layer mode.

## Tests (all pass)

- `konjnd_aggregation_step_runs_and_backprops` — 1-layer (legacy path
  through arch_forward/backward, use_2layer=false).
- `konjnd_aggregation_step_runs_and_backprops_2layer` — 2-layer; trains
  NaN-free, non-empty bake, aggregation MSE inside the loose ceiling.
- `konjnd_aggregation_2layer_w1_gradient_matches_finite_difference` —
  FD-checks the analytic `w1` gradient from `arch_backward` against a
  centered finite difference of the per-ref aggregation loss
  `L = w·(agg − t)²` in 2-layer mode; rel < 1e-3 on 6 spread `w1`
  entries.

## G5 weight sweep (V39 recipe + `--n-hidden-layers 2`)

Held-out eval via `bake_verdict` (features-root
`2026-05-15-full-features`, the same root V39's baseline was measured
on). Aggregation parquet `konjnd-dense.parquet` (1008 refs, S=5, K=8),
`--konjnd-aggregation-step-p 0.15`. Each run early-stopped (patience 50).

Bakes: `/mnt/v/output/zensim/bakes/v42_konjnd_agg_{w03,w01,w005,w002}_2026-05-26.bin`
(+ `.verdict.md` sidecars). Driver:
`scripts/v_next/run_g5_konjnd_agg_2layer_2026-05-26.sh`.

| agg weight | KonJND (G5) | CID22 | KADIK | TID | AIC-3 | AIC-4 | weighted goal |
|---|--:|--:|--:|--:|--:|--:|--:|
| **V39 baseline** | 0.420 | 0.879 | 0.925 | 0.932 | 0.802 | 0.905 | — |
| 0.02 | 0.574 | **0.874** | 0.866 | 0.869 | 0.789 | 0.908 | 0.366 |
| 0.05 | **0.722** | 0.351 | 0.821 | 0.827 | 0.209 | 0.525 | 0.013 |
| 0.10 | **0.855** | 0.254 | 0.714 | 0.697 | 0.139 | 0.694 | 0.088 |
| 0.30 | **0.857** | 0.035 | 0.261 | 0.324 | 0.271 | 0.475 | 0.088 |

(All values are aggregate SROCC. G1 dial is collapsed on every v42 bake
— these are rank-only bakes without the post-training PCHIP output
spline that gives V39 its G1=1.00; that is the absence of the spline
step, NOT a regression from the aggregation head. The CID22↔KonJND
tradeoff is the load-bearing signal.)

## Findings

1. **The mechanism works and is correctly wired.** KonJND SROCC moves
   monotonically with aggregation weight: 0.420 (V39) → 0.574 → 0.722 →
   0.855 → 0.857. The head clears the **0.70 G5 floor at w ≥ 0.05**
   (KonJND 0.722–0.857). The 2-layer gradient is verified numerically
   (FD test) and the live V39 recipe trains NaN-free with the head
   enabled on the 2-layer production arch.

2. **There is a sharp tradeoff and no single weight clears G5 while
   preserving the rest.** Between w=0.02 (CID22 0.874 ≈ V39, KonJND
   below floor) and w=0.05 (KonJND clears floor, CID22 collapses to
   0.351), the other corpora collapse catastrophically. The aggregation
   gradient destabilizes the shared encoder once it is strong enough to
   move KonJND. At w≥0.05, CID22/KADIK/TID/AIC-3 all crater.

3. **This is a recipe-tuning problem, not a wiring problem.** The lever
   is now usable on the production architecture (the task's blocking
   goal). Closing G5 without wrecking CID22 is open research — likely
   directions: a separate aggregation-only head that doesn't share the
   encoder gradient, lower learning rate on the aggregation step,
   gradient clipping/decoupling on the aggregation pass, or warm-up
   (train rank to convergence first, then anneal in a small aggregation
   weight). Those are follow-ons; the wiring + numeric verification +
   frontier mapping are complete.

## Verdict on the task

Wiring + tests: **done and verified.** G5 lever: **functional and
proven to move KonJND past 0.70.** A clean ship (G5 ≥ 0.70 AND CID22 ≈
V39) was **not** achieved at any of the 4 weights tried — the
aggregation head trades CID22 rank for KonJND under this recipe.
Committed in the worktree; **not pushed to main** (no weight beat V39
across the board, per the ship gate).
