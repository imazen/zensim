# V_24-thurstone+konjnd@0.02+LARGE+iwssim — EX-1 implementation

Implements PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM § 4 + 8 EX-1: Thurstone
Case V Gaussian-CDF pairwise NLL in place of the legacy logistic /
Bradley-Terry RankNet loss.

## Mechanics

- Per-pair loss: `L = -log Φ(d · target · (y_b − y_a))` where
  - `target = (mos_a - mos_b).signum()`
  - `d = Φ⁻¹(0.75) ≈ 0.6745` (Thurstone JND constant)
  - `Φ` is the standard normal CDF (Abramowitz-Stegun erf, max
    |err| ≈ 1.5e-7)
- Per-pair gradient: `∂L/∂(pred_diff) = clamp(-φ(u)/Φ(u), -10, 0) ·
  d · target` — gradient clip at -10 keeps Adam stable through the
  random-init transient (Mills ratio `φ/Φ ≈ |u|` for u → -∞
  produces unbounded grads otherwise; this was the dominant
  failure mode of the first 5-seed run, see CHANGELOG).
- Pair sampling: in-group uniform via the existing
  `train_mlp_with_tv` sampler. The doc-recommended pre-built pair
  file (§7 P0) is not required for SGD — same C(n,2) coverage is
  achieved asymptotically by sampling.
- Sensory eps drop: `|score_a − score_b| < ε` (default 5.0 on the
  0..100 score_zensim scale, ≈ 0.05 on raw normalised mix target).

## Implementation notes

- `LossKind::{RankNet, Thurstone}` on `MlpHyperparams`. Default
  RankNet = bit-identical to V_22-mix-LARGE bake bytes.
- CLI: `--loss thurstone --thurstone-d 0.6745 --thurstone-eps 5.0`.
- Composes with the T8.2 parallel-batch path (the chunked
  accumulator dispatches per-pair on `loss_kind`); determinism
  preserved.
- **Incompatible with Norm-in-Norm** (the NiN closed-form gradient
  is hardcoded to sigmoid). Trainer panics with a clear message if
  both are requested.
- **Aux content-class head**: flag plumbed but no-op (panics if >
  0). Per-row class labels are not threaded through `TrainingGroup`
  yet; queued follow-up.

## Test coverage

6 new tests in `mlp_train::tests`:
- `norm_cdf_anchors` — Φ(0)=0.5, Φ(0.6745)=0.75, Φ(1.96)=0.975, tails clamp inside (0, 1).
- `norm_pdf_anchors` — φ(0)=1/√(2π), φ(±1)=0.2420.
- `thurstone_loss_correct_ordering_low` — pred_diff=+5 → loss≈0, grad≈0.
- `thurstone_loss_wrong_ordering_pushes_correctly` — wrong → loss>1, grad pushes right way.
- `thurstone_loss_target_negative_mirrors` — target=-1 swaps the desired direction.
- `thurstone_numerical_gradient_check` — analytical ∂L/∂pred_diff vs centred
  finite-difference (1e-4 tol).
- `thurstone_jnd_at_unit_gap` — loss==-ln(0.75)≈0.288 at 1-JND.
- `thurstone_train_smoke_runs_and_recovers_ranking` — end-to-end
  trains an MLP on synthetic data and confirms SROCC > 0.85.

All 14 pre-existing trainer tests still pass.

## Training recipe (matches V_22-mix-LARGE)

Same 5-group structure as V_22:
| Group | Rows | Target | Train_w | Val_w |
|---|---|---|---|---|
| safesyn | 196,086 | mix_cv40_iw60 | 1.0 | 0.0 |
| kadid | 10,125 | mix_cv40_iw60 | 0.3 | 1.0 |
| tid | 3,000 | mix_cv40_iw60 | 0.3 | 1.0 |
| konjnd | 1,008 | PJND | 0.02 | 1.0 |
| cvvdp_iwssim_large | 73,300 | mix_cv40_iw60 | 0.5 | 0.0 |

Hyperparams: `--loss thurstone --thurstone-d 0.6745 --thurstone-eps
5.0 --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 0.001
--l2 0.00001 --leaky-alpha 0.01 --val-policy min --minibatch-size
256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0
--early-stop-patience 120 --norm-in-norm-weight 0.0`.

(Norm-in-Norm OFF since Thurstone is incompatible with the NiN
batch path. PWRC is on with the same threshold V_22 used; PWRC's
label-weighting term is loss-shape-agnostic.)

## Pair-file plan (deferred)

PSYCHOVISUAL § 7 P0 recommends a pre-built Thurstone pair file at
`/mnt/v/zen/zensim-training/thurstone-pairs/<corpus>_thurstone_pairs.parquet`.
For the EX-1 mechanism this is an optimisation — the in-group
sampler already constructs equivalent pairs at SGD time. Building
the explicit file would be useful for:
1. Reproducibility (same pair set across seeds).
2. Pre-filtering by per-pair `Δscale` to amplify near-threshold
   pairs.
3. KonJND ordinal-anchor pairs derived from PJND thresholds.

Status: NOT BUILT for this experiment. The current results
inherit the safe-synthetic + per-corpus training distribution
already in use.

## Results (filled in as bake_verdict completes)

See `v24_thurstone_5seed_panel.md` in the same directory for the
generated 5-seed CI table.

## Provenance

- Branch: `feat/ex1-thurstone-loss`
- Trainer source change: commit `9d5c0f9` (Thurstone loss +
  sequential path), `434c9c5` (parallel-batch composition),
  `aa6afc4` (gradient clip + extended patience).
- Trainer binary: built locally from this branch.
- Data: `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/`
  (same parquets as V_22-mix-LARGE).
- Bakes: `/mnt/v/output/zensim/ex1_thurstone_2026-05-18/bakes/`
- Logs: `/mnt/v/output/zensim/ex1_thurstone_2026-05-18/logs/`
- Verdicts: `/mnt/v/output/zensim/ex1_thurstone_2026-05-18/verdicts/`
