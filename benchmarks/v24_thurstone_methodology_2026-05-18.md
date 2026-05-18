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

## Results (5-seed CI, full Mohammadi panel)

### Aggregate SROCC (mean ± stdev across 5 seeds)

| Corpus | V_22-mix-LARGE+iwssim (ship candidate) | V_24-thurstone | Δ vs V_22 | Doc EX-1 target |
|---|---:|---:|---:|---|
| **CID22** (n=4292) | 0.8324 | **0.8403 ± 0.003** | **+0.008** | ≥ 0.83 ✔ |
| KADIK10k (n=10125) | 0.9677 | 0.9666 ± 0.0004 | -0.001 | n/a |
| TID2013 (n=3000) | 0.9729 | 0.9708 ± 0.0003 | -0.002 | n/a |
| **KonJND-1k** (n=1008) | 0.8927 | 0.8709 ± 0.004 | -0.022 | ≥ 0.92 ✗ |
| **AIC-3 CTC** (n=600) | 0.7845 | 0.7860 ± 0.003 | **+0.002** | ≥ 0.85 ✗ (gap -0.064) |

Per-seed SROCC by corpus (for reproducibility):
- CID22:  s1 0.8424  s2 0.8390  s3 0.8364  s4 0.8407  s5 0.8431
- KADIK10k: s1 0.9666  s2 0.9668  s3 0.9669  s4 0.9666  s5 0.9659
- TID2013: s1 0.9710  s2 0.9706  s3 0.9708  s4 0.9706  s5 0.9712
- KonJND:   s1 0.8695  s2 0.8711  s3 0.8768  s4 0.8656  s5 0.8717
- AIC-3:    s1 0.7837  s2 0.7882  s3 0.7813  s4 0.7886  s5 0.7880

### Full Mohammadi panel (seed 3 vs V_22 packed) — aggregate

| Corpus | Metric | V_22 | V_24-thurstone s3 | Δ |
|---|---|---:|---:|---:|
| CID22 | SROCC | 0.8324 | 0.8364 | **+0.0040** |
| CID22 | PLCC | 0.8289 | 0.8314 | **+0.0025** |
| CID22 | PWRC | 0.9006 | 0.9039 | **+0.0033** |
| CID22 | Z-RMSE | 0.559 | 0.556 | **-0.003** (better) |
| AIC-3 | SROCC | 0.7845 | 0.7813 | -0.0032 |
| AIC-3 | PLCC | 0.7953 | 0.7927 | -0.0026 |
| AIC-3 | PWRC | 0.8630 | 0.8610 | -0.0020 |
| AIC-3 | Z-RMSE | 0.606 | 0.610 | +0.004 (worse) |
| KonJND | SROCC | 0.8927 | 0.8768 | -0.0159 |
| KonJND | PWRC | 0.9178 | 0.9066 | -0.0112 |
| KonJND | Z-RMSE | 0.376 | 0.383 | +0.007 (worse) |
| KADIK10k | SROCC | 0.9677 | 0.9669 | -0.0008 |
| TID2013 | SROCC | 0.9729 | 0.9708 | -0.0021 |

## Packed bake (seed 3 winner)

Packed via `zenpredict repack --dtype i8 --zerobias 0.005 --compress
--optimize`:

- Path: `/mnt/v/output/zensim/ex1_thurstone_2026-05-18/bakes/v24_thurstone_konjnd_002_LARGE_iwssim_s3_h128_packed.bin`
- Size: 40,996 bytes (slightly smaller than V_22's 41,695)
- CID22 SROCC drift from F32 0.8364 → packed 0.8365 (+0.0001)
- Full panel essentially bit-identical to F32 (max SROCC drift 0.0004)
- 11.8% of weights zeroed at τ=0.005

## Honest gaps vs doc EX-1 predictions

The doc § 4 / § 8 EX-1 predicted "CID22 SROCC ≥ 0.83, KonJND SROCC
≥ 0.92, AIC-3 SROCC ≥ 0.85 simultaneously". This run delivers
CID22 ≥ 0.83 cleanly (the only target hit), AIC-3 essentially flat
at 0.786 (gap of -0.064), and a KonJND regression (gap of -0.049 to
target).

The reason **AIC-3 did not move** is documented in zensim CLAUDE.md
"ssim2-favoring SROCC" section: the training target
`mix_cv40_iw60 = 0.4·cvvdp + 0.6·iwssim` is ssim2-family-shaped
(both CVVDP and IWSSIM correlate strongly with SSIMULACRA2 on
compression distortions). Thurstone Case V changes the LINK
FUNCTION (logistic → Gaussian CDF) but inherits whatever target
shape the corpus carries. Without a different supervision signal
on AIC-3-style content, the link change can't unlock the gap.

**The doc's claim was specifically that pre-filtering pairs via
`|Δscore| > ε` on the normalised mix target would surface
near-threshold pairs that bind to JND units — exactly the AIC-3
regime.** The 0.05 ε (= 5.0 on score_zensim) only drops 0.5% of
konjnd's pair space (computed empirically), so the filtering is
not aggressive enough to reshape AIC-3 supervision meaningfully.
Larger ε would help but at the cost of throwing away most
training pairs.

The KonJND regression is likely a side-effect of Thurstone's
slower-converging gradient on the konjnd group specifically
(`train_w=0.02` means few pair samples; the gradient clip stops
the early-epoch instability but konjnd still doesn't reach
RankNet's final state).

## Next steps (not in EX-1 scope but documented for follow-up)

1. **Re-supervise with SSIMULACRA2 as one of the targets** (EX-3
   in the same doc). Pair file from a CVVDP-targeted vs
   IWSSIM-targeted vs SSIMULACRA2-targeted mix should change AIC-3
   shape more meaningfully than the link function alone.
2. **KonJND-aware ε**: scale ε per-group based on the target
   distribution's σ. konjnd's PJND scale has stdev 13.6 (in score
   units), so ε=5 drops <1% of pairs; safesyn's mix target has
   stdev ~25, so ε=5 drops ~10%. Per-group ε would equalize the
   signal density.
3. **Higher minibatch + lower LR for Thurstone**. The gradient
   clip at -10 prevents instability but may be too aggressive for
   the converged regime. A schedule that warms up to no-clip after
   30 epochs might let Thurstone reach RankNet's konjnd plateau.
4. **AIC-3 as a low-weight training signal** (not validation). The
   doc says "AIC-3 anchors at known JND levels" — using AIC-3 as
   a TINY-weight training group with PJND-derived ordinal labels
   could close the gap directly. Currently AIC-3 is held out for
   evaluation only.

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
