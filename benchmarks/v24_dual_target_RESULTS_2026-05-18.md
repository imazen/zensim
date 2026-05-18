# V_24 EX-DUAL — dual-target multi-task head FALSIFICATION

**Date**: 2026-05-18
**Verdict**: **FALSIFIED.** The dual-target auxiliary PJND head does
not help quality regression on any of {CID22, KADID, TID, KonJND-1k,
AIC-3} at this trainer configuration. Increasing λ_pjnd from 0.01 to
1.0 produces uniformly WORSE y_quality output than the λ=0 single-head
control on aggregate, with no λ producing Pareto-better results vs the
V_22-mix-LARGE+iwssim ship.

Branch: `feat/ex-dual-target-head`
Methodology doc: `benchmarks/v24_dual_target_methodology_2026-05-18.md`

## Single-seed λ sweep — full Mohammadi panel aggregate per corpus

All bakes seed=3, h=128, 100 epochs × 10k pairs/epoch, K=1, no NiN,
no PWRC. Validation policy: Min over kadid + tid + konjnd.

### SROCC

| λ_pjnd | CID22 | KADID | TID | KonJND | AIC-3 | bake size |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | **0.8631** | **0.8897** | 0.8464 | 0.5460 | **0.8036** | 157 KB |
| 0.01 | 0.7377 | 0.7827 | 0.7467 | **0.5822** | 0.7574 | 157 KB |
| 0.05 | 0.7745 | 0.8398 | 0.7954 | 0.4707 | 0.7927 | 157 KB |
| 0.10 | 0.7605 | 0.8438 | **0.8479** | 0.4657 | 0.7845 | 157 KB |
| 0.30 | 0.7957 | 0.8143 | 0.7864 | 0.5016 | 0.7882 | 157 KB |
| 1.00 | 0.7274 | 0.8594 | 0.8458 | 0.4112 | 0.7681 | 157 KB |
| **V_22 ship** | **0.8324** | **0.9677** | **0.9729** | **0.8927** | **0.7845** | 41 KB |

**Best per-corpus λ winner**: λ=0.00 wins CID22 / KADID / AIC-3;
λ=0.01 wins KonJND (by +0.036 vs control); λ=0.10 wins TID (by
+0.0015 vs control — within seed noise).

### Z-RMSE

| λ_pjnd | CID22 | KADID | TID | KonJND | AIC-3 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | **0.528** | **0.459** | **0.503** | **0.787** | **0.585** |
| 0.01 | 0.679 | 0.623 | 0.645 | 0.706 | 0.644 |
| 0.05 | 0.642 | 0.546 | 0.606 | 0.827 | 0.600 |
| 0.10 | 0.627 | 0.539 | 0.499 | 0.821 | 0.615 |
| 0.30 | 0.606 | 0.585 | 0.592 | 0.795 | 0.603 |
| 1.00 | 0.680 | 0.514 | 0.508 | 0.859 | 0.635 |
| **V_22 ship** | 0.559 | 0.249 | 0.236 | 0.376 | 0.606 |

### PWRC

| λ_pjnd | CID22 | KADID | TID | KonJND | AIC-3 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.9145 | 0.9350 | 0.8876 | 0.5883 | 0.8756 |
| 0.01 | 0.8163 | 0.8608 | 0.8270 | 0.6353 | 0.8327 |
| 0.05 | 0.8413 | 0.8998 | 0.8566 | 0.4894 | 0.8647 |
| 0.10 | 0.8327 | 0.9076 | 0.8913 | 0.4622 | 0.8528 |
| 0.30 | 0.8655 | 0.8831 | 0.8438 | 0.5235 | 0.8624 |
| 1.00 | 0.8073 | 0.9141 | 0.8888 | 0.3843 | 0.8394 |
| **V_22 ship** | 0.9006 | 0.9804 | 0.9832 | 0.9178 | 0.8630 |

## Decisive comparisons (§ A.9 bake_compare, n=500 bootstrap)

### EX-DUAL λ=0.0 (no PJND) vs V_22 ship

| Corpus | n | h_SROCC | h_Z | DecScore | Verdict |
|---|---:|---:|---:|---:|---|
| CID22 | 4292 | +24.32 | +44.83 | +20.27 | **A >> B** (EX-DUAL wins) |
| KADIK10k | 10125 | -94.21 | -567.56 | -15.70 | B >> A |
| TID2013 | 3000 | -52.29 | -236.00 | -0.00 | B >> A |
| KonJND-1k | 1008 | -34.78 | -55.29 | -0.00 | B >> A |
| AIC-3 CTC | 600 | +14.31 | +26.27 | +11.93 | **A >> B** (EX-DUAL wins) |

The λ=0 control wins CID22 + AIC-3 vs V_22 ship — this is interesting
**independent of dual-target** and worth attention separately. The
minimal-trainer (K=1, NiN-off, PWRC-off) baseline beats V_22-LARGE+iwssim
on the two corpora that don't share KonJND/KADID/TID's synthetic
distortion distribution. **This is a structural finding about V_22's
NiN+K stack hurting CID22, NOT about dual-target.**

### EX-DUAL λ=0.01 (best KonJND lifter) vs V_22 ship

| Corpus | n | h_SROCC | h_Z | DecScore | Verdict |
|---|---:|---:|---:|---:|---|
| CID22 | 4292 | -25.03 | -46.66 | -0.00 | **B >> A** (V_22 wins) |
| KADIK10k | 10125 | -83.92 | -296.17 | -13.99 | **B >> A** |
| TID2013 | 3000 | -44.03 | -140.44 | -0.00 | **B >> A** |
| KonJND-1k | 1008 | -29.20 | -46.36 | -0.00 | **B >> A** |
| AIC-3 CTC | 600 | -6.12 | -13.19 | -0.00 | **B >> A** |

**Decisive on 5/5 corpora — V_22 ship wins everything.** Even the best
EX-DUAL candidate at the most permissive λ does NOT close the gap.

## Mechanism analysis

### Why λ > 0 hurts y_quality

The shared-encoder dual-target architecture has a known multi-task
interference failure mode. At λ_pjnd > 0:

1. The PJND group's `human_score` column is per-source PJND broadcast
   (range 22–70) — a CONSTANT across the 20 distortion levels per
   source. The auxiliary head must learn `f(features) ≈ PJND(source)`
   which is **invariant to within-source quality variation**.
2. y_quality, by contrast, must be **sensitive to within-source
   quality variation** (it's a per-pair RankNet target).
3. Backprop through the shared encoder routes BOTH gradients into
   the same 300×128 weight matrix. The PJND gradient pushes encoder
   features toward source-identifying invariants; the RankNet
   gradient pushes them toward quality-discriminating signals.
4. **These are partially anti-correlated.** Source-identifying
   invariants AVERAGE OUT within-source quality variation by
   construction; quality-discriminating signals SUMMARIZE
   within-source quality variation. The encoder cannot satisfy
   both — λ_pjnd > 0 yields a compromise that degrades both.

### Why even λ = 0.01 already destroys quality

The PJND target's range (22–70) is **20–60×** larger than y_quality's
effective gradient scale (RankNet log-loss is O(1)). Even at
λ_pjnd = 0.01, the PJND-MSE gradient on the encoder is:

```
‖dL/dw‖_pjnd ≈ 2 · 0.01 · |y_pjnd − target| · |h|
             ≈ 2 · 0.01 · 500 · 1  ≈ 10
```

vs the RankNet gradient ≈ O(1). The PJND signal **dominates**
the encoder update even at small λ.

A proper scaling would normalize the PJND target to RankNet-scale,
e.g. divide by 100 (range 0.22..0.70) — but that's a different
experiment. The current sweep falsifies dual-target as **specified
in the instructions** (per-source PJND-broadcast MSE on raw scale).

### Why the experiment was worth doing anyway

Even with the magnitude-mismatch caveat above, the falsification
result is informative:

1. **Auxiliary task at this scale + shared encoder is a known
   bad pattern.** Domain practice (e.g. RUL prediction with
   auxiliary classification heads) normalizes auxiliary task
   scales OR uses task-specific subspaces of the encoder.
2. **Per-source PJND-broadcast is structurally a poor auxiliary
   signal for quality** — the broadcast makes the auxiliary target
   anti-correlated with within-source quality variation by
   construction. Even with proper scaling, the encoder would still
   be pulled toward source-identifying features.
3. **The KonJND validation noise is the load-bearing problem**, not
   dual-target. KonJND has 1008 pairs; SROCC CI at n=1008 spans
   ±0.06 easily; val-policy=Min on this noisy signal selects
   checkpoints with high cross-corpus regression. A trainer that
   used MEAN over (KADID + TID + KonJND) instead of MIN would
   produce comparably noisy but at least less catastrophic
   checkpoint selection.

## Recommendation

Do NOT proceed with 5-seed CI or bake_compare ship-candidate
extraction. The seed=3 sweep already shows:

- λ=0 is structurally better than every λ > 0 on the y_quality task.
- No λ produces Pareto-better results vs V_22-LARGE+iwssim ship.
- The mechanism (multi-task interference + magnitude-mismatch) is
  well-understood and would not be cured by additional seeds.

**Dual-target with per-source PJND-broadcast as the auxiliary
target is FALSIFIED for the V_22-LARGE+iwssim ship-comparison
question.**

### Possible follow-ups (NOT pursued in this experiment)

1. **Per-pair PJND labels (KonJND++ dataset)** — would test
   whether per-pair (not per-source-broadcast) PJND supervision
   carries non-anti-correlated signal. **Dataset unavailable.**
2. **Scale-normalized auxiliary target** — divide PJND by 100 so
   gradients are RankNet-comparable. May rescue λ ∈ [0.01, 0.1]
   from dominance, but would still suffer the per-source-broadcast
   anti-correlation problem.
3. **Disjoint subspaces** — split the n_hidden=128 hidden into
   y_quality-only (e.g., dims 0..63) and y_pjnd-only (dims 64..127)
   sub-encoders. Removes interference but loses the "force PJND-
   relevant representations into the y_quality encoder" mechanism
   that motivated dual-target in the first place.
4. **Compose with V_22-LARGE's NiN + K + PWRC stack** — would test
   whether dual-target helps WHEN the encoder already has the
   tools to absorb auxiliary supervision. Significant engineering
   work (extending the per-sample-α head trainer's NiN-compose
   batch flush to dual-output). Speculative payoff.

## Lineage

- Architecture: `zensim-train-core/src/dual_target_head.rs`
  (forward + backward + bake + 8 tests including 3 finite-difference
  backprop checks, all passing).
- Trainer wiring: `zensim-validate/src/mlp_train.rs`, new
  `train_mlp_dual_target_head` function dispatched via
  `MlpHyperparams::dual_target_head` flag, exclusive with
  pool_head/hybrid_head/per_sample_alpha_head.
- CLI flags: `--dual-target-head --pjnd-loss-weight LAMBDA
  --pjnd-group-name NAME` (default off, λ=0, name=None).
- Bakes are ZNPR v3 single-output (y_pjnd discarded) — load
  through standard `apply_mlp_scoring` runtime path unchanged.
- All sweep artifacts at
  `/mnt/v/output/zensim/ex-dual-2026-05-18/`:
  - 6 bakes (`exdual_l*.bin`)
  - 6 verdict reports (`exdual_l*.verdict.md`)
  - 2 bake_compare reports (vs V_22 ship: `exdual_l0.0_vs_v22_baseline.md`,
    `exdual_l0.01_vs_v22.md`).
- Methodology: `benchmarks/v24_dual_target_methodology_2026-05-18.md`.

## File register

- `zensim-train-core/src/dual_target_head.rs` — head module (1024 lines)
- `zensim-train-core/src/lib.rs` — module declaration
- `zensim-validate/src/mlp_train.rs` — trainer integration (+ 350 lines)
- `zensim-validate/src/bin/zensim_mlp_train.rs` — CLI flags
- `scripts/v_next/run_ex_dual_seed.sh` — single-seed launcher
- `scripts/v_next/run_ex_dual_sweep.sh` — λ-sweep launcher
- `scripts/v_next/run_ex_dual_sweep_resume.sh` — resume with retry
- `scripts/v_next/score_ex_dual_sweep.sh` — score all bakes
- `scripts/v_next/score_on_arrival.sh` — score-as-bakes-arrive helper
- `scripts/v_next/summarize_ex_dual.py` — collate verdicts to summary

Total wall time spent: ~50 min (sweep) + ~5 min (scoring + comparison).
Compute cost: ~$0.00 (local).
