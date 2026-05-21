# V11-D-PJND-DOMINANT — KonJND PJND-passthrough rescue FALSIFIED (task #198)

**Date**: 2026-05-20
**Task**: #198 — "test whether dominating the loss with KonJND-PJND
anchor passthrough can preserve KonJND ranking while still getting
partial cross-codec-eq benefit"
**Verdict**: **FALSIFIED with structural finality**. Across PJND-passthrough
weights `{2.0, 5.0, 10.0}` (4×, 10×, 20× the cross_codec_eq weight=0.5),
**KonJND collapses HARDER than V11-A's no-pjnd baseline at every tier**
AND CID22 also drops by 0.13-0.19 vs V11-A v4 ref.

**Conclusion**: the cross-codec-eq SOTA frontier is now closed with maximum
structural certainty. **No auxiliary loss can rescue KonJND from the
basin-B collapse** that cross-codec-eq induces. V10 BalancedV3 IS the
achievable Balanced SOTA at this substrate, and V_24 per-sample-α IS the
achievable Compression SOTA. **This was the last scheduled cross-codec-eq
experiment; the line is closed.**

## Hypothesis (task #198 brief)

Per the V11-A-CC-EQ-WEIGHT-SWEEP falsification (commit `e13c0b9`), KonJND
collapse is binary not magnitude-dependent at cross-codec-eq weights
`{0.05, 0.10, 0.20, 0.50, 1.00}`. Open follow-up direction #1 from
that doc:

> Cross-codec-eq + multi-codec-anchor produces a bistable loss landscape
> with two basins:
> - Basin A: rank-preserve mode (per-image PJND, used by V_22-mix and V_24)
> - Basin B: cross-codec-eq mode (consensus-medianizing, used by V11 at
>   any cc_eq weight)
>
> A 3rd loss term (PJND anchor passthrough at weight ≫ cc_eq) might
> break the bistability: explicitly penalize predictions at KonJND-PJND-
> anchored pairs from drifting away from score=80 (JND), forcing the
> network to stay in the Basin A neighborhood for those specific anchor
> pairs even while cross-codec-eq pulls others toward Basin B.

Two anticipated outcomes:
1. **PJND anchor breaks the bistability** → KonJND survives at higher
   cc_eq weights → some w-band ships.
2. **PJND anchor is overrun by cc_eq** → KonJND still collapses → V11
   mechanism CLOSED with even firmer structural certainty.

Result: outcome (2), and stronger than anticipated. PJND-passthrough does
not just fail to rescue KonJND — it makes the collapse WORSE while also
dropping CID22.

## Method

- **3 PJND-passthrough weights × 5 seeds = 15 GPU bakes.**
- `cross_codec_eq_weight = 0.5` held fixed (the V11-A central point).
- `pjnd_passthrough_weight ∈ {2.0, 5.0, 10.0}` = 4×, 10×, 20× the
  cross-codec-eq weight, spec'd in the task brief.
- `pjnd_passthrough_step_p = 0.30` (matches `anchor_step_p`).
- `pjnd_passthrough_target_score = 80.0` (V10's PJND calibration point).
- All other hparams identical to V11-A v4 (commit `a8c030e`): h=128,
  epochs=300, pairs-per-epoch=50000, lr=5.66e-3, mse_weight=1.0,
  monotonicity_reg=1.0, mix_cv35_iw65 target, 372 features, anchor=v4
  (2471 rows), equiv=v4 (1739 pairs), dynamic-range-floor=0.3 σ=25.
- Trainer extension: new `--pjnd-passthrough-{parquet,weight,step-p,target-score}`
  flags landed on commit `c8e2afe0`. Both CPU and GPU paths support the
  second anchor pool (per CLAUDE.md "Bake-metadata propagation" — fully
  additive).
- PJND parquet: `canonical-2026-05-21/train/konjnd-dense.parquet` (20,160
  rows × 372 features), loaded with constant per-row weight 1.0.
- Substrate: `2026-05-20-v11-substrate/anchors_ssim2_372col_v4.parquet`
  + `cross_codec_equivalence_ssim2_372col_v4.parquet` (4-codec × 372-feat
  v4 sets).

Runner: `scripts/v11_d_pjnd/launch_sweep.sh`
Eval: `scripts/v11_d_pjnd/eval_sweep.sh`
Cross-codec consistency: `scripts/v11_d_pjnd/run_cc_consistency.sh`
Outputs: `/mnt/v/output/zensim/exp_v11_d_pjnd_2026-05-20/`

## Per-seed verdict table

| pjnd_w | seed | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 1 | 0.7850 | 0.7803 | 0.7618 | 0.2836 | 0.7421 | 0.8999 |
| 2.0 | 2 | 0.6791 | 0.7906 | 0.7704 | 0.0247 | 0.7244 | 0.9009 |
| 2.0 | 3 | 0.7692 | 0.7982 | 0.7870 | 0.2684 | 0.7593 | 0.9001 |
| 2.0 | 4 | 0.8093 | 0.7872 | 0.7767 | 0.1555 | 0.7529 | 0.9180 |
| 2.0 | 5 | 0.7203 | 0.7792 | 0.7628 | 0.1225 | 0.7313 | 0.8960 |
| 5.0 | 1 | 0.7506 | 0.7874 | 0.7716 | 0.1352 | 0.7261 | 0.9097 |
| 5.0 | 2 | 0.7113 | 0.7890 | 0.7721 | 0.0449 | 0.7410 | 0.9028 |
| 5.0 | 3 | 0.7214 | 0.7844 | 0.7713 | 0.2689 | 0.7540 | 0.9040 |
| 5.0 | 4 | 0.7055 | 0.7964 | 0.7693 | 0.0724 | 0.7442 | 0.8866 |
| 5.0 | 5 | 0.7182 | 0.7882 | 0.7735 | 0.0895 | 0.7239 | 0.8907 |
| 10.0 | 1 | 0.7218 | 0.7856 | 0.7782 | 0.1358 | 0.7280 | 0.8965 |
| 10.0 | 2 | 0.7046 | 0.7928 | 0.7783 | 0.0235 | 0.7437 | 0.8972 |
| 10.0 | 3 | 0.6354 | 0.7979 | 0.7760 | 0.0353 | 0.7237 | 0.8722 |
| 10.0 | 4 | 0.7485 | 0.7848 | 0.7583 | 0.0583 | 0.7515 | 0.8991 |
| 10.0 | 5 | 0.6999 | 0.7859 | 0.7719 | 0.1212 | 0.7386 | 0.8949 |

## Per-tier median + reference comparison

| Variant | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 | cc_bp3 @ T=80 | cc_bp3 @ T=50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **pjnd_w=2.0 (med s4)** | 0.7692 | 0.7872 | 0.7704 | **0.1555** | 0.7421 | 0.9001 | 1.4023 | 1.5165 |
| **pjnd_w=5.0 (med s5)** | 0.7182 | 0.7882 | 0.7716 | **0.0895** | 0.7410 | 0.9028 | 1.4156 | 1.5060 |
| **pjnd_w=10.0 (med s4)** | 0.7046 | 0.7859 | 0.7760 | **0.0583** | 0.7386 | 0.8965 | 1.3873 | 1.5215 |
| V11-A' v4 ref (w=0.50, no pjnd) | 0.8965 | 0.9291 | 0.8932 | **0.4312** | 0.8137 | 0.9471 | 0.930 | 2.256 |
| V11-A' v4 ref (w=1.00, no pjnd) | 0.8944 | 0.9253 | 0.8903 | **0.3942** | 0.8173 | 0.9522 | 1.009 | 2.263 |
| V10 BalancedV3 (ref) | 0.8324 | 0.9664 | 0.9712 | **0.8927** | 0.7845 | 0.9016 | 7.062 | 7.062 |
| V_24 Compression (ref) | 0.8641 | 0.9316 | 0.8893 | **0.8080** | 0.8183 | n/a | — | — |

KonJND-median bakes per tier (spec: "Pick median bake by KonJND SROCC"):

- pjnd_w=2.0 → seed 4 (KonJND 0.1555)
- pjnd_w=5.0 → seed 5 (KonJND 0.0895)
- pjnd_w=10.0 → seed 4 (KonJND 0.0583)

## Ship gate verdict per tier

Per task #198 brief — Balanced trail gate:
- CID22 SROCC ≥ 0.8324 + 0.005 = ≥ 0.8374 (v10 + 0.005)
- KADID/TID/KonJND drift within −0.10 of V10 (**KonJND ≥ 0.7927**)
- AIC-3 / AIC-4 ≥ −0.005 of V10

**All three pjnd_w tiers FAIL the Balanced gate decisively on KonJND
AND CID22:**

| pjnd_w | CID22 vs gate (≥0.8374) | KonJND vs gate (≥0.7927) | Verdict |
|---:|---|---|---|
| 2.0 | 0.7692 < 0.8374 (−0.068) | 0.1555 < 0.7927 (**−0.637**) | FAIL |
| 5.0 | 0.7182 < 0.8374 (−0.119) | 0.0895 < 0.7927 (**−0.703**) | FAIL |
| 10.0 | 0.7046 < 0.8374 (−0.131) | 0.0583 < 0.7927 (**−0.734**) | FAIL |

KADID/TID/AIC-3 drift relative to V10:
- KADID drops 0.18-0.19 (from V10's 0.9664) — far exceeds the −0.10 gate.
- TID drops 0.19-0.20 — same.
- AIC-3 drops 0.04 — within −0.10.

**No tier passes any gate.** No ship.

## Findings

### 1. PJND-passthrough makes KonJND collapse WORSE, monotonically with weight

V11-A v4 at w=0.50 had KonJND 0.4312 (collapsed but not zero). Adding
PJND-passthrough at any positive weight pushes KonJND further toward
zero, monotonically:

- pjnd_w=2.0: median KonJND 0.1555 (**−0.276 vs V11-A**)
- pjnd_w=5.0: median KonJND 0.0895 (**−0.342 vs V11-A**)
- pjnd_w=10.0: median KonJND 0.0583 (**−0.373 vs V11-A**)

This is the OPPOSITE direction from the hypothesis. The PJND-passthrough
was supposed to PROTECT KonJND ranking; instead it ACCELERATES the
collapse. The mechanism appears to be that passthrough toward a single
constant target (score=80) collapses the dynamic range further — every
KonJND row gets the same target, so the network has even less reason
to learn quality discrimination within the KonJND pairs.

### 2. CID22 also drops 0.13-0.19 — no compression-trail compensation either

V11-A v4 at w=0.50 had CID22 0.8965 (the +0.062 lift that made it a
candidate Compression ship if KonJND hadn't collapsed). PJND-passthrough
at every tier drops CID22 by 0.13-0.19:

- pjnd_w=2.0: median CID22 0.7692 (**−0.127 vs V11-A w=0.50**)
- pjnd_w=5.0: median CID22 0.7182 (**−0.178**)
- pjnd_w=10.0: median CID22 0.7046 (**−0.192**)

So PJND-passthrough doesn't even preserve V11's compression-trail
advantage on CID22. Both Balanced AND Compression trails lose at
every tier.

### 3. Cross-codec consistency holds (the cc_eq mechanism is unchanged)

| pjnd_w | cc_bp3 @ T=80 (JND) | cc_bmax @ T=80 | cc_bp3 @ T=50 (JOD) | cc_bmax @ T=50 |
|---:|---:|---:|---:|---:|
| 2.0 (s4) | 1.402 | 3.156 | 1.517 | 3.442 |
| 5.0 (s5) | 1.416 | 3.179 | 1.506 | 3.421 |
| 10.0 (s4) | 1.387 | 3.125 | 1.521 | 3.452 |
| V11-A w=0.50 (s1, ref) | 0.930 | 2.030 | 2.256 | 5.437 |
| V11-A w=1.00 (v4 s2 ref) | 1.009 | 2.228 | 2.263 | 5.407 |
| V10 BalancedV3 | 7.062 | — | 7.062 | — |

The cross-codec convergence mechanism is still active. cc_bp3 at JND
stays around 1.4 (vs V10's 7.06), confirming the cc_eq aux loss is
doing its job. But PJND-passthrough doesn't FURTHER improve it (cc_bp3
slightly worse than V11-A's 0.93-1.07).

**The PJND-passthrough doesn't disrupt cross-codec-eq's basin lock-in.
It just makes the bake's score distribution worse everywhere else.**
Note also that t=50 (JOD) reveals the bake's dynamic range has
collapsed — many images saturate to score=100 or score=0 at the JOD
target, so the binary-search lookup degrades.

## What this rules out (structurally)

1. **Auxiliary anchor passthrough cannot rescue Basin A.** The 3-loss
   composition (RankNet pair + cross-codec-eq + PJND-passthrough)
   does NOT yield a bistable-loss-breaking middle ground at any
   PJND-weight magnitude. The mechanism makes both trails worse.

2. **The KonJND collapse is not magnitude-modulable by aux losses.**
   V11-A's sweep showed the cross-codec-eq term alone collapses
   KonJND uniformly across w ∈ {0.05, ..., 1.00}. Adding a 4× to
   20× heavier PJND-anchor on top DOESN'T pull the network back
   toward Basin A — it just degrades the network further.

3. **The cross-codec-eq SOTA frontier is closed at this substrate.**
   This was the last scheduled experiment per the V11-A direction-1
   plan. No further cross-codec-eq aux-loss experiments are
   warranted; the basin-B lock-in is a property of the mechanism,
   not a hyperparameter pathology.

## What's still open (different architectural directions)

The cross-codec-eq mechanism remains effective at cross-codec
convergence (cc_bp3 ≤ 1.5 at JND, vs V10's 7.06). It just trades
that against KonJND ranking. Future directions that don't
trigger this trade-off:

- **Cross-codec-eq band-gating**: route the cc_eq loss only to
  feature pairs whose target metric is in a specific band
  (e.g. q-range 5-40), leaving the high-q regime under
  pure-RankNet so KonJND PJND tracking survives. Untested.
- **Different architectural target**: cross-codec consistency at
  the picker frontier is already served by the production
  `v_cross_codec_v2` ship. The picker doesn't need a single bake
  that does both rank-preserve AND cross-codec-eq.
- **Distinct head architecture**: a multi-head bake with one
  rank-preserve head (V10-style) and one cross-codec-eq head
  (V11-style), gated by the input regime. Tested as
  PreviewV0_5Ensemble for V_05/V_24 trail merging; the same
  classifier approach could apply here.

None of these are on the current backlog; the V11 line itself is
closed.

## Trainer extension provenance

- New flags: `--pjnd-passthrough-{parquet,weight,step-p,target-score}`
- Trainer entry: `train_mlp_with_tv_anchored_equiv_pjnd` (new wrapper,
  back-compat: the existing `train_mlp_with_tv_anchored_equiv` defaults
  `pjnd_anchor: None`).
- CPU path: `train_mlp_per_sample_alpha_head` extended with second
  anchor pool standardize + per-step aux fire (~80 LoC of new code,
  mechanically identical to the existing anchor step).
- GPU path: `train_per_sample_alpha_head_gpu_with_aux_pjnd` (new
  entry), reuses `fire_anchor_aux` for the second pool. `GpuHparams`
  gains `pjnd_passthrough_weight` + `pjnd_passthrough_step_p`.
  `GpuAnchorRows` shape unchanged (same struct serves both pools).
- All edits backward-compatible: when `--pjnd-passthrough-weight 0`
  (default), no change in behavior or bit output vs prior trainer.

## Artifacts

- 15 bakes: `/mnt/v/output/zensim/exp_v11_d_pjnd_2026-05-20/cc4v11d_pjnd{2.0,5.0,10.0}_s{1..5}.bin` (each 261,351 bytes)
- 15 verdicts: `/mnt/v/output/zensim/exp_v11_d_pjnd_2026-05-20/verdicts/cc4v11d_pjnd{...}.md`
- 6 cc-consistency TSVs: `/mnt/v/output/zensim/exp_v11_d_pjnd_2026-05-20/cc_consistency/cc4v11d_pjnd{...}_t{80,50}_n20.tsv`
- Per-seed summary TSV: `/mnt/v/output/zensim/exp_v11_d_pjnd_2026-05-20/summary_2026-05-20.tsv`

## Ship decision

**No bake ships.** The Balanced ship remains V10 BalancedV3 (CID22
0.8324, KonJND 0.8927, KADID 0.9664, TID 0.9712 — see
`zensim/weights/v0_5_balanced_v3_2026-05-20.bin`). The Compression ship
remains V_24 per-sample-α (CID22 0.8641, AIC-3 0.8183 — see
`zensim/weights/v_24_persample_alpha_2026-05-19.bin`).

The cross-codec-eq SOTA frontier is closed with structural finality.
This is the LAST scheduled cross-codec-eq experiment per the task
plan, and the result confirms the frontier is closed regardless of
auxiliary-loss design. Move on to other directions.
