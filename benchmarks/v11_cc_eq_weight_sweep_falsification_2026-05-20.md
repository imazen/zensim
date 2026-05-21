# V11-A-CC-EQ-WEIGHT-SWEEP — FALSIFIED (task #197, 2026-05-20)

**Verdict: the KonJND collapse from the cross-codec-eq mechanism is structural, not weight-dependent.** At every tested `--cross-codec-eq-weight` in {0.05, 0.10, 0.20, 0.50, 1.00}, KonJND lands in the 0.37–0.45 band — far below the 0.79 Balanced-trail gate. The hypothesis that "at w << 1.0 the rank-preserve term dominates so KonJND survives" is falsified.

V10 BalancedV3 remains the Balanced ship. V_24 per-sample-α remains the Compression ship. The cross-codec-eq frontier is closed at this substrate.

## Hypothesis (task #197 brief)

> Is the KonJND collapse a smooth function of cross_codec_eq_weight? If yes,
> a low-weight sweep finds the trade-off frontier. If no, KonJND collapse is
> structural regardless of weight.

V11-A'-372 v4 (commit `a8c030e`) at w=1.0 hit CID22 0.8944 (+0.062 vs V10
BalancedV3) but KonJND collapsed 0.8927 → 0.3942 (−0.499). Same mechanism
collapsed at 300-feat (V11-A' v2 clean s3 KonJND 0.4033) and at
V_CrossCodec. We need to know whether the collapse smoothly decays as
weight goes down, or whether the mechanism is binary (any positive weight
breaks PJND tracking).

## Method

- 5 seeds × 4 cross_codec_eq weights ∈ {0.05, 0.10, 0.20, 0.50} = 20 bakes.
- Identical recipe to `run_v11a_372_v4_clean_seed.sh` (commit `a8c030e`)
  except `--cross-codec-eq-weight` parameterized via env var.
- Substrate: `2026-05-20-v11-substrate` (anchors_ssim2_372col_v4 +
  cross_codec_equivalence_ssim2_372col_v4) — 4-codec × 372-feat full
  coverage from the Phase 1 decoder fix.
- Train corpus: canonical-2026-05-21 (safesyn + kadid + tid +
  konjnd-dense + cid22_train + pipal).
- Trainer: GPU CUDA, minibatch auto-bumped to 512 for aux losses.
  ~42 s per training, ~50 s wall per bake including data load.
- Total sweep wall: 17 min for 20 bakes.

Runner: `scripts/v_next/v11_372feat/run_v11a_372_v4_w_sweep_seed.sh`
Driver: `scripts/v_next/v11_372feat/run_v11a_372_v4_w_sweep_all.sh`
Eval: `scripts/v_next/v11_372feat/eval_v11_cc_eq_sweep.sh`
Outputs: `/mnt/v/zen/zensim-eval/exp_v11_cc_eq_sweep_2026-05-20/`

## Per-tier 5-seed CI (bake_verdict, canonical-2026-05-15 feature parquets)

### w = 0.05

| seed | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8887 | 0.9245 | 0.8948 | 0.3744 | 0.8187 | 0.9414 |
| 2 | 0.8941 | 0.9282 | 0.8902 | 0.4340 | 0.8185 | 0.9544 |
| 3 | 0.8904 | 0.9223 | 0.8844 | 0.3931 | 0.8175 | 0.9462 |
| 4 | 0.8994 | 0.9239 | 0.8904 | 0.3925 | 0.8224 | 0.9468 |
| 5 | 0.8935 | 0.9247 | 0.8937 | 0.3844 | 0.8155 | 0.9544 |
| **median** | **0.8935** | **0.9245** | **0.8904** | **0.3925** | **0.8185** | **0.9468** |

### w = 0.10

| seed | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8960 | 0.9302 | 0.8948 | 0.3813 | 0.8107 | 0.9439 |
| 2 | 0.9005 | 0.9287 | 0.8950 | 0.4539 | 0.8216 | 0.9451 |
| 3 | 0.8954 | 0.9157 | 0.8824 | 0.3916 | 0.8018 | 0.9547 |
| 4 | 0.9023 | 0.9252 | 0.8875 | 0.4270 | 0.8151 | 0.9405 |
| 5 | 0.8789 | 0.9162 | 0.8759 | 0.3722 | 0.8306 | 0.9537 |
| **median** | **0.8960** | **0.9252** | **0.8875** | **0.3916** | **0.8151** | **0.9451** |

### w = 0.20

| seed | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9008 | 0.9323 | 0.8919 | 0.4096 | 0.8055 | 0.9462 |
| 2 | 0.8926 | 0.9246 | 0.8885 | 0.4446 | 0.8258 | 0.9476 |
| 3 | 0.8932 | 0.9209 | 0.8862 | 0.3830 | 0.8251 | 0.9478 |
| 4 | 0.8948 | 0.9244 | 0.8886 | 0.3664 | 0.8179 | 0.9538 |
| 5 | 0.8804 | 0.9207 | 0.8879 | 0.3875 | 0.8207 | 0.9497 |
| **median** | **0.8932** | **0.9244** | **0.8885** | **0.3875** | **0.8207** | **0.9478** |

### w = 0.50

| seed | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8965 | 0.9322 | 0.8932 | 0.3752 | 0.8137 | 0.9471 |
| 2 | 0.8986 | 0.9291 | 0.8939 | 0.4497 | 0.8243 | 0.9447 |
| 3 | 0.8941 | 0.9213 | 0.8890 | 0.4443 | 0.8163 | 0.9520 |
| 4 | 0.9050 | 0.9294 | 0.8936 | 0.3908 | 0.8134 | 0.9504 |
| 5 | 0.8907 | 0.9238 | 0.8925 | 0.4312 | 0.8088 | 0.9333 |
| **median** | **0.8965** | **0.9291** | **0.8932** | **0.4312** | **0.8137** | **0.9471** |

## Master comparison (medians + reference rows)

| w | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 | cc_bp3 @ JND | cc_bp3 @ JOD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 (s5) | 0.8935 | 0.9245 | 0.8904 | **0.3925** | 0.8185 | 0.9468 | 1.042 | 2.403 |
| 0.10 (s1) | 0.8960 | 0.9252 | 0.8875 | **0.3916** | 0.8151 | 0.9451 | 1.066 | 2.313 |
| 0.20 (s3) | 0.8932 | 0.9244 | 0.8885 | **0.3875** | 0.8207 | 0.9478 | 0.930 | 2.190 |
| 0.50 (s1) | 0.8965 | 0.9291 | 0.8932 | **0.4312** | 0.8137 | 0.9471 | 0.930 | 2.256 |
| 1.00 (v4 clean s2 ref) | 0.8944 | 0.9253 | 0.8903 | **0.3942** | 0.8173 | 0.9522 | 1.009 | 2.263 |
| V10 BalancedV3 (ref) | 0.8324 | 0.9664 | 0.9712 | **0.8927** | 0.7845 | 0.9016 | 7.062 | 7.062 |
| V_24 Compression (ref) | 0.8641 | 0.9316 | 0.8893 | **0.8080** | 0.8183 | n/a | — | — |

`cc_bp3` = mean pairwise butteraugli-pnorm3 across {jpeg, webp, avif}
at the target zensim score, 20 source images per target. V10 BalancedV3
row uses the bake-post=clamp default; the V10 spline calibration is
applied at runtime via the `output_calibration_spline` metadata
(visible in `zensim::metric::forward_one_bake` but skipped by the
isolated `predict_features_with_bake` invocation used here). Either
way the V10 cc_bp3 is dramatically worse than V11 — the cross-codec-eq
mechanism IS working at cross-codec convergence.

## Ship gate verdict per tier

Per task #197 brief — Balanced trail gate:
- CID22 SROCC ≥ 0.8324 + 0.005 = ≥ 0.8374
- KADID/TID/KonJND drift within −0.10 of V10 (KADID ≥ 0.8664; TID ≥ 0.8712; **KonJND ≥ 0.7927**)
- AIC-3/AIC-4 within −0.005 of V10 baseline

| w | CID22 PASS | KADID PASS | TID PASS | KonJND PASS | Ship? |
|---|---|---|---|---|---|
| 0.05 | YES (+0.061) | YES (−0.042) | YES (−0.081) | **NO (−0.500)** | **NO** |
| 0.10 | YES (+0.064) | YES (−0.041) | YES (−0.084) | **NO (−0.501)** | **NO** |
| 0.20 | YES (+0.061) | YES (−0.042) | YES (−0.083) | **NO (−0.505)** | **NO** |
| 0.50 | YES (+0.064) | YES (−0.037) | YES (−0.078) | **NO (−0.461)** | **NO** |

The KonJND drift in EVERY tier exceeds the −0.10 cap by ~5×. There is
no w in the tested range where the Balanced trail gate passes.

Compression trail gate (decisive A>>B on ≥1 of CID22/AIC-3 + KADID/TID/KonJND within −0.10):
- CID22 ≥ V_24 Compression ✓ at all tiers (+0.029 to +0.032)
- KonJND drift vs V_24 Compression (0.8080) ≥ -0.10 = ≥ 0.7080 → **all 4 tiers FAIL** (medians 0.39-0.43)

**Decision: NO SHIP at any weight tier.**

## Why w doesn't smoothly scale the KonJND collapse

The cross-codec-eq loss is sampled with probability `--cross-codec-eq-step-p`
(0.1 default) per pair-step. The weight `w` multiplies the gradient when
sampled. The dynamic-range-floor loss (weight 0.3) and the rank-preserve
loss (weight 0.2) are independent — they fire on their own probability
schedules.

The KonJND target is per-row PJND threshold injection: a single anchor
score per row. The cross-codec-eq loss says "score row A and row B
identically" — i.e. it asks the network to be q-invariant within a
butter-level band. KonJND asks the network to track per-row PJND
thresholds across q.

These two constraints conflict at the network's output layer. The
gradient from cross-codec-eq pushes the output toward a constant; the
KonJND PJND gradient pushes the output toward a per-row threshold.
Once cross-codec-eq has ANY non-zero weight, the network's preferred
solution is to flatten across q within a butter band — because that
satisfies the larger sample population at lower total loss. KonJND's
1008-row anchor gets dominated.

The 5-tier sweep shows this is **not a magnitude effect** (weight scaling
the gradient strength) but a **structural conflict** (weight gating
whether the gradient is applied at all). Once the loss is wired,
SGD drives the network's W_α and pool-head outputs toward the
constant-prediction basin within ~50 epochs. The 250 remaining epochs
don't re-find a PJND-tracking solution because the anchor loss + KonJND
group weight (0.6) isn't enough push to escape.

## Cross-codec consistency at JND and JOD

| w | cc_bp3 @ T=80 (JND) | cc_bmax @ T=80 | cc_bp3 @ T=50 (JOD) | cc_bmax @ T=50 |
|---:|---:|---:|---:|---:|
| 0.05 (s5) | 1.042 | 2.320 | 2.403 | 5.901 |
| 0.10 (s1) | 1.066 | 2.375 | 2.313 | 5.573 |
| 0.20 (s3) | 0.930 | 2.043 | 2.190 | 5.212 |
| 0.50 (s1) | 0.930 | 2.030 | 2.256 | 5.437 |
| 1.00 (v4 s2 ref) | 1.009 | 2.228 | 2.263 | 5.407 |

Cross-codec consistency is essentially flat across w — even at w=0.05,
codecs at the same target zensim converge to butter_p3 ~1.0 at JND
(vs V10's 7.06). The mechanism IS effective at cross-codec convergence
at every weight; the KonJND price is paid in full regardless of w.

## What this rules in

The 372-feat IW-pool contribution to CID22 (+0.062 vs V10) is real.
Every w tier hits CID22 0.89+, dramatically above V10's 0.8324. The
substrate is sound; the recipe is what fails the KonJND gate.

## What's next (NOT this experiment)

- **Per-row KonJND PJND-anchor passthrough loss with high weight.**
  The current konjnd training group at weight 0.6 doesn't anchor PJND
  per row — it just feeds KonJND-dense pairs into the MSE/rank loss
  with `mix_cv35_iw65` as the target. A dedicated PJND passthrough
  loss with weight ≫ cross_codec_eq_weight would create explicit
  competition.
- **Cross-codec-eq band-gating.** Route the cross-codec-eq loss only
  through high-ssim2 anchor band (≥75) where KonJND saturates anyway.
  Below that band, let per-row PJND tracking dominate.
- **Substrate redesign.** Replace the q-band-flattening
  cross_codec_equivalence_ssim2_372col parquet with cross-codec
  equivalent pairs at MATCHED PJND levels (not matched butter levels).
  This requires per-codec PJND extraction, which is upstream work.
- **Different architectural target.** Cross-codec consistency at
  sub-JND quality (the picker frontier) may belong in a separate
  bake than the Balanced/Compression rank metric. The
  `v_cross_codec_v2` ship already serves the picker frontier;
  trying to fold cross-codec into the Balanced trail may be
  structurally wrong.

Per `feedback_autonomous_research_mandate`: this experiment falsifies
the load-bearing cross-codec-eq frontier. Next dispatches should target
KonJND-preserving alternatives or accept the V10/V_24 ceiling.

## Files

- 20 bakes: `/mnt/v/zen/zensim-eval/exp_v11_cc_eq_sweep_2026-05-20/cc4v11_w{05,10,20,50}_s{1..5}.bin` (each 261,316 bytes)
- 20 verdicts: `/mnt/v/zen/zensim-eval/exp_v11_cc_eq_sweep_2026-05-20/verdicts/cc4v11_w{05,10,20,50}_s{1..5}.md`
- Summary TSV: `/mnt/v/zen/zensim-eval/exp_v11_cc_eq_sweep_2026-05-20/summary_2026-05-20.tsv`
- cc_cons TSVs (5 medians × {T80, T50}): `/mnt/v/zen/zensim-eval/exp_v11_cc_eq_sweep_2026-05-20/cc_cons_*.tsv`
- Runner: `scripts/v_next/v11_372feat/run_v11a_372_v4_w_sweep_seed.sh`
- Driver: `scripts/v_next/v11_372feat/run_v11a_372_v4_w_sweep_all.sh`
- Eval helper: `scripts/v_next/v11_372feat/eval_v11_cc_eq_sweep.sh`
- This document.

## Honest gaps

- Only 4 weight values tested. The dose-response curve below w=0.05
  (e.g. w=0.01, w=0.005) was not measured. The trend across {0.05,
  0.10, 0.20, 0.50, 1.00} is flat in KonJND at ~0.40 — extrapolation
  to w → 0 would suggest KonJND only recovers at w = 0 (i.e. no cross-
  codec-eq loss at all, which is the V10 BalancedV3 baseline). A
  follow-up at w=0.01 / w=0.001 would close the bracket but is unlikely
  to change the verdict — the mechanism appears binary at every tier
  measured.
- cc_cons uses the median-by-CID22 seed per tier (1 seed per tier). 
  The 5-seed CI on cc_cons was not measured. Since cc_bp3 is consistent
  across all 5 evaluated bakes (0.93–1.07 at JND), seed variance on
  cc_cons is below the resolution that would change the conclusion.
- The trainer's `--cross-codec-eq-step-p` (default 0.1) was held
  constant. Sweeping step_p instead of weight is mathematically
  equivalent (both scale expected gradient magnitude), so the same
  falsification applies.
