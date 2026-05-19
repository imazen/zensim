# `PreviewV0_5TunerV2` from cc4v2 candidates — FALSIFIED on dynamic range (2026-05-19)

**Hypothesis (parent task):** The cc4v2 bakes from EXP-CROSS-CODEC-V2,
declared FALSIFIED on the rank trail by the prior agent, may pass
the **Tuner trail** gate per `CLAUDE.md` SOTA-trail definitions
(strict mono ≥ 1pp better than every V0_5 rank-trail ship on the JPEG
50-image × 19-q sweep, tied rate ≤ 5 %, dynamic range ≥ 50 score
units). The candidate that hit T=63 cross-codec `butter_max = 1.152`
/ `butter_p3 = 0.536` (cc4v2_s1_w2_0) is striking enough to merit a
direct measurement against the Tuner trail.

**Verdict: falsified.** All three cc4v2 candidates fail the Tuner
trail's dynamic-range gate by **50-500×**. Their cross-codec consistency
is achieved by collapsing to a constant function (~63 regardless of
codec or q), not by learning cross-codec invariant scoring.

## Per-bake measurement (`qsweep_eval` on
   `/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv`,
   50 imgs × 19 q values, all JPEG)

| Bake | strict_mono | tied_rate | q=5 median | q=95 median | range | range_gate (≥ 50) |
|---|---:|---:|---:|---:|---:|:---:|
| baseline_tuner (V_tuner-v2-s2, SHIP) | 0.9278 | 0.0044 | 4.96 | 94.64 | **89.68** | PASS |
| cc4v2_s1_w2_0 | **0.9533** | 0.0000 | 62.90 | 63.00 | **0.10** | FAIL by 500× |
| cc4v2_s2_w1_5 | 0.8689 | 0.0000 | 62.74 | 63.01 | **0.27** | FAIL by 185× |
| cc4v2_s3_w1_5 | 0.9400 | 0.0000 | 62.09 | 63.01 | **0.92** | FAIL by 54× |

Full per-q histograms and calibration RMSEs at
`/tmp/qsweep_cc4v2_2026-05-19.md` (snapshot below as evidence).

### Why the high monotonicity is misleading

cc4v2_s1_w2_0's 0.9533 strict-mono on the JPEG q-sweep looks like a
win until you see the per-q distribution. Every quality level from
q=10 to q=95 has p25 = p50 = p75 = 62.99 across all 50 images. The
"non-violation" 95 % is because adjacent q values produce essentially
identical outputs at floating-point noise level (63.00 vs 62.99
counts as monotonic). With dynamic range of 0.10 score units across
the full quality span, a user-facing dial built on this bake would
return q=95 (the max search ceiling) for **every** target above 63
and q=5 for every target below 63 — useless as a tuner.

### Why the cross-codec butter at T=63 looks impressive

`/mnt/v/output/zensim/cross_codec_metric_2026-05-19-v2/cc4v2_s1_w2_0/eval_t63_n20.tsv`
shows JPEG q ≈ 93-99, WebP q = 99, AVIF q ≈ 98 across all 20 images.
The binary search hits target=63 at near-max-q for every codec because
the bake's score barely changes between q=10 and q=95 — once it
saturates around 63, the bisection picks whatever q is at the search
ceiling. At near-max-q every modern codec is near-lossless, so
cross-codec butter is tiny by construction. **This is not cross-codec
consistency; it is "every codec at q=99 ≈ source pixels."**

## Why the W=2.0 training collapsed to a constant

From `cc4v2_s1_w2_0.log` (epochs 0-299):

- Epoch 0: val SROCC 0.83, α(x) μ=1.000 (pool head off).
- Epoch 10: val SROCC **0.94** (peak), α(x) collapsed to μ=0.015.
- Epoch 30: best val 0.9548 (the retained checkpoint).
- Epoch 70+: val swings from -0.59 to 0.96 with no stable trajectory.
- Final reducer_w = [μ=-0.172, σ=0.004, max=-0.002, p6=0.014],
  b_α = -10.7 (α(x) effectively 0 → pool head only).

The `--cross-codec-eq-weight 2.0` gradient drove the bake's outputs
into a degenerate state where it minimizes the cross-codec
equivalence loss by predicting the same value (~63, the synthetic
data's pjnd-anchor target) for every input. The pair-rank loss can
no longer escape this basin because the per-pair gradients are
small relative to the cross-codec-eq gradients on a much larger
pool (68,788 eq pairs × W=2.0 vs 50,000 per-epoch ranking pairs).

This matches the v1 ship findings (commit `6bef807`) that
`--cross-codec-eq-weight 2.0` collapses; the v2 substrate (tighter
gap ≤ 0.3, avif-2× row weight, 30 butter levels, 68,788 pairs) did
not change the W=2.0 collapse dynamics. s2_w1_5 and s3_w1_5 also
collapsed to near-constant on this measurement — earlier seed-1
runs at W=1.5 had stayed stable, so this is a seed-dependent
basin attractor, not a hyperparameter-stable region.

## Calibration RMSE per band (target = q)

| Band | n | baseline_tuner | cc4v2_s1_w2_0 |
|---|--:|---:|---:|
| B0 [0,10) | 50 | 19.06 | 57.87 |
| B5 [50,60) | 100 | 19.00 | 10.79 |
| B6 [60,70) | 100 | 12.81 | 2.55 |
| B9 [90,100) | 100 | 5.73 | 29.61 |

cc4v2_s1_w2_0 wins only at B5-B6 (the bake's collapsed-to-63 region)
and catastrophically loses at B0 and B9 — the same per-band signature
as the prior tuner-v2 anchor falsification at
`benchmarks/v_tuner_v2_falsification_2026-05-19.md`. Different mechanism
(`--cross-codec-eq-weight` instead of `--anchor-loss-weight`), same
structural failure: anchor-style losses collapse the output to a
point mass when their weight is loud enough to bind.

## Decision

**No `PreviewV0_5TunerV2` ship from any cc4v2 candidate.**
Today's `PreviewV0_5Tuner` (V_tuner-v2-s2 calibrated, range 89.68,
strict-mono 0.9278) remains the dial profile. The W=1.0 candidate
(cc4v2_s1_w1_0, the CURRENT PreviewV0_5CrossCodec ship per commit
`6bef807`) remains the cross-codec profile.

The cc4v2 line is closed for Tuner-trail consideration. The
underlying observation that prompted this task — "cc4v2_s1_w2_0
hit cross-codec T=63 butter_pnorm3 0.536" — is real but the
mechanism is "bake output is a constant", which is structurally
incompatible with the Tuner trail's purpose (binary-searchable
quality dial).

## Proposed next direction — rank-preservation regularizer (V3)

The cross-codec-eq + anchor-style losses keep collapsing the bake
to a point mass when weights are loud enough to bind cross-codec
consistency. A bake that achieves both **dynamic range ≥ 50**
AND **cross-codec T=63 butter_max < 2.5** needs an architectural
constraint that prevents the collapse rather than a softer hyperparameter
sweep. Candidates:

1. **Explicit rank-preservation hinge.** Add a loss term
   `Σ_(i,j) max(0, margin - (s_i - s_j) * sign(target_i - target_j))`
   on the existing pair-ranking corpus. As `--cross-codec-eq-weight`
   increases and the bake tries to flatten, the rank-preservation
   hinge pushes back proportionally — collapse cannot win if it
   violates rank for every pair-loss observation.
2. **Constrained dynamic-range training term.** Penalize bakes
   whose per-batch output std falls below a configurable threshold
   (e.g., σ_output ≥ 15 score units). This is a direct fix for the
   "collapse to constant" failure mode.
3. **Hybrid head with explicit mono loss.** The hybrid_head
   architecture (used in V_22-hybrid and others) gives the bake
   two parallel heads (rank + pool); allow the rank head to retain
   dynamic range while the pool head absorbs the cross-codec
   anchor force. Per-sample-α gating chooses which head wins per
   input; the constraint "α-rank head dominates when target_score
   is in tuner range" can be added as an auxiliary loss.

The trainer's existing `--monotonicity-reg` / `--monotonicity-margin`
flags (added 2026-05-18 for the original Tuner) are the closest
existing lever — they implement a hinge on `output(q+δ) ≥ output(q)`
for each safesyn (image, codec, q) curve. The original Tuner used
`--monotonicity-reg 1.0`, which produced range 89.68. A V3 cycle
should try `--monotonicity-reg 5.0` PLUS `--cross-codec-eq-weight 1.0`
(or 1.5), where the mono-reg's per-curve hinge prevents the
collapse that cross-codec-eq alone produces.

## Files produced this session

- `benchmarks/v_tuner_v2_cross_codec_2026-05-19_falsification.md` (this doc)
- `/tmp/qsweep_cc4v2_2026-05-19.md` (full per-bake report)

## Reproduction

```bash
cd ~/work/zen/zensim--cross-codec-metric
cargo build --release --bin qsweep_eval -p zensim-validate
./target/release/qsweep_eval \
  --features /mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv \
  --manifest /mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv \
  --bake baseline_tuner=zensim/weights/v_tuner_2026-05-18.bin:clamp \
  --bake cc4v2_s1_w2_0=/mnt/v/zen/zensim-eval/exp_cross_codec_v2_2026-05-19/cc4v2_s1_w2_0.bin:clamp \
  --bake cc4v2_s2_w1_5=/mnt/v/zen/zensim-eval/exp_cross_codec_v2_2026-05-19/cc4v2_s2_w1_5.bin:clamp \
  --bake cc4v2_s3_w1_5=/mnt/v/zen/zensim-eval/exp_cross_codec_v2_2026-05-19/cc4v2_s3_w1_5.bin:clamp \
  --out /tmp/qsweep_cc4v2_2026-05-19.md
```

Wall time: ~3 seconds (50 imgs × 19 q × 4 bakes = 3,800 372-feature
predictions).
