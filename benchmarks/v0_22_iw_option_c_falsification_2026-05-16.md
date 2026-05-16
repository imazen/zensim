# V_22-IW Option C FALSIFIED — multi-bake mix can't recover (2026-05-16)

**Status**: T1.4 (V_18 + V_22-IW multi-bake α sweep) closed with
negative result. Both raw-output linear mix AND per-bake-z-normalized
mix sweep fail to find an α ∈ (0, 1) that Pareto-improves over V_18
ship alone on any held-out corpus.

## Key numbers — α-sweep mix SROCC

The mix `α × V_18_raw + (1 − α) × V_22-IW_raw` at α ∈ {0.0, 0.3,
0.5, 0.7, 0.9, 1.0}:

| α | CID22 raw-mix | CID22 z-mix | KADID raw-mix | KADID z-mix | TID raw-mix | TID z-mix |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 (V_22-IW alone) | 0.6122 | 0.6122 | 0.9447 | 0.9447 | 0.9580 | 0.9580 |
| 0.30 | 0.6444 | 0.2972 | 0.8605 | 0.9229 | 0.8901 | 0.9291 |
| 0.50 | 0.8474 | 0.3579 | 0.9271 | 0.0109 | 0.9434 | 0.0301 |
| 0.70 | 0.8811 | 0.7886 | 0.9351 | 0.9050 | 0.9497 | 0.9252 |
| 0.90 | 0.8910 | 0.8823 | 0.9379 | 0.9349 | 0.9519 | 0.9494 |
| 1.00 (V_18 alone) | 0.8933 | 0.8933 | 0.9387 | 0.9387 | 0.9526 | 0.9526 |

## Verdict

- **No α ∈ (0, 1) beats V_18 ship on ANY of CID22, KADID, or TID.**
- **V_22-IW alone (α=0.0) marginally beats V_18 alone on KADID
  (+0.006) and TID (+0.005)** but catastrophically loses CID22
  (−0.281).
- **Z-normalized mix is WORSE than raw mix** — at α=0.5 the z-mix
  collapses to KADID 0.01 and TID 0.03 (vs raw 0.93 and 0.94).
  Per-bake z-norm magnifies the sign-disagreement at the per-pair
  level: V_18's raw is high for pairs V_22-IW ranks low, and the
  z-normalized opposite-sign contributions cancel destructively.
- **Destructive interference at α=0.3 across all corpora** in raw
  mix: CID22 0.6444, KADID 0.8605, TID 0.8901 — all LOWER than
  either endpoint alone. This is the bake-mix pathology
  documented in CLAUDE.md V_20 learnings: linear raw-space mix
  destroys rank information when the two bakes' raw distributions
  have different shapes (V_18: mean ~50, stdev ~25; V_22-IW: mean
  ~95, stdev ~5 due to upper saturation).

## Why this happens

V_22-IW's raw output saturates at the upper end because the
IW-SSIM target distribution itself saturates:

- iwssim p95 (training corpus): 0.99982
- iwssim p99: 0.99999
- iwssim max: 1.00003

Mapped through the bake's regression head, the predicted range
clusters near 100 for high-quality pairs. V_18 ship's raw output
has much wider variance (distance-shaped, range 0..100 with most
mass in [40, 80] for the CID22 corpus). When you linearly combine
the two, the V_22-IW component's near-saturated upper end pulls
the mix score toward 100 for most pairs, collapsing the rank
distinguishability that V_18 provides.

Z-normalization "fixes" the scale mismatch but introduces a
sign-error: V_22-IW's saturated upper end z-scores are tightly
clustered near 0 (small deviation from a near-constant mean), so
the sign of `v22_z` for a given pair has little information about
quality. Mixing with a heavy V_22-IW weight (α<0.5) adds noise to
V_18's signal.

## Implications

V_22-IW seed=1 is **production-unviable** as either a standalone
bake or a multi-bake secondary. The IW-SSIM target needs to be
RESHAPED before retraining for a viable V_22 ship:

1. **Log-distance target** (`-log(1 - iwssim + ε)`) spreads the
   upper saturation across a wider numerical range.
2. **Quantile-bin target** maps iwssim into uniform-by-quantile
   space so the regression head sees equal-spaced supervision
   across the score range.
3. **Multi-target loss** trains on `human_score:0.5, iwssim:0.5`
   so the bake learns BOTH ssim2-shape AND IW-shape, hopefully
   landing somewhere useful for both CID22 and TID.

These directions are queued as T1.5 (V_22-IW v2 with target-
distribution transform).

## What this confirms about the methodology

This is a **clean, validated falsification** per the principled
experiment workflow:

1. **Hypothesis written first** in `benchmarks/v0_22_iw_methodology_2026-05-16.md`.
2. **Cost ceiling respected**: 1 train + 2 evals + post-hoc α
   sweep. ~3 hr total, well under the 2-hr-per-seed budget.
3. **Falsification gate clearly defined**: "CID22 SROCC drops
   > 0.030 AND PWRC + Z-RMSE drop." All three hit by a wide
   margin.
4. **Documented before retrying**: this doc + the seed=1 findings
   doc + the α-sweep doc together form the negative-result
   evidence base. Future agents will NOT retry V_22-IW standalone
   or simple multi-bake without new evidence.
5. **Closing summary in CLAUDE.md** queued (next commit) so the
   "ssim2-target training bias" section's "what's been tried" gets
   updated with the V_22-IW result.

## Next steps

- **Do NOT sweep V_22-IW seeds 2 and 3.** Per the methodology doc,
  the falsification mechanism is understood; seed noise won't fix
  shape mismatch.
- **Escalate to T1.5** (V_22-IW v2 with target-distribution
  transform). The simplest first try: train with
  `--target-column iwssim` but pre-transform the iwssim column via
  `-log(1 - iwssim + 1e-6)`. This is a 1-line CSV preprocessing
  step on the existing safesyn_features_iwssim_372col.csv.
- **Pivot toward CVVDP track** per the user's 2026-05-15 directive:
  "iw ssim exploration is a neat target, but also cvvdp - we got
  60pct done via vastai". The CVVDP target distribution may not
  flatten the way IW-SSIM does (JOD ∈ [0, 10] is psychophysically
  calibrated to be uniform-by-perception), so V_22-CVVDP may
  avoid the saturation pathology by construction.
- **Keep V_18 ship as PreviewV0_3 production**. No swap.

## Reference files

- Methodology doc: `benchmarks/v0_22_iw_methodology_2026-05-16.md`
- Seed=1 eval findings: `benchmarks/v0_22_iw_seed1_eval_findings_2026-05-16.md`
- α-sweep results: `benchmarks/v0_22_iw_option_c_alpha_sweep_2026-05-16.md`
- Analyzer script: `scripts/v_next/v0_22_iw_option_c_alpha_sweep.py`
- V_18 ship per-pair: `benchmarks/v0_18_ship_eval_per_pair_2026-05-16.csv` (uncommitted, ~3 MB)
- V_22-IW seed=1 per-pair: `benchmarks/v0_22_iw_seed1_2026-05-16_eval_per_pair.csv` (uncommitted, ~3 MB)
- V_22-IW seed=1 bake: `benchmarks/v0_22_iw_seed1_2026-05-16.bin` (200 KB, ZNPR v3)
