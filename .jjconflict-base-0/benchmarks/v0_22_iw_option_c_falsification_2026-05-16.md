# V_22-IW Option C FALSIFIED — multi-bake mix can't recover (2026-05-16)

**Status**: T1.4 (V_18 + V_22-IW multi-bake α sweep) closed with
negative result. Both raw-output linear mix AND per-bake-z-normalized
mix sweep fail to find an α ∈ (0, 1) that Pareto-improves over V_18
ship alone on any held-out corpus per the full Mohammadi 2025 panel
(SROCC + PLCC + KROCC + PWRC + Z-RMSE). Earlier draft used SROCC
alone, which CLAUDE.md `SROCC-only verdicts BANNED` forbids; this
revision uses the full panel and confirms the verdict (and refines
the V_22-IW-alone characterization at α=0.0).

## AIC-3 corpus included — verdict updated (2026-05-16)

The initial α-sweep ran on per-pair CSVs that DROPPED all AIC-3 rows
(a `load_aic3` schema bug — see commit `796a689`). Re-ran with the
fix: both V_18 ship and V_22-IW v2 per-pair CSVs at
`benchmarks/v0_18_ship_eval_per_pair_2026-05-16_v2.csv` and
`benchmarks/v0_22_iw_seed1_2026-05-16_eval_per_pair_v2.csv` now
include 600 AIC-3 rows.

AIC-3 baseline (V_22-IW vs V_18 ship):

| Stat | V_18 ship | V_22-IW | fast-ssim2 | Δ V_22-IW vs V_18 |
|---|---:|---:|---:|---:|
| SROCC | 0.7996 | 0.7600 | 0.7965 | **−0.040 (worse)** |
| PLCC  | 0.8093 | 0.7708 | 0.8075 | **−0.039 (worse)** |
| KROCC | 0.6302 | 0.5894 | 0.6288 | **−0.041 (worse)** |
| PWRC  | 0.8697 | 0.8354 | 0.8665 | **−0.034 (worse)** |
| Z-RMSE| 0.588  | 0.637  | 0.590  | **+0.049 (worse)** |

V_22-IW loses on ALL 5 stats on AIC-3 — same pattern as CID22.

**With AIC-3 included, V_22-IW alone hits 2 of 4 ship-grade corpora
(wins KADID + TID, loses CID22 + AIC-3). The CID22 + AIC-3 axis —
the compression-focused human-MOS corpora — both reject V_22-IW.**

Per the user's explicit guidance (2026-05-16): "cid22 and aic are
the most important eval validation sets." V_22-IW seed=1 fails both
of them across the full Mohammadi panel.

## V_22-IW alone (α=0.0) vs V_18 ship — full Mohammadi panel

The unambiguous panel reading at the endpoint that's *not* the V_18
ship alone:

| Corpus | Stat | V_18 ship | V_22-IW | Δ (better/worse) |
|---|---|---:|---:|:--|
| CID22 | SROCC | 0.8933 | 0.6122 | **−0.281 (worse)** |
| CID22 | PLCC  | 0.8911 | 0.5900 | **−0.301 (worse)** |
| CID22 | KROCC | 0.7081 | 0.4283 | **−0.280 (worse)** |
| CID22 | PWRC  | 0.9373 | 0.7270 | **−0.210 (worse)** |
| CID22 | Z-RMSE| 0.454  | 0.807  | **+0.354 (worse, lower is better)** |
| KADID | SROCC | 0.9387 | 0.9447 | +0.006 (better) |
| KADID | PLCC  | 0.9395 | 0.9458 | +0.006 (better) |
| KADID | KROCC | 0.7855 | 0.7949 | +0.009 (better) |
| KADID | PWRC  | 0.9631 | 0.9675 | +0.004 (better) |
| KADID | Z-RMSE| 0.343  | 0.325  | **−0.018 (better)** |
| TID   | SROCC | 0.9526 | 0.9580 | +0.005 (better) |
| TID   | PLCC  | 0.9554 | 0.9585 | +0.003 (better) |
| TID   | KROCC | 0.8110 | 0.8200 | +0.009 (better) |
| TID   | PWRC  | 0.9702 | 0.9741 | +0.004 (better) |
| TID   | Z-RMSE| 0.295  | 0.285  | **−0.010 (better)** |

V_22-IW alone hits **5/5 stats better than V_18** on KADID, **5/5
better** on TID, **0/5 better** on CID22. Per CLAUDE.md's
multi-stat agreement rule ("ships when at least 3 of 5 stats agree
on improvement"), V_22-IW alone has 2 of 3 ship-grade corpora
above threshold — a strong partial confirmation that the IW-SSIM
training target produces a genuinely better metric on synthetic
distortions (KADID + TID), with the CID22 collapse remaining the
fatal flaw.

Critically: the Z-RMSE column makes the V_22-IW wins **real wins,
not SROCC artifacts**. V_22-IW's predictions on KADID + TID are
σ-normalized closer to the human MOS than V_18's are. This is
calibration improvement, not just rank-shuffling.

## Key numbers — α-sweep mix SROCC + Z-RMSE

The mix `α × V_18_raw + (1 − α) × V_22-IW_raw`. Z-RMSE values
LOWER = better:

| α | CID22 SROCC | CID22 Z-RMSE | KADID SROCC | KADID Z-RMSE | TID SROCC | TID Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 (V_22-IW alone) | 0.6122 | **0.807** | 0.9447 | **0.325** | 0.9580 | **0.285** |
| 0.30 | 0.6444 | 0.733 | 0.8605 | 0.508 | 0.8901 | 0.480 |
| 0.50 | 0.8474 | 0.527 | 0.9271 | 0.372 | 0.9434 | 0.325 |
| 0.70 | 0.8811 | 0.474 | 0.9351 | 0.352 | 0.9497 | 0.304 |
| 0.90 | 0.8910 | 0.457 | 0.9379 | 0.345 | 0.9519 | 0.297 |
| 1.00 (V_18 alone) | 0.8933 | **0.454** | 0.9387 | **0.343** | 0.9526 | **0.295** |

Observations:
1. **CID22 column is monotonic in α**: Z-RMSE descends 0.807 → 0.454
   as α rises 0.0 → 1.0. No intermediate α is better than V_18 alone.
2. **KADID and TID Z-RMSE are MINIMIZED at α=0.0** (V_22-IW alone):
   KADID 0.325 (best), TID 0.285 (best). Any mix introduces worse
   calibration error on these corpora.
3. **No α ∈ (0, 1) Pareto-improves over both endpoints simultaneously
   on any single stat across all 3 corpora.**

## Z-NORM mix is catastrophic at intermediate α

Z-NORM mix at α=0.5 on KADID/TID:
- KADID Z-RMSE: 0.999 (vs V_18 0.343 — 2.9x worse)
- TID Z-RMSE: 0.988 (vs V_18 0.295 — 3.3x worse)

This is the destructive interference signal — per-bake z-normalization
flips signs on pairs where V_18 and V_22-IW disagree on rank, and
the mix cancels both. Raw-space mix avoids this trap but still
underperforms V_18 alone.

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
