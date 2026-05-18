# V_24 PJND-aware pair weighting — methodology + falsified results

**Date:** 2026-05-18
**Branch:** `feat/ex2-stdpool-head` (commit `qxzlowwu`)
**Status:** **FALSIFIED across both variants × 5-seed CI.** Hypothesis dead.

## TL;DR

Both variants catastrophically break KonJND learning (SROCC drops
from ~0.82-0.89 → 0.03-0.15 across 10 seeds total). CID22 actually
improves (+0.045 to +0.059 vs respective baseline) because the
trainer reallocates capacity AWAY from konjnd when boundary-only
weighting starves konjnd of useful gradient.

| Variant | KonJND SROCC mean±std | Δ vs baseline | bake_compare verdict |
|---|---|---|---|
| (a) per-sample-α + PJND | **0.1440±0.0082** | **−0.6784** | B>>A on KonJND, overall B 5-3 |
| (b) V_22-recipe + PJND | **0.0809±0.0513** | **−0.8118** | B>>A on KonJND/KADID/TID, overall B 8-4 |

**The boundary-pair hypothesis is wrong**: same-side ranking pairs
(both below threshold, or both above) carry KonJND ranking signal
that the gaussian-product weighting throws away. Without those,
the model can't learn the within-cluster ordering even though it
gets unlimited gradient on the cluster-boundary cases.

## Hypothesis

The user picked PJND-aware pair weighting as the next lever after
finding that weight-knob `konjnd_w` boost and per-sample α both
sit on the same CID22↔KonJND tradeoff axis (no Pareto expansion).
This lever weights **WHICH PAIRS** within each group contribute
gradient — orthogonal to "which group" (weight-knob) or "which
prediction path" (per-sample α).

**Hypothesis text (verbatim from user spec)**:

> KonJND learning is gradient-starved on the PJND boundary — most
> konjnd pairs are either clearly different (large score gap) or
> clearly same (PJND threshold), but the pairs at the JND boundary
> (where one image is just-above-threshold and another
> just-below) carry the actual JND signal. Boosting weight on
> those boundary pairs SPECIFICALLY should lift KonJND without
> warping the encoder for the whole konjnd group.

**Falsification**: KonJND SROCC fails to improve (within 5-seed CI)
vs the same recipe without PJND-weighting. If the boundary-pair
hypothesis is wrong, KonJND will be flat-or-worse.

## Per-pair weight definition

For konjnd group only, per-pair weight is a gaussian product on
(midpoint, gap):

```
mid    = 0.5 * (mos_a + mos_b)
gap    = |mos_a - mos_b|
w_mid  = exp(-((mid - threshold) / sigma_mid)^2)
w_gap  = exp(-((gap - gap_anchor) / sigma_gap)^2)
weight = (w_mid * w_gap) / Z
```

`Z` = empirical mean of `w_mid * w_gap` under uniform pair sampling
on the konjnd corpus. Normalization preserves the unweighted
gradient scale in expectation.

The factor multiplies into the existing PWRC `pair_weight` at every
gradient site (K=1 sequential, K>1 parallel-batch, NiN batch,
per-sample-α). For non-konjnd groups, the factor is 1.0 (no-op).

## Hyperparameters (initial config)

Empirically chosen from the konjnd corpus's bimodal cluster centers:

| Param | Value | Rationale |
|---|---|---|
| `threshold` | 45.0 | Midpoint of bimodal cluster centers (~31 below, ~58 above) — empirical PJND boundary in MOS space |
| `sigma_mid` | 8.0 | Tight gaussian around threshold |
| `gap_anchor` | 27.0 | 58 − 31 = typical straddling-pair gap |
| `sigma_gap` | 10.0 | Tight gaussian around anchor |
| `Z` | 0.329719 | Empirical mean, computed by build script |

Diagnostic builder:
`scripts/v_next/build_pjnd_pair_weights.py` materializes the
per-pair weight parquet for sanity check + reports the
distribution.

**Weight distribution sanity** (1,008-row konjnd corpus, 507,528
upper-triangle pairs):

```
p10:  0.0001    (heavily downweighted: same-side pairs far from boundary)
p25:  0.0002
p50:  0.1591
p75:  2.2628    (boundary-straddling pairs upweighted ~2x)
p90:  2.8210
p99:  3.0132
fraction w > 0.5: 0.4529
fraction w > 1.0: 0.4019
fraction w > 2.0: 0.2886
fraction w < 0.1: 0.4808
```

Sanity check: 48% of pairs receive negligible weight (both refs on
same side of threshold), 40% receive boost > 1.0 (straddling pairs).
Per-group total preserved by Z normalization (E[w | uniform] = 1.0).

## Recipes tested

Two variants, identical 5-group corpus, different head architectures:

### Variant (a): per-sample-α + PJND weighting

Mirror the V_24 per-sample-α recipe at
`scripts/v_next/run_per_sample_alpha_seed.sh` with `--pjnd-aware-pair-weighting`
added. Script:
`scripts/v_next/run_pjnd_pairweight_persample_seed.sh`.

Trainer flags: `--per-sample-alpha-head --norm-in-norm-weight 0.1
--pjnd-aware-pair-weighting --pjnd-threshold 45 --pjnd-sigma-mid 8
--pjnd-gap-anchor 27 --pjnd-sigma-gap 10 --pjnd-normalization-z 0.329719`.

### Variant (b): V_22-mix-LARGE recipe + PJND weighting

Same 5-group corpus, NO per-sample-α (legacy NiN K=256 head).
Isolates the pair-weighting effect from the per-sample-α
architecture. Script:
`scripts/v_next/run_pjnd_pairweight_v22recipe_seed.sh`.

## Inputs

- Branch base: `feat/ex2-stdpool-head` (has per-sample α + finetune infra).
- Per-sample-α baselines:
  `/mnt/v/zen/zensim-eval/v24_persample_alpha_2026-05-18/persample_seed{1..5}.bin`
- V_22-mix-LARGE baseline:
  `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed.bin`
- konjnd parquet (used for PJND weight diagnostic + training):
  `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/konjnd_mix_300col.parquet`
  (1,008 rows, score range [22, 70], bimodal at ~31/~58)
- Workspace:
  `/home/lilith/work/zen/zensim--pjnd-pairweighting`

## Pareto gate (stricter than prior experiments)

| Corpus | Threshold vs V_22-mix-LARGE baseline |
|---|---|
| CID22 SROCC | ≥ baseline + 0.000 (no regression) |
| KonJND SROCC | ≥ baseline − 0.005 (close the gap) |
| AIC-3 SROCC | ≥ baseline + 0.015 (preserve breakthrough) |
| KADID/TID SROCC | ≥ baseline − 0.04 (integrity guards) |

Stricter because we're testing whether a TARGETED lever can move
just KonJND without collateral damage to photo corpora.

## 5-seed CI results

### Per-corpus aggregate SROCC (mean ± std, n=5)

| Corpus | per-sample-α baseline | V_22-LARGE baseline | Var (a) per-sample-α + PJND | Var (b) V_22 + PJND |
|---|---|---|---|---|
| CID22 | 0.8548 | 0.8324 | **0.8780±0.0107** | **0.8854±0.0063** |
| KADID-10k | 0.9319 | 0.9677 | 0.9215±0.0083 | 0.9194±0.0061 |
| TID2013 | 0.8934 | 0.9729 | 0.8837±0.0045 | 0.8808±0.0096 |
| **KonJND-1k** | **0.8224** | **0.8927** | **0.1440±0.0082** | **0.0809±0.0513** |
| AIC-3 CTC | 0.8109 | 0.7845 | 0.8141±0.0040 | 0.8110±0.0022 |

### Per-seed KonJND SROCC (the load-bearing metric)

| seed | Var (a) | Var (b) |
|---|---|---|
| 1 | 0.1418 | 0.0279 |
| 2 | 0.1368 | 0.1462 |
| 3 | 0.1537 | 0.0440 |
| 4 | 0.1515 | 0.0631 |
| 5 | 0.1363 | 0.1231 |

KonJND falls from baseline 0.82-0.89 to 0.03-0.15 in EVERY seed, with
unusually low seed variance (0.008 std for variant a) — the
collapse is reproducible, not stochastic.

### α(x) distribution diagnostic (variant a only)

Per-sample α head learned values are recorded in the logs:

| Corpus | α mean | α p05 | α p50 | α p95 |
|---|---|---|---|---|
| CID22 | 0.774 | 0.514 | 0.809 | 0.923 |
| KADID-10k | 0.273 | 0.007 | 0.216 | 0.809 |
| TID2013 | 0.311 | 0.035 | 0.269 | 0.746 |
| KonJND-1k | 0.756 | 0.584 | 0.771 | 0.883 |
| AIC-3 CTC | 0.702 | 0.461 | 0.723 | 0.808 |

(Numbers from `pjnd_persample_seed1` — α distribution within
±0.05 of the baseline per-sample-α run, so the per-sample-α
mechanism is NOT what's breaking KonJND.)

### Pareto gate result

| Gate | Var (a) | Var (b) | Pass? |
|---|---|---|---|
| CID22 SROCC ≥ baseline | +0.0232 | +0.0530 | YES |
| KonJND SROCC ≥ baseline − 0.005 | −0.6784 | −0.8118 | **NO** (catastrophic) |
| AIC-3 SROCC ≥ baseline + 0.015 | +0.0032 | +0.0265 | (a) NO, (b) YES |
| KADID SROCC ≥ baseline − 0.04 | −0.0104 | −0.0483 | (a) YES, (b) NO |
| TID SROCC ≥ baseline − 0.04 | −0.0097 | −0.0921 | (a) YES, (b) NO |

**Both variants fail the Pareto gate**. The KonJND collapse is
the load-bearing failure; the pair-weighting lever does not
expand the Pareto frontier.

### bake_compare decisive verdicts

#### Variant (a): `pjnd_persample_seed1` vs `persample_seed1` baseline

```
Cross-corpus aggregate:
| Corpus    | A SROCC | B SROCC | Aggregate |
| CID22     | 0.8767  | 0.8548  | A>>B      |
| KADID     | 0.9282  | 0.9319  | B>>A      |
| TID       | 0.8892  | 0.8934  | promising |
| KonJND    | 0.1418  | 0.8224  | B>>A      |

Decisive cells: A wins 3, B wins 5
Overall winner: B
```

#### Variant (b): `pjnd_v22recipe_seed1` vs V_22-LARGE baseline

```
Cross-corpus aggregate:
| Corpus    | A SROCC | B SROCC | Aggregate |
| CID22     | 0.8914  | 0.8324  | A>>B      |
| KADID     | 0.9136  | 0.9677  | B>>A      |
| TID       | 0.8769  | 0.9729  | B>>A      |
| KonJND    | 0.0279  | 0.8927  | B>>A      |
| AIC-3     | 0.8097  | 0.7845  | A>>B      |

Decisive cells: A wins 4, B wins 8
Overall winner: B
```

### Falsification statement

The PJND-aware pair-weighting hypothesis is **dead** under the
tested gaussian-product configuration (threshold=45, sigma_mid=8,
gap_anchor=27, sigma_gap=10). The boundary-pair signal alone is
insufficient for KonJND — same-side pairs carry information the
trainer needs to rank within-cluster, and zeroing those out
across 48% of the pair space destroys the within-cluster
calibration.

**Two seed observations strengthen the falsification**:
1. **Identical reproducibility across architectures**: variant (a)
   per-sample-α and variant (b) legacy NiN head both fail in
   exactly the same direction with similar magnitude. The
   failure is in the gradient signal, not the head architecture.
2. **Trainer reallocates to photo corpora**: CID22 +0.02 to +0.05
   on both variants confirms the konjnd training time *is*
   being redirected — but the boundary pairs can't substitute
   for the cluster-internal rankings the encoder needs.

### Did NOT investigate (parked)

- **Gentler sigmas** (sm=20, sg=25 → no zero-weight pairs):
  diagnostic shows this would preserve same-side pair gradient
  and only mildly upweight boundary pairs. The
  `build_pjnd_pair_weights.py --sigma-mid 20 --sigma-gap 25`
  pre-check shows 51% of pairs in w > 1.0 (smoothly upweighted)
  and 0% with w < 0.1. This is essentially a "no-op" sweep —
  if it works, it shows the hypothesis was the right idea but
  needed less aggressive weighting. **However**, the user
  explicitly asked for the spec'd config (sm=8, sg=10) — the
  decisive falsification under those hyperparams stands.
  Parking the gentler sweep for a future session.

## Implementation notes

- New type `PjndPairWeightingConfig` in
  `zensim-validate/src/mlp_train.rs`. Resolved per-trainer-fn from
  group name → group index at training startup; passed through
  helpers via `Option<&PjndPairWeightingConfig>`.
- Multiplicative composition with PWRC: `pair_weight = pwrc_w · pjnd_w`.
- Non-target groups receive factor 1.0 (no-op).
- 5 gradient sites updated: K=1 sequential, K>1 parallel, NiN K=256
  buffer (per-sample-α path), NiN K=256 buffer (legacy path),
  parallel-batch path.
- 7 new CLI flags on `zensim_mlp_train`: `--pjnd-aware-pair-weighting`,
  `--pjnd-target-group`, `--pjnd-threshold`, `--pjnd-sigma-mid`,
  `--pjnd-gap-anchor`, `--pjnd-sigma-gap`, `--pjnd-normalization-z`.
