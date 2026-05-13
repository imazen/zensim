# Cycle-9b RankNet-pair-boost experiments — outcomes (2026-05-13)

## Summary

Follow-on to cycle-9. Cycle-9 (row-weight MSE boost) was conjectured
to fail because the boost only affected MSE loss term — RankNet pair
sampling was unweighted and carries most of the rank-correlation
signal. Cycle-9b tests the same hypothesis at the right loss term:
weight each RankNet pair's softplus loss by the boost factor of its
endpoints.

**Result: FALSIFIED.** 6-seed sweep at pair-boost 2.0 shows:
- CID22 mean +0.0026 (not significant, p=0.47)
- AIC-4 mean **-0.0043** (significant negative, p=0.02)

Pair-boost is a real lever but trades AIC-4 SROCC for tiny CID22
gain. Net negative. Pair-boost 2.0 seed=1 was an upper-tail
outlier (CID22 0.8687 = +1.1σ above 6-seed mean of 0.8629).

## Trainer change

Added flag `--low-q-pair-boost` and modified `ranknet_loss()` in
`scripts/v_next/train_v_next_mlp.py` (zensim commit `a700b10f`):

```python
# In ranknet_loss(), after computing softplus losses:
if low_q_pair_boost == 1.0:
    return losses.mean()
sqrt_boost = float(low_q_pair_boost) ** 0.5
ti = target[i_nz]
tj = target[j_nz]
boost_i = torch.where(ti < 50.0, low_q_pair_boost,
            torch.where(ti < 65.0, sqrt_boost, 1.0))
boost_j = torch.where(tj < 50.0, low_q_pair_boost,
            torch.where(tj < 65.0, sqrt_boost, 1.0))
weights = torch.maximum(boost_i, boost_j)
return (losses * weights).sum() / weights.sum()
```

## Experimental sweep

**6-seed sweep at pair-boost 2.0** (compared to V0_31+V0_32 baseline,
n=2 at boost 1.0):

| Seed | CID22 | AIC-4 |
|--:|--:|--:|
| 1 | **0.8687** | 0.9119 |
| 2 | 0.8665 | 0.9184 |
| 3 | 0.8646 | 0.9117 |
| 7 | 0.8567 | 0.9099 |
| 42 | 0.8556 | 0.9138 |
| 100 | 0.8654 | 0.9166 |
| **Mean (n=6)** | **0.8629** | **0.9137** |
| Std | 0.0054 | 0.0032 |

**Also evaluated**: pair-boost 5.0 seed=1 → CID22 0.8640 / AIC-4
0.9113. Smaller CID22 gain (+0.0012 vs V0_31), similar AIC-4 cost.

## Statistical analysis

Welch's t-test, pair-boost 2.0 (n=6) vs baseline V0_31/V0_32 (n=2):

| Metric | Baseline mean | Pair-boost mean | Δ | t | df | p |
|---|--:|--:|--:|--:|--:|--:|
| CID22 | 0.8603 | 0.8629 | +0.0026 | 0.79 | ~4 | 0.47 |
| AIC-4 | 0.9180 | 0.9137 | -0.0043 | -3.12 | ~6 | **0.02** |

**CID22**: not significantly different from baseline.
**AIC-4**: significantly LOWER with pair-boost.

The baseline n=2 is the weakest part of the comparison; with only
2 baseline seeds, the boost-1.0 distribution mean is poorly
estimated. But the direction (CID22 gain n.s., AIC-4 loss sig)
is clear regardless of baseline width.

## Cycle-9b verdict

**Pair-resampling boost is a real lever, but the trade is net
negative**: tiny CID22 gain (within noise) traded for measurable
AIC-4 loss. Not a Pareto improvement.

This refines the cycle-9 conjecture but doesn't crack the
CID22 ceiling either.

## Combined cycle-9 + cycle-9b verdict

| Lever | Seed-mean Δ CID22 | Seed-mean Δ AIC-4 | Stat sig? | Ship? |
|---|--:|--:|---|---|
| Row-weight boost 1.5 | -0.0006 | -0.0058 | trends neg | NO |
| Pair-resampling boost 2.0 | +0.0026 | -0.0043 | AIC-4 sig neg | NO |
| Pair-resampling boost 5.0 (1 seed) | +0.0012 | -0.0063 | — | NO |

**Neither boost variant breaks V0_16 SHIP's CID22 ceiling.** Both
require multi-seed confirmation; both fail it.

## Lessons learned (additive to cycle-9 doc)

1. **The "right loss term" argument was correct** but doesn't
   suffice. Pair-boost has a real effect (CID22 mean Δ +0.0026 vs
   row-boost mean Δ -0.0006, ~1.5σ different). But the effect
   isn't large enough to matter.

2. **Single-seed traps repeat.** Just as V0_34 (row-boost seed=1)
   appeared to dominate at seed=1, pair-boost 2.0 seed=1 also
   appeared to dominate. The seed-1 lottery isn't systematic luck
   — it's that anyone who runs seed=1 first sees attention-grabbing
   single-seed numbers. Cure: ALWAYS run ≥3 seeds before declaring
   a new lever works.

3. **Cross-corpus tradeoffs are usually real even if same-corpus
   gains are noise.** The AIC-4 -0.0043 was statistically real;
   the CID22 +0.0026 was not. When one moves significantly in a
   negative direction while the other moves insignificantly, the
   change is probably a net loss in disguise.

## Cycle-10 strategic options (carried forward unchanged)

| Option | Cost | Risk | Cycle |
|---|---|---|---|
| Data axis: JPEG-AI public corpus | Bandwidth + setup + train | Medium | 10a |
| Architecture axis: 300-feat input | Multi-tick infra work | Medium | 10b |
| Concordance filter: ssim2_butter | **Blocked** by missing data columns in synth CSV | — | 10c |

10c is blocked unless we preprocess the synth CSV to add
`score_ssim2` + `score_butteraugli_max` columns. The unified
parquets have these but they predate the 2026-05-12 CID22 purge
— would need a join + filter pass.

## Artifacts

Bakes:
- `/tmp/zensim_loop/bakes/v0_pairboost2p0_seed{1,2,3,7,42,100}_2026-05-13.bin` (6)
- `/tmp/zensim_loop/bakes/v0_pairboost5p0_seed1_2026-05-13.bin` (1)

Per-pair CSVs + eval logs: `/tmp/zensim_loop/v0_pairboost*_per_pair.csv`,
`v0_pairboost*_eval.log`

Run dirs: `/mnt/v/zen/zensim-training/2026-05-07/runs/*v0_pairboost2p0*`

Trainer change: zensim commit `a700b10f`.

Tick log entries: 513, 514 in
`~/work/zen/zenanalyze/zensim_champion_log.md`.

## Cycle status (overall)

| Cycle | Lever | Verdict |
|---|---|---|
| 7 | dssim co-training (V0_27) | FALSIFIED |
| 7 | cosine LR (V0_28) | FALSIFIED |
| 7 | smaller LR (V0_29) | FALSIFIED (underconverged) |
| 8 | KonJND-weight Pareto (V0_30/V0_31/V0_32/V0_26) | PARTIAL (V0_31 wins AIC-4) |
| 9 | Low-q row-weight boost (5 seeds at 1.5) | FALSIFIED |
| 9b | Low-q pair-resampling boost (6 seeds at 2.0) | FALSIFIED |

**V0_16 SHIP unchanged.** V0_26 (cycle-7) and V0_31 (cycle-8)
preserved as live-site alternatives. Cycle-10 needs user direction
on data axis or architecture axis (10c blocked).
