# V46: first --monotone-cbc bake — STRUCTURAL WIN, PANEL LOSS

## Bake
- File: `/mnt/v/output/zensim/bakes/v46_monotone_cbc_real_recipe_seed17_2026-05-26.bin` (261451 bytes)
- Recipe: `scripts/v_next/run_v46_monotone_cbc_on_real_recipe_2026-05-26.sh`
- Trainer commits: fa5c699 + bf92de5 + 26d6a1d (OOM fix)
- Date: 2026-05-26

## Verdict vs V39 (shipped Profile::A)

| Corpus | V46 SROCC | V39 SROCC | Δ |
|---|---:|---:|---:|
| CID22 | 0.7688 | 0.8793 | **−0.110** |
| KADID | 0.7569 | 0.9251 | **−0.168** |
| TID | 0.7943 | 0.9317 | **−0.137** |
| KonJND | 0.4253 | 0.4197 | +0.006 |
| AIC-3 | 0.6886 | 0.8023 | **−0.114** |
| AIC-4 | 0.7718 | 0.9051 | **−0.133** |

| G | V46 | V39 |
|---|---:|---:|
| G1 dial (p5..p95) | −4.1..69.9 (range 74) | −89.7..97.4 (range 187) |
| G7 CID22 ≥0.85 | **FAIL (0.77)** | PASS (0.88) |
| G8 Z-RMSE AIC-3 ≤0.80 | soft (0.74) | PASS (0.58) |
| G9 DS-AUC AIC-3 ≥0.70 | **FAIL (0.59)** | PASS (0.74) |
| Goals weighted | 0.256 | 0.714 |

## What this measures

V46 trains the same trainer with `--monotone-cbc` enabled but DOES
NOT replicate V32's hyperparams point-by-point. It uses:
- 1-layer (n_hidden=128 → 1) instead of V32's 2-layer (128 → 64)
- `--target-column mix_cv40_iw60` (single consistent metric) instead
  of V32's per-group-normalized `human_score`
- `--tanh-output-head-scale 20.0` instead of V32's 30.0
- NO auto-transforms (V32 uses YJ screen at
  `benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv`)
- 2 groups (safesyn + cid22_train) instead of V32's 5 groups +
  konjnd-dense

So the −0.10 to −0.17 SROCC delta vs V39 is a JOINT effect of (i) the
monotone-cbc constraint AND (ii) recipe differences. V46b (in flight)
isolates the constraint by matching V32's hyperparams point-by-point
+ enabling `--monotone-cbc`. The V46b ↔ V39 delta is the pure cost
of the constraint.

## Decision

V46 does NOT displace V39 at `ZensimProfile::A`. V39's shipping
known-limits (off-manifold inversion on OOD synthetic content,
characterized in `tests/metric_invariants.rs::v39_known_limit_violations`)
remain accepted.

## Next iteration paths

1. V46b: match V32 hyperparams exactly + `--monotone-cbc` (in flight).
   If V46b matches V39 within ~0.02 SROCC, the constraint is cheap
   and 2-layer + auto-transforms recover what 1-layer + no-transforms
   sacrificed; we can ship V46b as Profile::A_Strict (sibling profile,
   not a replacement).
2. If V46b still drops > 0.05 SROCC, soften the constraint:
   - softplus reparam `w = softplus(raw_w)` instead of hard ≥0 clamp;
     keeps weights smoothly positive but allows asymptotic 0 (a "dead"
     feature can effectively be excluded without dragging the loss).
   - Per-feature sign constraints from a learned sign mask, instead
     of global encoder ≥0.
   - Monotonicity only on the SSIM-derived feature subset (first
     dozen features), not all 372.
3. If softening doesn't recover SROCC, the V46/V46b numbers
   characterize a Pareto limit: pure-structural-monotonicity vs
   held-out-MOS-rank trade.

The trainer machinery (`--monotone-cbc`) and the synthetic-data
correctness test (`zensim-validate/tests/monotone_cbc_projection.rs`)
are correct + landed regardless of which iteration ultimately ships.
