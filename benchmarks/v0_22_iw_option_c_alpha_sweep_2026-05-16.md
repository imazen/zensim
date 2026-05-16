# V_22-IW Option C α-sweep — V_18 ship × V_22-IW seed=1 multi-bake (2026-05-16)

Post-hoc analysis of `α × V_18_raw + (1−α) × V_22-IW_raw` per-pair mix
at α ∈ 0.00, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00.

**Inputs**:
- V_18 ship per-pair: `/home/lilith/work/zen/zensim/benchmarks/v0_18_ship_eval_per_pair_2026-05-16.csv`
- V_22-IW seed=1 per-pair: `/home/lilith/work/zen/zensim/benchmarks/v0_22_iw_seed1_2026-05-16_eval_per_pair.csv`

**Joined rows**: 17,417 across 3 corpora.

## CID22 (n = 4,292)

Baselines (no mix): V_0_2 = 0.8676, V_18 ship = **0.8933**, V_22-IW = **0.6122**, fast-ssim2 = 0.8895, butter = 0.7412.

| α (V_18 weight) | SROCC raw-mix | SROCC z-mix | Δ raw vs V_18 | Δ z vs V_18 |
|---|---:|---:|---:|---:|
| 0.00 (= V_22-IW alone) | 0.6122 | 0.6122 | -0.2812 | -0.2812 |
| 0.30 | 0.6444 | 0.2972 | -0.2489 | -0.5961 |
| 0.50 | 0.8474 | 0.3579 | -0.0459 | -0.5355 |
| 0.60 | 0.8699 | 0.6344 | -0.0234 | -0.2590 |
| 0.70 | 0.8811 | 0.7886 | -0.0123 | -0.1048 |
| 0.80 | 0.8873 | 0.8557 | -0.0060 | -0.0377 |
| 0.90 | 0.8910 | 0.8823 | -0.0023 | -0.0111 |
| 0.95 | 0.8923 | 0.8890 | -0.0011 | -0.0043 |
| 1.00 (= V_18 alone) | 0.8933 | 0.8933 | +0.0000 | +0.0000 |

## KADIK10k (n = 9,805)

Baselines (no mix): V_0_2 = 0.8033, V_18 ship = **0.9387**, V_22-IW = **0.9447**, fast-ssim2 = 0.7967, butter = 0.5702.

| α (V_18 weight) | SROCC raw-mix | SROCC z-mix | Δ raw vs V_18 | Δ z vs V_18 |
|---|---:|---:|---:|---:|
| 0.00 (= V_22-IW alone) | 0.9447 | 0.9447 | +0.0060 | +0.0060 |
| 0.30 | 0.8605 | 0.9229 | -0.0781 | -0.0157 |
| 0.50 | 0.9271 | 0.0109 | -0.0116 | -0.9278 |
| 0.60 | 0.9324 | 0.8046 | -0.0063 | -0.1340 |
| 0.70 | 0.9351 | 0.9050 | -0.0035 | -0.0337 |
| 0.80 | 0.9368 | 0.9269 | -0.0019 | -0.0117 |
| 0.90 | 0.9379 | 0.9349 | -0.0008 | -0.0037 |
| 0.95 | 0.9383 | 0.9371 | -0.0004 | -0.0016 |
| 1.00 (= V_18 alone) | 0.9387 | 0.9387 | +0.0000 | +0.0000 |

## TID2013 (n = 3,000)

Baselines (no mix): V_0_2 = 0.8427, V_18 ship = **0.9526**, V_22-IW = **0.9580**, fast-ssim2 = 0.8460, butter = 0.6696.

| α (V_18 weight) | SROCC raw-mix | SROCC z-mix | Δ raw vs V_18 | Δ z vs V_18 |
|---|---:|---:|---:|---:|
| 0.00 (= V_22-IW alone) | 0.9580 | 0.9580 | +0.0055 | +0.0055 |
| 0.30 | 0.8901 | 0.9291 | -0.0624 | -0.0234 |
| 0.50 | 0.9434 | 0.0301 | -0.0092 | -0.9224 |
| 0.60 | 0.9475 | 0.8305 | -0.0051 | -0.1221 |
| 0.70 | 0.9497 | 0.9252 | -0.0029 | -0.0274 |
| 0.80 | 0.9510 | 0.9429 | -0.0015 | -0.0096 |
| 0.90 | 0.9519 | 0.9494 | -0.0007 | -0.0031 |
| 0.95 | 0.9523 | 0.9512 | -0.0003 | -0.0013 |
| 1.00 (= V_18 alone) | 0.9526 | 0.9526 | +0.0000 | +0.0000 |

## Cross-corpus Pareto picks (RAW-output mix)

For each α, raw-space mix SROCC per corpus.

| α | CID22 | KADIK10k | TID2013 |
|---|---:|---:|---:|
| 0.00 | 0.6122 | 0.9447 | 0.9580 |
| 0.30 | 0.6444 | 0.8605 | 0.8901 |
| 0.50 | 0.8474 | 0.9271 | 0.9434 |
| 0.60 | 0.8699 | 0.9324 | 0.9475 |
| 0.70 | 0.8811 | 0.9351 | 0.9497 |
| 0.80 | 0.8873 | 0.9368 | 0.9510 |
| 0.90 | 0.8910 | 0.9379 | 0.9519 |
| 0.95 | 0.8923 | 0.9383 | 0.9523 |
| 1.00 | 0.8933 | 0.9387 | 0.9526 |

## Cross-corpus Pareto picks (Z-NORMALIZED mix)

Per-bake z-normalization before mix — the offline `ensemble_mix`
tool's approach. Removes scale-mismatch between V_18 (mean ~50,
stdev ~25) and V_22-IW (mean ~95, stdev ~5 due to upper saturation).

| α | CID22 | KADIK10k | TID2013 |
|---|---:|---:|---:|
| 0.00 | 0.6122 | 0.9447 | 0.9580 |
| 0.30 | 0.2972 | 0.9229 | 0.9291 |
| 0.50 | 0.3579 | 0.0109 | 0.0301 |
| 0.60 | 0.6344 | 0.8046 | 0.8305 |
| 0.70 | 0.7886 | 0.9050 | 0.9252 |
| 0.80 | 0.8557 | 0.9269 | 0.9429 |
| 0.90 | 0.8823 | 0.9349 | 0.9494 |
| 0.95 | 0.8890 | 0.9371 | 0.9512 |
| 1.00 | 0.8933 | 0.9387 | 0.9526 |

## Decision aid (RAW-mix): how many corpora does each α beat fast-ssim2 on?

| α | wins vs ssim2 | total | corpora won |
|---|--:|--:|---|
| 0.00 | 2 | 3 | KADIK10k, TID2013 |
| 0.30 | 2 | 3 | KADIK10k, TID2013 |
| 0.50 | 2 | 3 | KADIK10k, TID2013 |
| 0.60 | 2 | 3 | KADIK10k, TID2013 |
| 0.70 | 2 | 3 | KADIK10k, TID2013 |
| 0.80 | 2 | 3 | KADIK10k, TID2013 |
| 0.90 | 3 | 3 | CID22, KADIK10k, TID2013 |
| 0.95 | 3 | 3 | CID22, KADIK10k, TID2013 |
| 1.00 | 3 | 3 | CID22, KADIK10k, TID2013 |

## Decision aid (RAW-mix): how many corpora does each α beat V_18 ship on?

| α | wins vs V_18 | total | corpora won |
|---|--:|--:|---|
| 0.00 | 2 | 3 | KADIK10k, TID2013 |
| 0.30 | 0 | 3 | — |
| 0.50 | 0 | 3 | — |
| 0.60 | 0 | 3 | — |
| 0.70 | 0 | 3 | — |
| 0.80 | 0 | 3 | — |
| 0.90 | 0 | 3 | — |
| 0.95 | 0 | 3 | — |
| 1.00 | 0 | 3 | — |

## Decision aid (Z-mix): how many corpora does each α beat fast-ssim2 on?

| α | wins vs ssim2 | total | corpora won |
|---|--:|--:|---|
| 0.00 | 2 | 3 | KADIK10k, TID2013 |
| 0.30 | 2 | 3 | KADIK10k, TID2013 |
| 0.50 | 0 | 3 | — |
| 0.60 | 1 | 3 | KADIK10k |
| 0.70 | 2 | 3 | KADIK10k, TID2013 |
| 0.80 | 2 | 3 | KADIK10k, TID2013 |
| 0.90 | 2 | 3 | KADIK10k, TID2013 |
| 0.95 | 2 | 3 | KADIK10k, TID2013 |
| 1.00 | 3 | 3 | CID22, KADIK10k, TID2013 |

## Decision aid (Z-mix): how many corpora does each α beat V_18 ship on?

| α | wins vs V_18 | total | corpora won |
|---|--:|--:|---|
| 0.00 | 2 | 3 | KADIK10k, TID2013 |
| 0.30 | 0 | 3 | — |
| 0.50 | 0 | 3 | — |
| 0.60 | 0 | 3 | — |
| 0.70 | 0 | 3 | — |
| 0.80 | 0 | 3 | — |
| 0.90 | 0 | 3 | — |
| 0.95 | 0 | 3 | — |
| 1.00 | 0 | 3 | — |

