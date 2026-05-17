# V0_24 dssim co-training v1 — FAILED, recipe-drift hypothesis

**Trained 2026-05-12 23:13Z. Eval 23:15Z.**

## Result

Bake: `/tmp/zensim_loop/bakes/v0_24_dssim03_2026-05-13.bin`
(120,683 bytes; ZNPR v2; 228 → 128 → 1; same arch as V0_16).

Training: 144,791 synth-clean rows, 300 epochs, batch 16384, lr 3e-3,
seed=1, **`--dssim-weight 0.3`**.

`dataset_metric_baseline --v04-bake v0_24...` on CID22 full 4292 pairs:

| Metric | SROCC |
|---|---:|
| V0_2 (linear baseline)  | 0.8676 |
| **V0_24 (this experiment)** | **0.8315** |
| fast-ssim2-gpu          | 0.8895 |
| butter pnorm3           | 0.7412 |

**V0_24 = -0.060 SROCC vs V0_16 (0.8919)** on CID22 full set. Hard
regression. Per-band, V0_24 LOSES to fast-ssim2 in every band:

| Band | V0_24 | ssim2 | V0_16 (ref) |
|---|---:|---:|---|
| B0 (<50)        | 0.3827 | 0.4418 | — |
| B1 [50,65)      | 0.3506 | 0.4694 | — |
| B2 [65,90)      | 0.6758 | 0.7722 | — |
| B3 (≥90)        | 0.0818 | 0.1121 | — |
| Near-PJND [58,68] | 0.3039 | 0.3908 | — |

## Root-cause analysis

V0_24's training command was the V0_16 recipe + `--dssim-weight 0.3`
PLUS one unintentional drift: **no TV pairs**. The training log showed
`0 adjacent-q pairs (0.0s)`. This is because:

- `--human-csv` rows are tagged with synthetic `q=0` in `load_human_csv`
- TV adjacency requires within-curve adjacent q values
- All 144,791 rows have q=0 → 0 adjacent-q pairs → TV disabled

V0_16's recipe presumably included `--sweeps v15r,v15rc` (or similar)
alongside the synth CSV, which DID provide real q values from
codec-sweep parquets → TV pairs existed → TV regularizer ran with
weight=20.

V0_24 v1 trained WITHOUT TV. That alone might explain the regression
(V0_X without TV has been measured at ~0.85 CID22 in earlier cycles,
ballpark-matching this 0.83 result).

## Hypothesis: not enough evidence yet

V0_24 v1's -0.060 vs V0_16 could be:
1. **All recipe-drift**: missing TV regularizer is the entire cause.
2. **All dssim**: dssim_weight=0.3 hurt CID22; would also fail with TV.
3. **Mix**: some of each.

Cannot attribute from this single run.

## Plan for V0_24 v2

Train with BOTH dssim weight + TV regularizer enabled:
- Include `--sweeps v12_zenavif,v12_zenjxl,v12_zenwebp,v13_zenjpeg,v14_zenpng`
  (the small codec-sweep parquets, ~75k rows total) to provide
  adjacent-q TV pairs.
- Sweep rows lack `dssim` column → loader needs to handle that.
- My current trainer patch fills missing dssim with 0, which makes
  `dssim_target = (1-0)*100 = 100` → wrong. Fix:
  use per-row mask so dssim loss only applies to rows that HAVE dssim.

Implementation:
```python
dssim_mask = ~torch.isnan(dst[idx])
if dssim_mask.any():
    err = (pred[dssim_mask] - dst[idx][dssim_mask]) ** 2
    dssim_mse = err.mean()
    loss = loss + cfg.dssim_weight * dssim_mse
```

This is one more trainer-patch change. Estimated 10 min implementation
+ 10 min training + 5 min eval. Defer to next tick.

## Status

V0_24 v1 archived but NOT shipped. V0_16 remains the ship.
Cycle-7 dssim co-training continues; v1 was a recipe-drift baseline.
