# TV regularizer + capacity sweep — 2026-05-10

Goal: find the trainer config that best balances **CID22 ssim2 fidelity** with
**smoothness** (low non-monotone q-step rate). Trained via
`scripts/v_next/train_v_next_mlp.py` (after fix `dd79a3c` for last-partial-batch
TV slice bounds) on unified v15r + v15rc (1.84M train / 233k val / 227k test).

Hyperparam constants: `--epochs 50 --batch-size 16384 --lr 3e-3 --rank-weight 0.5
--target ssim2 --loss mse_rank --seed 0`. RTX 5070 wall ~26 s per run.

## Results (val split = 232,902 rows, 220,644 adjacent-q pairs)

| run | hidden | tv-weight | val_srocc | test_srocc | val_mse | non-mono q-step % |
|---|---|---|---|---|---|---|
| 2026-05-07 baseline (no TV) | [64] | 0.0 | 0.9547 | 0.9814 | 42.08 | 8.26 |
| tv01_full_2026-05-10 | [64] | 0.1 | 0.9573 | 0.9817 | 39.38 | 7.96 |
| tv03_full_2026-05-10 | [64] | 0.3 | 0.9563 | 0.9815 | 39.12 | 8.23 |
| tv10_full_2026-05-10 | [64] | 1.0 | 0.9574 | 0.9813 | 38.24 | 8.21 |
| h128_tv03_2026-05-10 | [128] | 0.3 | 0.9590 | 0.9810 | 37.32 | 7.45 |
| h128x128_tv03_2026-05-10 | [128, 128] | 0.3 | 0.9537 | 0.9874 | 38.55 | 7.56 |
| h128_tv30_2026-05-10 | [128] | 3.0 | **0.9601** | 0.9812 | **34.71** | 7.54 |
| h128_tv100_2026-05-10 | [128] | 10.0 | 0.9574 | 0.9815 | 37.82 | 6.68 |
| **h128_tv300_2026-05-10** | **[128]** | **30.0** | 0.9584 | **0.9821** | 37.61 | **5.81** ← balance |
| **h128_tv1000_2026-05-10** | **[128]** | **100.0** | 0.9531 | 0.9782 | 43.08 | **4.26** ★← beats V0_2 floor |
| h192_tv100_2026-05-10 | [192] | 10.0 | **0.9622** | 0.9811 | 35.12 | 6.62 |
| h192_tv300_2026-05-10 | [192] | 30.0 | 0.9567 | 0.9822 | 39.54 | 5.66 |

Reference baselines:
- ssim2 ground truth: 5.08% non-mono (paper-level noise floor)
- V0_2 linear: **4.86%** non-mono (smoother than GT via per-q averaging)
- V0_4 mixed-supervision: 8.26%

## Findings (revised)

1. **TV weight DID work — it just needed to be 100x bigger than the original `--tv-weight 0.1` default smoke.** The TV term `relu(pred_lo - pred_hi).mean()` has values ~0.5–2 per batch, while MSE is ~40. With TV weight 0.1, the contribution to total loss is ~0.05 vs MSE's ~40 — utterly dominated. With TV weight 100, the term contributes ~50–200 — comparable to MSE.
2. **Capacity multiplier**: hidden=128 dominated hidden=64 across the board on the smoothness vs val_srocc Pareto. Hidden=192 doesn't reliably beat hidden=128 except at modest TV weights.
3. **Tradeoff curve at hidden=128**:
   - TV=3: val_srocc 0.9601 ★ (best), non-mono 7.54
   - TV=30: val_srocc 0.9584, non-mono 5.81 (good balance)
   - TV=100: val_srocc 0.9531, non-mono 4.26 ★ (beats V0_2 smoothness floor)
4. **The 228→N→1 LeakyReLU family CAN reach the V0_2 smoothness floor**, contrary to the prior tick's hypothesis — just needs aggressive TV weight (~100) and pays ~0.0016 in val_srocc.

## Production candidates

- **Smoothest**: `h128_tv1000_2026-05-10` — non-mono 4.26%, val_srocc 0.9531
- **Best balance**: `h128_tv300_2026-05-10` — non-mono 5.81%, val_srocc 0.9584
- **Best val_srocc with TV**: `h128_tv30_2026-05-10` — non-mono 7.54%, val_srocc 0.9601
- **Best hidden=192**: `h192_tv300_2026-05-10` — non-mono 5.66%, val_srocc 0.9567

The smoothness vs val_srocc Pareto front at hidden=128 spans: (val_srocc=0.9601, nonmono=7.54%) → (0.9584, 5.81%) → (0.9531, 4.26%).

## Next experiments (queued)

- **End-to-end CID22/KADID/TID validation**: bake `h128_tv1000`, `h128_tv300`,
  and `h128_tv30` to ZNPR v2; run `dataset_metric_baseline` on each. Confirm
  the synthetic-ssim2 improvements transfer to held-out human MOS.
- **Score-mapping refit**: TV-trained models may have a different distance
  distribution; refit `score_mapping_a/b` per
  `benchmarks/v04_calibrate_mapping_2026-05-01.md` if the default
  `(18.0, 0.7)` produces saturation/clipping.
- **Wider TV pair sampling**: increase `tv_bs` from `bs/4` toward `bs` to give
  every pair more supervision per epoch — could reduce TV weight needed.
- **Multi-scale TV**: penalize gaps of 1, 2, 3 q-steps simultaneously.

## Provenance

Output dirs in `/mnt/v/zen/zensim-training/2026-05-07/runs/`:
- 2026-05-07 baseline (no TV): `20260507T115414_v_next_ssim2_64h_full/`
- 2026-05-10 TV sweep: `2026051000{1257,1447,1536,1704,1754,2044,2131,2245,2334,2422,2516}_v_next_*2026-05-10/`

Each has `model.pt + scaler.npz + meta.json + predictions_val.parquet` (≤2 MB
each). Predictions parquet enables direct smoothness analysis per the
non-mono % column above without re-running the full pipeline.
