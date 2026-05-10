# TV regularizer + capacity sweep — 2026-05-10

Goal: find the trainer config that best balances **CID22 ssim2 fidelity** with
**smoothness** (low non-monotone q-step rate). Trained via
`scripts/v_next/train_v_next_mlp.py` (after fix `dd79a3c` for last-partial-batch
TV slice bounds) on unified v15r + v15rc (1.84M train / 233k val / 227k test).

Hyperparam constants: `--epochs 50 --batch-size 16384 --lr 3e-3 --rank-weight 0.5
--target ssim2 --loss mse_rank --seed 0`. RTX 5070 wall ~26 s per run.

## Results (val split = 232,902 rows, 220,644 adjacent-q pairs)

| run                            | hidden     | tv-weight | val_srocc | test_srocc | val_mse | non-mono q-step % |
|--------------------------------|------------|-----------|-----------|------------|---------|-------------------|
| 2026-05-07 baseline (no TV)    | [64]       | 0.0       | 0.9547    | 0.9814     | 42.08   | 8.26 (per doc)    |
| tv01_full_2026-05-10           | [64]       | 0.1       | **0.9573**| 0.9817     | 39.38   | 7.96              |
| tv03_full_2026-05-10           | [64]       | 0.3       | 0.9563    | 0.9815     | 39.12   | 8.23              |
| tv10_full_2026-05-10           | [64]       | 1.0       | 0.9574    | 0.9813     | 38.24   | 8.21              |
| **h128_tv03_2026-05-10**       | **[128]**  | **0.3**   | **0.9590**| 0.9810     | **37.32**| **7.45** ← best so far |
| h128x128_tv03_2026-05-10       | [128, 128] | 0.3       | 0.9537    | **0.9874** | 38.55   | 7.56              |

Reference baselines:
- ssim2 ground truth: 5.08% non-mono (paper noise floor)
- V0_2 linear: **4.86%** non-mono (smoother than GT via averaging)
- V0_4 mixed-supervision: 8.26%

## Findings

1. **TV weight has limited effect at hidden=64.** Going from 0 → 0.1 → 0.3 → 1.0
   moves non-mono from 8.26% → 7.96% → 8.23% → 8.21%. The TV pair sampling
   (4096/80200 per batch ~5%) is likely the bottleneck — many pairs see only
   ~2.5 epochs of supervision over 50 epochs.
2. **Capacity matters more than TV weight at this scale.** Hidden=128 with TV=0.3
   beats hidden=64 with TV=0.1 on *both* val_srocc (+0.0017) and non-mono
   (-0.51 pp). The hidden=64 architecture appears capacity-limited for the
   smoothness objective.
3. **Two-layer hidden=128,128 trades smoothness for test_srocc.** Best test_srocc
   (0.9874) but worst val_srocc (0.9537) — overfitting on the synthetic ssim2
   target without holding out enough capacity for the held-out distribution.
4. **All configs still significantly above V0_2's 4.86% smoothness.** The 228 →
   N → 1 LeakyReLU MLP family appears to have an inherent smoothness floor
   around 7-8% that further hyperparameter tuning won't break alone.

## Next steps

- Try wider TV pair sampling (increase `tv_bs` ratio in trainer) — 4096/80200
  random per batch is ~5%; bumping to 25% or all-pairs might let TV actually
  drive monotonicity.
- Try `--epochs 100` to give TV more iterations.
- Try multi-scale TV: penalize gaps of 1, 2, 3 q-steps (not just adjacent).
- Bake the current best (h128_tv03) and measure CID22/KADID/TID full-dataset
  SROCC to confirm the synthetic improvements transfer to held-out human MOS.

## Provenance

Output dirs in `/mnt/v/zen/zensim-training/2026-05-07/runs/`:
- `20260510T001257_v_next_tv01_full_2026-05-10/`
- `20260510T001447_v_next_tv03_full_2026-05-10/`
- `20260510T001536_v_next_tv10_full_2026-05-10/`
- `20260510T001704_v_next_h128_tv03_2026-05-10/`
- `20260510T001754_v_next_h128x128_tv03_2026-05-10/`

Each has `model.pt + scaler.npz + meta.json + predictions_val.parquet` (≤2 MB
each). Predictions parquet enables direct smoothness analysis per the
non-mono % column above.
