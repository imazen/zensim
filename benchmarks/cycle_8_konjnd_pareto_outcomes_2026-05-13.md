# Cycle-8 KonJND-weight Pareto experiments — outcomes (2026-05-13)

## Summary

Cycle-8 mapped the KonJND supervision-weight axis with 5 training
runs at constant base recipe (V0_26 recipe: h=128, TV=20, LR=3e-3
const, 300 epochs, seed=1 unless noted). The only knob varied:
`--human-csv konjnd:...:<WEIGHT>:0.0` where WEIGHT ∈ {0.0, 0.25,
0.5, 1.0}.

V0_16 (ship, no KonJND, different recipe) anchors the no-KonJND
end of the curve. V0_26 (full KonJND weight 1.0) anchors the
high-KonJND end. The cycle-8 NEW bakes are V0_30 (w=0.25) and
V0_31 (w=0.5), plus V0_32 (V0_31 with seed=42 for reproducibility).

## Pareto curve

| KonJND w | Bake | CID22 SROCC | AIC-3 SROCC | AIC-4 SROCC | JPEG-AI codec | Notes |
|--:|---|--:|--:|--:|--:|---|
| 0.0 | V0_16 SHIP | **0.8919** | 0.7965 | 0.9127 | 0.7951 | ship; different recipe family |
| 0.25 | V0_30 | 0.8702 | **0.8062** | 0.9159 | 0.7975 | cycle-8 new |
| 0.5 s=1 | V0_31 | 0.8628 | 0.8031 | **0.9176** | 0.8318 | **cycle-8 AIC-4 winner** |
| 0.5 s=42 | V0_32 | 0.8578 | 0.8081 | **0.9184** | 0.8416 | reproducibility check |
| 1.0 | V0_26 | 0.8639 | 0.8027 | 0.9097 | **0.8387** | cycle-7 candidate |

(All trained on `safe_synth_clean_features_with_dssim_qc.csv`
synth + `konjnd_aligned_features.csv` KonJND mixed supervision.
Eval: `dataset_metric_baseline --cid22 ... --pairs-tsv "AIC-3:..."
--pairs-tsv "AIC-4:..."`. AIC-4 column ranks AIC-4 aggregate
across all 6 codecs; "JPEG-AI codec" column is the AIC-4 subset
filtered to the `JPEG-AI` codec for the cycle-7 motivation.)

## Key findings

### 1. Pareto curve is CLIFF-then-FLAT, not smooth

Between w=0 and w=0.25, CID22 drops -0.0217 with almost no JPEG-AI
gain (+0.0024). This is a *fixed entry tax* on adding KonJND supervision.

Between w=0.25 and w=1.0, CID22 is essentially FLAT (V0_30 0.8702,
V0_31 0.8628, V0_26 0.8639 — all within seed variance). But JPEG-AI
scales monotonically from 0.7975 (w=0.25) to 0.8387 (w=1.0).

**Implication**: low KonJND weights aren't useful — the CID22 cost
hits regardless. Once the cost is paid, more weight is monotonically
better for JPEG-AI without further CID22 cost.

### 2. V0_31 is the AIC-4 ceiling

V0_31's AIC-4 SROCC 0.9176 (and V0_32 confirms at 0.9184, Δ=+0.0008
across seeds) is the BEST AIC-4 result of any bake tested. It beats:
- V0_16 (SHIP) by +0.0049
- V0_30 (w=0.25) by +0.0017
- V0_26 (w=1.0) by +0.0079

Per-codec AIC-4 SROCC for V0_31:
| Codec | V0_31 | V0_32 (s=42) | V0_16 (SHIP) | V0_31 vs V0_16 |
|---|--:|--:|--:|--:|
| AVIF | 0.9579 | 0.9644 | 0.9508* | +0.007 |
| JPEG-1 | 0.9184 | 0.9194 | 0.9018* | +0.017 |
| JPEG-2000 | 0.9329 | 0.9482 | 0.9238* | +0.009 |
| JPEG-AI | 0.8318 | 0.8416 | 0.7951 | +0.037 |
| JPEG-XL | 0.9697 | 0.9722 | 0.9742* | -0.005 |
| VVC | 0.9261 | 0.9384 | 0.9212* | +0.005 |

(* V0_16 per-codec extracted from prior per-pair CSV; aggregate
matches.)

V0_31 wins or ties every AIC-4 codec except JPEG-XL where V0_16 is
slightly better (-0.005).

### 3. Seed variance is small but matters

V0_31 (s=1) vs V0_32 (s=42), same recipe:
- CID22: 0.8628 vs 0.8578 — Δ=-0.0050 (small)
- AIC-3: 0.8031 vs 0.8081 — Δ=+0.0050 (small)
- AIC-4: 0.9176 vs 0.9184 — Δ=+0.0008 (negligible)

Seed variance is ~0.005 on aggregate CID22 SROCC. AIC-4 is stable
across seeds.

This means **AIC-4 leadership is reproducible**. CID22 has more
seed noise (some bands have only ~50-100 samples).

### 4. CID22 per-band breakdown (consistent KonJND failure mode)

| Bake | B0 | B1 | B2 | B3 | Near-PJND |
|---|--:|--:|--:|--:|--:|
| fast-ssim2 (truth) | 0.4418 | 0.4694 | 0.7722 | 0.1121 | 0.3908 |
| V0_16 (SHIP) | 0.40* | 0.42* | 0.75* | 0.10* | 0.36* |
| V0_30 (w=0.25) | 0.4005 | 0.4047 | 0.7444 | 0.0818 | 0.3306 |
| V0_31 (w=0.5) | 0.3982 | 0.4045 | 0.7277 | 0.0586 | 0.3353 |
| V0_26 (w=1.0) | 0.40* | 0.40* | 0.74* | 0.10* | 0.34* |

(* approximate; from prior per-pair CSVs)

All KonJND-trained bakes share the same B2 SROCC drop (~0.025-0.045
below ssim2). The B0/B1 difficulty is structural — neither V0_16
nor cycle-8 bakes match ssim2 there. The CID22 gain delta from
V0_31→V0_16 is concentrated in **B2 (high quality, n=2915)** which
is the bulk of the dataset.

## Cycle-7 + cycle-8 ship matrix (visible on live comparison site)

```
Bake                        CID22    AIC-3    AIC-4    JPEG-AI   Best for…
V0_16 SHIP                  0.8919   0.7965   0.9127   0.7951    CID22 raw aggregate
V0_26 cycle-7 candidate     0.8639   0.8027   0.9097   0.8387    JPEG-AI (specific codec)
V0_31 cycle-8 winner        0.8628   0.8031   0.9176   0.8318    AIC-4 (cross-codec)
```

All three available in the live site dropdown at
<https://imazen.github.io/zensim/site/compare.html> via
`compare.js` (commit `115b1020`).

## Cycle-8 verdict

**No ship change**. V0_16 still meets the goal #1 shipping bar
(CID22 SROCC > ssim2's 0.8895) and has the strongest CID22 fit
overall. V0_31 is a credible alternative ship candidate IF the
user prioritizes cross-codec breadth (AIC-3, AIC-4) over raw
CID22 aggregate.

The user can now toggle between V0_16, V0_26, V0_31 on the live
site and make an informed call about which trade matches the
product context.

## Cycle-7 + cycle-8 falsified variants (for the record)

These variants were tried and **rejected**:

| Variant | Recipe | CID22 | JPEG-AI | Verdict |
|---|---|--:|--:|---|
| V0_27 | V0_26 + dssim w=0.1 | 0.8658 | 0.7791 | dssim falsified (-0.060 JPEG-AI) |
| V0_28 | V0_26 + cosine LR | 0.8550 | 0.8242 | cosine LR falsified |
| V0_29 | V0_26 + LR 3e-4 const | 0.7499 | 0.6331 | underconverged at 300 epochs |

## What cycle-9 should investigate

The **B0/B1 CID22 ceiling** is the unsolved problem. All four
KonJND-trained cycle-8 bakes plus V0_16 score B0≈0.40 / B1≈0.40
against ssim2's 0.44/0.47. The ~0.05 gap is structural — neither
weight tuning nor recipe variation closes it.

Options for cycle-9:
1. **More B0/B1 training data**: KADID-10k has analytic-distortion
   labels in the low-quality range. Mixed-supervision experiment
   with KADID alongside synth+KonJND.
2. **Different feature representation**: the V_X 228-feat MLP
   may be lossy on low-quality artifacts. Experiment with 300-feat
   input (synth CSV has feat_0..299 today).
3. **B0/B1-targeted loss weighting**: bin the safesyn pairs by
   ssim2 q-bin and upweight low-q pairs in training.

None of these are cheap — all require trainer + bake-format work
or new data acquisition. Wait for user input before starting cycle-9.

## Reproducibility

Recipe (V0_31, V0_32):
```bash
python3 scripts/v_next/train_v_next_mlp.py \
  --sweeps NONE \
  --human-csv "safesyn:/tmp/zensim_loop/safe_synth_clean_features_with_dssim_qc.csv:1.0:0.0" \
  --human-csv "konjnd:/tmp/zensim_loop/konjnd_aligned_features.csv:0.5:0.0" \
  --target ssim2 --loss mse_rank --hidden 128 --epochs 300 \
  --batch-size 16384 --lr 3e-3 --weight-decay 1e-5 --rank-weight 0.5 \
  --tv-weight 20 --dssim-weight 0.0 --seed {1|42} \
  --tag v0_31_konjnd_w05_seed{1|42}_2026-05-13 \
  --out-dir /mnt/v/zen/zensim-training/2026-05-07/runs
```

Bakes:
- V0_30: `/tmp/zensim_loop/bakes/v0_30_konjnd_w025_2026-05-13.bin` (120,712B)
- V0_31: `/tmp/zensim_loop/bakes/v0_31_konjnd_w05_2026-05-13.bin` (120,712B)
- V0_32: `/tmp/zensim_loop/bakes/v0_32_konjnd_w05_seed42_2026-05-13.bin` (120,714B)

Per-pair eval CSVs:
- `/tmp/zensim_loop/v0_30_per_pair.csv` (5,192 rows)
- `/tmp/zensim_loop/v0_31_per_pair.csv` (5,192 rows)
- `/tmp/zensim_loop/v0_32_per_pair.csv` (CID22 4292; AIC-3+AIC-4 in separate CSVs)

Tick log entries: 502, 503, 504, 505, 506 in
`~/work/zen/zenanalyze/zensim_champion_log.md`.
