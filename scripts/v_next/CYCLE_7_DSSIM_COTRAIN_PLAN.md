# Cycle 7 — dssim co-training plan (2026-05-12, authorized)

## Motivation

Per `benchmarks/cycle_6_finals_2026-05-12.md` and
`benchmarks/aic_per_codec_v0_16_2026-05-12.md`:

- V0_16 wins fast-ssim2 aggregate on every public corpus by +0.002 to
  +0.005 SROCC.
- V0_16 wins or ties 14 of 21 per-codec comparisons across 3 held-out
  corpora.
- **Single biggest deficit**: JPEG-AI on AIC-4. V0_16 0.7951 vs
  fast-ssim2 0.8459 vs **dssim 0.9147**.
- dssim's multi-scale SSIM-derived structure captures something the
  zensim per-pair features + V_X MLP currently miss.

**Hypothesis**: adding dssim as an auxiliary loss head in
`train_v_next_mlp.py` lets V_X learn dssim's structure in parallel
with ssim2, without changing the input-feature surface (still 228
zensim per-pair features). Expected outcome: V0_24 (the dssim-cotrain
variant) maintains V0_16's CID22/AIC-3/AIC-4 aggregate wins AND closes
the JPEG-AI gap.

## Data availability

✅ **No new scoring needed**:
`/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv`
already has a `dssim` column (the synth generator scores it
alongside ssim2 + butter).

Sample header from that CSV:
```
source_path,decoded_path,codec,quality,width,height,
gpu_ssimulacra2,gpu_butteraugli,cpu_ssimulacra2,cpu_butteraugli,
size_bytes,run_id,dssim
```

The `dssim` column exists and is populated. dssim is a distance (0 =
identical, ~0.5 = max distortion). To make it loss-comparable with
ssim2 (a quality score, 0-100 scale), the trainer can either:

1. Negate-and-scale: `dssim_norm = (1 - dssim) * 100` → quality-like.
2. Train the network to output two heads (one for ssim2-target,
   one for dssim-target) with shared trunk.

Plan: use approach (1) — single output head, weighted-sum loss.
Network architecture stays identical to V0_16 (228 → 128 LeakyReLU →
1 Identity), so the bake binary stays the same format.

## Trainer changes

`scripts/v_next/train_v_next_mlp.py`:

1. **`build_arrays(df, target_col, dssim_col=None)`**: optionally
   return `dssim_full` array alongside `y_full`.
2. **`TrainConfig` adds `dssim_weight: float = 0.0`** (default 0
   preserves current V0_16 behavior).
3. **`train()` accepts `dssim_train`** and adds to the loss when
   `dssim_weight > 0`:
   ```python
   if dssim_weight > 0 and dssim_target is not None:
       dssim_norm = (1.0 - dssim_target) * 100.0  # quality-scaled
       dssim_mse = torch.mean((pred - dssim_norm[idx]) ** 2)
       loss = loss + dssim_weight * dssim_mse
   ```
4. **CLI**: `--dssim-weight 0.3` (suggested starting value, similar
   in magnitude to the existing `--rank-weight 0.5`).

## Recipe (proposed V0_24)

Same recipe as V0_16 EXCEPT:
- `--target ssim2` (unchanged)
- **`--dssim-weight 0.3`** (NEW)
- TV=20 seed=1 hidden 128 epochs 300 batch 16384 lr 3e-3 wd 1e-5
  rank-weight 0.5 (all unchanged)
- KonJND-aligned (unchanged)

Tag: `v_next_h128_ep300_tv20_seed1_dssimW03_2026-05-13`

## Expected wall

- Trainer convergence on cleanish synth (~145k rows after the
  2026-05-12 purge): ~10 min on RTX 5070 + 7950X.
- Bake + eval: ~5 min.
- Total: ~15 min from launch to V0_24 SROCC numbers in hand.

## Evaluation

Apply `dataset_metric_baseline --v04-bake v0_24_dssim_cotrain.bin
--max-pairs 20000 --pairs-tsv AIC-3:... --pairs-tsv AIC-4:...
--cid22 ...` to all 5 in-repo corpora; merge `score_zensim_v0_24`
columns into the parquets; compare per-codec vs V0_16.

**Success criteria**:
1. AIC-4 JPEG-AI SROCC > 0.85 (closes the gap from V0_16's 0.7951).
2. CID22, AIC-3, AIC-4 aggregate SROCC ≥ V0_16's (no regression).
3. Synth val SROCC ≥ 0.99 (no training collapse).
4. Non-mono q-step rate ≤ 6% (CLAUDE.md V0_16 gate is at 6%).

If V0_24 meets all 4, ship via the same V0_16 → V0_15 → V0_X swap
discipline.

## Risk

- dssim's CID22 SROCC is 0.8722 (vs ssim2's 0.8895). If V_X over-
  fits to dssim, CID22 could regress. Mitigation: dssim_weight=0.3
  is small relative to the ssim2 head; should be a soft pull, not a
  takeover.
- dssim on AIC-3 SROCC is 0.7884 (vs ssim2 0.7965). Same concern;
  same mitigation.
- The training distribution may not contain enough transformer-codec
  artifacts to actually move the JPEG-AI needle. Test reveals
  whether this is data-bound or architecture-bound.

## Authorization status (2026-05-12)

User explicitly authorized all 3 cycle-7 items including dssim
co-training. Plan executes autonomously.
## Data-prep gap for V0_24 training (discovered tick 487)

The trainer patch (`--dssim-weight 0.3` flag) is verified syntax-clean
and the loss-side wiring is correct. But there's a data-availability
gap: the trainer reads from unified parquets at
`/mnt/v/zen/zensim-training/2026-05-07/unified/` which carry
`score_zensim / score_ssim2 / score_butteraugli_max / score_butteraugli_pnorm3`
but NOT `dssim`. The synth CSV
`/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_perceptual_clean.csv`
HAS `dssim` (range [0, 0.2]) but isn't in parquet form and uses a
different schema (`source_path / decoded_path / codec / quality /
gpu_ssimulacra2 / dssim`).

**Next tick step**: write a small script that joins the two:
- Source rows by `(image_path, codec, q)` from unified parquets.
- Join `dssim` column from `training_safe_synthetic_perceptual_clean.csv`
  matched by `(decoded_path basename → image_path + codec, q)`.
- Emit `unified_v15rdc_zenjpeg.parquet` (or similar) with the dssim column.

Then re-run `train_v_next_mlp.py --sweeps v15rdc --dssim-weight 0.3`.

Alternative: add a `--dssim-csv` flag to the trainer that loads dssim
from a CSV separately and joins at train time. ~30 lines Python.

Either way the run is ~15 min once the data is in place.
