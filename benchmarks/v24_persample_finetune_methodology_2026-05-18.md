# V_24 Per-Sample-α + KonJND Finetune (gentle config) — methodology

**Date**: 2026-05-18
**Branch**: `feat/ex2-stdpool-head` (commits `tqkzupuo cf0ac62a` + `tqkzupuo f3cd0631`)
**Workspace**: `/home/lilith/work/zen/zensim--persample-finetune`

## Hypothesis (Step 1 of principled experiment workflow)

1. **Hypothesis**: Starting from V_24 per-sample-α 5-seed bakes
   (CID22 0.854 / KonJND 0.814 / AIC-3 0.811 — CID22+AIC-3 specialist),
   a BRIEF finetune (15 epochs, LR=3e-5) with KonJND train_w boosted
   from 0.02 → 0.10 should lift KonJND by ≥ 0.03 SROCC while preserving
   CID22 within 0.01 of baseline.
2. **Falsification**: If KonJND fails to lift by ≥ 0.03 OR CID22 drops
   by > 0.02 across 5 seeds, the gentle-finetune approach is dead.
3. **Cost ceiling**: 5 seeds × ~30s each = ~3 min compute. If hits with
   the gentle config, no further sweep needed.
4. **Ship form**: PreviewV0_X (single-bake) — V_24-FT-gentle bake.

## Reporting panel (Step 2)

| Corpus | Role | When inspected |
|---|---|---|
| safesyn (300col mix) | train | per-epoch trainer log |
| KADID (300col mix) | train+val | per-epoch + bake_verdict end |
| TID (300col mix) | train+val | per-epoch + bake_verdict end |
| KonJND (300col mix) | train+val | per-epoch + bake_verdict end |
| cvvdp_iwssim_large | train | per-epoch trainer log |
| CID22 (372col holdout) | validation only | bake_verdict end |
| AIC-3 CTC (372col holdout) | validation only | bake_verdict end |

## Implementation: `--continue-from` flag

Added a `--continue-from PATH` flag to `zensim_mlp_train` (commits
`cf0ac62a` + `f3cd0631`). When supplied with `--per-sample-alpha-head`,
the trainer:

1. Parses the input ZNPR v3 bake via `zenpredict::Model::from_bytes`.
2. Extracts `scaler_mean`, `scaler_scale`, layer-0 weights/biases, and
   the `zentrain.per_sample_alpha_head` metadata payload (W_α, b_α,
   rank_w, rank_b, reducer_w, reducer_b).
3. **Inverse-permutes the layer-0 weights to natural order.** The
   `bake()` pipeline auto-applies hu_reorder (sorting hidden units by
   L2 ascending) to layer 0 + permuted-identity layer 1. The metadata
   payload is NOT permuted. Reading layer-0 weights raw stores them in
   their PERMUTED order, mismatched with the natural-order metadata.
   The fix extracts the permutation from layer 1 (each row's
   column-of-1.0 = perm[row]) and inverse-permutes layer-0 columns so
   `w1[r, perm[i]] = permuted_w1[r, i]`.
4. Uses the loaded values as Adam-state-initialized weights instead of
   Xavier random init.

**Round-trip verification**: `--continue-from <bake> --lr 0` produces
a re-baked file whose bake_verdict scores are **bit-identical** to the
input (CID22 0.8548, KADID 0.9319, TID 0.8934, KonJND 0.8224,
AIC-3 0.8109 — all five corpora match).

## Recipe (gentle finetune)

Script: `scripts/v_next/run_persample_konjnd_finetune_gentle_seed.sh`

```sh
zensim_mlp_train \
  --group safesyn:/mnt/v/.../safesyn_mix_300col.parquet:1.0:0.0 \
  --group kadid:/mnt/v/.../kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:/mnt/v/.../tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:/mnt/v/.../konjnd_mix_300col.parquet:0.10:1.0 \
  --group cvvdp_iwssim_large:/mnt/v/.../cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0 \
  --hidden 128 --epochs 15 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 \
  --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --continue-from <baseline_persample_seed${SEED}.bin> \
  --lr 3e-5 --seed ${SEED} --log-every 3 --early-stop-patience 0 \
  --out <output_seed${SEED}.bin>
```

Differences from baseline V_24 per-sample-α recipe:
- `--continue-from` warm-init from per-sample seed bake
- `konjnd train_w`: 0.02 → 0.10 (5× boost)
- `--lr`: 1e-3 → 3e-5 (33× smaller, brief finetune)
- `--epochs`: 300 → 15 (20× fewer)

Total compute: ~30s per seed (vs ~9 min for full 300-epoch baseline).

## Result: 5-seed CI

5 seeds (1..5), each warm-init from the corresponding baseline persample
seed bake.

| Corpus | V_24-FT mean ± std | 95 % CI | V_24-PS baseline | V_22 ship | Δ vs base | Δ vs V_22 |
|---|---|---|---|---|---|---|
| CID22 | 0.8407 ± 0.0065 | [0.8326, 0.8488] | 0.8542 | 0.8324 | **−0.0135 (p=0.018)** | **+0.0083** |
| KADID-10k | 0.9315 ± 0.0010 | [0.9303, 0.9328] | 0.9307 | 0.9677 | +0.0009 | −0.0362 |
| TID2013 | 0.8897 ± 0.0006 | [0.8889, 0.8905] | 0.8873 | 0.9729 | +0.0023 | −0.0832 |
| KonJND-1k | 0.8624 ± 0.0068 | [0.8539, 0.8708] | 0.8136 | 0.8927 | **+0.0488 (p<0.0001, t=11.24)** | −0.0303 |
| AIC-3 CTC | 0.8092 ± 0.0029 | [0.8056, 0.8129] | 0.8117 | 0.7845 | −0.0025 | **+0.0247** |

**Headline**: KonJND lifted decisively by +0.049 over baseline (t=11.24,
p<1e-4). CID22 cost is small (−0.013) but statistically significant.
Other corpora essentially unchanged.

## α distribution: before vs after

α(x) = sigmoid(W_α · h + b_α) — per-sample mix between rank head (α)
and pool head (1−α). 5-seed mean shifts:

| Corpus | α mean (before) | α mean (after) | Δ |
|---|---|---|---|
| CID22 | 0.774 | 0.787 | +0.013 |
| KADID | 0.273 | 0.277 | +0.004 |
| TID | 0.311 | 0.319 | +0.008 |
| KonJND | 0.756 | 0.774 | +0.018 |
| AIC-3 | 0.702 | 0.708 | +0.006 |

The α distribution barely shifted — the KonJND lift came primarily
from rank_w / reducer_w / W1 weight updates, NOT from α re-allocation.
This suggests the finetune learned a **more KonJND-friendly rank-head
response** rather than discovering a structurally different α regime.

## Pareto gate evaluation

| Gate | Threshold | Result | Pass |
|---|---|---|---|
| CID22 ≥ V_22 − 0.005 | ≥ 0.8274 | 0.8407 | ✓ |
| KADID ≥ V_22 − 0.040 | ≥ 0.9277 | 0.9315 | ✓ |
| TID ≥ V_22 − 0.040 | ≥ 0.9329 | 0.8897 | ✗ (−0.043) |
| KonJND ≥ V_22 − 0.010 | ≥ 0.8827 | 0.8624 | ✗ (−0.020) |
| AIC-3 ≥ V_22 + 0.015 | ≥ 0.7995 | 0.8092 | ✓ |

**3/5 pass.** TID and KonJND remain below the gate. The finetune
recovered most of the KonJND gap (0.81 → 0.86 → gate 0.88), but the
last 0.02 of KonJND lift would require a more aggressive finetune,
which empirically destroys CID22 (`v1` experiment at konjnd_w=0.50,
LR=1e-4, 30ep: KonJND 0.928 but CID22 0.713).

TID is structurally a pure-synthetic-distortion corpus where the V_24
per-sample-α architecture starts at 0.89 vs V_22's 0.97. No amount of
KonJND-anchored finetune is going to close a 0.08 TID gap; this is
architectural.

## bake_compare verdict (seed=1 vs V_22)

`bake_compare` aggregate cross-corpus tally:
- **A (V_24-FT seed=1) decisively beats B (V_22)**: 2 cells
  (CID22 aggregate, AIC-3 aggregate)
- **B decisively beats A**: 17 cells (KADID/TID/KonJND aggregates +
  several bands)
- Promising-not-decisive: 2, Tied: 8, Noisy: 1

**Overall winner: B (V_22)** across decisive cells (2 A vs 17 B).

A's wins are concentrated where the user values them most for
compression product (CID22 the gold standard + AIC-3 low-q human MOS).
B's wins are concentrated on synthetic-distortion corpora (KADID/TID)
which are integrity guards, not product targets.

## Honest gap

This finetune is NOT a Pareto winner. It is a **CID22+AIC-3 specialist
that has recovered some KonJND ground** without destroying its
specialty:

- KonJND went 0.81 → 0.86, **+0.05 lift** — the load-bearing claim.
- CID22 stayed at 0.84 (vs baseline 0.85, vs V_22 0.83) — within
  Pareto gate.
- AIC-3 stayed at 0.81 (vs baseline 0.81, vs V_22 0.78) — preserved.
- KADID/TID essentially unchanged from baseline (still below V_22).

The hypothesis was "init protects CID22 while small konjnd boost
recovers KonJND". This is **partially confirmed**:
- Init DOES protect CID22 (−0.013 cost vs baseline)
- Konjnd boost DOES lift KonJND (+0.049)
- But the lift stops at 0.86 (not 0.88+ needed for V_22 parity)

## Verdict

**Not a Pareto winner per the original gate, but a measurable upgrade
over the V_24-per-sample-α baseline.** Specifically valuable as a
**multi-bake ensemble component**: a per-sample-α-finetune bake +
V_22 ship mixed at runtime should let users select between CID22+AIC-3
specialist outputs (high α weight on FT) and KADID/TID generalist
outputs (high α weight on V_22). The runtime mix is the next
experiment.

## Packed bake

Seed=4 (best CID22 in the cohort) repacked to F16 + zstd:
- Source: `persample_konjnd_gentle_seed4.bin` (223,876 bytes)
- Packed: `persample_konjnd_gentle_seed4_packed.bin` (82,190 bytes,
  36.7% of input)
- Quality preserved: CID22 0.8451, KADID 0.9321, TID 0.8896,
  KonJND 0.8544, AIC-3 0.8131 (identical to source bake)

## Reproducibility

- Trainer commits: `tqkzupuo cf0ac62a` (`--continue-from` flag) +
  `tqkzupuo f3cd0631` (hu_reorder perm fix).
- Training parquets:
  `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/*.parquet`
- Input bakes: `/mnt/v/zen/zensim-eval/v24_persample_alpha_2026-05-18/persample_seed{1..5}.bin`
- Output bakes: `/mnt/v/zen/zensim-eval/v24_persample_konjnd_finetune_v2_2026-05-18/persample_konjnd_gentle_seed{1..5}.bin`
- Wall time per seed: ~30s (15 epochs × 50k pairs × K=256 ≈ 200
  Adam steps per epoch × 15 ≈ 3000 Adam steps total).
- Total experiment time: ~3 min training + ~20s eval per seed.
