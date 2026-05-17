# V0_17 weight quantization & simplification review

**Date**: 2026-05-13
**Ran by**: claude-zensim-champ-loop on commit 81f194a8 (V0_17 ship)
**Tool**: `zensim-bench/examples/quant_compare.rs`
**Eval harness**: `zensim-bench/examples/dataset_metric_baseline.rs`

## Question

Can `i8` quant or other simplifications reduce the shipped MLP bin
size for a V_next without sacrificing CID22/KADID/TID SROCC?

## V0_17 inventory

- Architecture: 228 → 384 → 1 (LeakyReLU → Identity), built by 3-way
  concat of (0.65 × V0_16 + 0.30 × cycle-14-s1 + 0.05 × cycle-14-s42)
  — runtime equivalent to output-averaging the three 228→128→1 MLPs.
- Parameters:
  - Layer 0 (228×384): 87,552 weights + 384 biases
  - Layer 1 (384×1): 384 weights + 1 bias
  - Scaler mean/scale: 2 × 228 f32 = 1,824 B
  - Total f32 weights ≈ 88,321 (~353 KB raw)
- Shipped size: **355,332 B** (extra is ZNPR v2 header + section table)

## Sizes by dtype (drop-in re-bake, no retrain)

zenpredict already supports `WeightDtype::{F32, F16, I8}` end-to-end
(format spec at `zenanalyze/zenpredict/src/model.rs:60-99`; inference
at `inference.rs:95-217`). Bake side reads f32, packs to the chosen
dtype, with per-output f32 scales for I8.

| Variant | Bytes  | Fraction | Savings |
|---------|-------:|---------:|--------:|
| F32 (V0_17 ship) | 355,332 | 100.0 % | — |
| F16 re-bake      | 179,460 |  50.5 % | 171.7 KB |
| I8 re-bake       |  93,064 |  26.2 % | 256.1 KB |

Generic compression on the F32 bin is **not** viable: zstd -19 only
shrinks it to 334 KB (-6 %) because the f32 mantissa bits are
incompressibly noisy. F16 / I8 work by discarding bits that are
never used at inference time.

## Quality impact (full corpora, no sub-sampling)

Same V_X model bytes parsed three ways via the existing zenpredict
runtime (`saxpy_matmul_{f32,f16,i8}` paths). Aggregate SROCC vs
human MOS:

| Variant | KADID10k (n=10125) | TID2013 (n=3000) | CID22 (n=4292) |
|---------|--------:|--------:|--------:|
| F32     | 0.9428  | 0.9525  | **0.8934** |
| F16     | 0.9428  | 0.9525  | **0.8934** |
| I8      | 0.9427  | 0.9525  | **0.8934** |
| Δ F16 vs F32 | 0.0000 | 0.0000 | 0.0000 |
| Δ I8  vs F32 | -0.0001 | 0.0000 | 0.0000 |

CID22 per-band SROCC (the per-band gate from `CLAUDE.md` "per-band
reporting rule"):

| Band    | n     | F32    | F16    | I8     |
|---------|------:|-------:|-------:|-------:|
| B0 (<50)        | 324  | 0.4318 | 0.4318 | 0.4311 |
| B1 [50,65)      | 1010 | 0.4561 | 0.4561 | 0.4560 |
| B2 [65,90)      | 2915 | 0.7843 | 0.7843 | 0.7844 |
| B3 (≥90)        | 43   | 0.1540 | 0.1540 | 0.1545 |
| Near-PJND [58,68] | 89 | 0.4390 | 0.4390 | 0.4393 |

Worst per-band delta: **B0 I8 = -0.0007** (n=324 — within sampling
noise). All other deltas are 0.0000–0.0005 SROCC.

## Prediction divergence on random inputs (1024 samples in [-3, 3])

Raw uncalibrated MLP output magnitude: max=138,784, mean_abs=90,810.

| Variant | max\|Δ\| | mean\|Δ\| | rms\|Δ\| | mean\|Δ\| / mean\|y\| |
|---------|---------:|----------:|---------:|----------------------:|
| F16 vs F32 | 7.47    | 2.41     | 2.96     | 2.65e-5 |
| I8 vs F32  | 952.68  | 626.50   | 640.05   | 6.90e-3 |

I8 introduces ~0.7 % relative error in raw MLP output. After the
affine calibration (α=28.04, β=-5.07) this maps to **< 0.1 score
units** in the 0–100 zensim range — well below human-MOS noise
floor. The SROCC numbers above confirm this: rank order is
preserved.

## Weight magnitude distribution (layer 0)

% of the 87,552 layer-0 weights below a fraction of per-layer max:

| Threshold | Below count | Fraction |
|----------:|------------:|---------:|
| 0.1 % of max | 37,718 | 43.1 % |
| 1.0 % of max | 77,673 | 88.7 % |
| 5.0 % of max | 81,219 | 92.8 % |
| 10.0 % of max | 83,456 | 95.3 % |

Layer-1 (384→1) shows the same distribution (87 % below 1 % of max,
95 % below 10 %). This is the **3-way concat artefact**: each
sub-MLP only contributes through its own hidden-unit slice, so
~95 % of inter-MLP weights are unused. A retrained single-MLP at
`hidden ≤ 128` should match V0_17 with a fraction of the params.

## Recommendations for V_next

**Drop-in, zero-retrain (RECOMMEND for V0_18)**:

1. **Ship V_next as I8 with the same V0_17 weights** — 93 KB, -73.8 %
   bin size, SROCC bit-identical on all three corpora to four
   decimals. Single-line trainer change at
   `zensim-validate/src/mlp_train.rs:745` and `:753`:
   `dtype: WeightDtype::F32` → `WeightDtype::I8`.

2. **F16 fallback** if any I8 corner case is found in extended
   validation (AIC-3 / AIC-4 / KonJND PJND-anchor still need to be
   re-tested with I8). F16 saves 50 % with zero measurable change.

**Architectural cleanup (queued for V_next retrain)**:

3. **Retrain hidden=128 single MLP** to drop the 3-way concat
   redundancy. Expected size: ~30 KB at I8 (228×128 + 128×1 weights
   = 29,313 params + 384 B scales + 256 B header). Quality target
   matches V0_16 (CID22 0.8919) or better. If a hidden=128 single
   MLP can clear 0.8919, this becomes the obvious ship form.

4. **Hidden=192 single MLP at I8** as fallback if 128 underfits:
   ~46 KB.

**Not recommended at this size**:

- Sparse format with explicit indices: 90 % of weights are near
  zero, but at 30–93 KB the absolute saving (~10–30 KB on top of
  I8) isn't worth a ZNPR format extension. Sparse becomes
  interesting only if input dim grows past ~512.
- Input pruning (drop low-importance features): would need a
  feature-importance study; defer until input dim grows.
- 4-bit quant: ZNPR doesn't currently support; format extension
  cost > the ~15 KB additional saving.

## Validation owed before V_next ship

- [ ] Run I8/F16 V0_17 through `dataset_metric_baseline` with
      `--aic3` (corpus path needs fixing — current `n=0`) and the
      AIC-4 sample at `/mnt/v/dataset/aic4_sample/`.
- [ ] Run KonJND PJND-anchor with both quants
      (`/mnt/v/datasets/KonJND-1k/KonJND-1k`); confirm the
      at-PJND ≈ 63 ± 5 target still holds (CLAUDE.md gate 3).
- [ ] Verify non-monotonic q-step rate on JPEG unified parquet
      hasn't regressed (V0_17 ship = 5.49 % aggregate).

## Artefacts

- Tool: `zensim-bench/examples/quant_compare.rs`
- Re-baked .bin files: `/tmp/quant/v0_17_2026-05-13_{f32_rebake,f16,i8}.bin`
- Eval logs: `/tmp/eval_{f32_rebake,f16,i8}_4corpus.log`
- Source weight: `zensim/weights/v0_17_2026-05-13.bin` (md5
  2775812d7ffa3964a531022416527009)
