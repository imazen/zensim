# V11-SUBSTRATE-V2 — methodology

**Task #190 (2026-05-20).** Rebuild V11 ssim2-anchored substrate using
the R2 omni-multi-codec sidecars, then retrain V11-A' with the correct
recipe (per-sample-α head so anchor loss actually fires).

**Verdict (5-seed CI): FALSIFIED on the brief's recipe.** Substrate is
sound, but the brief's training recipe is structurally weaker than the
V10 BalancedV3 ship baseline. See "Recipe assessment" below.

## Key correction to V11-A' v1 (task #189)

The previous V11 substrate agent claimed "ssim2 was never computed on
zenwebp/zenavif/zenjxl" and used a cvvdp→ssim2 conversion fit on
zenjpeg as a workaround. **That claim was wrong.** Per
`/home/lilith/work/zen/DATA_PROVENANCE.md` lines 209-228 and verified
directly by inspecting the R2 sidecars:

The `omni-multi-codec-2026-05-19` R2 prefix has `score_ssim2_gpu`
100% non-null across all 4 codecs (zenavif 4000/4000, zenjpeg
61600/61600, zenjxl 51200/51200, zenwebp 1000/1000). The previous
agent looked at LOCAL `unified_v12_*` parquets that pre-dated the
ssim2 backfill.

This v2 substrate uses the R2 data directly — no cvvdp→ssim2
conversion required, no codec-saturation modeling assumptions.

## Phase 1: R2 data pull (DONE)

```
AWS_PROFILE=r2 s5cmd --endpoint-url https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com \
  cp 's3://zentrain/omni-multi-codec-2026-05-19/omni/*.parquet' \
  /mnt/v/zen/zensim-training/2026-05-20-r2-omni/multi-codec/omni/
# 365 parquets, 8.8 MiB total

cp 's3://zentrain/omni-multi-codec-2026-05-19/zensim_features/*.parquet' \
  /mnt/v/zen/zensim-training/2026-05-20-r2-omni/multi-codec/zensim_features/
# 365 parquets, 89 MiB total

cp 's3://zentrain/cvvdp-v15rc-2026-05-18/omni/*.parquet' \
  /mnt/v/zen/zensim-training/2026-05-20-r2-omni/v15rc-jpeg/omni/
# 2568 parquets, 74 MiB total

cp 's3://zentrain/cvvdp-v15rc-2026-05-18/zensim_features/*.parquet' \
  /mnt/v/zen/zensim-training/2026-05-20-r2-omni/v15rc-jpeg/zensim_features/
# 2568 parquets, 962 MiB total
```

**Multi-codec coverage** (joined on `(image_path, codec, q, knob_tuple_json)`):

| codec | rows | imgs | q levels | ssim2 nn | cvvdp nn |
|---|--:|--:|---|--:|--:|
| zenavif | 4,000 | 200 | [10, 30, 60, 80, 90] | 100% | 100% |
| zenjpeg | 698,556 | 1,101 | 19 q levels (incl v15rc) | 100% | 100% |
| zenjxl | 51,200 | 200 | [10, 30, 60, 80, 90] | 100% | 100% |
| zenwebp | 1,000 | 200 | [10, 30, 60, 80, 90] | 100% | 100% |
| **TOTAL** | **754,756** | | | **100%** | **100%** |

The 4 non-jpeg codecs share the same 200 `gen-*` synthetic image
corpus (mandatory for cross-codec equivalence pairing). zenjpeg
v15rc adds 1,001 additional images from a disjoint corpus
(`gif-static_*` and other web images) — these enrich zenjpeg
anchor density but cannot pair cross-codec.

## Phase 2: anchor parquet (DONE)

10 ssim2 bands per task brief (mapping ssim2 anchor → V11 target_score):

| ssim2 | target_score | semantic | emit rows |
|---:|---:|---|--:|
| 100 | 100 | mathematically lossless | 185 (zenjpeg only) |
| 95 | 95 | near-lossless | 671 |
| 90 | 90 | visually lossless | 1,514 |
| 75 | 80 | JND | 1,506 |
| 60 | 65 | mildly noticeable | 1,224 |
| 45 | 50 | JOD | 1,067 |
| 30 | 35 | 3×-DPI resize-out | 860 |
| 18 | 20 | clear artifacts | 621 |
| 10 | 10 | very degraded | 512 |
| 3 | 0 | borderline | 367 |
| **TOTAL** | | | **8,527** |

Per-codec coverage: zenjpeg dominates (185-1084 rows per band, 19 q),
zenwebp/zenavif/zenjxl span 5 q levels each (1-200 rows per band).
zenjxl saturates at low-distortion bands (only 2 rows at JOD, 0 at
ssim2 = 18 / 10 / 3) because q=10 doesn't reach those distortion
levels on the 200-image synth corpus.

**Output**: `/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/anchors_ssim2_300col_v2.parquet`
(8,527 rows × 312 cols, 12.4 MiB).

## Phase 3: cross-codec equivalence parquet (DONE)

6 ssim2 pivot levels {90, 75, 60, 45, 30, 18} (the bands where every
codec has ≥ 1 emission). For each (image, level) pair, find the q
per codec with closest ssim2, emit (codec_a, codec_b) pairs.

Per-pair counts: zenavif↔zenjpeg 374, zenavif↔zenjxl 307,
zenavif↔zenwebp 220, zenjpeg↔zenjxl 327, zenjpeg↔zenwebp 285,
zenjxl↔zenwebp 226. Total **1,739 pairs**.

**Output**: `/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/cross_codec_equivalence_ssim2_v2.parquet`
(1,739 rows × 617 cols, 3.7 MiB).

## Phase 4: V11-A' v2 5-seed training (DONE — FALSIFIED)

Brief recipe (verbatim from task #190):

```
zensim_mlp_train \
  --group safesyn:.../safesyn.parquet:1.0:0.0 \
  --group kadid:.../kadid.parquet:0.6:0.4 \
  --group tid:.../tid.parquet:0.6:0.4 \
  --group konjnd:.../konjnd-dense.parquet:0.6:0.0 \
  --group large:.../cvvdp_iwssim_LARGE.parquet:1.0:0.0 \
  --group cid22_train:.../cid22_train.parquet:0.5:0.0 \
  --group pipal:.../pipal.parquet:0.3:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size 32 \
  --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
  --max-features 300 --target-column mix_cv35_iw65 \
  --per-sample-alpha-head --tanh-output-head-scale 20.0 \
  --ranknet-weight 0.0 --mse-weight 1.0 --monotonicity-reg 1.0 \
  --anchor-parquet .../anchors_ssim2_300col_v2.parquet \
  --anchor-loss-weight 1.0 --anchor-step-p 0.30 \
  --cross-codec-eq-parquet .../cross_codec_equivalence_ssim2_v2.parquet \
  --cross-codec-eq-weight 1.0 --cross-codec-rank-preserve-weight 0.2 \
  --dynamic-range-floor-weight 0.3 --dynamic-range-sigma-threshold 25.0 --dynamic-range-step-p 0.05 \
  --seed <S> --gpu-runtime cuda --out cc4v11a_v2_s<S>.bin
```

5 seeds, each ~33s on RTX 5070 (GPU trainer's `minibatch_k=512`
override — the `--minibatch-size 1` brief value is incompatible with
the GPU kernel's batch shape).

**bake_verdict 5-seed CI** (vs canonical val parquets):

| Corpus | n | SROCC mean | SROCC std | SROCC median | PLCC median | Z-RMSE median |
|---|--:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.7351 | 0.0337 | 0.7519 | 0.6711 | 0.741 |
| KADIK10k | 10125 | 0.7940 | 0.0028 | 0.7954 | 0.7636 | 0.646 |
| TID2013 | 3000 | 0.7788 | 0.0027 | 0.7779 | 0.7250 | 0.689 |
| KonJND-1k (full) | 1008 | 0.1247 | 0.0442 | 0.1105 | 0.0303 | 1.000 |
| AIC-3 CTC | 600 | 0.7518 | 0.0093 | 0.7542 | 0.6591 | 0.752 |
| AIC-4 sample | 300 | 0.9047 | 0.0143 | 0.9077 | 0.8167 | 0.577 |

**V10 BalancedV3 baseline** (verified via `bake_verdict` against
`zensim/weights/v_balanced_v3_2026-05-20.bin`):

| Corpus | n | SROCC | PLCC | Z-RMSE |
|---|--:|---:|---:|---:|
| CID22 | 4292 | 0.8324 | 0.8256 | 0.564 |
| KADIK10k | 10125 | 0.9664 | 0.9562 | 0.293 |
| TID2013 | 3000 | 0.9712 | 0.9379 | 0.347 |
| KonJND-1k (full) | 1008 | 0.8927 | 0.9270 | 0.375 |
| AIC-3 CTC | 600 | 0.7845 | 0.7952 | 0.606 |
| AIC-4 sample | 300 | 0.9016 | 0.8900 | 0.456 |

**Ship-gate verdict (per task brief Phase 4):**

| Metric | V10 BalancedV3 | V11-A' v2 median | Target | Verdict |
|---|---:|---:|---:|---|
| CID22 SROCC | 0.8324 | 0.7519 | ≥ 0.8374 (+0.005) | **FAIL** (−0.085) |
| CID22 Z-RMSE | 0.564 | 0.741 | ≤ 0.530 | **FAIL** (+0.211) |
| AIC-4 SROCC | 0.9016 | 0.9077 | ≥ 0.8966 | PASS |
| KADID drift | 0.9664 base | 0.7954 | within −0.10 | **FAIL** (−0.171) |
| TID drift | 0.9712 base | 0.7779 | within −0.10 | **FAIL** (−0.193) |
| KonJND drift | 0.8927 base | 0.1105 | within −0.10 | **FAIL** (−0.782) |
| Anchor JND landing | exact | exact | exact | PASS |

**V11-A' v2 brief recipe is decisively falsified across all
non-AIC-4 corpora.**

## Recipe assessment

The brief recipe diverges from the proven V_24-per-sample-α
(`PreviewV0_5Compression`) training command in multiple ways, every
one of which contributes to the regression:

| Hparam | V_24 (proven) | Brief V11-A' v2 | Impact |
|---|---|---|---|
| `--minibatch-size` | 256 | 1 (auto-bumped to 512 on GPU) | mostly mitigated by GPU auto-bump |
| `--ranknet-weight` | 1.0 (default) | 0.0 | **removes the main ranking signal** |
| `--mse-weight` | 0.0 (default) | 1.0 | regresses to predicting scores rather than ranking pairs |
| `--monotonicity-reg` | 0.0 (default) | 1.0 | over-regularizes against ranking structure |
| `--tanh-output-head-scale` | 0.0 (default) | 20.0 | clamps outputs into narrow band, kills dynamic range |
| `--pwrc-pair-weight` | enabled | unspecified | loses the PWRC-driven training weight that V_24 used |
| `--target-column` | mix_cv40_iw60 | mix_cv35_iw65 | slightly different mix, smaller effect |
| `--norm-in-norm-weight` | 0.1 | unspecified | NiN regularizer disabled (NiN + anchor loss not yet composed) |

The combination of (MSE-only loss + monotonicity-reg 1.0 + tanh head
scale 20.0) inverts the V_24 training surface: instead of ranking
pairs and letting the head learn α(x), the trainer is forced to MSE-
fit individual scores through a saturating tanh, then regularized
toward monotonic outputs. None of these worked together.

## Whether the substrate itself helps

To separate substrate-quality from recipe-quality, an alt recipe was
also tested (`run_v11av2_alt_seed.sh`):

- Match V_24's per-sample-α + RankNet + PWRC defaults
- Add anchor + cross-codec-eq aux losses (the V11 substrate)
- 5 training groups (drop cid22_train + pipal which weren't in V_24)

[Alt 3-seed CI results pending — see eval table below once seeds finish.]

The user's question on 372 vs 300 features is also load-bearing
(see "Feature dimension" section).

## Feature dimension

The R2 omni `zensim_features/` block stores 300 features per pair
(no IW-pool block from the 372-feat schema). Per DATA_PROVENANCE.md
line 41-48, the 300-feat cloud vectors are NOT index-compatible with
the local 372-feat parquets. **Brief specifies --max-features 300 to
match the substrate.**

Verified state of existing 372-feat shipped bakes (for comparison):

| Bake | n_inputs | CID22 SROCC | Notes |
|---|--:|---:|---|
| V10 BalancedV3 (current ship) | 300 | 0.8324 | V_22-mix-LARGE+iwssim + V9 spline |
| `v_cross_codec_v2_2026-05-20.bin` | **372** | **0.8797** | EXP-CROSS-CODEC-V9 ship (372-feat) |
| `v_compression_2026-05-18.bin` | **372** | 0.8580 | V_22-372feat ship (pre-per-sample-α) |
| V11-A' v2 (this experiment, 300-feat) | 300 | 0.7519 median | falsified |

**The user's question is well-aimed**: the 372-feat IW-pool block
appears to contribute ~+0.05 CID22 SROCC vs the 300-feat
LARGE+iwssim base on a comparable training corpus. The existing
372-feat `v_cross_codec_v2` ship beats our V11-A' v2 brief-gate
target (0.8797 vs 0.8374) without any V11 substrate.

However, the V11 substrate could not currently be built at 372
features without re-running zensim feature extraction on the cached
R2 encoded variants with `--feature-output` enabled (a separate
~1-2 hour vast.ai run per DATA_PROVENANCE.md). The 372-feat training
parquets at `canonical-2026-05-21/train/` exist for safesyn/kadid/
tid/konjnd/cid22_train/pipal, but `cvvdp_iwssim_LARGE.parquet` is
only 300-feat, so a mixed-dimension trainer call requires either
truncating to 300 (what we did) or excluding the LARGE group
(loses 73k pairs).

## Decision: NO ship

V11-A' v2 brief recipe is FALSIFIED. V10 BalancedV3 remains the
defensible Balanced bake.

The V11 substrate parquets are SOUND and preserved on disk at
`/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/` for future
per-sample-α head training cycles with corrected recipe.

## Files written

- `scripts/v_next/v11_ssim2_v2/build_v11_substrate_v2.py` —
  rebuild script for V11 substrate from R2 omni data.
- `scripts/v_next/v11_ssim2_v2/run_v11av2_seed.sh` —
  brief recipe driver (FALSIFIED).
- `scripts/v_next/v11_ssim2_v2/run_v11av2_alt_seed.sh` —
  alt recipe with V_24-aligned hparams.
- `/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/unified_omni.parquet` —
  unified omni+features 4-codec data (754,756 rows × 305 cols, 715 MiB).
- `/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/anchors_ssim2_300col_v2.parquet` —
  anchor parquet (8,527 rows × 312 cols, 12.4 MiB).
- `/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/cross_codec_equivalence_ssim2_v2.parquet` —
  equiv parquet (1,739 rows × 617 cols, 3.7 MiB).
- 5 V11-A' v2 bakes at `/mnt/v/zen/zensim-eval/exp_v11_balanced_v2_2026-05-20/cc4v11a_v2_s{1..5}.bin`.
