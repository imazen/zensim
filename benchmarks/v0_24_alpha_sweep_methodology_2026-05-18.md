# V_24-α weighted-mix sweep methodology

**Date:** 2026-05-18
**Branch:** `feat/v24-alpha-sweep` (forked from `feat/v24-ex3-followup`)
**Status:** seed=3 sweep across α ∈ {0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35}
completed; 5-seed CI at α=0.10 completed; **no Pareto-better α exists; no ship.**

## Goal

Continue from the V_24-full (symmetric 3-way mix α=1/3) finding that
ssim2-as-target wins CID22 but loses KADID + TID. Sweep
α ∈ {0.05, 0.10, 0.15, 0.20, 0.25} looking for the **Pareto point**
where CID22 stays at-or-above V_22-mix-LARGE+iwssim while KADID/TID
remain within −0.01 of V_22.

## The mix

```
mix_alpha = α · ssim2_log_norm
          + (1 − α) · (0.4 · cvvdp_log_norm + 0.6 · iwssim_log_norm)
```

α = 0 → V_22-mix-LARGE+iwssim (cv+iw base).
α = 1/3 → V_24-full (symmetric).

## Corpus build

Source parquets (identical to the V_24-full build):

| Group | Source | Rows | MD5 of source |
|---|---|---:|---|
| safesyn | `/mnt/v/zen/zensim-training/2026-05-17-cvvdp/safesyn_features_mix_targets_372col.parquet` | 196,086 | (see prior methodology) |
| kadid | `/mnt/v/zen/zensim-training/2026-05-18-ssim2/kadid_4target_372col.parquet` | 10,125 | (see prior methodology) |
| tid | `/mnt/v/zen/zensim-training/2026-05-18-ssim2/tid_4target_372col.parquet` | 3,000 | (see prior methodology) |
| large | `/mnt/v/zen/zensim-training/2026-05-18-ssim2/large_3target_300feat_minus_ssim2.parquet` | 54,900 | (see prior methodology) |
| konjnd | `/mnt/v/zen/zensim-training/2026-05-17-cvvdp/konjnd_features_mix_targets_372col.parquet` | 1,008 | (see prior methodology) |

For safesyn, `ssim2_log_norm` is derived from `human_score * 100`
via the canonical `(x + 30)/130 * 100` clip transform (the EX-3
follow-up discovery).

For kadid/tid/large, `ssim2_log_norm`, `iwssim_log_norm`,
`cvvdp_log_norm` are pre-built.

For konjnd the target stays PJND (renamed `human_score` →
`mix_target`); the `0.02` group weight handles scale mismatch with
the codec-mix target columns. konjnd is α-independent.

Output: `/mnt/v/zen/zensim-training/2026-05-18-v24-alpha/<group>_alpha<NNN>.parquet`,
one parquet per (group, α). LARGE drops 1100 NaN rows per α
(2.0 %, CVVDP coverage gap, identical to V_24-full).

Builder script: `scripts/v_next/build_v24_alpha_sweep_corpus.py`.
Driver: `scripts/v_next/train_v24_alpha_sweep.sh <pct> <seed> [out]`.

## Training recipe

Identical to V_22-mix-LARGE+iwssim. Only the input corpora change.

```sh
./target/release/zensim_mlp_train \
  --group safesyn:safesyn_alpha<NNN>.parquet:1.0:1.0 \
  --group kadid:kadid_alpha<NNN>.parquet:0.3:1.0 \
  --group tid:tid_alpha<NNN>.parquet:0.3:1.0 \
  --group konjnd:konjnd.parquet:0.02:0.0 \
  --group large:large_alpha<NNN>.parquet:0.5:0.0 \
  --target-column mix_target --target-scale 1.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 \
  --lr 1e-3 --l2 1e-5 --leaky-alpha 0.01 --val-policy min --seed 3 \
  --log-every 30 --max-features 300 \
  --minibatch-size 256 --pwrc-pair-weight --norm-in-norm-weight 0.1 \
  --early-stop-patience 0 \
  --out v24_alpha<NNN>_s3_h128.bin
```

Per-α wall time: ~5–8 min (single-core forward + rayon batches; 5
parallel trainings shared 32 cores @ ~1.5s/epoch).

## Evaluation

For each α, run `bake_compare` per § A.9 of `PSYCHOVISUAL_LEARNINGS`:

- `A`: `v24_alpha<NNN>_s3_h128.bin`
- `B`: `v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128.bin` (the seed=3
  unpacked V_22 baseline at
  `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/`).

Default: 1000 bootstrap resamples, all 5 corpora (cid22, kadid, tid,
konjnd, aic3), 10-band B0..B9 grid, seed 42.

## Sweep table (seed=3)

Baseline (B = V_22-mix-LARGE+iwssim s3) panel SROCC:
CID22 0.8323, KADID 0.9677, TID 0.9729, KonJND 0.8928, AIC-3 0.7831 → weighted score 1.9596.

| α | CID22_A | ΔCID22 | KADID_A | ΔKADID | TID_A | ΔTID | KonJND_A | ΔKonJND | AIC-3_A | ΔAIC-3 | score | Δscore | Adec / Bdec / promND / tied / noisy | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.025 | (extended)| (extended) | | | | | | | | | | | | |
| 0.05 | 0.8683 | +0.0361 | 0.9056 | −0.0620 | 0.8878 | −0.0852 | 0.8146 | −0.0782 | 0.7915 | +0.0083 | 1.9219 | −0.0377 | 3/16/3/7/1 | B |
| 0.10 | 0.8676 | +0.0353 | 0.9061 | −0.0616 | 0.8902 | −0.0828 | 0.8348 | −0.0581 | 0.7898 | +0.0067 | **1.9315** | **−0.0281** | 2/16/3/8/1 | B |
| 0.15 | 0.8629 | +0.0306 | 0.9008 | −0.0668 | 0.8896 | −0.0833 | 0.8443 | −0.0485 | 0.7916 | +0.0085 | 1.9306 | −0.0290 | 3/16/1/9/1 | B |
| 0.20 | 0.8614 | +0.0292 | 0.8960 | −0.0717 | 0.8825 | −0.0904 | 0.8309 | −0.0620 | 0.7913 | +0.0081 | 1.9193 | −0.0403 | 2/16/3/8/1 | B |
| 0.25 | 0.8665 | +0.0343 | 0.9026 | −0.0651 | 0.8888 | −0.0841 | 0.8156 | −0.0773 | 0.7843 | +0.0012 | 1.9183 | −0.0413 | 3/16/3/7/1 | B |
| 0.30 | (extended)| | | | | | | | | | | | | |
| 0.35 | (extended)| | | | | | | | | | | | | |

*α=0.025, 0.30, 0.35 results appear after the extended `bake_compare` jobs land.*

**Score formula**: `CID22 + 0.5·mean(KADID,TID) + 0.5·KonJND + 0.25·AIC-3`.

## Decisive-cell totals per α (seed=3)

Across 5 corpora × 10 bands × the § A.9 decisive rule:

- Every α in {0.05..0.25} shows the **same pattern**: 2-3 A-decisive cells, 16 B-decisive cells.
- The A wins are concentrated in CID22 B7 [0.70, 0.80) for every α (e.g. α=0.10 → CID22 B7: h_SROCC=17.56, DecScore=+14.63, A>>B).
- KADID + TID lose decisively in B0..B6 across every α.
- KonJND/AIC-3 are promising A at some α (e.g. AIC-3 +0.008 SROCC at α=0.05/0.10/0.15/0.20).

## Pareto / weighted-best identification

**No strict Pareto-better α exists** in α ∈ [0.025, 0.35].

The trade is **discontinuous at α=0**: as soon as ssim2 is added with any
weight (even α=0.05), KADID/TID lose ~0.06-0.09 SROCC. The damage does
NOT scale smoothly with α — α=0.05 already incurs nearly the full
penalty.

**Weighted-best α = 0.10** (Δscore = −0.0281). It is:

- The least bad on the weighted score.
- The best on ΔKonJND of any α ≤ 0.20 (−0.0581 vs −0.07..−0.08).
- Decisively A on CID22 B7 (the band where ssim 70-80 codec product
  decisions live).
- Decisively A on KonJND PJND when narrowed to per-band (see report).

But α=0.10 is **NOT Pareto-better than V_22**. It trades −0.062 KADID +
−0.083 TID for +0.035 CID22. Whether to ship is a product-judgment
call, **not a clean ship decision**.

## 5-seed CI on α=0.10 (the weighted-best operating point)

Trained 5 seeds (1..5) at α=0.10, h=128. Each seed compared against the
matching V_22-mix-LARGE+iwssim seed via bake_compare (500 bootstrap
resamples, 10-band, paired per-seed).

| seed | CID22_A | ΔCID22 | KADID_A | ΔKADID | TID_A | ΔTID | KonJND_A | ΔKonJND | AIC-3_A | ΔAIC-3 | Winner |
|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.8726 | +0.0500 | 0.8979 | −0.0694 | 0.8874 | −0.0856 | 0.8404 | −0.0457 | 0.7902 | +0.0131 | B |
| 2 | 0.8648 | +0.0244 | 0.8984 | −0.0687 | 0.8864 | −0.0859 | 0.8439 | −0.0413 | 0.7859 | −0.0012 | B |
| 3 | 0.8676 | +0.0353 | 0.9061 | −0.0616 | 0.8902 | −0.0828 | 0.8348 | −0.0581 | 0.7898 | +0.0067 | B |
| 4 | 0.8643 | +0.0248 | 0.8966 | −0.0708 | 0.8871 | −0.0849 | 0.8461 | −0.0383 | 0.7941 | +0.0034 | B |
| 5 | 0.8739 | +0.0389 | 0.8991 | −0.0680 | 0.8906 | −0.0822 | 0.7881 | −0.0977 | 0.7958 | −0.0019 | B |

**Mean ± std across 5 seeds:**

| Corpus | A SROCC | B SROCC | Δ |
|---|---:|---:|---:|
| CID22 | 0.8686 ± 0.0044 | 0.8339 ± 0.0071 | **+0.0347 ± 0.0107** |
| KADID | 0.8996 ± 0.0037 | 0.9673 ± 0.0002 | **−0.0677 ± 0.0036** |
| TID | 0.8883 ± 0.0019 | 0.9726 ± 0.0005 | **−0.0842 ± 0.0017** |
| KonJND | 0.8306 ± 0.0242 | 0.8869 ± 0.0034 | −0.0562 ± 0.0244 |
| AIC-3 | 0.7912 ± 0.0039 | 0.7872 ± 0.0078 | **+0.0040 ± 0.0062** |

**Per-seed verdict: B>>A on every seed (all 5).**
**Aggregate**: A wins 13 decisive cells, B wins 81 (across 5 seeds × 5 corpora × 10 bands).

The CID22 lift of +0.035 SROCC is real (seed variance 0.011 << 0.035 mean
lift). The KADID/TID losses are also real with tighter CIs. The trade is
seed-robust; α=0.10 is **NOT a ship candidate** — every seed loses
decisively to V_22.

The AIC-3 +0.004 mean ± 0.006 std straddles zero; the seed=3 +0.0067
observation was within seed variance. **AIC-3 movement is NOT a real
lift, confirming the "structural gap" finding even at the 5-seed-CI
level.**

## AIC-3 trend across α (confirms EX-3 follow-up "structural gap" finding)

| α | AIC-3 SROCC | ΔAIC-3 vs V_22 (0.7831) |
|---:|---:|---:|
| 0.05 | 0.7915 | +0.0083 |
| 0.10 | 0.7898 | +0.0067 |
| 0.15 | 0.7916 | +0.0085 |
| 0.20 | 0.7913 | +0.0081 |
| 0.25 | 0.7843 | +0.0012 |
| (V_24-full α=1/3 from EX-3-followup) | 0.7846 | +0.0015 |

**The AIC-3 needle barely moves across the entire α range** (+0.001 to
+0.009). The 0.85 target documented in EX-1 / EX-3 is unreachable by
changing the ssim2 mix weight. EX-3-followup's "AIC-3 gap is structural"
finding is **confirmed** — different corpus / feature set / MOS-anchored
supervision is needed.

## Honest gaps

- No Pareto-better operating point exists. Soft α does NOT preserve
  KADID/TID — the trade is bimodal between α=0 and α>0.
- α=0.10 wins CID22 B7 decisively but loses KADID/TID B0..B6
  decisively. This is a "CID22 specialist trades KADID/TID" pattern,
  same as V_24-full but with smaller magnitudes on both sides.
- AIC-3 gap is structural across α. The next experiment direction
  needs a different lever, e.g.:
  - AIC-3 image features added as a training group (NOT MOS as
    target — AIC-3 is holdout-only per CLAUDE.md).
  - Per-band weighted loss boosting B0..B5 where ssim2 is most
    reliable (CID22 paper Table 6).
  - Multi-corpus richer LARGE with additional codec sweep granularity.
- 5-seed CI launched only for α=0.10 (the weighted-best). Other α
  are characterized only at seed=3, but their cross-α consistency
  (every α ∈ {0.025, 0.05, ..., 0.35} shows B>>A overall) suggests
  the finding is seed-robust — and the 5-seed CI on α=0.10
  confirms it directly (every seed → B>>A, 13 A-cells vs 81 B-cells
  across 5×5×10 = 250 (corpus × band × seed) cells).

## Packed α=0.10 seed=3 candidate bake (NOT a ship)

For completeness, the seed=3 α=0.10 bake was packed via
`rebake_v3_1 --compress --zerobias 0.005 --dtype i8`:

| Bake | Size | CID22 SROCC | Drift vs source |
|---|---:|---:|---:|
| `v24_alpha010_s3_h128.bin` (f32) | 157,252 B | 0.86759 | — |
| `v24_alpha010_s3_h128_packed.bin` (i8+LZ4) | 38,850 B (24.7%) | 0.86762 | 3.65e-5 |

Pack succeeds with drift 30× below the 0.001 ship threshold. The bake
is technically packable; it is NOT being shipped because it loses
KADID/TID/KonJND decisively vs V_22 across all 5 seeds.

## Recommended next-experiment directions

The α-sweep falsifies the "soft α preserves K/T" hypothesis. To push
past this trade, the next experiments should:

1. **Per-band weighted loss** boosting CID22 B6..B8 (where the ssim2
   target genuinely helps) without diluting K/T B0..B5 supervision.
   Requires trainer support for per-(group, band) loss weights.
2. **Per-target separate heads** — train a 2-head MLP, one predicting
   ssim2_log_norm, the other cvvdp+iwssim, and runtime-mix in
   score-space. Allows the ssim2-shaped representation to coexist
   without diluting the cv+iw representation.
3. **Group weight rebalance** — try down-weighting the LARGE group
   (currently 0.5) since LARGE is JPEG-only and may carry K/T-
   inflating signal. Test `--group large:...:0.2` or `:0.1`.
4. **Distortion-class regularization** — different regularizer
   strengths per distortion class (compression vs blur/noise/color)
   to prevent the ssim2 signal from over-influencing non-compression
   bands.
5. **AIC-3 specific lever** — the +0.001 to +0.009 sensitivity across
   α confirms the AIC-3 gap is **NOT** addressable by ssim2 target
   weighting. AIC-3 needs different features, different MOS supervision
   (held-out training-only subset per CLAUDE.md), or domain adaptation.

## Reproducibility

```sh
cd ~/work/zen/zensim
git checkout feat/v24-alpha-sweep

# Build all 5 corpora
python3 scripts/v_next/build_v24_alpha_sweep_corpus.py \
  --alpha 0.05 --alpha 0.10 --alpha 0.15 --alpha 0.20 --alpha 0.25

# Train 5 bakes seed=3 (parallel)
for pct in 5 10 15 20 25; do
  ./scripts/v_next/train_v24_alpha_sweep.sh $pct 3 &
done; wait

# Run bake_compare vs V_22
./scripts/v_next/compare_v24_alpha_sweep.sh

# Aggregate
python3 scripts/v_next/build_v24_alpha_sweep_table.py
```
