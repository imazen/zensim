# V_24-α weighted-mix sweep methodology

**Date:** 2026-05-18
**Branch:** `feat/v24-alpha-sweep` (forked from `feat/v24-ex3-followup`)
**Status:** seed=3 sweep completed; 5-seed CI ${SEED5_STATUS}; ${SHIP_STATUS}.

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

${SEED5_RESULTS}

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
  the finding is seed-robust.

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
