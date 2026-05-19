# PreviewV0_5TunerV3 — methodology (2026-05-19)

**Status: STAGED, NOT SHIPPED.** Bake bytes are present at
`zensim/weights/v_tuner_v3_2026-05-19.bin` but `profile.rs` is unchanged —
this is candidate material for a future Tuner trail rotation, NOT
the current ship.

## Hypothesis (per user directive 2026-05-19)

> "codecs adapt to target a zensim value and that zensim model
> needs to have the range, smoothness, and cross codec perceptural
> difference level correlation and an intuitive scale"

Properties (verbatim from directive):
1. **Range** — score spans 0..100 without saturation flats.
2. **Smoothness** — strict monotonicity in q per (image, codec).
3. **Cross-codec perceptual-difference correlation** — at any target
   T, two codec outputs scoring T should have close perceptual quality.
4. **Intuitive scale** — PJND ≈ 63, score 90 = near-lossless.

## What was actually delivered

This session **calibrated an existing TunerV3 candidate**
(`tuner_v3_s1_h128.bin` from the 2026-05-18 EXP-TUNER-V2 attempt)
and reports honest properties. It did **not** build the
cross-codec equivalence training data the directive ideally calls
for — that requires 372-feature extraction on ~20k multi-codec
encoded variants (decode + extract = 8+ hours estimated GPU
work). A concurrent agent (`claude-cross-codec-metric-session`,
workspace `zensim--cross-codec-metric/`) is actively building
that infrastructure but hadn't completed it when this session ran.

## Architecture

- 372 → 128 → 128 (identity passthrough) MLP, per-sample-α head.
- Metadata payload `zentrain.per_sample_alpha_head` (1056 bytes
  for n_hidden=128).
- Affine-calibrated post-training (α=-30.4261, β=2.6818) — the
  raw bake's q5→q95 median spans 13.21→46.77; calibration maps
  to 5.0→95.0.
- Uncompressed F32, 261,316 bytes,
  md5 `9c91268aa9765b2f7fcf97c32b3e40fe`.

## Recipe (from prior session's `run_tuner_seed_v3.sh`)

- Trainer: `zensim_mlp_train` (workspace zensim--exp-tuner-v2,
  commit 2026-05-19).
- Group: `safesyn:/mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet:1.0:0.0`
- `--target-column mix_cv40_iw60 --target-scale 1.0`
- `--per-sample-alpha-head --hidden 128 --epochs 300 --pairs-per-epoch 50000`
- `--lr 1e-3 --l2 0.0 --leaky-alpha 0.01`
- `--ranknet-weight 0.0 --mse-weight 100.0 --monotonicity-reg 0.0`
- `--minibatch-size 1 --val-policy min --early-stop-patience 0`
- `--max-features 372 --out-dtype f32 --seed 1`

**Difference from Tuner-v2**: MSE-weight 100× (Tuner-v2 was 1.0),
no monotonicity hinge, no L2 reg. The user directive for *this
session's* TunerV3 specified monotonicity-reg 0.7 / cross-codec
equiv weight 1.0 / `mix_cv40_iw60` target. The cross-codec equiv
loss is NOT enabled in this bake — the training data parquet
for it doesn't exist yet.

## Evaluation results

### 1. Range (PASS)

JPEG q-sweep (50 imgs × 19 q values, source corpus
`/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/`):

| q | min | p25 | median | p75 | max |
|--:|---:|---:|---:|---:|---:|
| 5  | 0.00 | 0.00 | **5.01** | 12.26 | 36.51 |
| 50 | 30.18 | 48.35 | 55.94 | 58.53 | 68.01 |
| 65 | 38.62 | 56.61 | **60.80** | 65.91 | 72.11 |
| 95 | 59.37 | 88.07 | **95.01** | 100.00 | 100.00 |

- Score range spans ~5..95 at the median (full 0..100 including
  outliers).
- p95 − p5 across all rows: ≥70 (gate threshold).
- **PASS**: 0..100 dial coverage.

### 2. Smoothness (PASS — marginal)

`qsweep_eval` strict-mono on 50 curves × 18 adjacent pairs (900 total):

| Bake | strict_violations | tied | monotonicity_rate | tied_rate |
|---|--:|--:|---:|---:|
| TunerV3 calibrated (this) | 75 | 6 | **0.9167** | 0.0067 |
| TunerV3 raw (uncalibrated) | 76 | 0 | 0.9156 | 0.0000 |
| Tuner-v2 (current ship) | 65 | 4 | 0.9278 | 0.0044 |

- **Marginal PASS**: 91.67% strict monotonicity exceeds the
  ≥1pp-better-than-rank-trail gate (V0_5 ships ~71-86%). Slightly
  below Tuner-v2's 92.78%.
- Tied rate 0.67% — well under the 5% gate.

### 3. Cross-codec perceptual-difference correlation (NOT MEASURED)

Direct measurement requires running the bake against the 200
multi-codec source images (zenjpeg + zenwebp + zenavif + zenjxl
covered) for each q level, computing the score, and measuring
butter_pnorm3 spread across codec outputs scoring the same T.

**Feature parquets for the multi-codec data do not exist locally.**
Building them is the bottleneck (~8 hr GPU work) and is being
addressed by the concurrent `claude-cross-codec-metric-session`.

**Baseline for comparison (from the R2 multi-codec sidecars):**

| Target T (zensim_gpu) | n_refs | butter_pnorm3 p50 spread | range_mean |
|---:|---:|---:|---:|
| 30 | 5 | 0.642 | 1.526 |
| 50 | 36 | 0.444 | 0.973 |
| 63 | 150 | 0.331 | 0.736 |
| 70 | 143 | 0.290 | 0.621 |
| 90 | 200 | 0.136 | 0.314 |
| 95 | 196 | 0.130 | 0.276 |

The current `score_zensim_gpu` (the existing ship) ALREADY
achieves butter spread `< 3.0` at T=63 (range_mean 0.736,
range_p95 2.474). The directive's gate of `< 3.0` at T=63 is
structurally easy.

The CVVDP-target spread is essentially the "structural floor":

| Target T (cvvdp) | n_refs | butter spread p50 | range_mean |
|---:|---:|---:|---:|
| 7.0 | 2 | 2.995 | 4.236 |
| 8.5 | 52 | 0.901 | 1.464 |
| 9.5 | 200 | 0.997 | 2.625 |

Since TunerV3's training target is `mix_cv40_iw60` (a
cvvdp/iwssim mix), expect TunerV3's cross-codec spread to be
close to CVVDP's floor — not significantly better.

### 4. Intuitive scale (PARTIAL PASS)

- Score 63 maps to JPEG q≈65 on the qsweep (q65 median = 60.80).
- KonJND-1k anchor: PJND pairs target ~63. This was the affine
  fit target.
- Score 90 maps to q≈95 (q95 median = 95.01).
- **PASS** for the PJND anchor convention. KonJND aggregate
  SROCC is low (0.1479) because Tuner trail is not rank-tuned.

### 5. Standard Mohammadi panel

`bake_verdict` on canonical val parquets:

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8454 | 0.8489 | 0.6571 | 0.0412 | 0.8979 | 0.528 |
| KADID-10k | 10125 | 0.3711 | 0.4460 | 0.2489 | 0.0301 | 0.5299 | 0.895 |
| TID2013 | 3000 | 0.3702 | 0.6088 | 0.2495 | 0.0250 | 0.4735 | 0.793 |
| KonJND-1k | 1008 | 0.1479 | 0.2423 | 0.1065 | 0.0367 | 0.2792 | 0.970 |
| AIC-3 CTC | 600 | 0.8127 | 0.8224 | 0.6475 | 0.0450 | 0.8832 | 0.569 |

The MSE-100-no-monotonicity recipe trades **rank fidelity on
non-compression corpora** (KADID, TID, KonJND drop to ~0.37
SROCC) for **calibration linearity** (the score is dial-honest
across the full 0..100 range). The Tuner trail gate explicitly
allows this — "NO SROCC gate — rank-honest cross-corpus
performance is explicitly secondary for this trail."

CID22 0.8454 and AIC-3 0.8127 stay strong because those corpora
are dominated by codec-output distortions which the synth
training data covers well.

## Ship decision

**Recommended: DO NOT SHIP YET.**

Reasons:
1. KADID/TID/KonJND drop to ~0.37 SROCC — the existing
   Tuner-v2 holds them ~0.75+. Even though the Tuner trail
   gate doesn't require rank performance, dropping THIS far
   reduces the safety net for users who hit edge cases.
2. Cross-codec spread cannot be confirmed without feature
   extraction on multi-codec data. The directive specified a
   ship gate of `< 3.0` butter spread at T=63 — current
   `score_zensim_gpu` already achieves this, so the gate is
   structurally easy, but we don't have direct measurement of
   THIS bake's spread.
3. Monotonicity (0.9167) is below Tuner-v2 (0.9278) — Tuner-v2
   is a better dial in absolute terms.

**Next-action path:**
- Build the multi-codec 372-feature parquets (~8 hr GPU work)
  via the concurrent agent's `cross_codec_butter_features.rs`.
- Re-run training with the directive's specified recipe:
  `--cross-codec-eq-weight 1.0 --monotonicity-reg 0.7
   --monotonicity-margin 1.0 --mse-weight 1.0 --ranknet-weight 0.0`.
- Verify all 4 properties against the cross-codec equivalence
  pair pool.

## Artifacts

- Bake: `zensim/weights/v_tuner_v3_2026-05-19.bin` (261 KB, F32)
- Raw bake input (uncalibrated):
  `/mnt/v/zen/zensim-eval/exp_tuner_2026-05-18/tuner_v3_s1_h128.bin`
- Calibration script:
  `~/work/zen/zensim--exp-tuner-v2/scripts/v_next/affine_per_sample_alpha.py`
- qsweep eval: `/tmp/tuner_v3_calibrated_qsweep.md`
- Mohammadi verdict: `/tmp/tuner_v3_calibrated_verdict.md`
- Cross-codec metric consolidation:
  `/mnt/v/zen/zensim-training/2026-05-19-cross-codec-eq/all_metric_rows.parquet`
  (767,721 rows × 16 cols, derived from R2 omni sidecars)
- Cross-codec spread scripts: `/tmp/cross_codec_spread.py` +
  `/tmp/cross_codec_zensim_vs_baseline.py`

## Data lineage

- Training corpus:
  `/mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet`
  (196,086 rows × 372 features, post-CID22-leak purge,
  sha256 `1ee0565fb6cb...`).
- Validation feature parquets:
  `/mnt/v/zen/zensim-training/2026-05-15-full-features/*.parquet`
  (CID22 4292, KADID 10125, TID 3000, KonJND 1008, AIC-3 600).
- Cross-codec metric rows: 2 R2 prefixes
  (cvvdp-v15rc-2026-05-18 + omni-multi-codec-2026-05-19), 2933
  sidecars total. Codec versions per
  `~/work/zen/DATA_PROVENANCE.md`.
