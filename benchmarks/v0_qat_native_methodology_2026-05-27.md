# v47-strict-QAT-native — ship methodology doc (2026-05-27)

This is the consolidated pre-ship methodology doc (CLAUDE.md shipping-policy
item #6) for the **QAT-native** axiom-clean metric bake, the strongest
candidate to replace the broken V39 at `Profile::A`. It supersedes the
scattered findings in `qat_fine_tune_2026-05-27.md`,
`v47_qat_native_verdict_2026-05-27.md`, and
`qsweep_qat_native_vs_v39_2026-05-27.md` by pulling them into one
verifiable record.

**Ship-form decision is USER-GATED** (see §h). This doc makes the swap a
one-line `include_bytes!` change once the user picks replace-vs-sibling.

---

## (a) Architecture + parameter count + size + hash

- **Bake**: `/mnt/v/output/zensim/bakes/v47_strict_qat_native_2026-05-27.bin`
- **sha256**: `d0ef7a3054d1ed9e70086d306cda69b71fc95072c6ef3351f362f27da096d4fc`
- **md5**: `802f0c4675bc4dfc32c6985fdfa5b6ad`
- **size**: 27,316 bytes (ZNPR v3, flags=3 = f16+compressed)
- **n_inputs**: 372 features (basic+peak+masked+IW-pool blocks)
- **Architecture**: `372 → 128 → 64` encoder (LeakyReLU 0.01) + per-sample-α
  head + tanh output pin (scale=30), **masked-monotone-by-construction**
  (W1≥0 on the 300 sign-safe features, rank_w≤0, α≡1 under `--monotone-cbc
  --monotone-strict` so the shipped net is `y = y_rank`).
- **Layer dtypes** (from `zenpredict inspect`):
  - L1 372→128: **f16**, 47,616 weights + 128 biases, 92.7% near-zero (zerobias)
  - L2 128→64: **f16**, 8,192 weights + 64 biases, 75.2% near-zero
  - L3 64→64 identity passthrough: **f32** (kept full-precision; the per-sample
    head reads it as the post-LeakyReLU hidden vector)
- **Calibration**: monotone PCHIP dial spline in `zentrain.output_calibration_spline`
  metadata, fit IN-PASS on the projected+quantized net (§d).

## (b) Trainer command + hyperparameters + inputs

One invocation, no Python post-steps:

```sh
zensim_mlp_train --manifest zensim/weights/manifests/v47_strict_qat.toml
```

Recipe (full provenance in the TOML):

| hyperparameter | value |
|---|---|
| seed | 17 |
| hidden / n_hidden_layers | 128 / 2 |
| per_sample_alpha_head | true |
| epochs | 200 (last **40 quantization-aware**) |
| qat_fine_tune_epochs / qat_tau | 40 / 0.005 |
| out_dtype | **f16** (+ zerobias + compress) |
| lr / l2 / minibatch | 1e-3 / 1e-4 / 32 |
| mse_weight / ranknet_weight | 0.6 / 0.6 |
| monotonicity_reg | 1.0 |
| tanh_output_head_scale | 30.0 |
| monotone_cbc / monotone_strict | true / true |
| monotone_feature_mask | `benchmarks/feature_sign_mask_2026-05-26.tsv` (300 pin_geq0 / 72 free) |
| target_column | `human_score` (per-group normalized to [0,1]) |
| anchor_loss_weight | 0.01 (spline-fit only — NOT a training target) |
| val_aggregate | geomean3 (SROCC·PLCC·PWRC) |
| auto_transforms | Yeo-Johnson cross-corpus-safe screen set |

Training inputs (canonical-2026-05-21, see §g for hashes). **CID22 human MOS
is NOT a training target** — `cid22_train` is the TRAINING-ONLY subset of the
broader CID22 library (MCOS/100), disjoint from the 49-ref held-out
validation set.

## (c) Lineage

Single trained net — **not** built from prior bakes (no ensemble/concat/KD).
The QAT mechanism: the last 40 epochs refresh the f32 forward scratch from
f16+zerobias COPIES of the master weights (straight-through estimator); Adam
keeps updating the f32 master so the net learns weights robust to packing.
The bake then stores those weights f16+zerobias directly — what trained is
what ships.

## (d) Calibration

Monotone PCHIP spline (`fit_monotone_spline`, ~18 quantile-bin knots) fit
in-pass over the per-row `multiband_anchor_dial100.parquet`. **The spline-fit
forward uses the projected (encoder≥0, rank_w≤0, α≡1) + f16-quantized net —
exactly what the bake ships.** (Forwarding the un-projected net inverts the
pred↔target correlation because projection flips signs → blur scored UP;
fixed in commit `742e8a7`, corr now +0.88.) Result: identity = 97.69 (the
dial max), 0 inversions, 0 above-identity, negative tail to −131.

**Global corruption-gate ordering verified on the ship bake** (gb82/dog,
clamp post): identity **97.69** > honest-q20 **40.36** > channel-invert-whole
**12.21** > block-zero-whole **0.00** — the regression-test ordering
(identity > honest-lq > broken decode) that V39 inverts (V39 identity = 0 on
every ref). Localized 8×8 defects are a separate global-metric limit → #33
(Approach-B local signal validated, op100 92.5%/81.2%).

## (e) Held-out panel (bake_verdict, full Mohammadi panel)

Aggregate per corpus (10-band + 4-band CID22 + step-5 tables in
`v47_qat_native_verdict_2026-05-27.md`):

| Corpus | n | SROCC | PLCC | KROCC | PWRC | Z-RMSE | DS-AUC |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8657 | 0.8591 | 0.6742 | 0.9782 | 0.512 | 0.8134 |
| KADIK10k | 10125 | 0.7933 | 0.7899 | 0.5959 | 0.9494 | 0.613 | 0.7249 |
| TID2013 | 3000 | 0.7927 | 0.8171 | 0.6024 | 0.9685 | 0.577 | 0.7753 |
| KonJND-1k | 1008 | 0.4185 | 0.3627 | 0.2872 | 0.7915 | 0.932 | 0.5413 |
| AIC-3 CTC | 600 | 0.7680 | 0.7845 | 0.5977 | 0.9334 | 0.620 | 0.7047 |
| AIC-4 sample | 300 | 0.8854 | 0.8768 | 0.7051 | 0.9756 | 0.481 | 0.8347 |

Goal scorecard (measurable subset): **G1 dial 0.97, G7 CID22 1.00**, G8 0.60,
G9 0.03, **G5 HF 0.23** (the characterized KonJND/AIC HF Pareto limit —
falsified across two architectures, see CLAUDE.md). Weighted 0.622.

## (f) Monotonicity / non-mono q-step rate

Measured on the real JPEG q-sweep (50 imgs × 19 q, the G3 measurement
bake_verdict can't do — `qsweep_qat_native_vs_v39_2026-05-27.md`):

| bake | monotonicity | non-mono | tied | dial median q5→q95 |
|---|--:|--:|--:|---|
| **qat_native** | **0.9433** | 5.67% | **0.33%** | 1.40 → 88.50 (every step ↑) |
| recal_negtail | 0.9378 | 6.22% | 0.44% | 4.61 → 88.43 |
| **v39 (shipped)** | 0.6767 | 32.3% | **53.6%** | broken: peaks q25, **→0.00 q55–q95** |

QAT-native is the **best dial of the field** — highest monotonicity, lowest
tied rate, clean monotone-increasing median across the full range. The 5.67%
non-mono is above the 6.0% advisory only at the top-of-dial tanh-pin window
where high-q increments compress (still strictly monotone, not a break).

## (g) Data-lineage table

| input | path (canonical-2026-05-21/train) | sha256 (prefix) | rows | CID22-contam |
|---|---|---|--:|---|
| safesyn | safesyn.parquet | `ad15cc79` | 196,086 | leak-purged (2026-05-12) |
| cid22_train | cid22_train_norm.parquet | `59c9888a` | 17,611 | TRAIN-ONLY subset, disjoint from 49-ref holdout |
| kadid | kadid.parquet | `83356f03` | 10,125 | dHash d≤10 audited |
| tid | tid.parquet | `6704de0c` | 3,000 | dHash d≤10 audited |
| konjnd_dense | konjnd-dense-norm.parquet | `5595a922` | 20,160 | n/a (PJND anchor) |
| anchor (spline) | multiband_anchor_dial100.parquet | `594b3df5` | — | NOT a training target |
| feature_sign_mask | benchmarks/feature_sign_mask_2026-05-26.tsv | `5d0e066d` | 372 | — |
| auto_transforms | …/screen_results_cross_corpus_safe.tsv | `ada1f1ce` | — | — |

All validation (CID22 49-ref / KADID / TID / KonJND / AIC-3 / AIC-4) is
held-out per `canonical-2026-05-18/val/`. CID22 human MOS never entered
training.

## (h) Honest gaps — what QAT-native does WORSE than V39, and the trade

| vs V39 | Δ | note |
|---|--:|---|
| CID22 SROCC | −0.014 | advisory; QAT-native still 0.8657 > 0.85 gate |
| KADID SROCC | −0.132 | the 72 dropped sign-flip features' analytic-distortion signal |
| TID SROCC | −0.139 | same — KADID/TID are ~95% non-compression synthetic distortions (integrity guards, not the primary compression target) |
| KonJND SROCC | −0.001 | both fail G5 0.70 (HF Pareto limit); f16 removes fine PJND precision — non-QAT recal keeps 0.485 if HF is the priority |
| AIC-3 / AIC-4 | −0.034 / −0.020 | minor |

**Why ship anyway (the trade):** V39 is *genuinely broken* at the regime
regression tests live in — it scores **identity = 0 on every reference**
(raw ≈ −90 → clamp) and **blur > identity** (31 above-identity on the blur
ladder), violating two of the three similarity axioms. Its dial is
non-invertible (53.6% tied, high-q → 0.00). QAT-native fixes both defects
by construction, is the best codec dial measured, keeps CID22 above gate,
and is 10× smaller (27 KB). The KADID/TID rank loss is on synthetic
non-compression distortions that are integrity guards, not the
compression-dial target that motivates zensim. For the user-facing
"type a target score" use case, QAT-native strictly dominates V39.

**The only reason NOT to replace V39 outright:** downstream consumers using
`Profile::A` purely for *ranking* (not the dial) on KADID/TID-like
analytic-distortion content would see lower rank-SROCC. If that matters, the
ship form is "add sibling profile, keep V39 too" rather than "replace."

### Ship-form options (USER decision)

1. **Replace V39 at `Profile::A`** — bake rotation (CLAUDE.md explicitly
   permits swapping the shipped weight to advance goal #1; no crate bump).
   Every consumer gets the axiom-clean dial. Recommended: V39 is broken.
2. **Add a sibling profile** (`Profile::ADial` or similar) — keeps V39's
   higher analytic-distortion rank for pure-ranking consumers. Cost: a new
   public enum variant (needs API approval).
3. **Hold** — keep gathering multi-ref corruption-gate data first.

The non-QAT `recal_negtail` (30 KB, KonJND 0.485) is the HF-priority
alternative if KonJND/PJND discrimination outweighs the native-packing +
CID22 gain.
