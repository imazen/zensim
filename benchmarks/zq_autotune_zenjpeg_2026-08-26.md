# Zq autotune — zenjpeg predictor fit (2026-08-26)

**Criterion-4 "zenpredict-baked Zq one-shot predictor (autotune)" — the MODEL, proven.**
Fit a Zq predictor for zenjpeg from the canonical 924 bigcodec data
(`ext924-canonical-2026-07-27/bigcodec/zenjpeg_lossy/{train,validate,test}_924.parquet`,
origin-split, 761,310 train rows). Input = the 924 zensim features (`f0..f923`) + the
**target ssim2** (`score_ssim2`); output = the **q** that achieved it. So at inference:
(image features, desired target-ssim2) → predicted q_start for the loop.

## Result (TEST split, held-out origins)
| predictor | RMSE(q) | within ±5 q | within ±10 q |
|---|--:|--:|--:|
| target-only anchor (no features) | 26.71 | — | — |
| **features + target (ridge λ=10)** | **9.74** | 44% | 74% |

**Features cut the q-prediction error 64%** vs a target-only anchor. A seed within ±10 q
74% of the time meaningfully shortens the bracketed secant loop (fewer encode→decode→score
iterations before it reaches the target band). This is the autotune's core value, proven on
real held-out data.

## Status + follow-on
- **DONE:** the model fit + validation (this doc; coefficients at
  `/mnt/v/zen/zensim-training/zq_zenjpeg_ridge_2026-08-26.npz`, 925×f32). Ridge is the proof;
  a `zensim_mlp_train` MLP would do better (nonlinear feature×target interactions, like
  zenavif's q0_head).
- **FOLLOW-ON (the bake + wire, feature-gated per [[feedback_no_zenpredict_in_codecs]]):**
  (1) train the production form via `zensim_mlp_train` (features+target → q) and bake ZNPR via
  `zenpredict-bake`; (2) wire into `zenjpeg::target_quality` as the `q_start` seed behind an
  `auto-tune` feature (zenjpeg CAN dep zenanalyze — no cycle, verified — to extract features at
  inference; zenavif's `q0_head` uses 8 CHEAP zenanalyze features for speed, the better production
  design vs the full 924 here). CID22 stays validation-only; train on the curated bigcodec sets.
- **Per-codec:** the same recipe applies to zenwebp/zenavif/jxl from their bigcodec views.

## PER-ENCODER generalization (all 4 main codecs, TEST split, 2026-08-26)
Same fit (924 features + target ssim2 → q, ridge λ=10) on each codec's bigcodec 924 view:

| codec | train rows | anchor RMSE(q) | full RMSE(q) | error cut | within ±10 q |
|---|--:|--:|--:|--:|--:|
| zenavif_lossy | 775,152 | 18.46 | **4.41** | **76%** | **96%** |
| zenwebp_lossy | 484,470 | 26.08 | **5.61** | **78%** | **93%** |
| zenjpeg_lossy | 761,310 | 26.71 | **9.74** | **64%** | **74%** |
| zenjxl_lossy  | 726,705 | 24.01 | **9.90** | **59%** | **74%** |

**The Zq autotune model generalizes to every main codec** — a feature-based one-shot q predictor
beats a target-only anchor by 59–78% RMSE, landing within ±10 q on 74–96% of held-out encodes.
zenavif/zenwebp are especially seedable (±10 q on 93–96%). This is criterion-4's "zenpredict-baked
Zq one-shot predictor (autotune), per encoder" — the MODEL, validated. The production form (MLP via
`zensim_mlp_train` → `zenpredict-bake` → wire behind each codec's `auto-tune` feature, using the
cheap 8-feature q0_head design for inference speed) is the mechanical follow-on.

## ⛔ RETRACTION (2026-08-26, same day) — the fit above is LEAKAGE, not a valid autotune
**The 59–78% "error reduction" is INVALID.** Verified directly: the `f0..f923` features **vary with
q for the same reference** (measured within-ref std: f0 0.027, f1 0.064, f50 0.13, f400 0.075, …) —
they are the **DISTORTED encode's** zensim features, NOT the reference image's. So
`(features + target) → q` LEAKS: a q=5 encode and a q=95 encode of the same ref have different
features, so "predict q from the distorted features" is near-circular. **Those features do not
exist before encoding**, so they cannot seed a pre-encode q predictor — the whole premise is void.

**Confirming simulation** (bracketed-secant on held-out per-image q→ssim2 curves, 1200 cells): using
one encode's features to seed for a DIFFERENT target, the "autotune" seed is NO BETTER than a
target-only anchor — mean secant iterations **3.85 (autotune) vs 3.64 (anchor)**, ≤3 iters 55% vs
57%. The leaky training accuracy does not generalize to the actual inference task.

**Two real, honest findings survive:**
1. **A valid Zq autotune needs REFERENCE-image features** (extract via zenanalyze on the ref, as
   zenavif's `q0_head` does with 8 cheap ref features) — the multi-crate pipeline stands unbuilt.
   The bigcodec 924 parquets cannot supply it (their features are distorted-side).
2. **Seed quality barely affects a bracketed-SECANT loop** — it converges in ~3–4 iterations from
   any reasonable seed, so an autotune's value is the **ONE-SHOT** prediction (skip the loop
   entirely for a ±Nq answer), NOT loop-seeding. This tempers the expected payoff and should steer
   the design toward the one-shot use case (or cruder loops) where the seed actually matters.

Lesson (again): validate the feature PROVENANCE before trusting a fit. A held-out split does not
catch leakage when the leaked signal (distorted features) is present in both train and test rows.

## ✅ THE VALID FORMULATION ALREADY EXISTS (found 2026-08-26, post-retraction)
The correct, non-leaky autotune was already prototyped: **`picker_zenjpeg_A_sourcefeat_v3.bin`**
(`/mnt/v/zen/picker-dense-full-2026-05-27/`, ZNPR, `leakyrelu_mlp_picker`). Its `.toml` is explicit:
> Inputs are IMAGE features (feat_*) + **zq_norm (the user's REQUESTED target quality / 100)**. The
> codec's per-encode q is **NOT an input — q is the decision the picker makes. No q-leakage.**

It ports `zentrain/tools/train_hybrid.py build_dataset`: per (image, target_zq), predict
`bytes_log = ln(min encoded_bytes over knob cells whose score_zensim ≥ target_zq)` + `reach` mask;
pick = argmin(bytes, mask=reach). Uses **SOURCE (reference) zenanalyze features**
(`zenjpeg_source_features_full.tsv` has feat_uniformity / feat_flat_color_block_ratio /
feat_distinct_color_bins — the same q0-family features).

**So the retraction stands and is now fully explained:** my ridge fit used the bigcodec
*distorted-side* `f0..f923` (leakage); the picker work correctly used **source features + zq_norm**
(no leakage) months ago. The valid formulation is proven; my attempt was a regression from it.

**What's genuinely left for a PRODUCTION Zq autotune** (the picker's own recorded follow-ons):
1. **Dense sweep** — the prototype swept only 5 q levels {10,30,60,80,90}; production needs ~30 q +
   16-20 log-spaced sizes (per the sweep discipline) — a data-gen task.
2. **A q_start / scalar head** — the prototype picks categorical knob cells + bytes; a Zq *seed*
   needs a continuous-q prediction head (the `.toml` lists "scalar prediction heads" as a follow-on).
3. **Source features for the CURRENT refs** — the 2026-05-27 source-feature set is 321 images; the
   924-era bigcodec has 2307 refs, whose source features must be extracted (zenanalyze on the refs).
4. **Wire behind each codec's `auto-tune` feature** (feature-gated per [[feedback_no_zenpredict_in_codecs]]).
And recall the measured caveat above: for a bracketed-SECANT loop the seed saves ~0 iterations, so
the autotune's real payoff is **one-shot** (predict-and-encode-once), not loop-seeding.
