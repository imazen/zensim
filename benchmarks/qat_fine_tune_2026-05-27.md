# QAT fine-tune in the trainer — methodology + findings (2026-05-27)

Goal (user directive): make the Rust trainer emit the packed+calibrated bake
NATIVELY (no Python recal/pack post-step). Implemented quantization-aware
fine-tune + native f16 storage + anchor-spline-on-the-shipped-net.

## Mechanism (3 commits)

1. **QAT forward STE** (`c5c5aed7`). `--qat-fine-tune-epochs N` (default 0 =
   off): the last N epochs refresh the f32 forward scratch (`WeightScratchF32`)
   from f16+zerobias COPIES of the master weights (`qat_quantize_copy` =
   `apply_zero_bias_per_layer_in_place` + `f32_to_f16_bits` round-trip,
   matching the bake). Adam keeps updating the f32 master → straight-through
   estimator. Forward/loss see the quantized weights → the net learns weights
   robust to packing. Clean 2-site change: the forward already reads the
   scratch, so no master/restore dance.
2. **Anchor-spline on the SHIPPED net** (`310c6aac` + fix `742e8a7`). Replaced
   the degenerate auto-spline (band-by-target + 0.01-pred-filter → 2 knots →
   broken dial) with `fit_monotone_spline` (quantile-bin → ~18 knots). **The
   spline-fit forward MUST use the projected (encoder≥0, rank_w≤0, α≡1) +
   quantized net** — i.e. exactly what the bake ships. Forwarding the
   un-projected net inverts the pred↔target correlation (projection flips
   signs) → spline direction wrong → blur scored UP to 2184. With the
   projected+quantized forward, corr = +0.88 → direction correct.
3. **Native f16 storage** (`55df657b`). `bake_per_sample_alpha_head_v3_2layer`
   now honors `--out-dtype`: encoder layers f16, identity passthrough kept
   F32, `compressed=true` when quantizing. Also fixed the latent bug where
   the 2-layer CPU path ignored `--out-dtype` entirely (every 2-layer bake
   was f32).

Recipe: `zensim/weights/manifests/v47_strict_qat.toml` (out_dtype f16,
qat_fine_tune_epochs 40, qat_tau 0.005) — one `zensim_mlp_train --manifest`
pass, no post-steps.

## Verification (full v47 recipe, 200 epochs, last 40 QAT)

The QAT **network** (no spline) is excellent — CID22 SROCC **0.8657**, ABOVE
the non-QAT recal's 0.8564 and close to V39's 0.8793. QAT did NOT hurt the
network. Blur ladder on the raw net: 0 inversions, 0 above-identity (monotone
preserved through quantization — signs survive f16+zerobias).

QAT-recal full panel (`v47_qat_recal_verdict_2026-05-27.md`):

| Corpus | QAT-recal | non-QAT recal-negtail | V39 |
|---|--:|--:|--:|
| CID22 SROCC | **0.8657** | 0.8564 | 0.8793 |
| CID22 Z-RMSE | **0.514** | 0.541 | 0.493 |
| CID22 DS-AUC | 0.8135 | 0.808 | 0.817 |
| KADID | 0.7933 | 0.8030 | 0.9251 |
| TID | 0.7927 | 0.7965 | 0.9317 |
| **KonJND** | **0.4185** | 0.485 | 0.4197 |
| AIC-3 | 0.768 | 0.770 | 0.8023 |
| AIC-4 | 0.8854 | 0.8902 | 0.9051 |
| identity / blur | 94.6 / 0 above-id, [−142,95] | 97.8 / 0 above-id, [−190,98] | broken (0) / 31 above-id |
| dial G1 | 0.98 | 0.99 | 1.00 |
| size | 34 KB | 30 KB | 257 KB |

## The QAT trade (honest)

QAT is a **trade, not a pure win**: +CID22 (+0.009, the gold standard) +
better CID22 Z-RMSE, but **−KonJND (−0.067)** — the f16+zerobias removes
fine-weight precision the near-lossless/PJND discrimination needs (both QAT
and non-QAT fail G5's 0.70 floor regardless — the characterized HF Pareto
limit). KADID/TID marginally lower (within noise of the strict-monotonicity
cost). The big win is **native one-pass packing** (no Python recal/pack) +
the CID22/calibration gain.

`qat_tau` sweep (0.005 → 0.002) in flight to test whether a gentler zerobias
recovers KonJND while keeping the CID22 gain + native packing.

## Status

- QAT network + spline-projection fix VERIFIED (CID22 0.8657, correct dial,
  0 above-identity, 34 KB).
- Native re-run (binary with all 3 fixes) in flight to confirm the
  one-pass native bake == the recal'd result without ANY post-step.
- `qat_tau=0.002` run in flight (KonJND-recovery test).
- "Flip to default": QAT is the v47_strict_qat.toml recipe standard. Whether
  to make it the GLOBAL trainer default is gated on the tau result (KonJND
  trade acceptability) — keeping it recipe-level (opt-in) is the safe default
  given the KonJND regression.
