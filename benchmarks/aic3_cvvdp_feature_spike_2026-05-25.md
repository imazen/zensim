# AIC-3 CVVDP-feature spike — feature-limit NOT confirmed (2026-05-25)

Decisive test of the "feature-information-limited" hypothesis on AIC-3: does
handing the MLP the raw CVVDP score as an input feature close the AIC-3 human-
JND SROCC gap? If yes, CVVDP-approximating CSF features are worth building.

Bakes (safesyn-trained, ssim2 target, per-sample-α head, h=128×2, seed=17):
`spike_cvvdp_feat_baseline_safesyn_2026-05-25.bin` (372) and
`spike_cvvdp_feat_cvvdp_safesyn_2026-05-25.bin` (373, f372 = raw CVVDP JOD).
CVVDP computed via `zen-metrics batch --metric cvvdp --gpu-runtime cuda` on the
600 AIC-3 pairs, row-aligned to `aic3_features_372col_2026-05-15.parquet`
(verified 600/600 by matching `score.jnd` + `ref_basename`).

## Results

| Scope | raw CVVDP alone | baseline (372) | +CVVDP (373) | delta |
|---|---:|---:|---:|---:|
| Pooled aggregate (600, 10 refs) | 0.7918 | 0.7865 | 0.7910 | +0.0045 |
| Per-ref averaged (n=10) | 0.9342 | 0.9475 | 0.9518 | +0.0043 |

(AIC-3-provided CVVDP on the PTC 5-image subset pools to 0.9606 — that is the
source of the "~0.96" figure that motivated the hypothesis; it is a 5-image
aggregate, not the full-set number.)

## Verdict — NOT feature-information-limited

- The safesyn baseline (no CVVDP) already scores per-ref SROCC **0.9475**,
  higher than raw CVVDP-alone (0.9342). No per-ref gap to close.
- Handing the MLP the *real* CVVDP score (the perfect such feature) adds only
  **+0.004** SROCC, mixed per-ref signs. CSF features approximating CVVDP would
  add even less.
- The apparent ~0.80 "gap" is a cross-ref **scale-calibration** artifact of
  pooling 10 source images with different JND scales — not a feature-information
  deficit. Bake outputs are near-constant on AIC-3's near-imperceptible
  distortions (std ≈0.05); the CVVDP feature shifts the mean but does not spread
  the distribution. Fixing the pooled number needs per-ref absolute-scale
  calibration, not better features.

Full verdict + artifacts (CVVDP scores TSV, augmented parquets, per-pair
predictions, reconstruction/verification scripts):
`/mnt/v/output/zensim/bakes/spike_cvvdp_feat_2026-05-25/AIC3_DECISIVE_VERDICT_2026-05-25.md`

Diagnostic-only code change: `bake_verdict` gained `--per-pair-output <path>`
(parquet-row-order `human<TAB>pred` dump) for per-ref SROCC computation.
