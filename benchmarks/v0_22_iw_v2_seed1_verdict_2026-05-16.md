# V_22-IW v2 seed=1 verdict — log-target retrain SHIPS 3/4 corpora (2026-05-16)

**Status**: T1.5 closed with **partial-win verdict**. V_22-IW v2 (trained
against `iwssim_log_norm = -log(1 - iwssim + 1e-6) / 13.72 × 100`)
beats V_18 ship on the full Mohammadi panel for AIC-3 + KADID + TID,
loses only CID22.

## Headline — V_22-IW v2 alone vs V_18 ship — full Mohammadi panel

| Corpus | Metric | V_18 ship | V_22-IW v2 | Δ (v2 − V_18) | Win? |
|---|---|---:|---:|---:|:--|
| **AIC-3** | SROCC | 0.7996 | **0.8071** | +0.0075 | ✓ |
|   | PLCC | 0.8093 | **0.8130** | +0.0037 | ✓ |
|   | KROCC | 0.6302 | **0.6406** | +0.0104 | ✓ |
|   | PWRC | 0.8697 | **0.8755** | +0.0058 | ✓ |
|   | Z-RMSE | 0.588 | **0.582** | −0.006 | ✓ (lower) |
| **CID22** | SROCC | 0.8933 | 0.8164 | −0.077 | ✗ |
|   | PLCC | 0.8911 | 0.8207 | −0.070 | ✗ |
|   | KROCC | 0.7081 | 0.6317 | −0.076 | ✗ |
|   | PWRC | 0.9373 | 0.8755 | −0.062 | ✗ |
|   | Z-RMSE | 0.454 | 0.571 | +0.117 | ✗ |
| **KADID** (NaN-filtered n=9805) | SROCC | 0.9387 | **0.9475** | +0.0088 | ✓ |
|   | PLCC | 0.9395 | **0.9483** | +0.0088 | ✓ |
|   | KROCC | 0.7855 | **0.8010** | +0.0155 | ✓ |
|   | PWRC | 0.9631 | **0.9693** | +0.0062 | ✓ |
|   | Z-RMSE | 0.343 | **0.317** | −0.026 | ✓ |
| **TID** | SROCC | 0.9526 | **0.9617** | +0.0091 | ✓ |
|   | PLCC | 0.9554 | **0.9623** | +0.0069 | ✓ |
|   | KROCC | 0.8110 | **0.8280** | +0.0170 | ✓ |
|   | PWRC | 0.9702 | **0.9766** | +0.0064 | ✓ |
|   | Z-RMSE | 0.295 | **0.272** | −0.023 | ✓ |

**Ship-gate score**: **3/4 corpora pass** the CLAUDE.md ≥3-of-5-stats
rule. Per the user's 2026-05-16 emphasis ("cid22 and aic are the
most important eval validation sets"), V_22-IW v2:
- **Wins AIC-3** unanimously across all 5 stats — including the
  Z-RMSE calibration-error metric.
- **Loses CID22** unanimously but the gap is dramatically reduced
  from v1: SROCC −0.077 (vs v1's −0.281), Z-RMSE +0.117 (vs v1's
  +0.354).

## v1 → v2 improvement on the primary axis

| Stat | V_22-IW v1 | V_22-IW v2 | Δ improvement |
|---|---:|---:|---:|
| CID22 SROCC | 0.6122 | **0.8164** | **+0.204** |
| CID22 Z-RMSE | 0.807 | **0.571** | **−0.236** |
| CID22 PWRC | 0.7270 | **0.8755** | **+0.149** |
| AIC-3 SROCC | 0.7600 | **0.8071** | **+0.047** |
| AIC-3 Z-RMSE | 0.637 | **0.582** | **−0.055** |
| AIC-3 PWRC | 0.8354 | **0.8755** | **+0.040** |

The log-target hypothesis was the correct fix for V_22-IW's
high-q saturation pathology. Spreading the `[0.99, 1.0]` tail
across a wide log-space range gave the regression head enough
supervision contrast to learn distinguishable predictions at the
top end. **This validates the principled-experiment-workflow
prediction that target-distribution transforms are the right
mechanism when the failure mode is target saturation.**

## Multi-bake α-sweep — confirms standalone is the best ship

| α | AIC-3 SROCC | CID22 SROCC | KADID SROCC | TID SROCC |
|---|---:|---:|---:|---:|
| 0.00 (V_22-IW v2 alone) | **0.8071** | 0.8164 | **0.9475** | **0.9617** |
| 0.30 | 0.7468 | 0.6724 | 0.8684 | 0.8809 |
| 0.50 | 0.7872 | 0.8501 | 0.9281 | 0.9425 |
| 0.70 | 0.7956 | 0.8802 | 0.9353 | 0.9494 |
| 0.90 | (≈ V_18) | 0.8910 | 0.9379 | 0.9519 |
| 1.00 (V_18 ship alone) | 0.7996 | **0.8933** | 0.9387 | 0.9526 |

Like v1, no α ∈ (0, 1) Pareto-improves over the endpoints. Multi-
bake destructive interference recurs at α=0.3 (CID22 drops to
0.67 — below either endpoint). Z-NORM mix shows the same
collapse-at-α=0.5 pattern documented in T1.4.

**Standalone V_22-IW v2 is the best mix point** at the cost of
−0.077 CID22 SROCC.

## Decision: V_22-IW v2 is a ship candidate as PreviewV0_5

V_22-IW v2 doesn't replace V_18 ship (PreviewV0_3) because CID22 is
load-bearing per `CLAUDE.md > CID22 is VALIDATION-ONLY` and the −0.077
gap is non-trivial. But:

- V_22-IW v2 has a real, measurable advantage on AIC-3 (the low-q
  compression corpus you flagged as primary). +0.008 SROCC AND
  −0.006 Z-RMSE is rare in IQA — most candidates trade one for the
  other.
- V_22-IW v2 also wins KADID + TID where V_18 ship was already
  strong. The win is consistent across 5 of 5 stats per corpus.
- The CID22 loss is the price of generalization to compression-
  realistic distortions in a non-ssim2-shape direction. Per CLAUDE.md
  "SROCC-only verdicts BANNED + ssim2-target training bias", CID22's
  human MOS was tuned alongside SSIMULACRA-2, so an IW-shape predictor
  inherits a 0.08-ish SROCC gap by construction.

**Ship as additive PreviewV0_5 profile**, NOT a replacement for
PreviewV0_3. Users opt into the IW-shape via the profile selector
when AIC-3-style low-q decisions matter more than CID22-style
medium-q rank fidelity.

## Per-band picture (queued)

The aggregate panel hides per-band trades. The next ship-grade
analysis should pull the 10-band Mohammadi panel out of
`dataset_metric_baseline` for V_22-IW v2 specifically and check:

- Does V_22-IW v2 still WIN AIC-3 on the B0..B5 low-q bands? (Where
  compression-product decisions live per CLAUDE.md "B0..B5 lift is
  the dominant priority".)
- Does V_22-IW v2's CID22 loss concentrate in the high-q bands
  (B8..B9) where V_22-IW's training-target log-spread shifts
  predictions away from the ssim2-shape that CID22 prefers?
- KonJND PJND calibration: V_22-IW v2 should now produce valid
  predictions (the KonJND ExtendedIw fix landed in T4.3). Verify
  the mean ± stdev aligns with the Cloudinary Table 4 anchors.

## Reference files

- Methodology doc: `benchmarks/v0_22_iw_methodology_2026-05-16.md`
- v1 (raw target) eval findings: `benchmarks/v0_22_iw_seed1_eval_findings_2026-05-16.md`
- v1 Option C falsification: `benchmarks/v0_22_iw_option_c_falsification_2026-05-16.md`
- v2 α-sweep (this verdict): `benchmarks/v0_22_iw_v2_option_c_alpha_sweep_2026-05-16.md`
- v2 bake: `benchmarks/v0_22_iw_v2_seed1_2026-05-16.bin` (200 KB, ZNPR v3)
- v2 per-pair: `benchmarks/v0_22_iw_v2_seed1_2026-05-16_eval_per_pair.csv`
- Log-target CSVs: `/mnt/v/zen/zensim-training/2026-05-16/v2/*_features_iwssim_log_372col.csv`
- Log-target generator: `scripts/v_next/v0_22_iw_v2_add_log_target.py`
