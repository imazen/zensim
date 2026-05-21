# Acumen Mode A — 4-architecture comparison (local CUDA)

**Date**: 2026-05-21
**Tracking**: [imazen/zensim#40](https://github.com/imazen/zensim/issues/40)
**Status**: ALL 3 variants FALSIFIED — castleCSF Mode A modulation
hurts more, the more invasive the application.

## Question

After Gate A Path B showed the original HfPost wiring (multiply
HF slots 10/11/12 by per-(scale, ch) castleCSF weight) loses
-0.012 SROCC on held-out CID22, the architectural finding doc
proposed 3 alternative wirings to test:

1. **HfPost** (original): scale slots 10/11/12 only
2. **WideModulation**: scale all 13 basic features per (scale, ch)
3. **AuxFeatures**: append 12 CSF weights as f228..f239 columns
   without modifying base features (let MLP learn to use them)

This benchmark tests all 3 against a no-acumen baseline using
identical hyperparams.

## Method

- Train: KADID (10125) + TID (2950) + AIC-3 (600) features
- Val (held-out): CID22 (4292)
- All 4 archs: `extract_acumen_features` with `--acumen-arch <variant>`
- Trainer: `zensim_mlp_train --hidden 64 --epochs 50
  --pairs-per-epoch 30000 --lr 0.001 --seed 1`
- AuxFeatures uses `--max-features 240`; others 228

## Results

| Arch | best epoch | KADID | TID | AIC-3 | **CID22 (held-out)** | Δ vs baseline |
|---|--:|--:|--:|--:|--:|--:|
| **baseline** (no acumen) | 30 | 0.9124 | 0.2635 | 0.9467 | **0.7044** | — |
| HfPost (slot 10-12) | 30 | 0.9101 | 0.2795 | 0.9493 | **0.6924** | **-0.0120 ↓** |
| WideModulation (slots 0-12) | 30 | 0.9101 | 0.2803 | 0.9312 | **0.6740** | **-0.0304 ↓** |
| AuxFeatures (228+12) | 30 | 0.9096 | 0.3189 | 0.9525 | **0.6159** | **-0.0885 ↓** |

## Pattern: monotonic degradation with invasiveness

- HfPost (12 features modulated): -0.012 SROCC
- WideModulation (52 features modulated): -0.030 SROCC
- AuxFeatures (12 cols added, no modulation but bigger MLP): -0.089 SROCC

**The more aggressive the acumen intervention, the worse the
held-out CID22 result.**

## Why AuxFeatures is the WORST

Intuition was that letting the MLP LEARN to use CSF weights as
context would help. Empirically it's the most damaging. Two
reasons:

1. **The aux columns are nearly-constant per reference image**
   — they only depend on ref's mean luminance, which has limited
   variation. Effectively per-ref bias, not per-pair signal.
2. **With only ~13.7k training pairs, the extra 12 dims overfit**
   — the MLP spends first-layer capacity on the constant aux cols
   instead of useful per-pair structure.

The MLP can't extract usable signal from features that don't vary
within a (ref, set-of-dists) cluster.

## Why HfPost / WideModulation lose

Information theory: post-hoc multiplicative scaling cannot ADD
information; it can only PRESERVE (with multiplier=1) or REMOVE
(with multiplier in [0,1)). The castleCSF weights at typical
viewing conditions include 0.03 factors that wipe HF signal the
MLP would otherwise use. Wider modulation = wider information
loss.

## Implications

**castleCSF Mode A as a feature-vector intervention does not
help any zensim quality-prediction MLP on these val corpora.**

The signal castleCSF carries (per-band perceptual weighting)
appears to be:
- Already captured implicitly by the multi-scale pyramid +
  per-scale stats the existing 228 features encode
- OR not directly correlated with quality-prediction utility
  on this distortion distribution

For acumen to add value to a quality metric, the intervention
must enter the pipeline **before pooling** — when band content
exists spatially, not when it's been collapsed to scalar
statistics. The three remaining architectural paths:

1. **CSF-weighted band energy DURING pyramid construction**:
   multiply each pyramid level's coefficients by the band's CSF
   weight before computing the SSIM/MSE/etc. stats. Touches the
   kernel, not the post-pool feature vector.
2. **Mode B (per-pixel L_adapt)**: spatial luminance variation
   adapts CSF weights spatially. Adds local-adaptation signal
   that image-mean Mode A can't.
3. **CSF as a parallel head**: separate MLP head that takes the
   CSF weights + reference statistics as input, blends with the
   main MLP output. Lets CSF inform the score directly rather
   than via input features.

## What was tried, what wasn't

| Tested | Result |
|---|---|
| HfPost (slots 10-12) | -0.012 |
| WideModulation (slots 0-12) | -0.030 |
| AuxFeatures (12 extra cols) | -0.089 |
| Mode B per-pixel | NOT TESTED (kernel work needed) |
| In-pyramid weighting | NOT TESTED (kernel work needed) |
| Parallel CSF head | NOT TESTED (architecture work needed) |

The three tested variants share a common failure mode: they treat
castleCSF as a post-extraction transform of pooled statistics.
The signal needs to enter spatially or as a separate inference
pathway to have a chance.

## Compute cost

- Extraction: 8 val parquets × ~1.5 min = ~12 min on local CUDA
- Training: 4 MLPs × 30s = 2 min on CPU
- Total: ~15 minutes
- Cost: $0 (local electricity)

vs vast.ai 3-parallel estimate: ~3 hours + ~$0.70. Local was the
right move — answer in 15 minutes instead of 3 hours.

## Verdict

**FALSIFIED — all 3 architectural variants of acumen Mode A as a
post-pool intervention hurt held-out CID22 SROCC** by 0.012,
0.030, and 0.089 respectively. The pattern is monotonic in
invasiveness: bigger intervention, bigger loss.

Recommend: do NOT ship any acumen Mode A variant. The next
investment (if pursued) must reach into pre-pool processing
(in-pyramid weighting) or change the inference architecture
(parallel CSF head). Both are substantially larger code
changes; defer until there's a stronger motivation than the
current null/negative signal.

The acumen FOUNDATION code (LUT, ViewingCondition, Phase 4
hooks, sweep image, GPU example) remains preserved on
`feat/acumen-foundation` + `feat/acumen-gpu` branches — useful
prerequisite for future Mode B / pre-pool work, and for any
HDR-IQA effort where viewing condition genuinely matters.
