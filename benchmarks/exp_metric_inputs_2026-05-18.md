# EXP-METRIC-INPUTS — FALSIFIED (2026-05-18)

**TL;DR**: Adding ssim2/cvvdp/iwssim per-pair scores as 3 additional MLP input
features (303 total) does NOT generalize. Wins CID22 (+0.007 / +0.04 vs
Compression / Balanced ship), but loses decisively on every other corpus —
including catastrophic -0.47 / -0.56 SROCC regression on KonJND.

## Hypothesis (FALSIFIED)

Adding 3 strongly-aligned metric scores (ssim2 / cvvdp / iwssim) as MLP inputs
gives the network a "warm start" toward perceptual ranking. Expected lift
+0.005 to +0.02 CID22.

## Result — fails both trail gates per § A.10

### vs Compression ship (V_24-per-sample-α s4)

| Corpus | exp-s4 | Compression | Δ | bake_compare verdict |
|---|---:|---:|---:|---|
| CID22 | 0.8711 | 0.8641 | +0.0070 | **A>>B** decisive |
| KADID | 0.8474 | 0.9316 | -0.0842 | B>>A decisive |
| TID | 0.8113 | 0.8893 | -0.0780 | B>>A decisive |
| KonJND | 0.3320 | 0.8080 | -0.4760 | B>>A decisive |
| AIC-3 | 0.8012 | 0.8183 | -0.0171 | B>>A decisive |

**Compression trail gate**: requires A>>B on ≥1 of {CID22, AIC-3} AND
not decisive B>>A on the other AND no single corpus regression worse than
-0.10 on {KADID, TID, KonJND}.

- CID22: A>>B ✓
- AIC-3: B>>A ✗ (fails the "not decisive B>>A on the other compression
  corpus" clause)
- KonJND: -0.476 (way worse than -0.10) ✗
- KADID: -0.084 ✓ (within tolerance)
- TID: -0.078 ✓ (within tolerance)

**FAILS Compression gate.**

### vs Balanced ship (V_22-mix-LARGE+iwssim)

| Corpus | exp-s4 | Balanced | Δ |
|---|---:|---:|---:|
| CID22 | 0.8711 | 0.8324 | +0.0387 |
| KADID | 0.8474 | 0.9677 | -0.1203 |
| TID | 0.8113 | 0.9729 | -0.1616 |
| KonJND | 0.3320 | 0.8927 | -0.5607 |
| AIC-3 | 0.8012 | 0.7845 | +0.0167 |

**Balanced trail gate**: requires A>>B decisive on CID22 AND not decisive
B>>A on any of {KADID, TID, KonJND, AIC-3}.

- KADID, TID, KonJND all regress decisively. **FAILS Balanced gate.**

## 5-seed reproducibility table

| seed | CID22 | KADID | TID | KonJND | AIC-3 |
|---:|---:|---:|---:|---:|---:|
| s1 | 0.8679 | 0.8311 | 0.8089 | 0.4044 | 0.7995 |
| s2 | 0.8743 | 0.8295 | 0.8187 | 0.3342 | 0.7990 |
| s3 | 0.8694 | 0.8260 | 0.8113 | 0.3866 | 0.8011 |
| s4 | 0.8711 | 0.8474 | 0.8113 | 0.3320 | 0.8012 |
| s5 | 0.8762 | 0.8371 | 0.8143 | 0.3167 | 0.7983 |
| median | 0.8711 | 0.8311 | 0.8113 | 0.3342 | 0.7995 |
| mean | 0.8718 | 0.8342 | 0.8129 | 0.3548 | 0.7998 |

Median CID22 seed = s4 (used for bake_compare).

## Baseline controls (from `baseline_panels_2026-05-18.md`)

Per-corpus aggregate SROCC of the raw metric features:

| Corpus | ssim2 | cvvdp | iwssim |
|---|---:|---:|---:|
| CID22 | 0.8895 | 0.8214 | 0.7836 |
| KADID | 0.8133 | 0.8339 | 0.8498 |

Even individual metrics outperform the trained MLP on KADID (8133 ssim2 vs
0.847 exp). The MLP didn't successfully integrate the metric inputs.

## Diagnosis — why does it lose so badly?

1. **f300/f301/f302 metadata mismatch between train and val**. The canonical
   training kadid/tid parquets carry constant per-image ssim2_gpu values
   (the upstream join in `2026-05-18-v24` joined ssim2 by
   `(image_path, codec, q, knob)` but the local KADID ssim2 sidecar has only
   `I01.png/kadid/0/{}` as the key → first-row wins per ref). The MLP
   trained on KADID never saw a varying ssim2 signal for that corpus.
   When val/kadid presents real per-pair ssim2 (range [0, 100], huge
   variance), the MLP extrapolates badly.

2. **KonJND has no metric supervision in training**. konjnd-dense has
   ALL-NaN for ssim2/cvvdp/iwssim, filled with safesyn-natural-scale means
   (constant input). The MLP learned that f300/f301/f302 carry no
   per-pair information for KonJND-distribution images. At inference
   on val/konjnd we also fill constants (no val join key) — so KonJND
   should be unaffected by the new inputs. The -0.47 SROCC regression
   suggests the per-sample-α head is reading a different `h` from the
   network and producing degenerate behavior.

3. **The per-sample-α head's pool head depends on all 128 hidden units**.
   Adding 3 inputs reshapes the network's first-layer weights, which
   propagates through the entire pool head's reducer weights. The net
   effect: a different surface entirely, not "old surface + extra signal".

4. **Training-corpus distribution shift on the metric inputs**. Safesyn
   (70% of training) has ssim2 in [-739, 99] with std 31.8. Val/CID22 has
   ssim2 in [25, 93] with std 10.7. Val/KADID has ssim2 in [-367, 100] with
   std 49.7 (from per-pair CSV which uses fast-ssim2 CPU, NOT GPU
   ssim2_gpu). The val/train distribution shift on f300 alone may be the
   dominant failure mode.

## Implementation notes

- 303 features = 300 base + ssim2 (f300) + cvvdp (f301) + iwssim (f302),
  all in natural metric scale (ssim2 ∈ [0, 100], cvvdp ∈ [0, 10], iwssim
  ∈ [0, 1]). Trainer's input scaler Z-scores per-feature.
- Imputed values for NaN: safesyn-natural-scale mean (ssim2=68.74 median,
  cvvdp=9.59, iwssim=0.97).
- Build script: `scripts/exp_metric_inputs/build_augmented_parquets.py`.
- Train script: `scripts/exp_metric_inputs/run_metric_inputs_seed.sh`.
- Bakes: `/mnt/v/zen/zensim-eval/exp_metric_inputs_2026-05-18/metric_inputs_s{1..5}_h128.bin`
  (225 KB each, 5 seeds).
- Verdict files: `verdict_s{1..5}.md`, `verdict_ship_compression.md`,
  `verdict_ship_balanced.md`, `bake_compare_canonical.md`,
  `bake_compare_augmented.md`.

## Falsification verdict

Both Compression and Balanced trail gates fail per § A.10. The 3 metric
inputs as additional MLP features do not produce a ship-grade improvement.

**Dead direction.** Future work using metric scores should NOT take this
form. Alternative directions worth trying:
- Late-fusion: train the existing 300-feature MLP, then blend its score with
  ssim2/cvvdp/iwssim via learned per-corpus weights (test-time post-hoc).
- Knowledge distillation: train the MLP to mimic an ensemble of (ssim2,
  cvvdp, iwssim, current zensim) on safesyn — let the MLP learn the
  consensus internally rather than reading the scores at inference.
- Per-corpus metric calibration: don't use a single global standardizer
  for ssim2/cvvdp/iwssim — instead, build per-corpus scalers in training
  so each domain's distribution is properly normalized. (This would
  require a corpus tag input or a separate head per corpus, both more
  complex than the current setup.)

## Data caveats

- **konjnd-dense training rows have ALL-NaN for ssim2/cvvdp/iwssim**, so
  the MLP gets a constant-input signal for that group during training.
- **val/konjnd has no per-pair dist_path join key**, so val f300/f301/f302
  are also constant — no metric supervision for konjnd at inference.
- **canonical train/kadid + train/tid ssim2_gpu is constant per image**
  (upstream join bug). 70% of the per-pair ssim2 variance in training
  comes from safesyn alone.
- **Val ssim2 column comes from fast-ssim2 CPU**, training uses GPU
  SSIMULACRA2 — different metric implementations with different score
  scales. Even when both columns are present, the trainer/inference scale
  mismatch breaks generalization.

Direct main commit: not pushed (waiting for jj sync).

— claude-session-exp-metric-inputs, 2026-05-18
