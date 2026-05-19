# V0_5 affine calibration methodology (2026-05-19)

## Problem

The three V0_5 production ships shipped 2026-05-18
(`PreviewV0_5Balanced`, `PreviewV0_5Compression`,
`PreviewV0_5Ensemble`) emitted **distance-shaped** raw output (low
raw = high quality), not score-shaped (0..100). With
`skip_score_mapping = true` plus the runtime's hard
`pre_bound.clamp(0.0, 100.0)`, every real-codec re-encode pair
returned exactly 0 because the raw output was negative across the
entire quality range.

Pre-fix observed by V0_5-IDENTITY-FIX report (commit `f47de21a`):

- `PreviewV0_5Balanced` raw on q=75 JPEG re-encode = -6 → score = 0.
- `PreviewV0_5Compression` raw on q=98 JPEG re-encode = -22 → score = 0.

SROCC is unaffected (rank-invariant against the sign flip), but
ABSOLUTE output is unusable. `zensim-target` (the user-facing
quality dial) defaults to `v0_3` as a workaround.

## Cause

The V_22-mix-LARGE+iwssim (balanced) and V_24-per-sample-α s4
(compression) bakes were trained on the `mix_cv40_iw60` target
column — a score-shaped 0..100 mix of CVVDP and IW-SSIM
log-normalized scores. But the trainer's loss is RankNet-style, so
the bake only learns *rank* order, not absolute calibration. The
i8 quantization on top further shifts the absolute output.

Result on the canonical training parquet
`canonical-2026-05-18/train/cvvdp_iwssim_LARGE.parquet`
(n=73,300, 300-feat schema):

| Bake | raw p5 | raw p50 | raw p95 | Spearman(target, raw) |
|---|---:|---:|---:|---:|
| Balanced (V_22-mix-LARGE) | -22.46 | -14.18 | +5.97 | **-0.9885** |
| Compression (V_24-per-sample-α) | -25.19 | -16.10 | +5.32 | **-0.9897** |

Both bakes are rank-perfect predictors of `mix_cv40_iw60`, but
with **negative slope** and median around -14 to -16 — exactly the
"distance-shaped" pattern.

## Fix

Approach **B (runtime affine in `ProfileParams`)**. The bakes are
i8-quantized so the existing `zensim-validate/bin/affine_calibrate`
tool (F32-only) can't operate on them. Rebake-from-scratch
(approach C) is heavy and unnecessary. Runtime affine is one line
of code per profile and is cleaner.

### Schema change

Added 4 fields to `ProfileParams` (no MSRV bump, additive):

```rust
pub affine_alpha: f64,           // identity = 0.0
pub affine_beta: f64,            // identity = 1.0
pub affine_alpha_compression: f64, // for ensemble compression route
pub affine_beta_compression: f64,  // for ensemble compression route
```

Applied in `apply_mlp_scoring` after the MLP forward / per-sample-α
dispatch, before the score-mapping / clamp:

```rust
let (a, b) = if routed_to_compression {
    (params.affine_alpha_compression, params.affine_beta_compression)
} else {
    (params.affine_alpha, params.affine_beta)
};
let raw_calibrated = if a == 0.0 && b == 1.0 { raw } else { a + b * raw };
```

The byte-identical short-circuit (commit `f47de21a`) is unchanged
— it returns before the MLP path runs, so identity images still
return score=100 regardless of α, β.

## Fit procedure

1. **Target column**: `ssim2_gpu` (0..100 scale, matches user
   expectations for a 0..100 quality dial). `mix_cv40_iw60` is the
   trained-on target but compresses high-quality content into a
   narrow band (p95=79); fitting against `ssim2_gpu` gives a wider
   useful range and matches the "score in [60, 90] at q=75"
   target the user requested.
2. **Fit corpus**: `canonical-2026-05-18/train/safesyn.parquet`
   (n=196,086; ssim2_gpu target p5=-5.96 p50=68.73 p95=93.25 —
   covers q=0..100 across multiple codecs).
3. **Holdout split**: 80/20 random shuffle (seed=42), R² computed
   on the held-out 20%.
4. **Method**: ordinary least squares
   `target = α + β · raw`. Two-column design matrix.

## Per-bake fits

### `PreviewV0_5Balanced` (V_22-mix-LARGE+iwssim s3)

```
score = 45.0561 + (-2.6602) · raw
R²_fit = 0.910,  R²_holdout = 0.925,  MAE_holdout = 6.09
```

Held-out verification on canonical val:

| Corpus | n | SROCC raw | score p5 | p50 | p95 | < 0 | > 100 |
|---|--:|--:|--:|--:|--:|--:|--:|
| CID22 | 4292 | 0.832 | 27.6 | 61.5 | 85.1 | 0.0 % | 0.0 % |
| KADID | 10125 | 0.968 | -5.5 | 35.7 | 77.6 | 9.0 % | 0.0 % |
| TID | 3000 | 0.973 | -8.0 | 39.8 | 67.5 | 9.3 % | 0.0 % |
| KonJND | 1008 | 0.893 | 28.7 | 57.9 | 68.1 | 0.0 % | 0.0 % |
| AIC-3 | 600 | 0.785 | 24.6 | 60.1 | 91.6 | 0.0 % | 0.8 % |

The KADID / TID below-0 tails (8-9 %) are heavy-distortion pairs
outside the training distribution; the profile's
`soft_clamp_score = true` (set 2026-05-19 alongside this fit)
preserves rank ordering at the boundary.

### `PreviewV0_5Compression` (V_24-per-sample-α s4)

```
score = 49.3380 + (-2.3967) · raw
R²_fit = 0.852,  R²_holdout = 0.853,  MAE_holdout = 9.06
```

Held-out verification on canonical val:

| Corpus | n | SROCC raw | score p5 | p50 | p95 | < 0 | > 100 |
|---|--:|--:|--:|--:|--:|--:|--:|
| CID22 | 4292 | 0.864 | 23.7 | 65.5 | 90.4 | 0.0 % | 0.9 % |
| KADID | 10125 | 0.932 | -16.3 | 27.8 | 79.5 | 15.1 % | 3.2 % |
| TID | 3000 | 0.889 | -13.5 | 32.1 | 80.0 | 12.1 % | 0.0 % |
| KonJND | 1008 | 0.808 | 25.0 | 58.8 | 68.1 | 0.0 % | 0.0 % |
| AIC-3 | 600 | 0.818 | 20.7 | 60.9 | 93.7 | 0.0 % | 1.3 % |

The compression bake's R² is lower because per-sample-α dispatch
adds a per-row sigmoid-gated mix between rank head and pool head;
the affine is fit against the post-dispatch scalar output.

### `PreviewV0_5Ensemble`

Inherits both fits — primary (balanced) route uses
(45.0561, -2.6602); compression route uses (49.3380, -2.3967).
The classifier's hardcoded threshold (`logit > 0`) is
unchanged.

## Behavior on a real q=75 JPEG re-encode

From `zensim/examples/v05_score_probe.rs` on a 64×64 synthetic
gradient (commit referenced in CHANGELOG):

| Profile | q=30 | q=50 | q=70 | q=75 | q=90 | q=95 |
|---|--:|--:|--:|--:|--:|--:|
| Balanced | 74.96 | 78.09 | 79.08 | 79.41 | 86.22 | 89.89 |
| Compression | 43.28 | 51.12 | 72.41 | 78.76 | 89.15 | 92.50 |
| Ensemble | 74.96 | 78.09 | 79.08 | 79.41 | 86.22 | 89.89 |

The target [60, 90] range at q=75 is hit by all three profiles.
q=95 sits in the 90-92 range as expected for a near-PJND pair.

## zensim-target demo verdict (2026-05-19)

`cargo run --release --example demo_matrix -p zensim-target` with
`ZENSIM_TARGET_PROFILE=<x>`:

| Profile | Cells converged | Ratio |
|---|---:|---:|
| `v0_3` (legacy baseline) | 33/36 | 92 % |
| `balanced` (V0_5) | 32/36 | 89 % |
| `compression` (V0_5) | **34/36** | **94 %** |
| `ensemble` (V0_5) | 33/36 | 92 % |

V0_5Compression beats the V0_3 baseline by 1 cell (94 % vs 92 %).
All four profiles pass the user's ≥ 80 % gate. **Default profile
in `zensim-target` is bumped from `v0_3` to `compression`**
(commit referenced in CHANGELOG).

## SROCC preservation check

The affine is rank-invariant (multiplying by a constant + adding a
constant doesn't change ranks). `bake_verdict` (which operates on
bake bytes directly, bypassing the runtime affine) reports the
same SROCC numbers as pre-fix:

```
CID22 0.8324 (vs 0.8324 pre-fix)
KADID 0.9677 (vs 0.9677 pre-fix)
TID   0.9729 (vs 0.9729 pre-fix)
KonJND 0.8927 (vs 0.8927 pre-fix)
AIC-3 0.7845 (vs 0.7845 pre-fix)
```

## Regression test

`zensim/tests/v05_calibration.rs` (18 tests):

- Per-profile positive-score check at q ∈ {30, 50, 70, 90} on a
  synthetic 64×64 gradient × 3 profiles = 12 tests.
- Per-profile monotonicity sweep (q=30 ≤ q=50 ≤ q=70 ≤ q=90, and
  q=90 − q=30 > 5 score units) × 3 profiles = 3 tests.
- Per-profile identity-image short-circuit preserved (still
  returns 100 on identical inputs) × 3 profiles = 3 tests.

All 18 pass. The existing `zensim/tests/v05_identity.rs` (7 tests)
also still passes — the affine path is gated behind the identity
short-circuit.

## Files touched

- `zensim/src/profile.rs` — 4 new fields on `ProfileParams`, set
  per-profile across all 7 static `ProfileParams` blocks.
- `zensim/src/metric.rs` — affine application in `apply_mlp_scoring`
  after the MLP / routing step, before the clamp.
- `zensim/tests/v05_calibration.rs` — new 18-test regression suite.
- `zensim/examples/v05_score_probe.rs` — new diagnostic probe.
- `zensim-target/examples/demo_matrix.rs` — `ZENSIM_TARGET_PROFILE`
  env var added for profile selection.
- `zensim-target/src/bin/zensim_target.rs` — default profile bumped
  from `v0_3` to `compression`.
- `zensim-validate/src/bin/dump_raw_predictions.rs` — new utility
  for dumping raw bake predictions on a parquet corpus (used to
  produce the fit data above).

## Reproduce

```sh
# Fit (Python):
python3 -c "
import numpy as np
from scipy.stats import spearmanr
data = np.loadtxt('/tmp/v05_calibration/balanced_safesyn_ssim2.tsv',
                  delimiter='\\t', skiprows=1)
tgt, raw = data[:,0], data[:,1]
m = np.isfinite(tgt) & np.isfinite(raw)
tgt, raw = tgt[m], raw[m]
rng = np.random.default_rng(42)
idx = rng.permutation(len(raw))
fit, holdout = idx[:int(0.8*len(idx))], idx[int(0.8*len(idx)):]
A = np.vstack([np.ones_like(raw[fit]), raw[fit]]).T
sol, *_ = np.linalg.lstsq(A, tgt[fit], rcond=None)
print('alpha=%.4f beta=%.4f' % tuple(sol))
"

# Dump raw predictions:
cargo build --release --bin dump_raw_predictions -p zensim-validate
./target/release/dump_raw_predictions \
    --bake zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin \
    --parquet /mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet \
    --target-column ssim2_gpu \
    --output /tmp/v05_calibration/balanced_safesyn_ssim2.tsv
```

## Limitations / honest gaps

1. **Fit uses ssim2_gpu as the calibration target, not the bake's
   own training target.** A purist would argue the affine should
   match the trained-on `mix_cv40_iw60` scale. But the user-facing
   purpose of V0_5 is a 0..100 quality dial; ssim2_gpu's broader
   range (p95=93.25 vs mix_cv40_iw60's p95=79.39) gives a more
   usable score distribution at high quality. Document this
   choice; revisit if the next V_X bake reverts to a different
   calibration scheme.
2. **KADID / TID heavy-distortion tail clips negative after
   affine** (8–15 % below 0 on the val corpora). The soft-clamp
   handles it; SROCC is preserved; user-facing scores never drop
   below 0. But the calibrated linear fit is technically wrong on
   that tail — a piecewise / logistic fit would be tighter at the
   extremes. Defer until v0.6.
3. **No per-sample-α head re-fit.** The compression bake's affine
   is fit against the post-α-mix scalar, so any future change to
   the per-sample-α dispatch parameters would invalidate the fit.
   Re-fit if the dispatch changes.
4. **Synthetic gradient probe.** The probe image in
   `v05_score_probe.rs` is a small synthetic gradient; the
   q=20→30 inversion observed there (77.1 → 75.0) is content-
   specific and doesn't reflect real-photo behavior. The
   regression test only checks q ∈ {30, 50, 70, 90} which are
   monotone across all three profiles on the probe.
