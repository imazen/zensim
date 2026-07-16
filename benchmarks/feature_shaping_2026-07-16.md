# Feature shaping HURTS the compression holdouts (2026-07-16)

**User asked** (a) what the IW feature extremes are, (b) to vary the per-feature
shaping "since we have so many", (c) to test more shapers incl. composites, and
(d) to use Rust for reproducibility when an advantage is found.

## The IW extremes

`benchmarks/feature_health_sweep_2026-07-16.md` (Rust-extracted features,
`column_audit.py --mode features`) found the raw features CAN explode —
**f38 = 3.57e6 on kadis**, f129 = 3.3e4 on safesyn — in features
{12,38,51,77,90,116,129,155} (basic/peak/masked blocks, NOT the IW-pool block
228-371 the older note guessed). But they are **transform-handled**
(f38→winsor_p99 clips, f129/f90→yeo_johnson compresses) and there are **no NaN/Inf
rows and no leaks** (cid22_train's human_score==ssim2_gpu is the documented
ssim2-anchoring). So the raw extremes are real but not an unhandled drag.

## The shaping sweep — less is more (the advantage)

Every prior run used the fixed 2026-05-25 screen at `--auto-transforms-min-lift
0.05`. Sweeping the knob on the psa+tanh base (seed 13):

| shaping | CID22 | AIC-3 | KonJND | nonphoto | dial mono | G1 |
|---|---|---|---|---|---|---|
| base (min-lift 0.05) | 0.8559 | 0.7900 | 0.4805 | 0.9515 | 0.929 | ✓ |
| min-lift 0.02 (more) | 0.8276 | 0.7784 | 0.2543 | 0.9466 | 0.947 | ✓ |
| min-lift 0.20 (fewer) | 0.8584 | 0.8080 | **0.7228** | 0.9492 | 0.947 | ✓ |
| **no transforms** | **0.8680** | 0.7995 | **0.7680** | 0.877 | 0.959 | ✓ |

**Removing the transforms lifts CID22 +0.012, AIC-3 +0.010, and KonJND +0.29
(0.48→0.77 — clears the G5 0.70 floor that two architectures failed in May), and
improves the dial** — the *only* casualty is nonphoto (0.95→0.88). The current
min-lift 0.05 screen is **over-shaping**: it drags every compression holdout to
help one non-photo axis. `min-lift 0.20` is the middle ground (keeps nonphoto,
recovers most of KonJND).

This inverts the 2026-05-14 D3 assumption ("don't trim transforms aggressively")
— that was a different base (V_20 IS single-MLP, ssim2 MSE target); on the
psa+tanh rank base, more shaping hurts.

## Reproducibility

The advantage is **already a Rust recipe knob** (`--auto-transforms-min-lift`) —
no new code, no Python.

## Seed-confirm — the s13 CID22 lift was a mirage; KonJND edge is modest+noisy

Same lesson as the triplet. no-transform vs base across seeds {13,7,23}:

| metric | none s13 | none s7 | none s23 | none mean | base mean |
|---|---|---|---|---|---|
| CID22 | 0.868 | 0.853 | 0.835 | **0.852** | ~0.853 |
| KonJND | 0.768 | 0.581 | 0.711 | **0.687** | ~0.526 |

- **CID22: NO robust advantage** — 0.852 ≈ base 0.853; at s23 no-transform is
  *worse*. The s13 +0.012 was a favorable draw.
- **KonJND: a real but modest edge** — no-transform ≥ base at all 3 seeds
  (+0.16 mean, range +0.01…+0.29). "Less shaping helps the near-threshold axis"
  holds, but it is far smaller than s13's +0.29 and noisy.
- **Composite winsor→cbrt HURTS** (CID22 0.839, KonJND 0.393 at s13) — adding a
  shaper is the wrong direction; confirms "less is more".

## Corpus-mix variants (directive: different holdout vs train sets) — mix is FINE

Dropping the machine-metric mega-corpora HURTS (they're load-bearing, not drag):

| variant | CID22 | AIC-3 | KonJND | nonphoto |
|---|---|---|---|---|
| full (all) | 0.8559 | 0.7900 | 0.4805 | 0.9515 |
| no_kadis | 0.8511 | **0.6922** | 0.3745 | 0.9503 |
| no_bigcodec | 0.8359 | 0.7890 | 0.4546 | **0.7586** |
| no_mega (both) | 0.8232 | 0.8226 | 0.4442 | **0.7621** |

kadis holds up AIC-3, bigcodec holds up nonphoto + CID22. No mix change helps.

## Verdict

Nothing here robustly beats B. On the primary CID22 gate every variant (base,
no-transform, min-lift 0.20, corpus-drops) sits on the same **~0.85 seed-noise
plateau**; B (0.876) is at the top of it with a clean dial. The single real
(small) lever is **less feature-shaping → modestly better KonJND**, defensible as
a minor recipe tweak (raise `--auto-transforms-min-lift`), not a B-beater. The
discipline (seed-confirm) caught two s13 mirages this session (triplet, no-transform)
— the model is noise-limited near B, and the right conclusion is that B's recipe
is close to the achievable frontier on this feature set + these corpora.
