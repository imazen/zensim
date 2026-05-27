# V47 masked-monotone — STRICT fixes the correctness defect (2026-05-27)

First retrain using the per-feature sign mask (300 pin≥0 / 72 free, from
benchmarks/feature_sign_mask_2026-05-26.tsv). Two variants on the
V32-faithful recipe + --monotone-cbc.

## Blur-ladder monotonicity (OOD synthetic, raw post-spline)

| Bake | inversions | above-identity | identity=max? |
|---|--:|--:|---|
| V39 (shipped A) | 27 | 31 | NO (blur scores ≥ identity) |
| **V47-strict** | **0** | **0** | **YES** ✓ |
| V47-partial | 0 | 0 | (degenerate, all ~constant) |

V47-strict is **monotone-by-construction** — the V39 correctness defect
(blur ≥ identity on OOD content) is FIXED. identity is the unique max at
every content; score strictly decreases with blur.

## Held-out panel (bake_verdict)

| Corpus | V47-strict | V47-partial | V39 |
|---|--:|--:|--:|
| CID22 | 0.8547 | 0.0 (dead) | 0.8793 |
| KADID | 0.8030 | 0.0 | 0.9251 |
| TID | 0.7965 | 0.0 | 0.9317 |
| KonJND | **0.4850** | 0.0 | 0.4197 |
| AIC-3 | 0.7700 | 0.0 | 0.8023 |
| AIC-4 | 0.8902 | 0.0 | 0.9051 |

- **STRICT**: CID22 0.855 (−0.024 vs V39, above the 0.85 G7 floor),
  KonJND **+0.065** (monotonicity helps near-lossless rank), AIC-4 −0.015.
  Cost: KADID −0.122, TID −0.135 — the analytic-distortion-ranking signal
  carried by the 72 dropped sign-flip features. Trained healthily
  (val 0.90, no collapse).
- **PARTIAL is degenerate** — the 72 free features let the optimizer
  cancel the spread; raw preds collapse to ~50 (range 0.05) → spline maps
  everything to ~0 → all-zero panel. Out.

## Remaining: dial calibration (G1)

STRICT ranks well but its dial is COMPRESSED on real content: `y_pre`
clusters near 0 (tanh-pin score ~50) and only spreads wide on extreme
synthetic blur. The auto-spline fit only 2 degenerate knots
(pred≈49.97/49.99) and maps the val distribution to a negative band
(G1 p5=−42, p95=−15). The bake is monotone with real spread — it needs
the dial stretched to [0,100].

Fix (in flight, v48): retrain strict with a SMALLER `--tanh-output-head-scale`
(30 → ~5) so the narrow real-content `y_pre` spread maps across the full
[0,100] dial. Monotone amplification preserves the correctness guarantee.

## Status vs the goal

The #1 goal — Profile::A monotone-by-construction (never blur > identity)
— is ACHIEVED IN PRINCIPLE by V47-strict. Two items before it could
ship as A: (1) dial calibration (v48 tanh-scale), (2) the KADID/TID rank
cost (−0.12/−0.14) is the price of strict monotonicity — a user call,
since CID22 (the compression gold standard) stays competitive and KADID/
TID are integrity guards, not the primary compression target.

## v48 (tanh=5) — FALSIFIED; dial is structural, not a tanh-scale knob

Hypothesis: a smaller `--tanh-output-head-scale` (30→5) would amplify the
compressed real-content `y_pre` spread across [0,100]. Result: WORSE.
Monotonicity held (0 inversions) but the panel cratered — CID22
0.855→0.574, KonJND 0.485→0.030, all corpora down — and the dial got
NARROWER ([−33,−6.7]). A steeper sigmoid saturates + vanishing-gradients
the encoder. Smaller tanh scale is the wrong lever.

### Root cause (characterized)

The monotone bake has `y_pre = rank_w·h + rank_b ≤ rank_b` (rank_w ≤ 0,
h ≥ 0). Real distorted content has SMALL error features (subtle
artifacts) → small h → `y_pre ≈ rank_b ≈ 0` → tanh-pin ≈ 50. The rank
signal lives in a ~0.05-wide pred band around 50. The auto-spline
forwards the anchor, takes median-pred per target band — but every band
maps to ~49.9, so the "strictly increasing in pred" filter keeps only
2 knots and the spline degenerates (maps the 0.05 band → [0,1.7], val
content extrapolates to negatives → G1 fail).

This is a TRAINING-TIME dial-spread problem, not a calibration or
tanh-scale one:
- auto-spline: degenerate on the compressed distribution (can't fix).
- tanh-scale: smaller saturates (v48 falsified); larger narrows further.
- dynamic-range-floor (the G1 lever): rides the cross-codec-eq substrate,
  which PANICS on 2-layer (`multi-layer/skip + cross_codec_eq: not yet
  wired`). Not a drop-in for the 2-layer V32-faithful recipe.

Candidate fixes (each a distinct experiment):
1. **1-layer + dynamic-range-floor** — drop to 1-layer (where cross-codec-eq
   / dynamic-range-floor ARE wired) + add the dial-spread regularizer.
2. **rank_b spread incentive** — force/init rank_b large positive so the
   identity case anchors at 100 and distortion spreads down.
3. **Calibration refit on a real corpus** — fit the spline on CID22 preds
   (not the degenerate anchor); but the bake's 0.05 real-content band
   means a ~2000× gain → hypersensitive dial.

## Decision framing

The CORRECTNESS goal is ACHIEVED: v47-strict is monotone-by-construction
(0 inversions, identity=max), competitive with ssim2/cvvdp on CID22/AIC-3,
and BEST-of-field on KonJND (0.485 vs cvvdp 0.048, iwssim 0.186, V39 0.420).
Its only deficits are KADID/TID (the benchmark V39 is the outlier-high
there vs the conventional baselines) and the structurally-compressed dial.

Two paths:
- **Ship v47-strict as a sibling profile** (e.g. Profile::A_Strict / a
  "monotone similarity" profile) for the regression-test / general-
  similarity use case where monotonicity > dial range; keep V39 as
  Profile::A for the codec quality-dial (good range, accepts OOD
  inversion). Different use cases, different profiles. (User anticipated
  this: "ships as a sibling profile for users that need the guarantee.")
- **Pursue the dial fix** (1-layer + dynamic-range-floor is the most
  promising) to get ONE bake that is both monotone AND full-dial — then
  it could replace V39 at Profile::A outright.

## v49 (target-scale) — NO-OP; dial compression is constraint-vs-MSE, not scale

Hypothesis: the recipe fed [0,1] targets against a [0,100] output (scale
mismatch). FALSIFIED — v49 came out BIT-IDENTICAL to v47-strict (CID22
0.8547, same spline knots, same blur ladder). Root: the V47/V46b recipe
has NO `--target-scale` flag → it used the DEFAULT (100.0), so targets
were already `human_score × 100 = [0,100]`, matching the pinned output.
load_parquet applies target_scale (parquet_loader.rs:170); the recipe
never overrode it. My "recipe used 1.0" premise (from the V39 manifest)
was wrong for V46b/V47.

So the MSE (weight 0.6) ALREADY pushes the output toward the [0,100]
targets. The dial still compresses because the **monotone constraint
(W1≥0 on the sign-safe set, rank_w≤0, tanh pin) fights that pressure**:
y_pre ≤ rank_b, real content has small error features → small h-swing →
narrow y_pre → compressed pinned output. Raising MSE weight to push
harder fights RankNet (V37 falsified: CID22 0.88→0.55).

### Falsified dial levers (all measured)
- target-scale (v49): no-op — already 100.
- tanh-scale smaller (v48): saturates, panel craters.
- auto-spline: degenerate on the compressed pred band (2 knots).
- more MSE weight: fights RankNet (V37, prior).
- dynamic-range-floor: rides cross-codec-eq, panics on 2-layer.

### Conclusion: a fundamental tension of THIS monotone parameterization
Strict monotonicity (W1≥0 + tanh pin) + full real-content dial
resolution are in tension for the per-sample-α arch. A unified
monotone+full-dial single bake needs a DIFFERENT monotone output
architecture (research): e.g. a monotone-but-unbounded head + a separate
learnable monotone calibration, or injected identity-anchor pairs to pin
the top + a 1-layer dynamic-range-floor substrate. Not a quick flag.

### What ships
v47-strict is the deliverable for the CORRECTNESS goal: monotone-by-
construction (0 inversions), competitive with ssim2/cvvdp (CID22 0.855,
KonJND 0.485 best-of-field). It is sibling-shippable as a "monotone
similarity" profile for the regression-test / general-similarity use
case (rank + monotonicity matter, absolute dial resolution does not).
V39 stays Profile::A for the codec quality-dial (full range, accepts the
OOD inversion). Unifying them is open architecture research.
