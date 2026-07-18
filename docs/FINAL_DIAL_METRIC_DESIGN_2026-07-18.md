# Final dial metric — design for closed-loop diffmap + one-shot codec targeting (2026-07-18)

The metric zensim ships must be optimal on THREE axes at once. This spec fixes the
architecture that satisfies all three, and the evals that gate it. It is the synthesis of
the 2026-07 findings (pin sweep, perceptibility classification, diffmap coherence, the
additive-vs-MLP tradeoff).

## 1. What each axis demands

| axis | demands |
|---|---|
| **Dial** | monotone in true quality (targeting converges); smooth C1 (no flats, no seams); bounded + calibrated to [neg, 100] (absolute — "85" is a fixed quality); coherent across the WHOLE range incl. corruption (below) + near-lossless (above) |
| **Closed-loop diffmap** | diffmap(x,y) = ∂score/∂distortion(x,y); must be the scalar's spatial gradient so per-block refinement serves the target. MEASURED: pooled coherence is already ~1.0; spatial coherence is capped ~0.66 **because the scalar is non-additive** — a linear/additive scalar makes the diffmap its EXACT gradient |
| **One-shot targeting** | score↔quality invertible + low-variance so `q_hat(target)` lands within δ in ONE encode; monotone fallback so a binary search converges if it misses |

## 0. De-risk result (CONFIRMED 2026-07-18)

The central bet — additive core → exact diffmap — is **measured and confirmed**
(`diffmap_block_coherence`, additive-vs-full ΔS per 64px block):

| pair | additive-scalar diffmap | full (non-additive) scalar | SSE (codec default) |
|---|--|--|--|
| gb82 honest q20 | **0.9876** | 0.8661 | −0.4488 |
| gb82 honest q10 | **0.9876** | 0.8933 | −0.5820 |
| CID22 mozjpeg q30 | **0.9862** | 0.9139 | +0.3034 |

An additive scalar's diffmap is a **near-exact spatial gradient (~0.987)**; the current
non-additive scalar caps at 0.87–0.91 (+0.07–0.12 left on the table); the codec's SSE
default is unreliable (−0.58 to +0.30). Not 1.000 only because of the winsor transforms +
f16 + multiscale blur — negligible. **The "go additive" decision is locked.**

**L1 quality — no tradeoff (CONFIRMED 2026-07-18).** A basic-only additive linear model
(f0–155, `--max-features 156`) vs a full-372 control, same recipe/seed:

| | CID22 | imazen26 | nonphoto | HF |
|---|--|--|--|--|
| **basic-156 (additive core)** | **0.8978** | 0.8353 | 0.8531 | **0.4446** |
| full-372 control | 0.8876 | 0.8411 | 0.8605 | 0.3394 |

The additive basic core **beats** the full model on CID22 and near-lossless, at ~0.006 cost
on imazen26/nonphoto. So the non-additive peak/max features (f156–371) add nothing to CID22
— they can move to L2 (the severe floor, where max/p-norm is exactly what's wanted) at **no
quality cost**. Both risks the design carried — "does additive cost the diffmap?" (no, it
*gives* it) and "does basic-only cost quality?" (no, it *helps*) — are retired.

## Relaxation (user 2026-07-18): discontinuity below the codec-targetable range is fine

Codecs never emit corruption or deep negatives as a *quality setting* — those are
out-of-band. So the dial only has to be smooth **within the range a codec can target**;
below the lowest honest q, a discontinuity is acceptable. This **removes the hardest
constraint** (a single C∞ function spanning corruption→near-lossless): L2's severe floor and
the negative tail may be a hard drop below the codec range, and L3's squash only has to be
smooth *inside* the targetable band. Coherence (correct ordering, monotone descent) still
holds below; only *continuity* is waived there.

## 2. The decision: additive core, not an MLP

The session's central tension: MLP precision (non-additive pooling, 372-feat depth) vs
everything the closed loop needs (additive → exact diffmap; linear → smooth + deterministic +
collapse-immune; simple → near-lossless survives). The diffmap finding is decisive: **the
0.66 spatial ceiling is non-additivity, and only an additive scalar removes it.** B (linear)
already delivers CID22 0.876, near-lossless HF 0.61 (best of any arch), collapse-immunity,
and additivity. The depth-MLP's small holdout edge is not worth forfeiting the diffmap, the
smoothness, and the near-lossless zone. **Go additive.** Recover precision additively (a
GAM, below), never by re-introducing non-additive pooling.

## 3. Architecture (5 layers, additive-preserving)

```
features f_0..f_371  (XYB multi-scale; BOTH mean-pooled AND max/p-norm blocks)
  │
  ├─ L1 ADDITIVE CORE          raw_core = Σ_k g_k(f_k)          (linear = B; or monotone GAM)
  │                            → precision (CID22 ~0.876) + near-lossless (0.61)
  │                            → diffmap = Σ_k g'_k(f_k)·local_k(x,y)  EXACT spatial gradient
  │
  ├─ L2 SEVERE FLOOR (butteraugli-analog, additive)
  │        raw = raw_core − λ·Φ(maxpool/​pnorm of local error)
  │        → fires ONLY on perceptible localized breakage (M≈0 on honest images)
  │        → gates the ~36 PERCEPTIBLE corruption families; correctly skips the 8 subtle
  │        → its diffmap term concentrates at the worst block = where the corruption is
  │
  ├─ L3 ASYMMETRIC SMOOTH SQUASH    y = squash(raw)
  │        saturating on the BOTTOM (corruption/negatives pinned — coherence)
  │        near-LINEAR on top (near-lossless keeps gradient — precision + smoothness)
  │        one C∞ function; no tanh double-squash, no piecewise seam
  │
  ├─ L4 MONOTONE CALIBRATION SPLINE   score = pchip(y)
  │        full-range anchor (corruption/neg → honest → near-lossless)
  │        lower extrapolation FLOORED → negatives (monotone descent)
  │        upper capped ≤100; SROCC-invariant; makes the dial ABSOLUTE
  │
  └─ score ∈ [neg, 100]   monotone, smooth, corruption-safe, invertible
```

Why this satisfies all three axes:
- **Dial:** L3 (smooth, pinned bottom / linear top) + L4 (monotone, full-range, negatives) =
  smooth C1 monotone dial with working negatives and no flats/seams. L2 keeps perceptible
  corruption below honest; L1 keeps near-lossless alive (0.61, not the MLP's dead ~0).
- **Diffmap:** L1 is additive → diffmap is its EXACT spatial gradient in the honest/product
  zone (resolves the 0.66 ceiling). L3's squash contributes only a per-image scalar gain (no
  spatial rank change). L2 adds a correction concentrated at the corruption. So
  `diffmap(x,y) ∝ Σ_k g'_k(f_k)·local_k(x,y) − λ·∂M/∂(x,y)` — computed from the SAME weights
  as the scalar, coherent by construction. `DiffmapWeighting::ModelCoherent` reads g'_k/scaler_k
  (once, for linear) instead of the stale V0_2 SSIM weights.
- **One-shot:** monotone-by-construction (L3 sign-consistent + L4 monotone) → invertible. Low
  variance (linear, deterministic, no seed collapse) → `q_hat(target)` is a reliable function.

### The coherence↔precision split, concretely
Precision and coherence want different signals (per the perceptibility study): the honest
zone rewards the fine MEAN-based core (L1); the severe zone needs the robust MAX/order-stat
(L2, a butteraugli-max analog). L2 magnitude-gates naturally — subtle corruption → small M →
correctly not gated; perceptible → M spikes → pinned by L3's floor. "Learn from butteraugli"
= L2. Coarse ordered bands (near-lossless ≥ subtle > honest q20 > q10 > perceptible-corruption
> garbage→neg) are the coherence spec; we guarantee band ORDER, not within-garbage rank.

## 4. Eval suite (the ship gates)

### Eval A — Dial quality
Dense multi-codec q-sweep (q0 + step-1 near-lossless + JND-zone step-2 + injected perceptible
corruption + negative-tail). Report per codec:
- **Monotonicity** ≥ 93% non-decreasing adjacent-q, **tied ≤ 5%** (no flats — the near-lossless
  dead-zone failure), **dynamic range** spans [neg,100] (G1).
- **Calibration**: score↔achieved-ssim2/human is stable, low per-bin residual.
- **Corruption safety** (redefined): the ~36 PERCEPTIBLE families rank below honest q10; the 8
  SUBTLE families stay near-lossless (over-gating them is a FAIL, not a pass).
- **Negatives**: worse-than-worst-codec < 0, monotone descent, no clamp-flat.

### Eval B — Closed-loop diffmap
- **Spatial** `SROCC(diffmap_block, ΔS_refine)` (`diffmap_block_coherence` tool) — must beat
  SSE (codec PSNR default) AND, with the additive core, approach ~1.0 in the honest zone.
- **Pooled** `SROCC(mean diffmap, 100−score)` across the sweep — ~1.0.
- **Closed-loop convergence (the real test):** simulate the loop — target score → refine the
  top-diffmap blocks → re-score → repeat. Measure (a) monotone approach to target (no
  oscillation), (b) steps-to-converge, (c) that it beats an SSE-guided loop. A coherent
  diffmap converges fast and monotonically; an incoherent one oscillates / stalls.

### Eval C — One-shot codec targeting
- Train `q_hat = R(source_features, target_score)` (the RD/picker companion; zensim is the
  ground-truth it's measured against). For a grid of targets × images: encode ONCE at q_hat,
  measure `|zensim(encode) − target|`. Report residual p50/p95; gate **p95 ≤ 5 dial units**.
- **Fallback**: if one-shot misses, a dial binary-search converges in ≤ N steps (needs the L3+L4
  monotonicity — this is why Eval A's monotonicity is load-bearing for targeting, not cosmetic).

## 5. Honest tradeoffs

- **Give up**: the depth-MLP's ~0.004–0.01 holdout edge on some corpora. Worth it for exact
  diffmap + smoothness + determinism + a live near-lossless zone.
- **Near-lossless rank stays ~0.61 (B's level), not higher** — a feature-discriminability limit,
  not fixable by architecture; the dial only needs near-lossless to MOVE smoothly, which L1+L3
  deliver. Improving it is a separate feature-engineering axis.
- **Corruption gate ≈ butteraugli level on the perceptible subset, not 100%** — by design; 100%
  was over-gating the 8 subtle families.

## 6. Build plan (mostly assembling existing pieces)

1. **L1 core**: BVLS/lasso linear (B-family, exists) or a monotone GAM (per-feature monotone
   splines, additive) — target CID22 + near-lossless. `train_minmax`/linear trainer.
2. **L2 severe term**: weight the existing peak/max feature blocks (or add an explicit maxpool-
   error term); calibrate λ on the PERCEPTIBLE corruption subset (below q10) with a held-out
   gate, verifying honest scores are undisturbed (M≈0).
3. **L3 asymmetric squash**: add the output head to `zensim-train-core` + the runtime dispatch
   in `zensim/src/metric.rs` (alongside the tanh head); fit pin-bottom / linear-top.
4. **L4 spline**: `bake_dial_refit` on a full-range anchor (corruption→honest→near-lossless) with
   floored negative extrapolation.
5. **Diffmap**: implement `DiffmapWeighting::ModelCoherent` for the additive core (g'_k/scaler_k,
   computed once) + the L2 concentration term; validate with `diffmap_block_coherence`.
6. **Evals A/B/C** as the ship gates; the closed-loop convergence sim (B) and the one-shot
   residual (C) are the two NEW harnesses to build.

The order that de-risks fastest: **L1 (have it) → Diffmap ModelCoherent + Eval B** (prove the
additive core gives the exact diffmap the closed loop needs — the thing no MLP can) → **L2+L3+L4**
(corruption + smooth dial + negatives) → **Eval C** (one-shot). If Eval B confirms additive →
exact diffmap, the architecture decision is locked and the rest is calibration.

## L2/L3 reframe — corruption is a PRESERVE-THROUGH-DIAL problem (2026-07-18)

Measuring the basic-156 additive core's RAW-output corruption gate returned a surprise:
**85.6% @q20** (above butteraugli's 72%, far above shipped-B's dialed 18%). The distortion
features (edge/HF/MSE) the additive core already uses register corruption directly — so the
corruption ranking is IN the additive raw output. **B's 18% was its dial SPLINE remapping the
corruption ranking away** (the spline was fit for honest quality), NOT the model being blind.

Consequences for the design:
- **L3 (asymmetric bottom-pin) is the load-bearing corruption mechanism**, not L2. Its job is
  to PRESERVE the raw ranking's corruption/negative floor through the dial mapping — precisely
  what pinning the bottom does (vs a spline that remaps corruption upward).
- **L2 (max/p-norm floor) narrows to localized breaks the mean misses** — from this run, only
  `edge_border_all_k4` (33%) is clearly weak; a small targeted supplement, not a co-equal layer.

**CAVEAT — verify before locking L2 scope.** The per-region pattern here (sq8/sq16/sq64 = 100%,
whole/frac2 = 69%) is INVERTED vs the full-corpus multimetric analysis (small regions hardest
for ssim2/cvvdp). This 222-recipe held-out subset may be skewed, or the additive core's HF/edge
features may genuinely catch localized breaks the SSIM-based metrics miss. Re-measure on the
full 672-recipe corpus + spot-check actual raw scores for small- vs large-region recipes before
committing L2's scope. The 85.6% headline (corruption-signal-in-raw-output) is robust; the
per-region attribution is not yet.
