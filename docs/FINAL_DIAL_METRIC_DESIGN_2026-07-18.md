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

**L1 quality — RETRACTED 2026-07-18 (the "additive core" was an MLP).** The table below
claimed a basic-only *additive* model beats full-372 at CID22 0.8978. That is WRONG: the
`--max-features 156` bake it measured (`L1_linear_mf156.bin`) is **not additive** — `zenpredict
inspect` shows every "linear"/"basic-156" bake in this campaign is a `156→128→1` **LeakyReLU
MLP**. The `zensim_mlp_train` binary has no linear mode (`--n-hidden-layers 0` is never consumed
by train-core; it always emits a 128-unit hidden layer). So the row below compares two MLPs, and
the "additive core" label is false.

| ~~model~~ (ALL MLPs — mislabeled) | CID22 | imazen26 | nonphoto | HF |
|---|--|--|--|--|
| ~~basic-156 "additive core"~~ (156→128→1 MLP) | 0.8978 | 0.8353 | 0.8531 | 0.4446 |
| full-372 control (372→128→1 MLP) | 0.8876 | 0.8411 | 0.8605 | 0.3394 |

**What a GENUINELY additive basic-156 actually scores (measured 2026-07-18,
`additive_basic156_probe.py` — the linear solver restricted to f0..155, `n_layers=1` identity,
`additive=True`, `diffmap_basic_fraction=1.0`, verified):**

| model | additive? | CID22 | dial-mono | note |
|---|---|--|--|--|
| **additive basic-156** (best of 12: shaped-lasso) | **✓ yes** | **0.8563** | 0.973 | first real additive-156 |
| additive-372 **B** (`b_sdr_linear`) | ✓ yes | 0.8764 | smooth | ships as Profile B |
| basic-156 **MLP** (`Ebothg` winner) | ✗ no | 0.8939 | 0.984 | the promoted "winner" |
| basic-156 MLP (`L1_linear_mf156`) | ✗ no | 0.8978 | 0.474 | the mislabeled row above |

**Both "no tradeoff" conclusions are FALSE; there is a real tradeoff:**
- **additive costs ~0.038 CID22** vs the basic-156 MLP (0.856 vs 0.894) — the LeakyReLU
  nonlinearity is doing real ranking work, it is not free.
- **basic-156 costs ~0.020 CID22** vs additive-372 B (0.856 vs 0.876) — so f156–371 DO help an
  additive model's CID22; "they add nothing" was an artifact of comparing two MLPs whose
  nonlinearity absorbed the feature loss.

The one claim that SURVIVES: **additive was never the dial problem** — additive + output spline
gives a smooth dial (0.973), as B already showed. The gap is CID22 *rank*, not dial smoothness.
So the honest closed-loop choice is a genuine trade: exact-gradient diffmap (additive B/basic-156,
CID22 0.86–0.88) **vs** best rank (MLP, CID22 0.894 but input-dependent diffmap whose coherence
is unmeasured). This is a decision to surface to the user, not a solved "go additive." Detail:
`benchmarks/additive_vs_mlp_correction_2026-07-18.md`.

## Relaxation (user 2026-07-18): discontinuity below the codec-targetable range is fine

Codecs never emit corruption or deep negatives as a *quality setting* — those are
out-of-band. So the dial only has to be smooth **within the range a codec can target**;
below the lowest honest q, a discontinuity is acceptable. This **removes the hardest
constraint** (a single C∞ function spanning corruption→near-lossless): L2's severe floor and
the negative tail may be a hard drop below the codec range, and L3's squash only has to be
smooth *inside* the targetable band. Coherence (correct ordering, monotone descent) still
holds below; only *continuity* is waived there.

## 2. The decision: additive core, not an MLP

> **⚠ CORRECTION 2026-07-18 — this decision was made on debunked evidence; it is now a genuine
> tradeoff, not a slam-dunk.** §1's "additive costs no quality" was measured on mislabeled MLPs
> (see the retraction above). The real numbers: additive tops out at CID22 0.876 (B, 372-input)
> / 0.856 (basic-156); the MLP reaches 0.894. So "the depth-MLP's small holdout edge is not
> worth forfeiting the diffmap" understates the edge — it is ~0.038 CID22, and it buys real
> rank. The diffmap argument (0.987 exact-gradient for additive, unmeasured for MLP) still
> favors additive for the *closed loop specifically*; but this is a trade the user should make
> explicitly (exact diffmap + 0.86–0.88 rank vs input-dependent diffmap + 0.894 rank), not a
> settled "go additive." The prose below is preserved as the original reasoning.

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


## CORRECTION (user 2026-07-18): no external metric at runtime
zensim is ONE self-contained forward pass — butteraugli cannot run at runtime. The corruption
"severe floor" must use zensim's OWN XYB max/p-norm PEAK features (f156-371, the butteraugli
analog), not an external metric. The "butteraugli Stage-2" is OFFLINE-only (zensim-regress test
harness). CAVEAT (E7-lite): those peak features do NOT auto-gate corruption — max-norm pooling
inherits a localization bias (over-weights small intense breaks vs whole-image corruption), so
in-scalar corruption gating needs deliberate perceptibility-calibrated weighting and is an OPEN
question. Runtime corruption-safety may not be cleanly achievable within zensim's features;
corruption may stay an offline/regression concern.

## Post-campaign refinement (2026-07-18) — the recipe, not the layers

The E1–E6 + E-MSE experiment campaign (`benchmarks/final_metric_experiments_2026-07-18.md`)
simplified the 5-layer design to a **recipe** on the additive core:

**Winning recipe = B's smooth-dial least-squares (BVLS) fit, restricted to BASIC features.**
- **Additive/basic → exact diffmap (0.987)** — confirmed; B is only 62% basic (→ 0.87 diffmap),
  and basic-only costs no CID22 (E2), so a basic-only B is exact-diffmap + same quality.
- **Dial monotonicity is the LOSS/fit, not the architecture.** B's BVLS least-squares → dial-mono
  0.98 with full range. RankNet → jittery (0.47). Naive MSE-SGD → COLLAPSE (near-constant, fake
  0.998, CID22 0.085). So reproduce B's constrained least-squares, not RankNet or MSE-SGD.
- **Corruption is a SEPARATE guard, not an in-scalar layer.** The additive core's apparent
  85% gate was a RankNet over-reaction to localized HF (E2: perceptually inverted; E-MSE: drops
  to 27% under a non-RankNet loss). Real corruption gating = butteraugli-max Stage-2 on the
  PERCEPTIBLE subset (72%). L2/L3 of the original design are retired.
- **Negatives** (severe tail) need winsor to tame the un-clipped feature explosion (E5: raw range
  to −2.9M; RankNet even inverts the tail) + the spline floor; this is the below-codec-range
  regime where the relaxation permits coarseness.

So the build reduces to: **(1) reproduce B's BVLS fit on basic-only features → smooth dial +
CID22 + exact diffmap; (2) wire the ModelCoherent diffmap (B's basic sensitivities); (3) keep
butteraugli as the separate corruption guard; (4) spline-floor for negatives.** No asymmetric
squash, no in-scalar corruption layer.
