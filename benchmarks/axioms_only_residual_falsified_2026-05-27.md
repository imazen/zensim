# Axioms-only residual-from-identity — FALSIFIED (2026-05-27)

Tested the user's "enforce only the absolute axioms (identity=max +
bounded), keep the encoder expressive" hypothesis via a residual-from-
identity metric: score = f(λ·g), g = Σ wⱼ(hⱼ(x) − hⱼ(x_id))², h =
UNCONSTRAINED encoder, x_id = standardized identity. Probe binary:
`zensim-validate/src/bin/residual_identity_probe.rs` (trains on
cid22_train+kadid+tid+konjnd, evals panel + blur ladder).

## Result: falsified on the panel; A2 free, A3 not gained

| Form | A2 id=max | A3 no-inversions | Panel (CID22 / KonJND) | Failure |
|---|---|---|---|---|
| linear `100−λg` | ✓ (construction) | ✗ (6/2/6/2 invs) | 0.50 / −0.01 | blows up (−25000) |
| RBF `100·e^−λg` | ✓ | trivial (all→0) | 0.62 / −0.27 | saturates to {0,100} |

- **A2 (identity is the unique max) is free** in both — 0 above-identity
  on every blur ladder, by construction.
- **A3 (no heavier-blur-above-lighter inversions) is NOT gained** — the
  linear form still inverts; the RBF only avoids it by saturating
  everything below identity to 0 (degenerate, no resolution).
- **The panel craters either way** — the distance-from-identity has a
  pathological dynamic range (g: 0 → huge); linear → blows up negative,
  exp → saturates to 0. Neither maps to a usable dial → SROCC 0.5-0.6.

## Conclusion: no free lunch (the real answer)

Enforcing the axioms STRUCTURALLY costs the human-MOS panel, by EITHER
route:
- full feature-monotonicity (v47-strict): A1+A2+A3 ✓, panel −0.12/−0.14
  (KADID/TID) + compressed dial.
- identity-anchoring residual: A1+A2 ✓ (not A3), panel craters
  (blow-up/saturation).

The expressive, well-calibrated metric (V39) is exactly the one that
violates the axioms off-manifold: it saturates raw >100 → clamped to
100, so on OOD content identity AND light distortion both read 100
(TIE — violates "unique max"), masking the raw inversions. You get
expressive-and-faithful OR structurally-axiom-clean, not both, in
these parameterizations.

## Middle grounds (catch/bound — the pragmatic path)

The defect that actually matters for the general-similarity / regression-
test use case: a clearly-degraded image scoring at the MAX (100), so a
broken decode passes as "identical." Bounding THAT doesn't require
structural monotonicity:

1. **Runtime saturation guard** — for the regression-test use case, treat
   score ≥ (100 − ε) on a NON-identical pair as suspect: the metric is
   saturating. Cheap (no retrain); flags/caps the exact failure
   (degraded reading as identical). The runtime already clamps ≤100 and
   special-cases score(x,x)=100; this adds "non-identical ⇒ score < 100"
   as a guard.
2. **Monotone-envelope clamp** — score_final = min(V39_score,
   envelope(simple_guaranteed-monotone_feature)) where the envelope is a
   monotone ceiling on a raw SSIM-family feature. Keeps V39's resolution
   below the envelope; bounds the gross OOD over-scoring. One guaranteed-
   monotone feature anchors the ceiling without constraining the whole
   net.
3. **Ship v47-strict as a sibling monotone profile** — the cleanest
   axiom-satisfying option (A1+A2+A3 by construction), competitive-ish
   panel (CID22 0.855, KonJND best-of-field), for callers that need the
   guarantee over dial resolution. V39 stays Profile::A.
4. **Soft-monotonicity + identity-anchor training** (open research) — an
   expressive net trained with a SOFT monotonicity penalty (reduce not
   eliminate inversions) + identity examples (anchor A2), accepting
   bounded residual non-monotonicity a runtime guard catches.

Recommendation: (1) the runtime saturation guard is the cheapest fix for
the actual failure (degraded-reads-as-identical) and needs no retrain;
pair with (3) v47-strict sibling for callers wanting the hard guarantee.
A single expressive+axiom-clean bake is open architecture research
(input-convex / monotone-subspace nets).
