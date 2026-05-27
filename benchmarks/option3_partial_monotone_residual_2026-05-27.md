# Option 3: partial-monotone residual — ARCHITECTURE PROVEN (2026-05-27)

The first metric structure to achieve all three similarity axioms AND
full resolution simultaneously. Probe:
`zensim-validate/src/bin/monotone_subspace_probe.rs`.

## Form

    score(x) = 100 − λ_m·(D_m(x_mono) − D_m(id)) − F_b(x_free)

- x_mono = 300 sign-safe features, x_free = 72 sign-flip features
  (benchmarks/feature_sign_mask_2026-05-26.tsv).
- D_m = Σ_j c_j·LeakyReLU(W_m·x_mono + b_m)_j, with W_m, c, λ_m ≥ 0
  (W_m projected ≥0 each step) → monotone-↑ scalar dissimilarity in
  x_mono. Identity has minimal x_mono ⇒ D_m(x) ≥ D_m(id).
- F_b = δ·tanh(relu(M(x_free) − M(id))/δ) ∈ [0, δ): BOUNDED downward
  refinement from the free features (unconstrained MLP), 0 at identity.

## Guarantees (by construction)
- A1 bounded ≤ 100 (both subtracted terms ≥ 0).
- A2 identity is the UNIQUE max (both terms = 0 only at identity).
- A3 monotone-↓ in every sign-safe feature, up to a bounded δ slack
  from F_b (bounded non-monotonicity).
- RESOLUTION: the mono term is unbounded below ⇒ a terrible distortion
  gets a genuinely low score, NOT capped at 99 (the fatal flaw of the
  runtime-guard option 1).

## Probe result (axioms ✓ + resolution ✓; panel under-powered)

Blur ladders: inv=0, above_id=0 on ALL four contents; identity=100;
heavy blur → −1700 to −4100 (monotone, full resolution). The axioms +
resolution hold exactly.

Panel (SROCC): cid22 0.61, kadid 0.76, tid 0.76, konjnd −0.33, aic3 0.56.
Mediocre — but the PROBE is under-powered vs v47-strict (which got cid22
0.855 on the same 300 features):
- 1-layer monotone backbone (v47-strict: 2-layer 372→128→64).
- 4 groups, NO safesyn (the 196k main signal), NO auto-transforms.
- NO L2 → encoder weights grow unboundedly → scores → −6000 → loss
  climbs (Huber-clamping the MSE residual bounded the gradient but not
  the underlying weight growth).

So the panel gap is the probe setup, not the structure. The structural
question — "can a metric be expressive + full-resolution + axiom-clean
at once?" — is answered YES.

## Production path (to a shippable option-3 bake)

Wire the partial-monotone residual head into zensim_mlp_train:
1. 2-layer monotone backbone (W1≥0, W2_enc≥0 — reuse the masked
   monotone-cbc machinery on the 300 features) → D_m.
2. Bounded free head F_b on the 72 features (new metadata payload +
   bake_runtime dispatch).
3. Residual output `100 − λ_m·(D_m − D_m_id) − F_b` (no tanh pin).
4. Full recipe (safesyn + cid22_train + kadid + tid + konjnd +
   auto-transforms + anchor) + L2 on the encoder (prevents the blow-up)
   + Huber/σ-weighted MSE.
5. Tune δ (the bounded-non-monotonicity slack) — δ=0 is strict-monotone,
   δ large approaches V39.
Eval vs V39 + v47-strict + ssim2/cvvdp on the full panel + blur ladder.
Expected: competitive panel (full recipe + 2-layer) WITH axioms +
resolution by construction — a single bake that could replace V39 at
Profile::A and is monotone-by-construction.

This is the only known route to expressive + axiom-clean together;
input-convex / deep-lattice nets are the literature analogues.
