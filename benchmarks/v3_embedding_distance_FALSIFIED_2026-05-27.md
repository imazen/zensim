# V3 isotropic embedding-distance + unconstrained control — FALSIFIED (2026-05-27)

## Hypothesis

V2 (partial-monotone) capped CID22 at 0.625 because the unbounded monotone
term drowned the bounded free head. **Drop the monotone constraint, keep only
A1+A2** via an identity-anchored embedding distance
`score = 100 − λ·‖φ(x) − φ(id)‖` (φ unconstrained). A1 (≤100) + the
defect-half of A2 (nothing exceeds identity) hold by construction; φ is
free → should recover the panel toward the ~0.88 expressive ceiling.

## Result — FALSIFIED (worse than V2)

| corpus | V2 (partial-monotone) | V3 (embedding distance) |
|---|--:|--:|
| CID22 | 0.625 | **0.0826** |
| KADID | — | −0.0595 |
| TID | — | −0.2025 |
| KonJND | −0.349 | −0.2245 |

Near-random rank. A1+A2 held (above_id=0) but the panel collapsed, and blur
ladders were non-monotone (color_blocks dipped 100→41→38 then *recovered*
to 65). Log: `v3_embedding_distance_FALSIFIED_2026-05-27.log`.

**Mechanism**: the isotropic RMS distance scores by *how far* from identity,
not *which way* — but human MOS is highly directional (same displacement,
different distortion type → very different perceptual cost). Requiring P≥0
with P(id)=0 also forces ∇P(id)=0 — a flat spot at identity, exactly the
high-quality regime where rank matters most.

## Unconstrained control — DIVERGED (recipe ceiling unmeasurable)

Same 372→64→32 trunk + free linear readout (no axiom). It **diverged**:
scores exploded to +21836, loss rose to 88, SROCC strongly anti-correlated
(CID22 −0.68). The hand-rolled f64 probe recipe (no grad clipping,
cosine-restart LR, unbounded output) **cannot stably train an unconstrained
head** — so the probe recipe can only train *constrained* architectures and
**cannot measure the unconstrained ceiling**.

## Conclusion — the probe arc is recipe-confounded; v47-strict is the answer

Probe SROCC is comparable only AMONG probes (best axiom-clean = V2 0.625);
the absolute gap to V39's 0.879 is dominated by the weak hand-rolled recipe,
not the axiom constraints — proven by `v47-masked-strict`, which is the SAME
axiom-clean monotone architecture trained with the PRODUCTION recipe and hits
**CID22 0.855**. The probes usefully (a) ruled out isotropic-distance, (b)
confirmed A1+A2-by-construction is achievable, (c) showed the partition holds
the axioms — but the production path (v47-strict + dial recal, see
`v47_strict_recal_methodology_2026-05-27.md`) is the deliverable, not a probe.

Do NOT retry isotropic embedding distance, and do NOT use the hand-rolled
probe recipe to draw absolute-SROCC conclusions about any architecture.
