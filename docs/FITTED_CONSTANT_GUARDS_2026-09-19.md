# Keeping fitted constants sane, identified and generalized (2026-09-19)

User direction 2026-09-19: "we might want to figure out how to ensure trained values stay sane and generalized and
don't collapse or overfit". This is the contract for any fit that produces **physical constants** — exponents,
knees, sharpnesses, per-level or per-channel weights, output maps — as opposed to a learned head's weights.
It applies first to DVIFM's `g, P, C₀, β, ς` (`../zenpapers/docs/iqa-methods/dvifm-zensim-feature-design.md`,
fits in `benchmarks/dvifm_screen2d_*`), and to every later constant fit unless a preregistration says otherwise.

It is written from measured failures, not in the abstract. Three are already on record:

- **Scale confound.** Phase 2d's first grid scored each (C₀, β) cell through a FIXED output map: start loss 908 vs
  target variance 172 on CID22-A — worse than predicting the mean — so the surface ranked cells by rescaling, not
  by masking. Fixed by refitting the map inside every cell (Amendment 1).
- **Boundary pile-up.** With the map refit, 10 of 12 plane-levels put β at the grid ceiling (20.6) and several put
  C₀ at the floor; refinement then pulled Y′ l0 back to β = 3.8 *and* improved the loss. A coordinate sweep with
  other levels unfitted produces boundary values that are artifacts of the sweep order.
- **Map degeneracy.** The fitted map repeatedly collapsed to a straight line (A ≈ 1.5e9, λ ≈ 6e-6 — an exp
  evaluated in its linear regime). Numerically survivable, meaningless as a shape, and a sign the family is
  over-parameterised for the data.
- **Earlier, elsewhere:** the v50 kb25 fit collapsed to CID22 0.64 while its own val statistic looked healthy
  (`DATA_SPLITS.md`), and Profile B's winsor bounds fitted on one corpus clamped 245 of 372 features constant on
  another — both "the fit looked fine on its own terms" failures.

## The five questions every constant must answer

A fitted constant ships only with all five answered, in its spec JSON provenance block and in the benchmark record:

1. **Is it identified?** Report the loss change at ±1 grid step in each axis, and a profile interval: the set of
   values whose refit loss is within the seed-to-seed noise of the optimum. A constant whose interval spans most of
   the grid is NOT identified — freeze it at its prior and say so. Shipping an unidentified fitted number is the
   most common form of fake precision.
2. **Is it sane?** Each constant has a registered plausible range with a literature source (masking slope
   β ≈ 0.6–0.7: Legge & Foley 1980, Watson & Solomon 1997; g ∈ [0.2, 2]; P > 0; ς > 1). Fit freely AND with a
   penalty pulling toward the prior, with the penalty weight chosen on the development leg, never on the fit loss.
   If the two fits are indistinguishable on the development leg, ship the prior-pulled one.
3. **Does it agree across domains?** Fit on at least two disjoint domains (e.g. CID22-A human, SafeSyn
   ssim2-teacher, imazen26-crops ssim2, TID/KADID human). A constant is "generalized" only when the domains'
   profile intervals overlap; then ship the pooled value. Where they do not overlap, ship the prior and record the
   disagreement as the finding. This is the cheapest and strongest guard we have.
4. **Is it stable under resampling?** Bootstrap over REFERENCES (never rows — rows within a reference are not
   independent), ≥20 refits with the grid fixed and only the refinement rerun; report the 10th–90th percentile
   band. For the large source families, also leave-one-family-out.
5. **Did it earn its own parameter?** Default to TIED constants: shared across levels, and Cb tied to Cr. A
   per-level or per-plane split must beat the tied model on the development leg by more than the seed noise
   (explicit nested-model comparison). DVIFM's own luma-only configuration is ~29 parameters; a 3-plane model with
   untied everything is ~100. Untie only what pays.

## Collapse and degeneracy detectors — run on every fit, refuse on failure

Cheap, structural, and independent of the loss. Each one names the shipped simplification it implies:

| Detector | Trips when | What to ship instead |
|---|---|---|
| Mixture collapse | any level or channel weight < 1% of the simplex | drop that term; refit without it |
| Masking off | interquartile range of `v` over TRAIN blocks < 0.05 | drop the visibility parameters for that (plane, level); weight = 1 |
| Masking is a gate | fraction of blocks with `v ∈ (0.05, 0.95)` < 5% | one explicit threshold, not (C₀, β, ς) |
| Map degeneracy | the output map's exponential runs in its linear regime (λ·E ≪ 1 over the data) | the linear map, with one parameter |
| Parameter degeneracy | \|corr\| > 0.95 between two parameters in the finite-difference Hessian (e.g. `g` vs `C₀`: φ_g(s·x) = s^g·φ_g(x), so a contrast rescale trades against the knee; `β` vs level weight; `P` vs the map) | reparameterise, or fix one at its prior |
| Target clamped | any target produced by clipping a signed teacher (e.g. `clip01(ssim2)`) — measured 7.2% of the imazen26 leg | refit on the RAW signed teacher, or a monotone squash; never a clamp |
| Saturation | > 5% of TRAIN rows at the score ceiling or floor, or any feature clamped constant by a guard on the fit corpus (the Profile B winsor failure) | widen the guard corpus, or the feature is dead here |
| Dead input | a term's removal changes the development metric by less than seed noise | remove it; record it as dead |

A detector tripping is a **result**, not an error: "the smooth masking curve does not earn its three parameters at
this level; a flatness gate does the same work" is a publishable finding and a cheaper kernel.

## Process rules

- **Fit, development, decision are separate.** Constants come from the fit rows; splits and penalty weights from a
  source-disjoint development leg; holdouts decide nothing and are read once, frozen, after everything is fixed.
  No constant is ever selected by a holdout number.
- **Budget by the slowest arm** and pair seeds across arms (the 2026-09-19 fairness contract,
  `PREREG_SCALES_PLANES_2026-09-19.md`); report per-seed paired differences, never bare means. Seed spread is not
  a confidence interval.
- **Every fitted-feature arm carries a permuted-column control** of the same width. Without it, "our features
  hurt" cannot be separated from "any N extra inputs hurt at this data size" — which is exactly what the DVIFM
  add-on screens could not separate at 8,327 rows.
- **Prefer the smaller model when the evidence ties.** Ship the tied, prior-pulled, fewer-parameter variant.
- **In code:** assert `v` non-increasing in C, the score monotone in every block error and exactly at the scale
  ceiling for an identical pair, bounded outputs, no NaN at C = 0, and that a declared constant lies inside its
  registered plausible range. Convex mixtures use a simplex parameterisation so non-negativity cannot be violated
  by an optimiser step.
- **Provenance:** the spec JSON records, per constant: value, prior and its source, profile interval, which
  domains agreed, ±1-step sharpness, tied-or-free, and which detectors were checked. A constants file without that
  block is not evidence.

## Cache and storage discipline for constant fits (user, 2026-09-19: "we need to be compact")

A constants fit needs the *distribution* of block statistics, not every block. Phase 2d wrote 60 GB of block cache
for 6.5k plane-rows (~971 KB per row per plane) — measured, and far too much: SafeSyn at 180k rows × 3 planes would
be ~520 GB against ~102 GB free. Storage rules for every later fit:

- **Cap blocks per row at WRITE time** (deterministic stride or reservoir, cap + seed in the cache header), target
  ≤ 40 KB per row per plane. The fitter already stride-subsamples for its Adam step; caching the full set buys
  nothing but disk.
- **Quantise the cached record.** The 18 values per block are extrema and a peak error on a bounded axis: f16, or
  i16 in the kernel's own integer domain, is enough for fitting and halves or quarters the cache. Record the
  quantisation and check it against a full-precision cache on one small domain.
- **Cache summaries where a summary suffices.** A (C̃, m) 2-D histogram per (plane, level) — say 256×256 bins in the
  integer log domain, a few hundred KB per domain — supports the whole C₀ × β grid exactly, because the grid's loss
  is a sum of per-block terms. Keep per-block records only for the parameters that need gradients through
  individual blocks, and only on the subsample.
- **One domain live at a time.** Fit a domain, write its constants and surfaces, then release its cache (mirror to
  Tower first if it cost real compute) before extracting the next.
- **Budget stated up front**, checked with `df -h /mnt/v` before each stage, and ≥80 GB kept free. A phase that
  needs more than ~25 GB of scratch needs a redesign, not more disk.

## Where this is applied

- Phase 2d (`benchmarks/dvifm_screen2d_prereg_2026-09-19.md` + Amendment 1): per-cell map refit, widened grid with
  edge extension, convexity assertions, committed loss surfaces.
- Phase 2f (SafeSyn constants, `~/tmp/devin/dvifm_phase2f_safesyn_prompt.md`): the cross-domain agreement gate (3)
  and the cap-adequacy check are the point of that phase.
- Any future constant fit: cite this document in the preregistration and state which guards are in force.
