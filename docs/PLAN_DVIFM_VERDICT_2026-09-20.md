# DVIFM: what we can extract, what to record, and the experiments that settle it (2026-09-20)

User direction 2026-09-20: an integer kernel is cheaper to build than one test is to run, so stop gating it behind
screens. The three questions this program answers, and nothing else:

- **Q1 — is anything in DVIFM useful to zensim?** (a mechanism we adopt, not a feature block we bolt on)
- **Q2 — can DVIFM standalone beat zensim?** (held-out, not in-sample)
- **Q3 — does it give us spatial steering we do not have?** ([plan](PLAN_SPATIAL_STEERING_DVIFM_2026-09-19.md))

Companions: [fitted-constant guards](FITTED_CONSTANT_GUARDS_2026-09-19.md),
[joint core](PLAN_JOINT_CORE_SET_2026-09-19.md), [scales/planes prereg](PREREG_SCALES_PLANES_2026-09-19.md).

## 1. What the 2026-09-19 fits actually let us extract

Two independent 3-plane fits exist: CID22-A (25 human refs, in-sample SROCC 0.949) and TID+KADID (human, 0.922).
Both ran the corrected protocol (per-cell map refit, 18×18 grid with edge extension, multi-start refinement) and
both recorded per-(plane, level) sharpness = the loss change one grid step away. Read across the two domains:

| What | Extractable? | Evidence |
|---|---|---|
| **Luma level-0 knee `C₀ ≈ 0.04–0.08`** | **YES — the one portable constant** | CID22 0.0789, TID 0.0430; sharpness 2.2 and 1.5 MSE per grid step — the only cell where moving the knee costs real loss in BOTH domains |
| Knee at coarser levels | No | disagrees by 4–6 orders of magnitude between domains (e.g. Y′ l2 both at the grid floor but with opposite β; Cb l0 CID22 at the ceiling 3.0, TID at 0.002) |
| **Exponent β anywhere** | **No — unidentified** | one-step sharpness ≤0.05 MSE on losses of 19–84, i.e. <0.3% of the loss, at 13 of 15 cells; best case Y′ l3 = 0.195 (1%) |
| **β's two-state behaviour** | **YES, as a shape, not a value** | 10 of 15 cells want β ≥ 16 (a step: full weight below the knee, ~none above); the rest want β ≤ 0.9 (masking off). No cell wants a psychophysical 0.6–0.7 slope |
| **Chroma-dominant channel mix** | **YES, as a direction** | both domains put most weight on chroma: CID22 (Y 0.287, Cb=Cr 0.357 each → 71% chroma), TID (Y 0.167, Cb=Cr 0.417 → 83%). Contradicts the talk's "luma weighted higher" |
| Level weights | No | Y′ l1 = 0.266 (CID22) vs 0.0001 (TID); Cb l0 = 0.138 vs 0.541. Three cells collapsed below the 1% floor |
| Output map shape | Linear | the fitted `A·exp(−λE)+B` ran in its linear regime on CID22 (λ ≈ 4e-9); TID produced healthy λ. Ship the 1-parameter linear map and say so |

**Consequence for the kernel, and it is a simplification, not a loss:** a two-state visibility — weight 1 below a
per-(plane, level) knee, weight ~0 above it, or uniformly 1 when that level wants masking off — reproduces what the
data supports. No `pow`, no exp/log, no 3-parameter curve: one integer compare per block. That is why the integer
kernel is now built first rather than last.

## 2. What to record, so this is usable later

Every constants artefact carries a spec JSON with, per (plane, level):

```
plane, level,                         # ycbcr_y|cb|cr or xyb_y, 0..4
mode,                                 # "gate" | "off" | "curve"   <- the simplification actually supported
c0, c0_interval,                      # value + profile interval (loss within seed noise of the optimum)
c0_sharpness_pm1,                     # MSE change one grid step down / up
beta, beta_interval, beta_sharpness_pm1,
sigma, g, P,                          # with the same three fields each
identified,                           # bool per constant, from interval vs grid span
prior, prior_source,                  # e.g. beta 0.65, Legge & Foley 1980
domains_agreeing,                     # ["cid22a","tidkadid","core-v1"] whose intervals overlap
tied_or_free,                         # which constants were tied across levels/channels
detectors_tripped                     # mixture-collapse | masking-off | gate | map-degenerate | clamped-target
```

Plus, once per artefact: the channel simplex, the level simplex, the output-map form and parameters, the block grid
(n, phase), the band mode (lap|local|boxres), the plane normalisation (min, span per channel, and how derived), the
feature-set identity, the fit domain with row-key hash, the git commit, and the tool hashes. And for serving: the
**integer LUT layout** — index derivation from the float/fixed bits, table size, Q format of the stored values, and
the measured max error against the real-valued function over the full input domain.

A constants file without the interval, the sharpness and the agreeing-domains fields is not evidence and may not be
baked. This is the guards doc's five questions, expressed as a file format.

## 3. The experiment program — each experiment ≤1 h on `joint-core-v1`

Built first, in parallel, because it is cheap: **the i16 kernel** (integer domain, integer log-mapped tables,
two-state visibility as the default mode, i64 accumulation ⇒ bit-identical across strips, threads, SIMD tiers and
architectures). It is a prerequisite for X3 and X6 and makes every other run faster.

| # | Question | Experiment | Decides |
|---|---|---|---|
| X1 | Q2 | Standalone 3-plane DVIFM vs `fast-ssim2`, B, D and both Rev3 ensembles on the core's development legs + one frozen read of the sealed CID22-B 24 refs; SROCC/KROCC/PLCC with a paired bootstrap over references | whether DVIFM standalone is competitive at all |
| X2 | Q1 | The **two-state** constants (gate/off per level, from §1) vs the fully fitted curve vs the priors — same arms otherwise | whether the curve earns its parameters, and fixes the constants we ship |
| X3 | Q1/Q2 | i16 kernel vs f64 oracle: per-feature deviation, rank preservation, and measured cost at 64²…4096² with α+β·pixels | whether the cheap form is the same metric |
| X4 | Q1 | **Pooling transplant:** zensim's own XYB decomposition with DVIFM's 5×5 block-peak × two-state-visibility pooling (arm H2 of the tradeoff study) vs production zensim, leaders' recipe, 5 paired seeds, permuted control | the highest-value single idea: block pooling without adopting the pyramid |
| X5 | Q1 | **Chroma transplant:** zensim + DVIFM chroma terms only (Cb/Cr), given both domains put 71–83% of weight on chroma | whether the chroma finding transfers to our model |
| X6 | Q3 | Steering S1–S3: exact additive block map, rectangle-gain prediction vs actual finite-edit gain (against the current attribution map as control), and block-rank agreement with ssim2 ∧ butteraugli | whether the map is better than what we steer with today |

Order: X3 (validates the kernel), X2 (fixes constants), X1 (the standalone verdict), X4, X5 (the transplants), X6
(steering). X4 is the one most likely to produce something we ship, because it adopts a mechanism rather than a
feature block — and the add-on screens already showed that bolting 30 columns onto basic228 is redundant.

**Kill criteria, declared now.** If X1 shows standalone DVIFM below `fast-ssim2` on the sealed refs AND X4 shows no
paired gain from the pooling transplant AND X6 shows the map no better than the current attribution map, the family
is recorded as a negative result, the kernel stays default-off as research surface, and the program ends. Any single
one of those passing is worth the next round.
