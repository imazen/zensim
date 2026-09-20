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

## 4. Why these sampling kernels, and the block-edge-contrast experiment

**The kernels are not interchangeable and they answer different requirements.**

- **2×2 box (zensim's scale step).** Exact area average, so it matches the "pixel = area" sensor/display model and
  preserves DC exactly, at 1 add per input pixel. Its response is |cos πf|: it zeroes Nyquist but passes 0.71 at
  half-Nyquist, so most of the band that folds under 2:1 decimation survives the filter. That is the theoretical
  root of the measured aliasing — F1 swings 1.62× over codec-grid phases at level 3 on box scales against 1.12× on
  the binomial pyramid.
- **Binomial [1 2 1]/4 (DVIFM's, Burt–Adelson).** Response cos²(πf): 0.5 at half-Nyquist, zero at Nyquist, so it
  attenuates the folding band far better at 4 adds and 2 shifts. Repeated binomial convolution approaches a
  Gaussian, and the Gaussian is the kernel for which coarsening creates no new extrema (scale-space causality,
  Koenderink/Lindeberg) — structure at a coarse level is inherited, never invented. Its coefficients are powers of
  two, so it is **exact in integer arithmetic**, which is why the i16 kernel can be bit-identical everywhere.
- **Mitchell–Netravali (B=C=1/3).** Justified empirically, not information-theoretically: Mitchell and Netravali
  mapped the cubic (B,C) plane into blur / ringing / blocking regions by subjective study and picked the balance
  point. That is a statement about human preference, which is why it belongs on the **data** side (renditions).
- **Lanczos.** Maximises stopband attenuation by windowing a sinc, and pays in ringing. For metric work that is
  actively harmful: ringing is a visible artifact, so a Lanczos-resampled reference makes the metric measure the
  resampler. It is why the Lanczos renditions were excluded from the core.

**Hence the split we have landed on, stated as a rule:** binomial inside the metric (integer-exact, no ringing,
no invented structure, good folding-band attenuation), Mitchell for preparing image data (perceptual preference),
Lanczos nowhere, box only where its exact-DC property is the point.

## 5. X7 — block-edge contrast (user question, 2026-09-20)

Worth testing, and cheap, but note the tension: **every low-pass step smears exactly the feature we want.** A DC
step at an 8-pixel codec boundary becomes a ramp after [1 2 1], so the block-peak statistic on a band plane is least
sensitive at the boundary where blocking artifacts live. Three additive terms, each one change:

1. **Unfiltered level-0 peak.** Compute the 5×5 block peak on the RAW difference plane (no band, no blur) alongside
   the band planes. Costs nothing new — the difference already exists — and it is the only term that sees a step at
   full amplitude.
2. **Across-boundary vs within-block contrast.** Per 5×5 block, the contrast measured across its interior columns
   and rows against the contrast within the neighbouring 3×3 corners: a step confined to one line is blocking;
   the same contrast spread over the block is texture. This is the discriminator the current edge-discounted
   contrast does not provide — that discount exists to stop an edge MASKING error, not to detect the edge itself.
3. **Phase-agnostic lattice.** zensim's existing blockiness slot (v2 idx 25) is oriented but fixed-phase, so it
   sees a codec whose partition is offset only weakly. Run the term over all 8 phases and keep the maximum, or run
   it on the coprime 5-grid so alignment averages out; report both against the fixed-phase control.

Gate: on TRAIN codec pairs, per codec, does any of the three raise within-image agreement with ssim2 ∧ butteraugli
on JPEG and WebP (where blocking dominates) without hurting AVIF/JXL? Run as variants inside X4, whose pooling
machinery already exists; the permuted-column control applies to each added term.

## 6. Why the constants came out unphysical — the loss and the fitting loop need iteration

User question, 2026-09-20: "our loss and learn must need iteration if we aren't getting enough psychovisual
constants — but I thought we did??" Both halves are right, and the second deserves a correction.

**We never actually measured psychovisual constants.** The 2026-09-19 phase-2 fit reported β = 0.604–0.658 per
level, which looks exactly like Legge & Foley's 0.62 and Watson's 0.7 — but it was initialised at 0.65 with a prior
pulling it there, and the whole fit moved the loss by 2.3%. Those numbers were the prior, not evidence. The
phase-2d fit dropped that prior, and with the prior gone the data did not constrain β at all: one grid step changes
the loss by under 0.3% at 13 of 15 cells, and the optimum runs to the grid edge. So the honest statement is that the
fit is uninformative about β, not that β is 20.

**Five reasons the loss cannot see a masking exponent, each a concrete iteration:**

1. **Pooled MSE against MOS is the wrong objective for a within-image weighting.** The exponent changes how blocks
   are weighted *inside* an image; pooled MSE mostly cares about each image's overall level, and the per-cell refit
   of the output map (A, B, λ) absorbs level error before β ever sees a gradient. **Iteration: fit the constants
   against a within-reference ranking loss** — pairwise hinge over same-reference pairs, or Spearman surrogate —
   with pooled MSE kept only as a reported diagnostic.
2. **Coordinate-wise sweeps let levels compensate for each other,** which is what produces boundary optima that
   later sweeps undo (Y′ l0 went 20.6 → 3.8 and the loss *improved*). **Iteration: joint optimisation** over all
   (plane, level) constants and the head together, with the grid used only for initialisation.
3. **Too many free constants for the data.** 15 knees + 15 exponents + 15 sharpnesses + weights on 2,192 rows.
   **Iteration: fit TIED first** — one β shared across levels, Cb tied to Cr — and untie only what beats seed noise
   on the development leg. A tied β is identifiable where 15 free ones are not.
4. **The prior was silently dropped** between phase 2 and phase 2d. **Iteration: always fit both** (free and
   prior-pulled toward 0.65) and report the pair; ship the prior-pulled one when the development leg cannot tell
   them apart.
5. **Our contrast axis is not the axis the psychophysics is stated on.** Legge–Foley-style exponents describe
   threshold elevation against masker contrast *relative to local mean luminance* at a given spatial frequency. Our
   C̃ is a band-plane amplitude range over a 3×3 corner, globally normalised by the channel's sRGB-cube span — a
   band amplitude, not a Weber or Michelson contrast. No exponent fitted on that axis is comparable to the
   literature. **Iteration: define contrast as band amplitude ÷ local mean (a Weber-like ratio) and refit** — this
   is the variant whose exponent can be compared to published values at all, and it is the one that would let us
   claim a psychovisual constant honestly.

Until at least (1), (3) and (5) are done, "the exponent wants a step" should be read as "this loss cannot
distinguish a step from a slope", which is a statement about our fitting, not about vision.

## 7. X8 — SafeSyn as a constants domain, alone and as majority weight

User, 2026-09-20: "we can also try safesyn alone or safesyn as majority weight". Legitimate and cheap, and it
attacks the identifiability problem from the data side rather than the loss side:

- **SafeSyn alone** (`/var/tmp/zensim-validation-2026-09-15/recovery/tables/safesyn_{fit,development}.parquet`,
  141,054 / 38,758 rows, all-TRAIN, zero holdout exposure, **signed** ssim2 targets down to −744 — never clipped).
  64× more rows than CID22-A, and 25 distortion types × 5 severities, which is exactly the contrast-by-noise
  coverage a masking exponent needs. The catch is stated on every table: the target is a metric teacher, so a
  constant fitted here is fitted to SSIMULACRA2's opinion of masking, not a human's.
- **SafeSyn-majority core variant** as an arm in X1/X2: the same model fitted on a core reweighted so SafeSyn
  dominates, against the 75%-photography core. This is a deliberate exception to the composition rule, run as a
  comparison rather than a replacement, and it measures how much the teacher's own distortion distribution is
  driving the constants.
- Judge both on the TRAIN-side development legs; the sealed CID22-B read stays single and is spent in X1.
  Report the constants three ways — CID22-A, TID/KADID, SafeSyn — with profile intervals, and mark a constant
  portable only where all three intervals overlap. That is the first test with enough rows to expect an answer.
