# Paper-faithful steerable-pyramid IW weight estimator — spike methodology (2026-05-15)

**Worktree**: `~/work/zen/zensim/.claude/worktrees/agent-ae0e10c9dcc6401ac`
**Agent**: `claude-session-iw-pyramid-spike`
**Status**: spike — research code, NOT ship-grade.

## Why this spike exists

The V_20a IW-SSIM bake catastrophically failed CID22 (aggregate SROCC
0.187 at k=8, vs V_18 ship 0.8880). The previous divergence analysis
identified four ways the current `zensim/src/iw_pool.rs`
implementation departs from Wang & Li 2011 (the IW-SSIM paper):

1. **Spatial-domain variance** instead of wavelet GSM scale parameter
2. **5×5 box window** instead of 11×11 Gaussian
3. **6-output multi-statistic block** instead of a single
   weighted-mean SSIM
4. **MLP-fed features** instead of an MS-SSIM-weighted scalar

The paper's emphasis is on #1 — a wavelet-domain estimator gives
per-orientation directional sensitivity that a scalar spatial
variance cannot. Vertical, horizontal, and diagonal edges contribute
to subband energy at different orientations; their weights differ.
This spike tests whether the wavelet path carries different signal
from the spatial path on our V_18 4-scale XYB pyramid.

## Paper claim

**Wang & Li 2011** ("Information Content Weighted Structural
Similarity Index", IEEE Trans. Image Processing vol. 20 no. 5, May
2011, pp. 1185-1198): IW-SSIM gives **+0.006 SROCC weighted-average
lift over MS-SSIM** across six IQA databases (LIVE, A57, IVC,
Toyama, TID2008, CSIQ).

The paper's weight formula (eq. 18 in §III-C) for the per-pixel
weight `w_n(i)` at level n is:

> w_n(i) = log(1 + σ²_x(i) / σ²_n)

where `σ²_x(i)` is the **local variance of wavelet coefficients at
pixel i**, and `σ²_n` is the noise-floor parameter (the additive
visual noise variance at level n).

The GSM (Gaussian Scale Mixture) framework (§II-B and eqs. 5-8)
models a per-band wavelet coefficient vector `c_i` at location i as
`c_i = z_i · u_i` where `u_i ~ N(0, C_U)` is a fixed-covariance
Gaussian and `z_i` is the **per-pixel scale parameter** (a positive
random scalar). The maximum-likelihood estimate of `z²_i` from a
patch of L wavelet coefficient vectors is:

> z²_i^ML = (1 / (L · K)) · Σ_{j∈patch} c_j^T C_U^{-1} c_j

where K is the dimensionality of the coefficient vector (= 6
orientations in the default steerable pyramid). The paper uses an
11×11 Gaussian window weighted patch around each pixel.

For our adaptation, we treat `σ²_x(i)` as the per-pixel local
variance of subband coefficients (single-band view: maximum across
6 orientations gives directional sensitivity).

## Aggregation across scales

The paper combines per-scale weighted-SSIM into a single scalar
using the same exponents as MS-SSIM (Wang & Bovik 2003):

> β_n = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  for n = 1..5

IW-SSIM at level n: `iw_ssim_n = Σ_i w_n(i) · SSIM_n(i) / Σ_i
w_n(i)` — a pure weighted mean.

Multi-scale combination: `IW-SSIM = Π_n iw_ssim_n^β_n` (geometric
mean with MS-SSIM weights).

The paper uses 5 scales. Ours has 4. The spike skips the multi-scale
collapse — we just compute weights at scale 0 (the finest) and feed
them through the existing zensim feature pipeline as a drop-in
replacement for the spatial-variance estimator.

## Reference implementations consulted

1. **Wang's original Matlab** (U Waterloo, `iwssim` package): the
   `buildSCFpyr.m` / `reconSCFpyr.m` steerable-pyramid builder uses
   Simoncelli's 1995 design with 6 orientations and 5 scales by
   default. `iwssim.m` calls this to build per-band coefficient maps,
   then runs an 11×11 Gaussian window over each subband to estimate
   `σ²_x` and feed the log-info-content weight.

2. **pyiqa / IQA-PyTorch** (`pyiqa/archs/ssim_arch.py`, `IWSSIM`
   class): a clean Python re-implementation. Uses the same 5-level,
   6-orientation steerable pyramid; computes `σ²_x` via 11×11
   Gaussian (sigma=1.5) on per-band coefficients; uses
   `σ²_n = 0.4` as the noise-floor parameter (the paper's
   recommendation for natural images).

3. **scikit-image**: no IW-SSIM (basic SSIM only). Not useful.

## Our adaptation plan

The full steerable-pyramid construction is ~200 LOC and would
require a 5-level decomposition that doesn't align with our 4-scale
XYB pyramid. For a SPIKE (research code, no shipping), we ship an
**approximation that captures the paper's directional signal at the
finest scale** while reusing zensim's existing pyramid layer:

### The approximation: 4-orientation oriented-gradient + log-info-content

Use 4 oriented gradient kernels (0°, 45°, 90°, 135°) on the
reference plane at scale 0, then compute per-orientation local
variance over a 9×9 Gaussian window (close approximation to the
paper's 11×11). At each pixel:

> σ²_p(i) = max over k=0..3 of  Var_{patch}(g_k(i))

The max-across-orientation captures the "directional max" that
gives diagonal edges different weight from horizontal+vertical
edges of the same total energy. (Sum would also work; max is
what the paper recommends per §III-B's discussion of "dominant
orientation".)

Then apply the paper's log-info-content weight (already in
`iw_pool.rs::info_log_sigma_e_sq`):

> w(i) = log₂(1 + σ²_p(i) / σ²_e)

This re-uses the log transform field added in main commit
`c23f178c`; the spike only changes how `σ²_p` is computed.

### Why this is the right spike (not the full steerable pyramid)

- **The interesting signal is directional max**, not the
  steerable-pyramid filter shape itself. If 4 orthogonal oriented
  gradients carry the same directional signal as 6 Simoncelli
  steerable subbands (to leading order), the spike answers the
  research question.
- **A full Simoncelli steerable pyramid is ~200 LOC of FIR design**
  (steerable basis filters, pyramid recursion, oriented subbands).
  Implementing it correctly in a 4-hour spike is unrealistic.
- **The paper's hypothesis is directional sensitivity**, not "the
  Simoncelli filter is the only correct way to get it". A 4-tap
  oriented gradient catches diagonal vs axis-aligned at the same
  zeroth-order level.
- **If the 4-orientation approximation shows decorrelation from the
  spatial path, the full steerable pyramid is a follow-up that
  warrants the LOC investment.** If the 4-orientation
  approximation correlates ~1:1 with spatial variance, the full
  steerable pyramid won't recover signal either.

A full Simoncelli implementation is a documented follow-up (Step 5
of this doc); the spike's purpose is the A/B comparison.

## What the spike measures

- **A/B weight-map correlation** between
  `IwWeightKind::LocalVariance` (current) and the new
  `IwWeightKind::SteerablePyramidLogGsm` on a real KADID image
  pair.
  - Pearson correlation between the two weight maps after
    normalising both to unit max.
  - **Decision rule**: if r > 0.95, the wavelet path adds
    negligible signal to spatial variance — don't bake. If r <
    0.85, the wavelet path is decorrelated enough to warrant a
    training run. r ∈ [0.85, 0.95] = mixed; warrants a follow-up
    spike with the full Simoncelli pyramid.
- **Synthetic diagonal-vs-axis-aligned test**: an image with the
  same total edge energy split across diagonal vs horizontal+
  vertical orientations should produce **different** weight
  distributions under the steerable path and **identical**
  distributions under spatial variance.
- **Reference-impl smoke test**: weight at the center of a single
  step edge (an isolated transition from 0 to 100, no other
  signal in the patch). Compare to a hand-computed expected value
  under the paper's formula. Match to ±5 % is the gate.

## Cost estimate

The runtime perf doc
(`benchmarks/extended_iw_runtime_perf_2026-05-15.md`) shows the
current spatial-variance IW pool adds **+13–15 % per-pair compute
at 512²–1024²**, and the combined extended+IW path is **+25–28 %**.

The 4-orientation oriented-gradient path replaces a single 5×5 sum
pass with 4 oriented gradient convolutions + 4 local-variance
passes. Rough cost: **3× the current local-variance pass**, but
still small compared to the basic SSIM map computation. Expected
overall: **+25–30 % per-pair compute vs standard** when IW is
enabled (was +13–15 % for spatial). The combined extended+IW path
would land around **+35–40 %**.

A full Simoncelli pyramid would be **~2× the oriented-gradient
cost** (6 orientations × 5 scales × steerable filter taps), so
roughly **+50 % vs standard** for the full IW path. That's the cost
to beat with whatever lift the spike measures.

## What this spike does NOT do

- **Does NOT train a bake.** The acceptance gate is the A/B
  weight-map comparison + the synthetic directional test. Training
  is a SEPARATE follow-up that should only happen if the weight
  maps are sufficiently decorrelated.
- **Does NOT touch `IwWeightKind::LocalVariance`.** That kind stays
  semantically identical so V_20a bakes remain reproducible.
- **Does NOT ship a new profile variant.**
  `ProfileParams.compute_iw_features` already exists; the spike
  changes WHICH weights are computed, not the runtime wiring.
- **Does NOT compute σ²_x against a steerable subband.** The
  approximation uses oriented gradients at the input scale, which
  is materially different from the paper (no subband bandpass).
  This is documented as a known divergence — see "Known divergences
  from the paper" below.

## Known divergences from the paper

| # | Paper (Wang & Li 2011) | This spike | Why |
|---|---|---|---|
| 1 | 5-level Simoncelli steerable pyramid | 4-orientation oriented gradients at scale 0 | LOC budget; spike tests whether directional signal matters at all |
| 2 | 6 orientations | 4 orientations | Same — 4 captures the axis-vs-diagonal split |
| 3 | 11×11 Gaussian window | 9×9 Gaussian window | Faster, similar SP support |
| 4 | σ²_e (noise floor) ≈ 0.4 on luminance | σ²_e configurable, default chosen by experiment | Our XYB-X channel range differs from luminance |
| 5 | Multi-scale geometric mean with MS-SSIM weights | Single-scale weights at scale 0 (spike) | The MS-SSIM aggregation is a SEPARATE follow-up |
| 6 | Spatial variance estimator NOT in the paper | Kept as `IwWeightKind::LocalVariance` for V_20a reproducibility | Existing experiments depend on it |

If the spike measures meaningful weight-map decorrelation (r <
0.85), the next iteration replaces (1)–(3) with a full Simoncelli
pyramid (~200 LOC follow-up) and re-runs the A/B.

## Falsification criterion for the larger IW-SSIM direction

Per CLAUDE.md "experiment-rigor policy" (push to paper claim before
falsifying):

- The paper claims **+0.006 SROCC lift** over MS-SSIM.
- Our setup is not MS-SSIM, it's a 372-feat MLP-fed pipeline. The
  comparable claim translates to: **paper-faithful IW weight
  estimator should at least match V_18 ship on CID22** (not 0.187
  like the V_20a falsification).
- The spike DOES NOT train a bake, so it can't falsify the larger
  direction. What it CAN falsify is: if the wavelet weight map
  correlates 1:1 with spatial variance, the IW direction is dead
  regardless of what the bake measures — there's no extra signal to
  train on.

## Implementation references

- Paper PDF: not pulled into this worktree; pyiqa's
  `pyiqa/archs/ssim_arch.py` `IWSSIM` class is the most accessible
  re-implementation.
- Spike code lives at `zensim/src/iw_pool.rs` (new
  `IwWeightKind::SteerablePyramidLogGsm` variant + helpers).
- Tests: `zensim/src/iw_pool.rs` `tests` module.
- A/B comparison: `zensim-validate/examples/iw_pyramid_ab.rs`
  (new).

## Decision gate (post-spike)

After the spike runs:

1. **r > 0.95 (weight maps correlate ~1:1)** → IW direction
   FALSIFIED at the weight-estimator level. Document, archive
   weight maps, do not train a bake. Recommendation: pivot to a
   different B0-B5 lever (see CLAUDE.md V_20 learnings: the
   distortion-manifold direction was also falsified; we're out of
   IW-direction levers absent a fundamentally different
   mechanism).
2. **r ∈ [0.85, 0.95] (mixed)** → spike INCONCLUSIVE; warrants a
   full Simoncelli pyramid follow-up (~200 LOC). Don't train a
   bake on the 4-orientation approximation; train on the full
   pyramid if and when that's built.
3. **r < 0.85 (decorrelated)** → wavelet path carries different
   signal. Worth training a 372-feat bake against the new
   estimator (~$0.30 GPU). Run the trainer; eval at full Mohammadi
   panel; compare to V_18 ship.
