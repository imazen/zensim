# Full-resolution Y with coarse XYB: prototype and sampling analysis

Registered September 12, 2026, before implementation. Concrete caller:
models declaring basic/peak feature IDs through Rust `BakeScorer`, and
`research::extract` requests for those same IDs. No public API additions.

Extend the existing feature planner and folded extraction owner. If no requested
input needs full-resolution X/B, specialize the walk with a const boolean to
skip those channels' gather and moment kernels. Keep conversion, downsampling,
all full-resolution Y features, and all coarser XYB arithmetic unchanged.
Support basic/peak requests first; other families conservatively retain X/B.
Uncomputed IDs must be absent from the plan's populated set. Ensemble unions
must restore X/B if any member needs it. No existing model is pruned or refit.

Validation: retained-feature bit parity on identity, impulses, grid patterns,
odd dimensions and small reflected inputs, serial and threaded; model serving
versus full extracted inputs; existing plan/serving tests. Benchmark matched
full versus subset extraction and diagnostic Rust model serving on deterministic
synthetic pixels at 1024 and 2048, one and eight threads. Reuse existing benchmark
owner, interleave arms, record revision/binary/affinity and warm samples.
These are performance and arithmetic checks, not model-quality qualification.

Additional user request: mathematically compare 2× and 3× sampling of codec
block artifacts, including sampling phase, attenuation/aliasing, isolated
corruptions, and compute. This does not authorize silently replacing the current
pyramid with a 3× pyramid or reusing weights with different scale semantics.

## Mathematical model of one-third dimensions

Here “1/3” means width/3 and height/3, hence **one ninth of the pixels**.
Model the error in an already-converted plane as `e[x,y]`. For a separable
normalized filter `h`, stride `d`, and integer phase `(φx,φy)`, the observed error is

```text
z[m,n] = Σ_i Σ_j h[i] h[j] e[d m + φx + i, d n + φy + j].
```

This is an independent linear sampling model, not a prediction of the nonlinear
zensim score. It separates lattice alignment from filtering and pooling. The
current Rust pyramid uses 2× box reduction (`feature_v2_stream` →
`blur::downscale_2x_into`). The July 26 feature-gap review in `zenpapers`
§2.8 already identifies aliasing in that pyramid; its older global-feature
inventory is superseded by later implementations.

### Which codec grids matter

DCT JPEG uses 8×8 component blocks; subsampled chroma has a different footprint
in delivered-image coordinates. [T.81 §4.3](https://www.w3.org/Graphics/JPEG/itu-t81.pdf).
Lossy WebP/VP8 has 4×4 transform blocks within its larger coding structure.
[RFC 6386](https://www.rfc-editor.org/rfc/rfc6386).
AV1 has variable transform sizes and rectangular shapes, so an AVIF image is
not one uniform grid. [AV1 tool description, transform table](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf).
JPEG XL VarDCT also has multiple transform sizes/shapes; its Modular mode is
outside a DCT-grid model. [libjxl transform strategies](https://github.com/libjxl/libjxl/blob/main/lib/jxl/ac_strategy.h).

Thus `B = 4,8,16,32,64` is a set of useful **local artifact periods**, not an
assertion that every codec boundary produces an error or that all errors are
block-aligned. Deblocking, resampling, cropping and chroma reconstruction can
move or spread the error. Apply the following separately to each axis.

### Lattice alignment: coprimality helps coverage

For block period `B`, sample starts visit `(φ + dm) mod B`. There are exactly
`B/gcd(B,d)` distinct residues. Boundary offsets inside the sampling bins visit
`(kB − φ) mod d`, with `d/gcd(B,d)` distinct offsets. The joint pattern repeats
after `lcm(B,d)` source pixels.

| B (source pixels) | Residues visited, d=2 | Residues visited, d=3 | 3× joint repeat |
|---:|---:|---:|---:|
| 4 | 2 of 4 | 4 of 4 | 12 pixels |
| 8 | 4 of 8 | 8 of 8 | 24 pixels |
| 16 | 8 of 16 | 16 of 16 | 48 pixels |
| 32 | 16 of 32 | 32 of 32 | 96 pixels |
| 64 | 32 of 64 | 64 of 64 | 192 pixels |

Example, `B=8,d=3`: starts visit `0,3,6,1,4,7,2,5`; boundary offsets cycle
`0,2,1`. The boundary's output coordinate is `8k/3`: **fractional and alternating
in phase**. Flooring these positions gives gaps `2,3,3,…`. An existing detector
that only checks an integer 4-pixel output grid cannot be reused unchanged.
All block phases being visited does not mean all source samples survive: a
single periodic pattern is sampled across many blocks, not fully recovered
inside each individual block.

### Boundary mixing: the cost of that coverage

Take non-overlapping `d×d` boxes and an otherwise constant, piecewise-block
signal. At `d=2` with origin-aligned even block boundaries, boxes never cross a
boundary. A one-pixel change of origin makes them cross every boundary in that
axis. At `d=3` with power-of-two `B`, two of every three boundaries are inside a
box, and one is between boxes.

For `B≥d`, over a complete repeating pattern, the fraction of output boxes
crossing a vertical boundary is `q = 2/B` for `d=3`. For a complete square grid,
the fraction crossing either axis is `1−(1−q)^2`:

| B | Boxes crossing a vertical boundary | Boxes crossing either axis |
|---:|---:|---:|
| 4 | 50% | 75% |
| 8 | 25% | 43.75% |
| 16 | 12.5% | 23.4375% |
| 32 | 6.25% | 12.1094% |

These fractions count mixed boxes, not lost signal. A simple edge gives a more
useful response measure. An isolated step of height `Δ` either survives as one
output jump or is split into `uΔ` and `(1−u)Δ`. The sum of squared first
differences per edge, divided by `Δ²`, becomes

```text
R(u) = u² + (1−u)².
2× box: aligned 1; shifted one pixel 1/2.
3× box: offsets 0,1,2 give 1,5/9,5/9; complete-cycle mean 19/27 ≈ 0.704.
```

So 3× averages away some fixed alignment dependence, but reduces this particular
edge-energy statistic about 30% relative to aligned 2×. This edge calculation
assumes separated steps whose responses do not overlap (e.g. B≥8 here).
Unnormalized absolute first-difference sum still preserves `|Δ|` for each
monotone isolated step. Max, L2 and higher-order pools react differently;
pixel-count normalization also changes. No single weight correction restores
all of them. One isolated boundary still has phase variation; only the repeated
grid visits the full cycle.

### Frequency response and aliasing

For a d-tap box, at source frequency `f` cycles/pixel,

```text
|H_d(f)| = |sin(π d f) / (d sin(π f))|,  H_d(0)=1.
2D separable amplitude = |H_d(fx) H_d(fy)|.
Output frequency = d f modulo 1, folded into [0,1/2].
```

Filter amplitude at the fundamental `f=1/B`:

| Artifact period B | 2× box | 3× box |
|---:|---:|---:|
| 4 | 0.707107 | 0.333333 (above the 3× Nyquist limit) |
| 8 | 0.923880 | 0.804738 |
| 16 | 0.980785 | 0.949253 |
| 32 | 0.995185 | 0.987190 |
| 64 | 0.998795 | 0.996790 |

These are filter gains, not phase-independent RMS guarantees after decimation.
For a product sinusoid varying in both axes, multiply the gains; square again
for power. At B=8, the 3×/2× **power ratio** is about 0.759 for a one-axis
sinusoid and 0.576 for a product sinusoid. Sharp block edges also have harmonics,
which suffer more attenuation/aliasing than this fundamental alone suggests.

3× has a lower Nyquist limit: `1/6` rather than `1/4` cycles/source-pixel.
A period-4 sinusoid aliases to an output period of 4 samples (12 source pixels)
under 3×. A proper antialias filter suppresses it instead. Misalignment cannot
preserve genuine high-frequency information beyond the new sampling limit.
The box is especially weak as an antialias filter: its first zero is `1/d`,
above its Nyquist limit `1/(2d)`.

At checkerboard frequency `(1/2,1/2)`, a 2× box cancels exactly; a 3× box retains
amplitude `1/9` (power `1/81`) as an alias. That is useful residual evidence,
but it is not faithful reconstruction. The 3× box instead has exact zeros at
source frequencies `1/3` and `2/3` in either axis.

For a non-symmetric kernel substitute its actual
`H(f)=Σ_j h[j] exp(−2πifj)`. For example, normalized `[1,2,4]/7` has
`|H(1/2)|=3/7` and `|H(1/3)|=1/√7`, so it avoids those particular box zeros
while passing more alias energy. Its centroid is `10/7` samples from the first
tap, so spatial back-projection must carry that offset. Reversing the taps has
the same frequency magnitude and opposite phase behavior. It does not remove
the rank loss: one linear output per nine input samples leaves a nullspace of
dimension at least eight per interior 3×3 cell. Kernel asymmetry moves blind
patterns; it cannot eliminate all of them.

### Salt-and-pepper and the full-resolution escape hatch

One isolated signed impulse of amplitude `a` becomes `a/4` under 2× box or
`a/9` under 3× box. Output mean squared error relative to input MSE is `1/4`
or `1/9`, respectively. Independent zero-mean pixel noise has the same variance
reduction. Opposite-sign impulses inside one box can cancel completely.
Full-resolution Y avoids these losses for its luma-like error signal; it does
not protect pure X/B error. A low coarse-error threshold is therefore not a
safe condition for deciding whether finest-scale chroma was needed.

A cheap guard should accumulate full-resolution `Σ(error²)` and/or maximum
absolute error before reduction, per spatial region and including X/B. For
nonnegative normalized weights, `mean(error²) ≥ mean(error)²`; the difference
is exactly the within-filter variance that reduction hides. Such a guard is
a separate possible follow-up, not included in this channel-skip prototype.

### Compute and spatial interpretation

Count channel-pixels relative to one original-size plane, ignoring edges:

| Feature plan (spacings in source pixels) | Work units |
|---|---:|
| Full XYB at 1,2,4,8 | 3.984375 |
| Y at 1; XYB at 2,4,8 (implemented subset) | 1.984375 |
| Y at 1; XYB at 3,6,12 (one 3× step, then 2×) | 1.437500 |
| Y at 1; XYB at 3,9,27 (three 3× steps) | 1.374486 |

The 3,6,12 alternative removes another **27.56% of these work units** relative
to the 2,4,8 subset. This is not an end-to-end speed prediction: full-resolution
conversion and reduction still inspect the source; filter cost, halo, SIMD
tails, allocations and scheduling remain. A larger filter can consume the
saved budget. The scales and physical support change, so this requires newly
identified features and refitting, unlike simply omitting unneeded 1× X/B.

For JXL spatial steering, keep density in original-image coordinates and
integrate it over actual encoder regions. At 3× an 8-pixel block covers 8/3
coarse cells, so use overlap areas rather than integer reassignment. Correct
area/centroid accounting can preserve a map integral; it cannot undo filtering
or recover a missed localized error. Some current blockiness kernels have an
integer lattice: they would need an explicit redesign, not a new scale number.

The useful next experiment is **Y at 1 plus XYB at 3,6,12**, paired against
2,4,8 with a fixed filter, plus a full-resolution chroma guard. Before any
model-quality claim, sweep all nine translations of the same pair (both images
translated together), horizontal/vertical grids, impulses, phase reversals,
codec outputs, and HDR. Report worst-phase target-score drift and spatial
misallocation as well as mean accuracy and speed. Coprimality is a reason to
test 3×; it is not enough to select it for the one-number product dial.


## September 13 amendment: bounded training screen (before fitting)

The user requests learning/training/testing in under five minutes. Add a
`feature-screen` stage to the existing `run_full_eval.sh` owner, delegating
only orchestration to `scripts/lib/feature_screen.py`. Rust still owns
extraction (`extract_features_372col`), fitting/baking (`zensim_mlp_train`),
complete final pixel/cache scoring (`BakeScorer` through the extractor audit),
and statistics (`panel`). No new model, scorer, resizer or statistic owner.

The committed recipe `feature_screen_2026-09-13.json` pins the September 8
264-pair canonical JXL packet and SSIMULACRA2 labels. Every source is T2 train.
Eight existing fit origins remain fit; split the former four calibration
origins into dev 1214/6064 and inner-test 8462/9066. These images were already
examined in earlier experiments; this is a development screen, not a new
unseen or protected holdout. Source/family roles and bytes must validate.
Never use the packet's `human_score` field (a row key) as the training target.
Fit explicitly to `ssim2`, scale 1, as a metric proxy.

Register `v1screen_rev3` as fresh canonical 372-column Rev3 extraction,
with exact extractor/decoder binary hashes and source diff per run. Fit two
H32 f32 MLPs, seed 4004, 120 epochs, 4096 stratified within-reference pairs per
epoch, RankNet plus MSE; selected IDs are full228 and y190. The dev group
selects checkpoints; inner-test is not loaded by the trainer. This is an
instrument recipe, not a tuned competitive model claim. No per-test spline
fit or calibration is served. Rust panel's logistic rescale is descriptive
and cannot establish absolute target error.

The 300-second deadline includes input admission, fresh extraction or verified
cache, both fits, final pixel/cache audits, and per-role/per-reference panels.
One-time compilation, existing codec generation and existing judge labels
are outside the iteration budget. Failures/timeouts remain incomplete.
Cache identity includes producer, arithmetic revision, decoder/extractor
binary, pixel/label manifest hashes and roles; table bytes are rechecked.
Filter/scale variants are not servable feature identities yet and must not
masquerade as these unchanged-scale channel subsets. This packet does not
measure corruption, HDR, spatial interventions, target steering or full-size
latency. Those remain follow-up packet expansions and qualification gates.


## Measured channel specialization

The implemented planner specializes the existing folded walk once with
`const FULL_RES_XB: bool`. For eligible basic/peak requests, absence of every
full-resolution X/B ID skips gather and H/V moment work for those channels.
Conversion and coarse pyramids remain unchanged. A requesting ensemble member
restores the channels. Masked/IW/wider requests conservatively use the full
path. A subset has explicit populated IDs, not a misleading family shorthand.

Full228 retains all basic/peak IDs; y190 removes 38 finest X/B IDs. The fixture
bakes use nonzero checksum weights and are not trained quality models. Costs
include BakeScorer planning, allocation, extraction and forward inference.
The rawfold comparison separately reuses scratch and is not the serving baseline.

| Geometry | Threads | Full228 ms | Y190 ms | Paired time change |
|---|---:|---:|---:|---:|
| 1024² | 1 | 27.893 | 15.146 | −45.68% |
| 2048² | 1 | 89.287 | 53.530 | −40.06% |
| 1024² | 8 | 5.414 | 3.907 | −28.06% |
| 2048² | 8 | 18.295 | 13.482 | −26.17% |

AMD Ryzen 9 9950X3D, CPU0 or physical cores0–7 on its 96MiB L3 CCD,
Rev3, release dynamic SIMD dispatch, no target-cpu=native, 30 paired rounds.
One call per sample; 1–2 outliers per group. Paired delta 95% intervals (ms):
ST1MP [−12.7483,−12.7288], ST4MP [−35.8116,−35.7067],
MT1MP [−1.5725,−1.4690], MT4MP [−4.8613,−4.7016].
These are synthetic-image engineering measurements, not full corpus p95.

The dedicated opt-in benchmark groups disable zenbench0.1.9's background
process-name scan because it mistakes its own exclusive-lock heartbeat for a
competing benchmark and stalls 30 seconds per round. They retain the exclusive
process lock and paired statistics; run only on an otherwise quiet host.
Stalled/preliminary attempts are preserved but excluded. The final files are
`st-measured.json` and `mt8-measured.json`, not earlier similarly named files.

Correctness checks pass retained-feature bit parity on impulses, checkerboards,
colored grids, identity, small/reflected and odd images; serial/threaded SDR,
HDR and auto-HDR routing; planner normalization/ensemble unions; full-feature,
research, baked pixel, cached and spatial surfaces. The 190-input integration
fixture reports 30 max terms absent from additive density, while finite-max
refinement coverage is complete. This distinction remains visible.

## Actual zenresize kernels, including 1.5×

User-requested main was fetched and pinned at
[`e3975fb9d6d6b7baa96038a0eb8e27febb37c012`](https://github.com/imazen/zenresize/commit/e3975fb9d6d6b7baa96038a0eb8e27febb37c012).
The existing benchmark now reads actual `F32WeightTable` coefficients and runs
actual float resizers. It tests Box, Triangle, Mitchell, RobidouxFast,
RobidouxSharp, Lanczos2 and Lanczos3 at 1.5×, 2× and 3× reduction.
Signed f32 planes preserve negative filter lobes; the clamped integer plane
resizer is not an appropriate stand-in for signed XYB.

| Filter | 1.5× ms | 2× ms | 3× ms |
|---|---:|---:|---:|
| Box | 2.483 | 1.695 | 1.039 |
| Triangle | 3.758 | 2.650 | 1.468 |
| RobidouxFast | 3.759 | 2.651 | 1.919 |
| RobidouxSharp | 6.288 | 4.630 | 2.984 |
| Mitchell | 6.291 | 4.630 | 2.984 |
| Lanczos2 | 6.291 | 4.628 | 2.985 |
| Lanczos3 | 9.130 | 7.075 | 4.673 |

One 1152² gray f32 plane, CPU0, reusable resizer/output, 30 interleaved rounds.
These are **resampling alone**: no color conversion, feature kernels, image
pair or model. The filter benchmark is not integrated into a serving pyramid.
Cost of a two-image multichannel resampling plan cannot be inferred from one
plane's cost without measuring the combined work and buffers.

For the requested three kernels at 3×, amplitude at period8 is
Triangle .64760, RobidouxSharp .66506, Mitchell .63736. Actual diagonal
checkerboard RMS is respectively .012346, .0001038, .0000922. Thus the sharper
cubic retains slightly more period8 signal while both cubics suppress this
particular Nyquist alias much more than Triangle. Triangle is about twice as
fast here. Neither stronger suppression nor sharper passband alone proves
better corruption detection: a scoring model can lose the very error it needs
to detect. Test salt/pepper, colored impulses and grids through fitted models.

All actual interior coefficient sums are within 1e-6 of unity. Polyphase
predictions match the actual 2D checkerboard MSE within 1e-6. Impulse tests
cover every source phase, preserve signed min/max and record MSE ratios.
Complete coefficients, gains and measurements are in the linked JSON record.

### Rational 1.5× sampling

For reduction `d=3/2`, each output axis retains 2/3 samples; area is 4/9.
Pixel-center sampling has centers `.25,1.75,3.25,…`, with two output phases
per three source pixels. For power-of-two block B, these centers visit 2B
half-pixel residues modulo B and repeat after 3B source pixels / 2B outputs.
A boundary's output coordinate `2kB/3` cycles through three fractional phases.
The source-frequency Nyquist limit is 1/3, versus 1/4 at 2× and 1/6 at 3×.
It retains a wider band, with a larger feature workload.

Actual Box at 1.5× alternates taps `[1]` and `[.5,.5]`: it is **not** area
averaging. Its impulse peak varies .25–1 by 2D phase and checkerboard RMS is
.5. At 1.5× the system is polyphase, so a single row's transfer function does
not describe the entire image. For a separable diagonal checkerboard, if
`a_p = |H_p(pi)|²`, aggregate output MSE is `(mean_p a_p)²`. This explains the
Box result despite one phase transmitting the checkerboard unchanged.
Triangle and the requested cubics materially reduce this phase problem.

Explicit sample-work alternatives (one full plane =1):

| Channel/scales plan | Work units | Reduction versus full XYB |
|---|---:|---:|
| Full XYB at1,2,4,8 | 3.984375 | baseline |
| Y1 plus XYB2,4,8 (implemented) | 1.984375 | 50.20% |
| All XYB at1.5,3,6,12 | 1.770833 | 55.56% |
| Y at1,2,4,8; XB at1.5,3,6,12 | 2.508681 | 37.04% |
| Only finest XB replaced by1.5; other scales unchanged | 2.873264 | 27.89% |
| Y1 plus XYB1.5,3,6 | 2.750000 | 30.98% |

These are plane-sample counts, not predicted milliseconds. Separate channel
pyramids add buffer/scheduling work. Keeping all existing coarse values in the
finest-only variant requires an independent branch from the original plane.
Changing filters/scales needs an explicit pyramid/feature identity and a model
refit, unlike the implemented channel omission, which preserves retained values.

My next filter candidates are Triangle for cost and RobidouxSharp/Mitchell as
matched-cost alternatives. First wire one explicit pyramid variant through
Rust extraction, BakeScorer and spatial coordinate mapping; then compare
freshly trained variants in the bounded screen. Do not choose on resizer
frequency responses alone.

## Reproduction and chronology

[Structured benchmark data](data/fullres_y_subset_2026-09-13.json) pins CPU,
revision, binaries, source hashes, full coefficients, phases and confidence
intervals. Private evidence is under `~/work/zensim-validation-2026-09-12/fullres-y-subset/`.
The subset binary/source copy predates only the RobidouxSharp benchmark-arm
addition; the final kernel binary has its own separate source and binary hash.
`filter-cost-sharp.json` / `filter-response-sharp.json` are final kernel results.

Build with `cargo build --release -p zensim --bench extract_paths_bench --features custom-profiles,feature-regime-v2,training`;
run the resulting release bench executable with `ZEN_XP_SUBSET=1`,
`ZENSIM_FORMULA_REV=3`, `ZEN_XP_SIZES=1024,2048`, `ZEN_XP_ROUNDS=40`,
`ZEN_XP_WALL_S=120`, `ZENBENCH_RESULT_PATH=<fresh.json>`, and the registered
thread/affinity settings. For kernels set `ZEN_XP_FILTERS=1` instead and
`ZEN_XP_FILTER_RESPONSE=<fresh-response.json>`. The harness converges at30 rounds.
Use the workspace run-heavy wrapper and a quiet host; never reuse output paths.

Scoped formatting, focused Rev1/Rev3 parity and serving tests, plan/research
suites, CI-exact Clippy, portable and wasm32-wasip1 checks passed. The WASI check
retains pre-existing unused SIMD-import warnings in fused.rs. No public API
was added; no named model changed. September 12 preregistration comes first;
September 13 kernel additions, measurements and fast-screen amendment follow.
Older prose saying no refit applies to the channel prototype at registration;
the bounded screen is the later explicitly requested refit experiment.


## Final bounded learning results

The final cold run, including fresh extraction and Parquet conversion, takes
**8.75 seconds**; verified-cache run **8.47 seconds**. Both include two fits
and 528 complete final BakeScorer pixel/cache audits. These are wall-clock
iteration costs, not a quiet-machine inference benchmark. One-time release
builds took about12–28 seconds for the trainer changes and23 seconds for the
already-warm nested extractor dependency build; clean builds may take longer.

| Inner-test metric (44 pairs / two training origins) | Full228 H32 | Y190 H32 |
|---|---:|---:|
| Signed Spearman | .963775 | .985905 |
| Raw Pearson | .978391 | .989833 |
| Logistic-rescaled PLCC | .983888 | .990815 |
| Logistic-rescaled Z-RMSE | .178785 | .135227 |

The reduced model can learn and serve; this tiny one-seed proxy test does not
establish superiority. Each arm preserves all twelve identities exactly100,
with no distorted score above100 and observed minima around−24.8/−24.9.
Per-reference/full six-stat panels and signed/raw correlations are retained.
The panel's logistic fit is never applied to the served model and its errors
are not a target-score accuracy bar. No new score calibration is claimed.

A poisoned cache, overlapping fit/test origin and exhausted deadline all fail
with `FAILED_OR_INCOMPLETE`. Existing full-eval stage recovery/reuse tests pass.
Registry tests validate the new era's slots/hash. Final trainer logs confirm
`--no-auto-eval` and Rev3 admission. Preliminary CSV input was refused because
that loader has no reference IDs; default Snappy Parquet was refused because
its codec was not compiled. The final recipe uses uncompressed f64 Parquet.
The first successful preliminary fit also exposed automatic full-verdict launch;
that evaluator refused the revision before scoring. The final two runs explicitly
suppress it and do not enter protected evaluation defaults.

[Structured run, hashes, stages, panels and controls](data/feature_screen_2026-09-13.json).
[Runnable recipe and setup](../docs/FULL_EVAL.md#five-minute-feature-development-screen-september-13-2026).
The remaining margin inside five minutes should buy a more discriminating
fixed training-development packet: corruption impulses/swaps/grids at several
sizes and phases, then spatial pixel interventions and reachable codec target
ladders. Preserve the cheap one-seed screen; use multiple seeds and full
qualification only for promising candidates. Filter/scale variants must first
serve completely in Rust with explicit semantics, and then get fresh features.
