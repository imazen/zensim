# Basic feature and head decomposition — September 14, 2026

Registered before fitting. The preceding [scale frontier](product_scales_2026-09-14.md)
finds basic228 substantially better than local120 on the same TRAIN scalar
development rows, but slower prepared maps and only 2/5 spatial passes. Separate
HF signals, L8 pools, hard-max pools and head capacity before adding more features.
The question is consequential: max queries have real retained-map cost, while
the fused scalar kernel may still compute pools that a model does not consume.
Feature counts alone cannot establish savings.

Reuse the exact admitted product-TRAIN tables and hashes, family partitions,
fresh native Rev3 SSIMULACRA2 codec targets and human rank supervision. No new
extraction era or filter. EVAL/TEST are never read; calibration remains unused.
The previous TRAIN controls are disclosed reused development evidence. This is
not a new independent validation instrument or the feature set's ultimate ceiling.

Exactly four new profiles, each with paired seeds 7101/7103/7107:

| Profile | Inputs | Head |
|---|---|---:|
| basic156 | All basic features, including 36 HF ratios | 128 |
| basic192l8 | basic156 plus 36 L8 pools | 128 |
| basic192max | basic156 plus 36 hard-max pools | 128 |
| basic228 | All basic and peak pools | 32 |

The recipe records every canonical ID. Compare basic156 against the preceding
local120/H128 to estimate the HF-family contribution; compare both 192-input
arms against basic156 to separate the peak families; compare basic228/H32 with
the preceding basic228/H128 for capacity. Retain y60/H32 as the spatial control
and full944/H128 as the scalar reference. Do not choose a best seed.

Keep the established corrected uniform mixture weights, sample seeds +10000,
32 epochs, 8192 draws/epoch, plain heads, MSE weight 1, f32 export, no early
stopping and no auto-eval. Verify actual sampling digests against the existing
Rust replay. Densify through the existing Rust owner and verify raw/packed
public-API scores exactly. Every reported ensemble uniformly composes all three
seeds through BakeScorer. Twelve fits, at most two concurrent one-thread workers;
no adaptive fit extension. This is a bounded screen, targeting about five minutes
for fitting/scalar assessment; native interventions and timings are separate.

Assess every seed and ensemble with canonical Rust full panels and scatter:
human, codec, distorted-only, high quality, per-codec/source/class/reference.
Preserve signed rank, Mohammadi panels, envelopes, clumping, saturation, raw
tails and undefined groups. MAE is supplementary. Report paired seed deltas;
seed spread is not source-level uncertainty. Only 57 high-quality distorted
pairs and eight human development sources limit conclusions. No incomplete
research composite becomes a shipping verdict.

Audit all four complete ensembles on the same five metadata-selected TRAIN JXL
pairs, original bitstreams, and native decoded pixels. Verify canonical consumed
features and full pixel/cached scalar parity. Run the same block-32 finite repair
instrument and unchanged M2 >= .99 / M3f >= .70 / all finite / complete-support
requirements. Inspect saved component evidence if a family worsens repair
prediction; do not weaken gates or fit on any EVAL failure.

Measure all new complete ensembles and local120, y60 and basic228/H128 controls
interleaved through the existing benchmark at 1024²/2048², one pinned worker,
at least 30 accepted rounds, scalar and official prepared-map calls. Record
dispersion and contention. Existing summary-only timing cannot prove p95.
Advance to broader TRAIN coverage only if a variant passes all five spatial
cases and trades scalar assessment against measured cost favorably. Do not
consume EVAL on a variant failing these TRAIN gates. Otherwise record the
failed hypothesis and the evidence determining the next correction.

Stop after twelve fits, four ensembles, full scalar panels, twenty native
audits, twenty spatial cases and the registered timing comparison (or explicit
measurement failures). No new optimizer, kernel or controller. Full frozen
qualification, corruption composition, representative larger spatial cases,
native bounded targeting/RD, HDR and p95 remain separate requirements.

Artifacts: `~/work/zensim-validation-2026-09-14/product-peaks/`.

## Results

All twelve fits and four uniform ensembles completed. Actual sampling digests
match replay. Rust exported raw/packed predictions are bit-identical on 31,548
rows; all twenty native pixel/cache audits pass. Full Rust panels and scatter
cover every seed and ensemble on the admitted TRAIN development populations.
Fitting, packing and score export took 214 seconds; complete scalar panels
took another five seconds. Spatial interventions and timing were separate.

| Ensemble | Human signed SROCC | Distorted codec signed SROCC | Codec seed range | High-quality SROCC | Spatial pass / 5 |
|---|---:|---:|---|---:|---:|
| basic156_h128 | 0.8838 | 0.8998 | 0.8934–0.8968 | 0.6145 | 2 |
| basic192l8_h128 | 0.8772 | 0.9156 | 0.9076–0.9151 | 0.5613 | 4 |
| basic192max_h128 | 0.9064 | 0.9489 | 0.9400–0.9498 | 0.6477 | 1 |
| basic228_h32 | 0.8851 | 0.9549 | 0.9501–0.9533 | 0.6541 | 4 |

Descriptive paired source-bootstrap intervals below use 1,000 identical cluster
draws per comparison and the Rust signed-rank owner. All 23 codec development
sources are distinct admitted families. Intervals are not adjusted for multiple
comparisons and do not qualify superiority on these repeatedly used TRAIN rows.

| Candidate minus control | Human delta [95% interval] | Distorted codec delta [95% interval] |
|---|---|---|
| basic156_h128 − local120_h128 | +0.0184 [+0.0065, +0.0305] | +0.0016 [-0.0079, +0.0109] |
| basic192l8_h128 − basic156_h128 | -0.0066 [-0.0145, +0.0023] | +0.0157 [+0.0082, +0.0227] |
| basic192max_h128 − basic156_h128 | +0.0226 [+0.0156, +0.0309] | +0.0490 [+0.0349, +0.0630] |
| basic228_h32 − basic228_h128 | -0.0197 [-0.0299, -0.0109] | +0.0016 [-0.0021, +0.0053] |

## Distribution and measured cost

| Ensemble | Mohammadi OR | Geometry out4 | Raw clump | Raw OLS residual p99 | Distorted >100 |
|---|---:|---:|---:|---:|---:|
| basic156_h128 | 0.0145 | 0.0797 | 0.1500 | 44.34 | 2 |
| basic192l8_h128 | 0.0080 | 0.0717 | 0.1442 | 44.60 | 0 |
| basic192max_h128 | 0.0007 | 0.0645 | 0.1768 | 34.30 | 0 |
| basic228_h32 | 0.0000 | 0.0688 | 0.1928 | 34.28 | 4 |

These are distinct diagnostics: OR is not the shape-normalized out4 fraction,
and the OLS residual percentile is not absolute target error. Complete raw and
normalized geometry, coverage, saturation, per-codec/source/class panels and
every seed remain in the hashed results. No single table replaces the full
assessment or supplies an incomplete composite with a qualification label.

| Ensemble | Scalar median 1024² / 2048² ms | Prepared median 1024² / 2048² ms |
|---|---:|---:|
| basic156_h128 | 16.39 / 78.22 | 59.69 / 237.26 |
| basic192l8_h128 | 16.40 / 78.28 | 61.96 / 248.10 |
| basic192max_h128 | 16.45 / 78.65 | 83.15 / 328.95 |
| basic228_h32 | 16.37 / 77.98 | 84.13 / 339.33 |
| y60_h32 | 7.97 / 37.16 | 26.52 / 107.59 |
| local120_h128 | 15.41 / 74.77 | 55.28 / 221.32 |
| basic228_h128 | 16.43 / 78.52 | 85.50 / 339.86 |

The existing benchmark interleaves complete ensembles and controls with one
pinned worker, fixed textured RGB8 inputs and at least 30 accepted rounds.
Results retain dispersion and owner reliability flags. These are medians, not
p95; absolute or relative p95 shipping gates remain unmeasured. No new memory
claim is made. Full numerical output is in the [summary](product_peaks_2026-09-14.results.json)
and the immutable artifact packet.

Removing max pools from basic228 lowers the prepared median from 85.50 to
61.96 ms at 1024² and from 339.86 to 248.10 ms at 2048². Scalar medians stay
essentially unchanged: the fused scalar walk still computes peak pools. Reducing
the head to H32 likewise does not materially reduce complete inference cost.
The max-map cost is real, but the measured scalar accuracy loss rules out
assuming those features are expendable. No p95 gate inherits a pass from these
median comparisons.

## Interpretation and next correction

Adding the 36 HF ratios to local120 improves human development rank by .0184
(descriptive paired interval [.0065, .0305]), while the codec change .0016
has an interval crossing zero. Adding hard maxima to basic156 improves codec
rank by .0490 [.0349, .0630], substantially more than the L8-only addition
.0157 [.0082, .0227]. Max features carry useful scalar information on this packet;
removing them to obtain a cheap map loses that information. L8-only addition
has no clear human-rank gain here. These are matched-recipe effects, not causal
proof about a family's optimal attainable quality under every training recipe.

Reducing basic228 from H128 to H32 preserves measured codec rank (delta .0016,
interval [-.0021, .0053]) but loses human rank (-.0197 [-.0299, -.0109]). The
smaller head passes all five M2 checks and four M3f checks. Its one M3f failure
is the web-screenshot TRAIN case, row 5316: M2 1.0000, M3f .6739. It is still
FAIL, not rounded into a pass. The basic156, L8-only and max-only H128 models
pass 2/5, 4/5 and 1/5 complete spatial cases respectively. Every block is finite;
there are 588 actual pixel repair interventions across the four ensembles.
Basic228/H32 also gives four distorted TRAIN pairs scores above 100; basic156
has two. These identity-ordering failures are separate from spatial coherence
and remain blockers even if the map approximation is corrected.

Saved-data oracle substitutions distinguish head linearization from map
approximation without fitting or repeating image comparisons. For basic228/H32's
failed case, replacing the predicted max contribution with the observed feature
change makes M3f .6478, whereas replacing all non-max contributions gives .9809.
Replacing only edge or L8 contributions gives .7470 or .7496; replacing SSIM
alone gives .6922. These diagnostics are not runtime inputs or deployable scores.
They implicate the non-max approximation in this case and do not prove that a
particular arithmetic correction will work across images or real codecs.

The existing attribution owner already documents the relevant mathematical
limit and tests it in `l8_finite_moment_removal_and_near_zero_coefficients`.
For a positive p-th moment with pooled feature F=(S/n)^(1/p), removing fraction
r of its frozen moment mass drops the feature by F[1-(1-r)^(1/p)]. Its local
linearization predicts only Fr/p. For 0<=r<=1, the exact drop lies between that
linearization and p times it. L8 can therefore underpredict a finite removal by
up to eightfold even with a mathematically correct derivative. Signed model
contributions can cancel, so this single-feature bound is not a score-error
bound. Actual pixel replacement also changes blurred neighborhoods and coarse
footprints; a frozen-signal correction alone is not an exact pixel repair.

The next justified mechanism experiment is to separate finite L2/L4/L8 pool
curvature from changed-neighborhood effects on all five TRAIN cases, using the
existing diagnostic and attribution owners. Only advance a correction if it
preserves scalar/feature semantics and improves the complete served map under
the unchanged gates, with measured preparation/query cost. This is a named
approximation question, not permission to invent another scoring path or
change the feature era silently. A higher-capacity spatially stable head may
still be needed to recover the human-rank loss.

The descriptive bootstrap initially refused the prior local120 control's older
six-place score export because targets were not bit-exact. The corrected run
uses that campaign's already verified round-trip `assessment-exact` export;
models, rows, draws and exact assertions are unchanged. The failed log and
correction receipt are retained. No model is qualified, no EVAL/test data were
opened, and no calibration or gates changed. The full production goal remains
active, including native targeting/RD, corruption, HDR and tail latency.
