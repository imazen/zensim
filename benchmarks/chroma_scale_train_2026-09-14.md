# Quarter-resolution chroma improves some ranks but fails the quality screen

September 14, 2026. Adding quarter-resolution B to the fast y60/H32 feature set
improves the fresh ensemble's human rank from .8315 to .8358 and codec rank
from .9244 to .9327. All three ensembles retain zero robust native preference
conflicts. However, added B produces one more distorted-codec outlier, worsens
human raw clumping and maximum residual, and lowers native map association with
Butteraugli. Adding both X and B also fails human-rank noninferiority against
the frozen original control. **Neither added-chroma arm advances.**

[Served comparison, raw scatter and A/Bs](/zensim/reports/chroma-scale-train-2026-09-14/index.html).
[Full results](chroma_scale_train_2026-09-14.results.json).
This follows the [four-contract filter diagnostic](native_filter_2026-09-14.md).
It is a new TRAIN feature-availability comparison, not a new feature formula,
EVAL qualification, or claim that a feature set has reached its capacity.

## Registered experiment

Three nested native Rev3 box-pyramid read sets share a plain H32 head:

| Profile | Feature availability | Inputs |
|---|---|---:|
| y60 | Y at full/half/quarter/eighth resolution; X/B at eighth | 60 |
| y70 | y60 plus quarter-resolution B | 70 |
| y80 | y70 plus quarter-resolution X | 80 |

The added IDs are104..113 for B and78..87 for X. Canonical basic slots use
`scale*39 + channel*13 + local`. Each ten-signal block contains SSIM and
artifact/detail mean/L4/L2 plus MSE; variance/texture/contrast ratios and peaks
remain excluded. All model bakes explicitly retain their declared IDs.
Changing input width also changes initialization trajectories; this is not
an isolation of feature information from optimizer behavior.

Use the unchanged admitted TRAIN packet:7,000 human and7,947 codec fitting
rows;1,000 human,1,629 codec and264 native development rows. Human development
uses eight source-separated KADID TRAIN references; TID remains fit-only. Codec
source-family partitions, historical reserved-family and suffix8 exclusions
remain unchanged. No native-local pairs supply fitting or checkpoint selection.
No calibration, new sources, extraction, encodes, EVAL or TEST occurs.

Nine scientific fits use initialization seeds7101/7103/7107 and the previously
verified disjoint sampling windows, with a new preflight. Uniform within-reference
sampling supplies32 epochs of8,192 attempts, human rank-only and codec rank plus
signed SSIM2 absolute supervision. Both fitting groups have explicit checkpoint
weight1 and mean selection; development rows never select checkpoints. The native
fitting group is omitted. No TV term, transform, nonnegative constraint or loss
sweep is added. SSIM2 is a teacher, not independent human quality evidence.

All fits use the canonical Rust trainer, f32 ZNPR output, Rust densification,
and complete public ensemble scoring. A fixed first-seed y60 repeat reproduces
all2,893 development scores exactly. Every scientific sampler digest matches
its replay, and raw/packed scores match on all three populations. No best seed
is substituted for a complete equal-weight three-member ensemble.

## Measured scalar and native behavior

| Ensemble | Human SROCC | Codec SROCC | Human out4 | Distorted-codec out4 | Native conflicts | Own-map rank median | Butteraugli map rank median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fresh y60 | .83150 | .92442 | 2/1000 | 76/1380 | 0/79 | .5941 | .5618 |
| y70 | .83582 | .93274 | 0/1000 | 77/1380 | 0/79 | .7912 | .4926 |
| y80 | .83223 | .93262 | 5/1000 | 80/1380 | 0/79 | .7529 | .4809 |
| Frozen original y60 | .83894 | .92555 | 3/1000 | 76/1380 | 0/79 | .5574 | .5750 |

`out4` is the canonical normalized scatter-envelope statistic, not a raw
score-error threshold. Own-score map association and independent peer responses
must remain separate. The new y70/y80 maps pass the .70 own-map diagnostic line
on6/8 cells versus3/8 for fresh y60, while Butteraugli association worsens.
All ensemble native M2 checks pass. These are native intervention diagnostics,
not matched rate–distortion allocation or target-loop qualification.

Human raw clumping rises from .115 to .147 with y70; maximum fitted raw residual
rises from131.21 to147.94. Codec raw p99 residual improves from50.66 to48.61,
and maximum from70.24 to64.00. The opposing directions explain why rank or one
normalized outlier statistic alone would be misleading. Human MOS was used for
within-reference ranking, so these raw units do not establish calibrated dial error.
All raw/normalized tails, density, saturation, ranges, per-reference/source/class/
codec and near-lossless panels are retained, including undefined sparse statistics.

Across the three recorded seeds, human rank ranges .82591–.83471 for y60,
.82437–.83832 for y70 and .82223–.82992 for y80. These are seed ranges, not
confidence intervals over independent source populations. Three seeds and four
repeatedly inspected native families cannot establish broad generalization.
The historical original y60 remains an exact matched control; its older nearby
sampling windows are not newly claimed to be independent replicas.

## Integrity and chronology

Assessment covers all nine members, three new complete ensembles and the frozen
original ensemble:17,537 canonical Rust panels, plus complete raw scatter. The
native replay produces104 model/cell measurements and includes D separately as
frozen Rev1 context. All272 decoded pixel/peer rows and the original control's
scores/maps reproduce exactly;3,536 pixel/cache comparisons cover all264 unique
native development rows for every new/control composition. All244,800 consumed
feature values match the native float64 cache exactly, with matching scorer
float32 casts. Packed input IDs/revisions are verified. The later
[layout-sensitive inspection](chroma_scale_train_2026-09-14.input_support.json)
confirms nonzero structural paths for every declared input, with asymmetric
and disconnected-output controls. This does not prove functional necessity
or nonzero sensitivity everywhere. Neutral encoding
identity and full additive/refinement coverage remain checked by the Rust owner.

The initial control check used the earliest six-decimal TSV exports and refused
at a string mismatch. The later existing full-precision assessment is the
correct chronological authority and exactly reproduces the fresh control exports.
The failed wrapper/output are preserved. No model, data, tolerance or gate was
changed to resolve this provenance mismatch.

Fit, repeat, packing and score export took160 seconds; full scalar assessment
6 seconds and native replay/rank assessment8 seconds. These are stage durations,
not a claim about all orchestration, plotting or qualification wall time.

## Runtime and decision

The idealized active channel-pixel work is87/64 full-image equivalents for y60,
91/64 for y70 and95/64 for y80: additions of4.6% and9.2%. This ignores shared
conversion/pyramid work, buffers, feature dependencies and spatial ownership;
it is not a measured speed claim. Complete-ensemble scalar/prepared timing uses
the existing public-API benchmark with explicit30-round, one-call measurements,
five admitted native geometries and synthetic1024²/2048² inputs. Prepared calls
include score/map and one rectangle query; reference setup is outside the timed
call, and worker reuse across many comparisons is not established by this setup.

All four suites completed 1,260 individual-call observations, with12/18/6/4
flagged rounds in native scalar/prepared and synthetic scalar/prepared runs.
None passes quiet acceptance. The following p95 values are descriptive only:

| Ensemble | Synthetic scalar p95, 1MP / 4MP ms | Synthetic prepared p95, 1MP / 4MP ms |
|---|---:|---:|
| y60 | 10.42 / 38.54 | 26.71 / 105.11 |
| y70 | 10.74 / 40.38 | 27.36 / 108.80 |
| y80 | 10.94 / 40.26 | 27.55 / 111.92 |

The observed median paired ratio to y60 spans .965–1.066 for y70 and
.973–1.110 for y80 across all registered cases/modes. These noisy observations
do not establish a speedup or pass the declared1.15x cost gate.
Runtime outcomes are recorded in the companion results and timing evidence.
Flagged rounds remain present and block quiet-performance acceptance. A gallery
render overlapped part of prepared timing and is documented; no quiet-machine
claim or selective removal of noisy rounds is made. Modes execute separately,
so a prepared/scalar ratio is not a within-round paired mode comparison.

Registered quality continuation requires human/codec rank within.005 of both
fresh and frozen controls, no increase in native conflicts or new failing class,
no increase in M2/map failure counts, and nonincreasing human/distorted-codec
out4. y70 fails codec outliers; y80 fails outliers and frozen-control human rank.
The one-row y70 tail difference is small, but the gate remains failed. Raw tails
and peer-map tradeoffs also remain explicit. No post-result relaxation or
immediate feature/head/loss sweep is warranted. No model advances or qualifies.

Complete model artifacts, commands, manifests, all seed panels, native measurements,
raw timing rounds and selected failure A/Bs are served with the report. Broader
human/composite qualification, calibrated target loops, native allocation RD,
corruption composition and controlled latency remain requirements of the full
product goal. This failed recipe does not prove a feature-capacity ceiling.
