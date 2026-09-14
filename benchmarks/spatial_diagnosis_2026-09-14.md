# Spatial feature diagnosis and frozen controls — September 14, 2026

Following the [wider TRAIN screen](spatial_coverage_2026-09-14.md), use the
existing Rust saved-data analysis on all 69 reports. Register six additional
family-diagnostic pixel calls on every image with failing M3f (report row2445,
screenshot row5324), including all three models as matched controls. The
existing family diagnostic runs with finite moments off; its scalar, actual
gains, feature linearization, density and baseline rectangle predictions must
match the previous enabled run's uncorrected baseline exactly. No new scorer,
model fit, threshold, source admission, calibration, EVAL or TEST access.

The saved analysis substitutes *observed* per-family feature responses. These
are unavailable-to-runtime oracles, not improved models or shipping maps.
Four extra basic-family diagnostic maps per call are counted separately.
All six pixel calls and 75 saved analyses complete. All required parity fields
match exactly, and the existing analysis verifies family reconstruction.
Degenerate absent-family correlations reported as zero by the historical owner
are not evidence of poor correlation; their observed contributions are zero.
The small residual in absent L8 controls comes from subtracting separately
accumulated densities, not an active L8 feature.

## Two different limitations

| Model / image | Served finite M3f | Observed-max oracle | Observed-nonmax oracle |
|---|---:|---:|---:|
| basic228/H128 / report | 0.7597 | 0.9208 | 0.7744 |
| basic228/H32 / report | 0.5840 | 0.9844 | 0.5689 |
| y60/H32 / report | 0.9880 | 0.9880 | 1.0000 |
| basic228/H128 / screenshot | 0.5313 | 0.4608 | 0.8066 |
| basic228/H32 / screenshot | 0.5195 | 0.5886 | 0.6674 |
| y60/H32 / screenshot | 0.8653 | 0.8653 | 0.9998 |

For basic228/H32 on the report, max-response error is the main rank limitation:
the oracle rises .5840→.9844. The actual implementation freezes the base signal
and removes maxima by source ownership. Pixel repairs also change blurred
neighborhoods and may create new maxima. This known approximation is not an
exact finite-edit model, even when every declared feature has map support.

The screenshot cannot be rescued by correcting max response alone. Its
largest H32 disagreement at `[24,0,32,8]` has an actual score gain +6.9272,
feature-gradient estimate +4.7482, disabled rectangle estimate −4.6206 and
enabled estimate −4.1921. Family contributions expose strong cancellation:

| Family | Observed feature-gradient contribution | Disabled predicted contribution |
|---|---:|---:|
| SSIM | +1.8889 | +0.2548 |
| edge | +6.9579 | +1.4222 |
| MSE | +0.1941 | +0.0873 |
| HF | +0.0925 | +0.4256 |
| L8 | +8.2745 | +0.9878 |
| max | -12.6598 | -7.7983 |

Errors in edge, L8 and max terms cancel imperfectly; changing only one term can
worsen total rank. The high M2 is a rank result, not accurate gain magnitude.
Full-resolution and half-resolution B-channel detail peaks contribute strongly
here (IDs170/173/191), alongside basic and other-scale terms. This is a local
response diagnostic, not a model-wide feature importance estimate. It does not
prove a kernel arithmetic bug, nor justify reducing chroma weights globally.

Earlier bin1 follow-ups left these rectangle ranks unchanged. The present
family and root-curvature oracles also show that the earlier five-case root
correction cannot simply be generalized to all inputs. The prediction needs
better finite neighborhood response or a differently trained feature/model
regime; a blanket max removal is not established as sufficient.

## Existing family subsets also fail the wider screen

After the diagnosis, separately preregister the three already-trained controls
that omit entire suspect families: local120/H128 (no HF/peaks), basic156/H128
(no peaks), and basic192l8/H128 (no hard maxima). Reuse all 23 pairs, blocks,
bin8, finite moments and original gates. No new fitting or seed selection.
These are separately trained subsets, not coefficient-zeroing ablations.

All 69 new calls / 4,878 repairs have finite supported refinement and exact
native pixel witnesses. Base scores match their previous Rust cached rows
within the original 1e-10 tolerance. Same Rust panel reproduces saved ranks.
The complete six-model comparison reuses the earlier three models' evidence:

| Model | Both gates /23 | M2 failures | M3f failures | Worst M3f | Human TRAIN SROCC | Codec TRAIN SROCC |
|---|---:|---:|---:|---:|---:|---:|
| basic228/H128 | 17 | 5 | 1 | 0.5313 | 0.9048 | 0.9532 |
| basic228/H32 | 17 | 4 | 2 | 0.5195 | 0.8851 | 0.9549 |
| y60/H32 | 23 | 0 | 0 | 0.7166 | 0.8389 | 0.8780 |
| local120/H128 | 18 | 2 | 3 | 0.3150 | 0.8654 | 0.8982 |
| basic156/H128 | 19 | 2 | 2 | 0.6053 | 0.8838 | 0.8998 |
| basic192l8/H128 | 20 | 3 | 2 | 0.4852 | 0.8772 | 0.9156 |

Scalar columns are unchanged context from the complete
[TRAIN](product_train_2026-09-14.md), [scale](product_scales_2026-09-14.md) and
[peak](product_peaks_2026-09-14.md) panels/scatter. Read their near-lossless,
percentile, envelope, saturation, outlier and seed evidence; these two columns
cannot replace the composite or qualify a model. Codec agreement uses a
SSIMULACRA2 proxy, not independent human evidence. The packet's repeated TRAIN
development use and eight human references remain limitations.

Y60 is luma-focused, not luma-only: it keeps ten local Y slots at the first
three scales and all three channels' ten local slots at the eighth scale.
It alone passes all 23 spatial cases here, but sacrifices scalar quality.
The richer subsets improve rank and still fail spatially. No existing model
in this comparison resolves the full product tradeoff.

| New control | Failed row | M2 | M3f |
|---|---:|---:|---:|
| local120/H128 | 388 | 0.9951 | 0.4262 |
| basic192l8/H128 | 388 | 0.9822 | 0.6873 |
| basic156/H128 | 2436 | 0.9880 | 0.8469 |
| local120/H128 | 2445 | 0.9996 | 0.5769 |
| basic156/H128 | 2445 | 0.9907 | 0.6796 |
| local120/H128 | 4588 | 0.9882 | 0.9794 |
| basic192l8/H128 | 4588 | 0.9500 | 0.9588 |
| local120/H128 | 5316 | 0.9814 | 0.7875 |
| local120/H128 | 5324 | 0.9936 | 0.3150 |
| basic156/H128 | 5324 | 0.9959 | 0.6053 |
| basic192l8/H128 | 5324 | 0.9889 | 0.4852 |
| basic156/H128 | 6577 | 0.9869 | 0.9593 |

The tiny screenshot fails every richer model, including local120 without HF,
L8 or hard maxima. Do not spend the next iteration on removing maxima alone
or another capacity sweep. Next isolate chroma versus luma and scale-specific
finite neighborhood response on the actual failing pixels, using the existing
retained feature/kernel diagnostics. Keep scalar quality, model nonlinearity
and corruption detection separate in that investigation. A new feature era
requires mathematical parity and retraining; no inference formula changes here.

## Reproducibility and scope

Artifacts: `~/work/zensim-validation-2026-09-14/spatial-diagnosis/` contains
original protocols, source-report hashes, all per-feature arrays, saved oracle
analyses and `family-controls/` with exact models/tables/score witnesses.
[Compact results](spatial_diagnosis_2026-09-14.results.json) include every new
control disposition and the matched scalar-panel source hashes. The immutable
prior 69-case packet is referenced by hash; none of its results is overwritten.

The 69 control calls take 24.93 seconds including native decoding/verification
at two single-worker cases under a 16GiB cap. This is not a latency benchmark.
Together with six family-diagnostic calls this increment performs 75 complete
candidate cases and 5,088 repairs, plus zero-pixel saved analyses. The controls
contribute 5,016 candidate pixel comparisons / 138 candidate maps; family calls
add 216 comparisons / six candidate maps plus 24 separate diagnostic maps.
No new library code or build is needed; use the pinned existing binaries.

The [combined served A/B gallery](http://localhost:3300/zensim/reports/spatial-diagnosis-2026-09-14/gallery/index.html)
contains 138 checks from six frozen models, with all 24 failures retained.
The original 30-cell nomination still has seven unavailable cells, and only
five source families are represented. TRAIN results are not new EVAL rows.
No model qualifies. The full production goal remains active, including native
bounded targeting/RD, corruption, HDR, runtime tails and frozen qualification.
