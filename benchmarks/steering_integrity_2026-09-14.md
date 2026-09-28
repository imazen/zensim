# Steering and integrity model development — September 14

User directive: use complete assessment (Mohammadi, envelopes, tails,
clumping, saturation) through a single evaluation path, and develop an
all-purpose steering control with real-time bug detection.

Before implementation: migrate scatter math from gauntlet/outlier Python to
`zenstats::scatter`, consumed by existing Rust `bake_verdict` and `panel`.
Emit full-population statistics before per-pair plot subsampling; the renderer
reads them. Existing legacy artifacts stay legacy; missing metrics are not
passes. Version the tie correction: tied predictions share their mean target
order statistic, rather than acquiring arbitrary ordering from input rows.
Reference occupancy and model occupancy are distinct; degenerate scales are
unmeasured, not arbitrary epsilon denominators. Preserve raw score residuals
and exact floor/ceiling mass alongside normalized shape. No eval-fitted model
calibration. New zenstats API caller/signatures are registered in its README.

Prepared steering next: inactive corruption companions preserve perceptual
maps; activation returns a typed integrity failure. No finite differencing
through the corruption threshold and no corruption gradient enters allocation.
Register API details before those edits. Evaluate whole compositions through
BakeScorer, including every new head and spline, before product claims.

The September 13 split contract remains absolute. The existing corruption
trainer's legacy mode and mixed-data canonical loader are not approved defaults.
Admit train and eval separately before reads; exclude all earlier test roles.
Freeze severity/disposition labels using pixels and provenance before fitting,
separating valid native low-quality encodes from recoverable, catastrophic and
ambiguous damage. Fit/calibrate on train only. Check native false activation
by codec and quality band, catastrophic detection and actual real-bug coverage.

The frozen minimal/wide study is a baseline, not a cross-corpus selection.
No model is qualified until the existing scorecard's human, target, corruption,
native spatial RD, latency and memory requirements are measured and pass on
allowed eval data. Changes here do not lower those bars.

## Prepared-companion API registration

Concrete caller: `SteeringSession::compute` used by codec reconstruction loops.
Add the unit variant `ZensimError::CorruptionDetected` to the existing
non-exhaustive error enum (retain Eq and Copy). `prepare_steering` admits
already-servable companions whose feature requirements the existing plan
covers. Each reconstruction uses the same retained extraction, computes the
perceptual map without a corruption derivative, and checks the restored head
on those exact features. Strict activation rejects the reconstruction even
when min(P,C) would leave a negative P unchanged. Exact identity remains 100.
Restore the companion before propagating any error; repeated calls must never
silently lose protection. No change to ordinary composed scalar scoring.

## Bounded severity-aware D228 companion prototype

Freeze before looking at candidate outputs. Reuse the eight admitted fit origins
2010/1054/6068/6610/7066/9380/8206/8384 and two training-calibration origins
1214/6064. Exclude 8462/9066 entirely because they acquired test roles. Original
validation origins remain eval only and are not opened for fitting. Read only
explicit admitted PNG paths; do not load historical mixed feature parquets.

For this first prototype, use the established D228/Rev1 feature and scalar
contract, the existing HGB100/31-leaf factory, honest fit weight4, seed4101,
f32 inputs and train-calibrated isotonic output. Keep the fixed probability
activation0.9 (head score threshold10). This is one fitted head, not a seed
lottery; measure it through the complete Rust D+ZCTH surface. It is a baseline
composition for the new serving path, not a replacement for the Rev3 model
study or proof of a fast all-purpose model.

Severity admission is deliberately explicit and provisional: whole-region,
opaque channel/layout or documented real-bug reproductions with RGB RMSE
>=0.10 of full scale AND >=15% of pixels changing by >=16 codes are the
catastrophic-proxy fit stratum. Whole opaque flip/rotation qualifies under
the same pixel tests; one-pixel shifts do not. Native honest outputs and
matched JPEG anchors remain negatives irrespective of perceptual quality.
Inert defects, local/weak effects, normal aliasing, tone changes, sparse
impulses and other ambiguous operations retain their own records and are
excluded from binary fitting. Their exclusion is not successful detection;
report the entire disposition inventory. These pixel thresholds are an
engineering screen, not validated human severity or a universal salience rule.
Review representative pixels before fitting and preserve all source/hash/
operation/extent/disposition reasons. Do not apply this head as a substitute
for perceptual penalties on valid-but-bad reconstructions.

Advance only if training-calibration has zero native false activations,
<=1% overall honest activations, >=95% catastrophic-proxy recall and >=90%
real-bug-proxy recall, plus complete Rust pixel/cache parity. Otherwise keep
the failure and do not open eval merely to search for a better-looking result.
No threshold sweep, eval relabeling or model selection is authorized by a fail.
A passing prototype still needs broader native low-quality negatives, actual
bug fixtures, and the full product scorecard on admitted eval.

Pre-fit visual review amendment: the contact sheet shows full mirror/layout
failure on a mostly-white website falling below the blanket RMSE/extent bar.
Whole opaque flip/rotation therefore also enters the proxy-positive stratum
when >=0.5% of pixels change materially. This is a semantic layout override,
recorded before any head fit/output. Preserve the earlier proposed labels and
the contact sheet. Weak/sparse chromatic effects and neutral-page inert swaps
remain explicitly outside the catastrophic-proxy claim.

Prepared real-model audit: extend the existing extractor audit with the explicit
`ZENSIM_AUDIT_PREPARED_STEERING=1` diagnostic. On every requested reconstruction,
compare actual prepared rejection with the frozen head's activation; for every
accepted pair compare the scalar and all 32px refinement queries with the base
perceptual worker, including finiteness. Retain the same decoded-pixel hashes.
This is pipeline parity, not a new finite-intervention or native-RD claim.

Cost instrument registration: existing `ssim2_speed_bar` gains opt-in
`ZEN_S2_PREPARED=1` paired base/head prepared-worker arms. Full score+map,
reference cache outside the timed reconstruction loop, identical geometry and
thread count. A fired head on the fixture fails the run instead of timing a
rejection as a successful map. Use at least 30 rounds, pinned single-thread
1024², retain dispersion and machine contention. This bounded cost measurement
does not supply missing p95/worker-memory or full image-size coverage.

Full-catalog diagnostic, frozen model: score all 6,024 original eval attempts,
including excluded weak/ambiguous and inert operations. No new fit or threshold.
This exposed 11 duplicate pixel groups with one admitted positive operation and
one excluded/unlabelled operation (e.g. the Adam7 reproduction and nearest
aliasing produce identical pixels). The first report guard refused any differing
disposition. Correct it to distinguish missing labels from contrary labels:
retain all catalog dispositions/families, prefer an existing known binary label
for that exact pixel identity, and count once. A valid/catastrophic conflict
still refuses the report. Do not manufacture negative labels from exclusions,
change the fitted labels, or claim the pixels reveal the software cause.

Reproducibility closure, before replay: the original fit manifest pinned a
build-tree extractor subsequently rebuilt for the prepared audit. Preserve that
manifest and original head unchanged. Freeze current tools in a separate packet
and replay the exact same train-only recipe once, with no parameter/label/seed
changes. Compare all predictive ZCTH bytes (everything except provenance JSON)
and all training predictions. This is a determinism check, not a second candidate
or another use of eval. The original eval remains bound to the original head.

## Results and remaining product gates

Status: **COMPLETE_UNQUALIFIED prototype**, with one new head and one deterministic
replay of that same recipe. This is not a new perceptual model or a default
replacement. All fitting/calibration uses train; eval is gate/diagnostic only.
No test segments or mixed historical feature tables were opened.

Training has 900 unique fit pairs and 124 training-calibration pairs. Calibration
has zero activations on 80 honest outputs, 44/44 catastrophic proxies and 27/27
real-bug proxies detected. The final ZCTH is 182,532 bytes, 100 trees / 5,548
nodes, reading D228 under Rev1 with f32 inputs. Full train Python/Rust parity:
zero ULP decision-function difference, zero probability difference and identical
activation. The replay retains every predictive byte and all 1,024 training
predictions exactly; only the provenance JSON differs. Frozen tool copies and
a replay manifest close the original build-tree binary pin limitation. The
original head remains the evaluated artifact.

| Frozen eval stratum | Result |
|---|---:|
| Admitted severe proxies | 175/180 detected (97.22%) |
| Real-bug reproduction subset | 109/113 detected (96.46%) |
| Honest reconstructions | 0/320 activated or lowered |
| Native JXL controls | 0/168 activated |
| Native AVIF controls | 0/136 activated |
| Worst stored native knob per origin/codec | 0/16 activated |
| Matched low-quality JPEG anchors | 0/16 activated |

The eight eval origins span photo, document, graphic and screen content. The
lowest per-origin severe recall is 17/18; these small origin counts are not a
population guarantee. Five misses remain: Adam7 buffer failure (2403), HEIC
chroma motion-compensation shift (6823), zero-B (9113), grayscale JPEG sampling
strip overflow (8113), progressive JPEG AC truncation (8183). There was no
threshold/model tuning after these misses.

All 6,024 original eval attempts were then scored, including unlabelled cases;
5,679 pixel-unique pairs remain. Besides the 320 honest and 180 positive pairs,
there are 5,173 weak/ambiguous pairs (1,086 activated) and six unique inert pairs
(zero activated). The ambiguous activations are **unresolved specificity**,
not successful bug detections or known false positives. Distortion family is
not itself a binary label. This is the strongest reason not to call the head
all-purpose or an operational error oracle. Identical pixels produced by
separate operations retain all catalog families/dispositions; no pixels can
establish the software cause.

Prepared public-API eval checks 500/500 pairs: 175 typed integrity rejections,
325 accepted with exact base/perceptual score/features/maps, 16,224 finite
rectangle queries. This proves composition and map preservation only; it does
not substitute for finite interventions, M2/M3 or native codec RD.

| 1024², single-thread public path, 30 rounds | Mean ms | Observed min–max ms |
|---|---:|---:|
| D scalar | 26.903 | 26.223–28.164 |
| D + integrity scalar | 26.911 | 26.216–28.328 |
| D prepared score + map | 59.355 | 58.459–60.648 |
| D + integrity prepared score + map | 59.185 | 58.204–60.964 |

Reference preparation is outside these reconstruction timings. The fixture is
synthetic textured SDR, no activation. There is no detectable head overhead
within dispersion, and no claimed speedup. No p95, per-worker memory or broad
size/thread qualification is supplied. D/Rev1 remains a comparatively expensive
baseline, separate from the Rev3 minimal/wide feature study.

Assessment now keeps rank-normalized geometry **and raw score density**: 20
raw equal-width min/max bins, with absolute range/slope alongside them. A smooth
power-law compression can preserve ranks perfectly while putting >70% of raw
scores in one bin; the regression test catches exactly that case. Robust ±4-MAD
outlier share is also insufficient alone: a wider error cloud can reduce its
outlier fraction. Read p99/span, maximum/span, raw absolute residual tails,
full Mohammadi panels and named composite floors together. No new pass threshold
or weighting was selected from this eval. Missing peers/range/axes are INCOMPLETE.

Next: register operational severity labels and broaden train-only controls for
honest low-quality and spatially allocated encodes, including JPEG/WebP; keep
this eval frozen. Qualify the perceptual base and complete composition through
allowed human panels, HFNL/dial and raw/shape tail gates. Then per-image codec
bounds, frozen codec-specific 1/2/3-shot seeds, native spatial allocation/RD,
HDR and p95/worker-memory costs. Existing qualification requirements stand.
