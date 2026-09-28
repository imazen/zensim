# Corruption activation and low-quality boundaries

September 13, 2026, after the steerable-subset study and the user's dual-head
clarifications. This supersedes earlier zero-floor composition and the use of
every synthetic corruption as a catastrophic label. Historical measurements
remain unchanged and do not qualify the new composition.

## Registered serving change

Concrete callers: `BakeScorer::score_features` and its pixel callers, and
`Zensim::compute_with_corruption` returning `CorruptionVerdict`. Change the
existing public `corruption_head::gate_score` behavior, with no signature or
wire-format change:

```
activated = C < T
score = activated ? min(P, C) : P
```

P and C are quality-oriented scalar scores: lower is worse; equality at T is
inactive. The existing tree head supplies C = 100*(1-p); its probability
deadband corresponds to T = 100*(1-probability_deadband). A ZNPR catcher can
also supply a negative C. Preserve negative P; never raise it. T belongs to
the model configuration, not the end user's one target-score dial.

Activation is an integrity rejection even when P is already below C and the
minimum leaves its number unchanged. Count false activations independently
of honest score lowering. Fit/calibrate the catcher for a separated inactive
and failure response; a thresholded minimum alone does not make a model bimodal
or establish a reliable operating threshold.

## Label policy before the next fit

Separate defect cause, visible severity and required disposition. A confirmed
implementation bug is not automatically a catastrophic image failure, and a
low perceptual score is not evidence of a bug.

| Disposition | Examples and evidence | Treatment |
|---|---|---|
| Valid reconstruction | Native encodes across the attainable per-image range, including the codec floor, strong blocking/ringing, lost chroma/detail and normal spatial allocation | Catcher negatives, including deliberately difficult low-quality examples; perceptual score/map must still price their damage |
| Recoverable defect | Localized or weak row/grid/impulse defects whose visible effect remains comparable to valid low-quality reconstruction | Keep a separate reviewed stratum. Perceptual damage remains relevant; no automatic catastrophic positive or universal below-q20 requirement |
| Catastrophic failure | Material channel/layout/stride mistakes, large missing/repeated regions, destroyed structure, or severe impulse/grid damage inconsistent with accepted reconstruction | Catcher positives when supported by the actual pixels and failure provenance; activation rejects the candidate |
| Ambiguous or visually inert | Cause known but effect negligible, or indistinguishable from a valid encode at this quality | Preserve and report separately; do not force a binary label or count as successful catastrophic detection |

Examples describe review criteria, not automatic operation-name labels. A tiny
affected area can still destroy salient content; whole-image channel swaps can
be inert on neutral content. Very low quality changes the relevant comparison,
but must not become a blanket exemption for genuine catastrophic failures.
Pixel-identical valid and buggy executions cannot be separated by a metric
that sees only the image pair; pipeline diagnostics own that distinction.

Before fitting, create a versioned label manifest referencing immutable row,
source and pixel identities, operation/codec settings, affected extent, severity,
disposition and review reason. Review source-matched native quality ladders and
failure severity sweeps together. Establish codec-attainable bounds first;
codec q values and a universal q20 anchor are not a common quality ruler.
Freeze labels without looking at candidate catcher outputs. Keep existing
manifests and report old/new strata and exclusions so relabeling cannot silently
turn previous failures into passes. No relabeling or new fit has occurred here.

## Calibration and evaluation

Use canonical training origins, with a separate training-origin calibration
split; preserve untouched evaluation origins. Tune one model threshold subject
to honest false-activation constraints, then freeze it. Report activation,
score lowering and catastrophic recall separately by codec, origin, quality
band, defect extent and reviewed disposition. Include the lowest valid quality
band explicitly; pooled false-positive rates cannot hide its failures.

Extend the existing zero-native and <=1% overall honest score-lowering limits
to false activation as well. The legacy fitting
screen now checks activation as well as lowering, but its old binary labels and
q20 ordering remain historical diagnostics until the reviewed manifest exists.
The previous 95% detection, 90% real-bug detection and 99% below-q20 figures
cannot be reinterpreted as catastrophic-stratum passes without new evaluation.
Register catastrophic/low-quality strata and ordering anchors before fitting;
do not drop difficult rows based on a candidate's scores.

Successful steering must use the perceptual branch only. An active catcher
returns a failure disposition, not a corruption gradient. Recheck the catcher
on every actual reconstruction, including low-quality candidates. The current
prepared API still refuses companions; enabling inactive companions and a typed
failure result is remaining implementation work. Its refusal is not evidence
that the model needs a spatially differentiable corruption head.

No model is qualified by this contract change. Existing catcher thresholds and
the steerable-subset candidates require evaluation under the new composition.

## Implementation validation

`cargo test -p zensim --lib --all-features`: 476 passed, eight ignored. Tests
cover strict threshold neighbors/equality, negative tails, active verdicts
without score lowering, ZNPR composition through `BakeScorer`, and tree pixel/
cached-row parity. These fixtures prove serving behavior, not trained detection.
`just clippy`, `just api-doc-check`, `just lint-scripts` (612 scripts), formatting
and diff checks pass. The public signature snapshots are unchanged.
`scripts/serving_matrix.sh` passes native and portable configurations, including
builds without v2. Logs: `~/tmp/zensim-corruption-threshold-2026-09-13-serving/`.
No extractor arithmetic, model bytes or prepared-steering behavior changed.
