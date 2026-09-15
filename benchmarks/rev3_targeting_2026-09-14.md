# Frozen Rev3 scalar targeting — September 14, 2026

This comparison measures the same nine frozen MT913 compositions and matched
B/D through the existing `demo_matrix` / `SeedCurve` / Rust target-search owners.
It does not fit models or change their failed ranking/dial qualifications.
Native spatial RD and HDR remain separate requirements.

The bounds instrument now accepts explicit, hash-bound complete compositions.
Manifest mode replaces implicit named B/D, so Rev3 ensembles and matched Rev1
controls can run in separate matching processes. No new scorer, controller,
public API or calibration algorithm is introduced. Changing any member, order,
weight or executable invalidates calibration reuse.

## Admission and chronology correction

The initial run reused the September 8 twelve-source TRAIN authority. A later
reservation audit found that origins 8462/9066 and suffix-8 families must remain
reserved. I stopped the first Rev3 evaluation and retained its completed seed
fit and partial measurements as **inadmissible**, not product evidence. This
was an admission error; checking the old canonical split alone was insufficient.

The replacement removes 6068, 9066, 8462 without replacement or quality-based
selection. Nine TRAIN families remain; all eight validation families are
unchanged. Their original split/family authorities and source hashes verify.
All eleven frozen compositions, knob/target grids, policies and gates stay
unchanged. Prior evaluation exposure is disclosed; the correction follows
pre-existing reservations, not model outcomes. No secret holdout is accessed.

Sources are the original opaque RGB8 sRGB renditions, at most 256 pixels on the
long side. Their PNG color metadata is checked before scoring; this is not
native-depth, wide-gamut or HDR qualification. Codec repositories are fetched
and at their remote main revisions. Unrelated working changes are preserved.
AVIF has active unrelated production edits and is deferred to a separately
pinned run; JPEG, WebP and JXL are included here.

## Frozen measurement protocol

Fit codec/model-specific seeds solely on the nine TRAIN families, using the
existing median/envelope `SeedCurve` owner and 21 native-knob samples per image.
For each validation image/codec/model, measure the same 21-point ladder before
steering. Keep fixed requests −10/30/70/90/99 and five additional witnessed
scores, in original model units. Only requests with a measured witness within
one score unit enter steering success/failure counts. An unwitnessed range is
not proof of impossibility. Bounds and witness knobs are not controller inputs.

Compare midpoint and frozen TRAIN-curve policies at actual 1/2/3 encode budgets.
Independently reproduce every emitted bitstream and score, accounting separately
for that extra encode. Preserve independent SSIMULACRA2/Butteraugli judgments,
negative targets, endpoint/interior errors, under/overshoots and source results.
One score unit is an instrument stopping band, not a validated universal
perceptual tolerance. Model-specific witness targets cannot support cross-model
pooled-error comparisons as if requests were identical. Small-image operating
times are not production latency qualification.

Artifacts: `~/work/zensim-validation-2026-09-14/rev3-targeting/`; only the
`admitted/` replacement supplies the authoritative result. `STOPPED.json`
records the earlier run's disposition; no failed experiment is erased.

## Measured results

All 11 compositions complete all registered cells. The admitted runs perform
**9,648 steering cases**, 15,995 search encodes and 9,648 separately counted output
verification encodes. TRAIN ladders add 1,134 encodes; evaluation bounds add 1,008,
shared across models within each revision. Total admitted encoder work is 27,785
full encodes. The stopped initial run is additional inadmissible work retained
separately. 138 steering cases have negative targets.

Of 1,320 fixed image/codec/model requests, 288 have a witness, 823 lie outside
the sampled envelope and 209 are unwitnessed inside it. Exclusions are coverage
limits, not codec-impossibility claims or hidden targeting failures.

The following shows **three-shot TRAIN-curve median / p95 absolute error**, in
original score units. Each model has its own witnessed target population; these
columns describe individual models, not a controlled cross-model ranking. Full
1/2/3-shot, endpoint/interior, class, paired-family and hit-rate tables come from
the existing analyzer and are retained in the replay bundle.

| Frozen composition | JXL | JPEG | WebP |
|---|---:|---:|---:|
| MT913_full944_h128_ens5 | 0.632 / 9.403 | 0.605 / 2.877 | 0.630 / 8.221 |
| MT913_full944_h256_ens5 | 0.826 / 6.475 | 0.566 / 3.057 | 0.970 / 8.316 |
| MT913_local120_h128_ens5 | 0.836 / 3.669 | 0.533 / 4.380 | 0.601 / 3.308 |
| MT913_selected619_h128_ens5 | 0.679 / 7.286 | 0.653 / 6.084 | 0.678 / 8.749 |
| MT913_y40_h32_ens5 | 0.077 / 1.033 | 0.421 / 2.256 | 0.327 / 3.402 |
| MT913_y40_h128_ens5 | 0.126 / 0.931 | 0.465 / 2.875 | 0.391 / 1.725 |
| MT913_y60_h32_ens5 | 0.111 / 1.224 | 0.404 / 1.829 | 0.424 / 1.677 |
| MT913_y60_h128_ens5 | 0.131 / 1.207 | 0.311 / 2.955 | 0.413 / 1.916 |
| MT913_linear60 | 0.406 / 0.948 | 0.339 / 2.060 | 0.335 / 2.642 |
| MT914_matched_B | 0.355 / 3.024 | 0.539 / 3.053 | 0.484 / 3.229 |
| MT914_matched_D | 0.254 / 1.653 | 0.417 / 2.951 | 0.375 / 1.690 |

Target-search medians alone would hide the wide models' tails: full944/H128
has three-shot JXL p95 error 9.403 and WebP 8.221. The narrower models have tighter
tails on many of their own witnessed populations, but their incomplete score
range and failed human ranking still prevent shipping. No winner is selected
or recipe adapted to this assessment. Native spatial allocation, AVIF targeting,
corruption specificity, full-range near-lossless behavior, HDR and production
latency/memory qualification remain outstanding.

All output re-encodes are byte-exact and reproduce their reported score. A separate
canonical extractor audits the first emitted result per composition and codec:
**33/33 decoded pixel hashes and f32 complete scores match exactly**. This
independent sample verifies all members are served consistently; it does not
claim independent re-decoding of every output. The extractor's one-member
ensemble refusal is retained; singletons are audited via its standalone-bake
option, with identical model bytes.

Four Rust example tests and four analyzer tests pass. Six actual negative controls
refuse changed members, weights, order, formula or family overlap before any
source pixel access. Root and targeting-example Clippy, script lint and board
render/data gates pass. No Rust public API changed.
