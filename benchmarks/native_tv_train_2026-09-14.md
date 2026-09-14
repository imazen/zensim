# Native pair-margin recipe: no advancement, September 14, 2026

The registered plain local120/H128 TV recipe fails the native continuation screen.
Human and codec rank improve modestly, but the complete ensemble has **10/79**
robust native conflicts versus **9/79** for its fresh control. This is a failed
TRAIN screen, not a statistically established population regression or an EVAL
result. No model is promoted, and no shipping gate is relaxed.

[Exact A/B gallery and all model comparisons](/zensim/reports/native-tv-train-2026-09-14/index.html).
The prior [signed contribution diagnosis](native_contributions_2026-09-14.md)
motivated clean pair supervision rather than another capacity/constraint sweep.

## What was actually compared

Six scientific fits: local120/H128 plain, control vs the existing Rust TV hinge,
three paired initialization seeds 7101/7103/7107. TV uses 214 robust native pairs
from eight TRAIN fitting families, ordered by same-buffer SSIM2 and Butteraugli
agreement, not by codec q. Weight .5, raw margin .25, every 50 loss-bearing main
pairs, batch 32. The native table has 428 endpoint rows, train/selection weights
zero. Human/codec groups retain their original supervision and scaler mass.

The corrected recipe selects checkpoints using the equal mean of the human
and codec **fitting** groups, explicitly excluding native. All fits use 32 epochs,
8,192 attempts/epoch, uniform sampling, MSE weight1, no early stop or auto-eval,
Rev3 full944 source features with the same120 declared input IDs, and f32 packs.
The eight scientific/integrity fits plus packing/public scoring completed in
**203.37 seconds**. Full panels took about3s and native replay7.70s.
These are workflow elapsed times, not qualified inference latency benchmarks.

TV consumes the same RNG as main sampling and adds Adam updates. This tests a
complete recipe, not an isolated loss term with identical ordinary draws. SHA
derived sampling seeds have disjoint conservative1,216,320-word windows including
TV draws. Control digests match subset_sim; TV digests are actual trainer records.
The fixed first-seed TV repeat reproduces its digest and all served predictions.

## Checkpoint-selection trap caught before accepting the experiment

With every validation weight zero, the historical trainer selects the best
checkpoint using the average of **all** group reports, including a group labeled
report-only. Early-stop=0 still exports best_bake. Adding native therefore changed
the control even when its train and validation weights were both zero. The fixed
removal check failed:968/1,000 human development scores changed, maximum138.3954
points, despite matching sampling digests and all32 reported main trajectories.

The first two fits plus one integrity fit remain diagnostic artifacts in
`fits-separated/`; no further seeds ran there. Replacement `fits-explicit/` sets
human/codec selection weights1 and `--val-policy mean`, preserving the original
two-group checkpoint rule. Its no-native control reproduces all2,893 human/codec/
native development scores exactly. No development labels choose checkpoints.
This corrects the recipe without changing historical trainer arithmetic.

The synthetic Rust regression reverses the TV pair on an axis absent from main
training and checks the served preference under both distance and score polarity.
All four arms pass; the full7-test polarity suite, Clippy and script lint pass.
CLI/plain-loop comments and the playbook now describe the actual contracts.

## Native results

Four unchanged TRAIN development families, eight cells,272 retained JXL outputs.
All nine Rev3 candidates/control models run through complete Rust pixel/map APIs;
frozen D uses Rev1 on the exact same pixels.79 robust consensus comparisons use
SSIM2 .1 / Butteraugli quality .005 margins and opposite model change >.1.

| Model | Conflicts /79 | Own-score mass rank median | SSIM2 map rank | BA quality map rank | M2 failures | Mass cells below .70 |
|---|---:|---:|---:|---:|---:|---:|
| NTV914_local120_control_ens3 | 9 | 0.469 | 0.366 | 0.315 | 1 | 6 |
| NTV914_local120_tv_ens3 | 10 | 0.503 | 0.429 | 0.315 | 0 | 5 |
| NTV914_local120_control_s7101 | 10 | 0.585 | 0.487 | 0.479 | 0 | 4 |
| NTV914_local120_tv_s7101 | 13 | 0.650 | 0.419 | 0.374 | 0 | 4 |
| NTV914_local120_control_s7103 | 11 | 0.525 | 0.203 | 0.018 | 1 | 6 |
| NTV914_local120_control_s7107 | 7 | 0.540 | 0.319 | 0.224 | 1 | 6 |
| NTV914_local120_tv_s7103 | 10 | 0.654 | 0.431 | 0.382 | 1 | 4 |
| NTV914_local120_tv_s7107 | 7 | 0.579 | 0.409 | 0.253 | 0 | 5 |
| PT914U_local120_h128_plain_ens3 | 10 | 0.494 | 0.376 | 0.300 | 0 | 5 |
| D_frozen_revision1 | 0 | 0.638 | 0.535 | 0.525 | 0 | 5 |

The TV ensemble has photo/document/graphic/screen conflicts2/0/5/3; control
2/0/5/2. Its own-score map association improves .469→.503; SSIM2 association
improves .366→.429 and Butteraugli remains .315. High internal consistency does
not establish independent quality. The .99/.70 native lines are mechanism
diagnostics, not a passed repair, allocation, or matched-RD gate.

## Full scalar assessment and tails

All six members and both complete ensembles have **10,792** canonical
panels over human1,000, codec1,629 and native264 TRAIN-development rows. All
panels preserve their actual populations and counts, including reference, source,
codec, class, ladder, distorted, identity and sparse high-quality views. The
9,840 undefined sparse-statistic entries are explicit nulls with paths;
they are not zeros or passes. Full panel/scatter results are served in
`evidence/assessment-final/RESULT.json`. This is not a complete shipping composite.

| Ensemble | Human SROCC | Codec SROCC | Distorted codec SROCC | Native pooled SROCC |
|---|---:|---:|---:|---:|
| NTV914_local120_control_ens3 | 0.869654 | 0.939399 | 0.900676 | 0.621048 |
| NTV914_local120_tv_ens3 | 0.873543 | 0.939776 | 0.901294 | 0.621048 |
| PT914U_local120_h128_plain_ens3 | 0.865363 | 0.937908 | 0.898233 | not in original scalar packet |

Native pooled rank is identical between the two fresh ensembles while their
local conflict counts differ. Pooled rank alone misses the steering defect.

Human raw residual p99 improves73.91→71.12, but maximum residual worsens159.59→
163.51, raw clumping increases .132→.153, and raw coverage falls .70→.65.
Geometric out4 remains .013. Human MOS and native score units are not identical;
these are the canonical diagnostics, not a newly invented absolute-error target.
All raw/normalized scatter and range fields remain visible in the evidence.

## Did the fitting pairs learn?

A separately registered post-screen diagnostic scores the214 fitting pairs
through the same public Rust scorer, without fitting or changing any gates.

| Ensemble | Opposite preferences >.1 | Margins >=.25 | Mean hinge |
|---|---:|---:|---:|
| NTV914_local120_control_ens3 | 13/214 | 96/214 | 0.093340 |
| NTV914_local120_tv_ens3 | 11/214 | 95/214 | 0.101355 |

The exported TV ensemble does not satisfy most fitting margins, and its mean
hinge is worse than control. This experiment therefore does **not** establish
a feature-set capacity ceiling or a pure generalization failure. It establishes
that this fixed training/selection recipe does not repair the product behavior.
Do not infer that another weight, more features, or dropping chroma will fix it.

## Provenance, checks and stopping rule

The registered continuation checks pass for rank tolerances, no newly failing
class, M2 count and map-failure count, but fail required conflict improvement.
Stop this recipe; no immediate margin/weight sweep or EVAL retuning. The separate
coarse-B sampling/phase response remains a concrete feature question on saved
TRAIN pixels; broader feature, targeting, corruption, native RD and runtime
qualification remain open. The cumulative memory audit is30/99 full reads;
related transcripts and the remaining documentation audit are not complete.

All272 fresh decoded pixel/peer rows match the retained packet. The frozen
product control reproduces272 scores and all maps exactly. Every new member
and ensemble matches cached native inference:2,176 pixel comparisons, including
neutral duplicates joined by exact decoded hash, covering264 unique rows each.
All17,358 raw/packed human/codec/native predictions agree. No EVAL, TEST,
TERMINAL, new encodes, feature extraction or calibration was used.

The first binary-hash preflight refused after Cargo rebuilt the CLI owners;
current binaries and source hashes were then pinned before fitting. A checkpoint
diagnostic initially expected32 log lines rather than64 (stdout/stderr duplication);
it was corrected to32 unique epochs, preserving the failed log. A summary check
initially treated deduplicated neutral rows as distinct cached rows; it now joins
their exact pixel hashes and checks every neutral response, with no skipped rows.

Artifacts: `~/work/zensim-validation-2026-09-14/native-tv-train/`. Start with
`PROTOCOL.md`, `CHECKPOINT_AMENDMENT.md`, `TOOLS.json`, corrected fit registration,
`INTEGRITY.json`, full assessment, native manifest/result, `SUMMARY.json` and
fit-pair diagnostic. The adjacent results JSON binds these by hash. Commands
are in the immutable registrations and model sidecars; private local drivers
orchestrate existing Rust owners. The LAN report publishes recipes, scores,
models and diagnostic failures. No model or package release is made.
