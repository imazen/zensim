# Prospective core ordering repair — 2026-10-16

Status: design for a new TRAIN-only experiment, not a repaired or qualified model.
The [measured diagnosis](../benchmarks/addrfix_2026-10-16.md) finds 97 strict
inversions before the output curve. Removing K changes no A7r result. Fix the
core ordering before fitting another scalar calibration; preserve the current
feature sets, architectures and original bakes as controls.

The subsequent cheap f32 checkpoint control improves the C ladder from 19/39
to 25/39 on AVIF-rav1e, reaching that bar and passing all five ladder codecs.
All three f32 models still fail standard AVIF/JPEG, and f32 W-C loses one svt
window. Check packing/pruning effects before paying for a new fit; preserve
separate intervention identities and select the final packed model on TRAIN.
The original fit receipts show production selects epoch 119 (`last.bin`) while
the recorded best curve epoch is 0; all three best checkpoints differ from the
selected final checkpoint. The retained best comparison therefore changes the
training epoch as well. A same-epoch f32 comparison must precede attributing
regressions to a packing operation.
The same-epoch replay is complete: all gate counts are unchanged, and all 97
original bad edges remain strict inversions in f32. Packing did not create
these failures. Prioritize the TRAIN-only core ordering loss below; the best
checkpoint comparison remains a separate exploratory observation.

## Population and generation

Extend the existing E34 reference admission in
[e34_augmentation.py](../scripts/rev4_featpot/e34_augmentation.py): canonical
imazen-26 TRAIN in both split owners, the named source exclusions, broad-family
exclusions, native dHash screen against all registered assessment references,
and family closure. The previously measured eligible pool has 913 origins;
re-resolve paths and hashes before reuse. Keep existing development families out
of fitting. Never turn an assessment reference into a training image because
this investigation exposed its failure.

Select reference-only feature clusters within photo, screen, line art, mixed
and tiny content strata, retaining singleton clusters. Require at least 50
references per stratum and reserve at least 20% of families within every stratum
for TRAIN-internal model selection. If that coverage cannot be admitted, report
the missing stratum instead of treating a smaller set as representative. Freeze
origin/family membership and the fit/selection roles before any generated scores
or candidate results are inspected.

Use zenmetrics/zenfleet's existing plan-cell bridge and persisted artifacts,
with imazen zenavif-rav1e and zenjpeg encoders. Do not reuse the historical E34
third-party JPEG encoder for this training dataset. Cover native q0–20 at step 1
and q22–100 at step 2 (61 settings), every supported production mode and CPU-cost knob,
and the 20 registered sizes 32,40,48,64,80,96,128,160,192,256,320,384,512,640,
768,1024,1280,1536,2048,4096. Use a declared imazen resampling kernel and never
upscale; record which origins reach each size and any rejected endpoint/mode.
Reduce image count before size or quality resolution, while retaining the
minimum content coverage. First verify encoded bytes, native decoded pixels,
metric maps and per-cell manifests actually land before scaling generation.
The low-q density covers adjacent settings: the measured failed AVIF edges span
q0–6 (standard q0–5), and failed JPEG edges span q0–16.

Score the same native decoded pairs with pinned zenmetrics CVVDP standard_fhd
as the scoring teacher. Retain butteraugli scores and native maps as steering
diagnostics; SSIMULACRA2 is a low-priority comparison, not an admission filter.
Record reference/distorted pixel hashes, encoder and metric build commits,
display parameters, feature formula/layout and output polarity in every join.
No human target or protected population enters this experiment. Any distributed
execution needs its separate coordinator authorization.

## Loss and implementation owner

Reuse `zensim_mlp_train --tv-pairs-file` and its existing `TvRegularizer` in
[mlp_train/mod.rs](../zensim-validate/src/mlp_train/mod.rs), rather than adding a
second ranking trainer. A pair is an observed worse/better quality preference,
not an assertion that a higher encoder q always looks better. Among the 97
diagnosed failed edges, 44 also decrease under the active mentor; blindly
forcing every gate-panel q edge upward would encode those disagreements as
training truth. Gate rows are never pair-generation inputs.

Build within-reference, same-size, same-codec/mode teacher-resolved preferences
on the new TRAIN ladders, including adjacent and longer-range pairs. Teacher
resolution and the positive raw-output separation margin must be frozen from
TRAIN-only score distributions; the gate's 0.5 mentor window margin is not a
CVVDP JOD or neural raw-output margin. The existing hinge is
`max(0, s*(prediction[better]-prediction[worse]) + margin)`, with `s=-1` for a
score and `s=+1` for a distance. A zero-margin hinge permits flat collapse, so
include an explicitly positive margin arm and preserve the original rank loss
and training populations. Register the finite weight/margin search, seeds and
selection objective before fitting; do not choose them on these A7r rows or the
new matched human ranking.

The pair TSV indexes the exact concatenation of trainer groups. Bind that
ordering by SHA, require every index in range before launch, require zero
dropped pairs in the trainer receipt, and independently check both polarity
directions on real worse/better inputs. Native features must match the served
extractor, including derived inputs. Compare the unchanged-loss control against
the new loss with identical group order, seeds and feature/architecture choices.

## Selection and confirmation

Select only on the frozen TRAIN-internal reference families, using teacher
agreement, strict ordering and separation, identity and cross-reference rank
to detect collapse. Fit K using TRAIN only after the core is frozen. Check
final packed bytes as well as f32 checkpoints through the existing bake owner;
packing, pruning and calibration are separate interventions, not an assumed
single quantization effect.

Then replay the original A7r panels with unchanged live mentor bars as an
explicitly exploratory regression check. A pass there does not establish
generalization. Confirmation needs a separately frozen untouched ladder panel
and the adopted ranking, steering and runtime gates; any new assessment exposure
is logged before access. Existing matched CVVDP ranking results cannot select
this repair's hyperparameters or checkpoint.


E36 prospective clarification (coordinator, 2026-10-17): the TRAIN selection
split has no human ranks, so no cross-reference rank acceptance rule is added
to selection. Cross-reference ranking is protected at the registered human
assessment by noninferiority and its dataset and W2 guards. Plain N emits a
higher-is-better raw score `pin-g`; its ladder improvement is raw(better) minus
raw(worse). This is the existing score-polarity hinge convention, independent
of the `nonneg_distance` flag's name.
