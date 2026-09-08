# Cleanup scientific controls — registered September 7, 2026

Registered before new encoding or training results. This extends the frozen
three-seed A_plain reproduction, not a new-model qualification or a promise
of improvement. The recipe contains the documented historical KonJND/KADID
split overlap and runs with an explicit historical-replay reason.

## Matched floor-data control

Replay `scripts/bestofall_wave.sh` H_anchorlad at seeds 4004/4005/4006, then
repeat exactly that recipe with rav1e training ladders appended to its existing
four-codec anchor group and TV pair stream. Keep architecture, group weights,
loss weights, optimizer, epochs, packing anchor and evaluation instruments fixed.
H versus A changes architecture and loss/data together; it does not isolate
any single constraint. The paired H versus H+rav1e comparison isolates the
added ladder data and its consequent sampling distribution.

Select eight references from the existing September 5 anchor sources by
ascending pixel count, breaking ties by filename. Preserve original PNG bytes;
no resizing or image selection based on model outcomes. The selected sources
are 8288, 8434, 8414, 7004, 7042, 7050, 7052 and 7058. Recheck their existing
training-split and CID22-neighbor records. Encode rav1e q=0..30 inclusive,
then 50/70/90/100 with the frozen ladder zenmetrics binary. Preserve duplicate
bitstreams; the existing instrument owner identifies saturated settings.
Use the frozen feature extractor and pin binaries, inputs and outputs by SHA.
Unknown decoder/feature provenance remains an explicit historical limitation.

The existing ladder builder, feature builder, TV pair builder, Rust trainer,
Rust packer and Rust BakeScorer evaluation remain the owners. Append new rows
and pairs without changing old row indices or old pair order. Record the actual
number of distinct floor settings and usable pairs; insufficient coverage is
a result, not permission to change the preregistered source selection.

Report all three paired seeds, CID22 held-out SROCC, the separately labeled
KonJND memorization guard, identity/negative tails, every codec floor gate,
research composite, elapsed training and artifact bytes. No winner is selected
from a single seed. A floor improvement cannot waive a rank or identity failure.
Retain the component if evidence does not justify its removal. This control
answers whether the missing codec's floor supervision helps; it does not claim
that eight references establish content-wide perceptual calibration.

## Actual target loop

Use the existing zensim-target controller, extended to call BakeScorer for
candidate models. Compare B, D and the fixed seed-4004 candidate on identical
held-out source bytes, codecs, target scores, tolerance and pass limits.
Record actual encoded bytes, achieved score, target error, passes and wall time;
measure fixed-pair scoring separately from encoding. Use explicit bounded
source and target manifests, retain failures and unreachable targets, and
report independent available judges separately. No interpolation of cached
scores is presented as an actual encoding loop, and missing judges or performance
measurements remain missing qualification evidence.

The operative cleanup plan's noninferiority/product gates still apply. These
experiments may close a hypothesis with a negative result; they cannot turn a
failed or unmeasured gate into a qualified model.

### Concrete target-loop inputs (registered before loop execution)

Use CID22 gold original PNGs `1025469.png`, `1044329.png`, `1189261.png`
(the first three 512-square references by pixel count/name) and
`adriankierman-report-page.png` (text/graphics coverage). No resizing. These
filenames are absent from both CID22's 201-reference training table and the
bigcodec training table. This is a four-reference diagnostic, not a powered
content-wide comparison. Compare current B, D and H_anchorlad seed 4004 at
JPEG/WebP/AVIF, targets -10/30/70/90/99, tolerance 1, budgets 3 and 8. The
current target adapters fix their other encoder settings; pin the complete
binary/dependency identities. Use SSIMULACRA2, Butteraugli pnorm3 and fixed B
as independent judges. No recipe/target adaptation follows these results.


## Completed results — September 7 local / September 8 UTC

All six registered H runs completed through the cleaned Rust trainer, packer,
declared-ID conversion and `BakeScorer` evaluator. The three unchanged H controls
exactly reproduce **every** stored `rank`, `dial`, `corruption`, `per_pair`,
`gates`, `composite`, and ladder `checks`, `measured`, `contract`, `regression`
field from September 6. Together with the separate three-seed A_plain replay,
this validates preservation of two competitive historical recipes after cleanup.
The numbers below are means ± **sample standard deviation across three seeds**;
they are not confidence intervals over content.

| Recipe | CID22 SROCC, 49 held-out references | KonJND overlapping memorization guard | Research composite |
|---|---:|---:|---:|
| A_plain unchanged | 0.889095 ± 0.002369 | 0.499735 ± 0.006024 | 0.872872 ± 0.000383 |
| H constrained control | 0.874350 ± 0.001800 | 0.498477 ± 0.025017 | 0.861040 ± 0.003165 |
| H + rav1e ladders | 0.875244 ± 0.000919 | 0.518783 ± 0.019913 | 0.863230 ± 0.001895 |

Paired CID22 deltas for seeds 4004/4005/4006 are +0.000094, −0.000951,
+0.003540 (mean +0.000894, sample SD 0.002350). This small screen does not
establish perceptual noninferiority or a robust improvement. No seed was selected
as a winner. H versus A is a joint architecture/loss/data change, not an isolated
ablation of the nonnegative-distance mechanism.

All six H runs pass C1–C6, including identity exactly 100, no grid cell above
identity, and working negative tails. **Every run still fails all five codec-floor
regression gates.** Mean represented-ladder fractions and the fixed mentor bars:

| Codec | H control | H + rav1e | Required mentor fraction |
|---|---:|---:|---:|
| rav1e AVIF | 0.1624 | 0.1795 | 0.6410 |
| SVT AVIF | 0.8120 | 0.8547 | 1.0000 |
| JPEG | 0.5214 | 0.5556 | 0.6667 |
| JXL | 0.3718 | 0.3718 | 0.9615 |
| WebP | 0.8974 | 0.8974 | 1.0000 |

Thus the missing eight-reference rav1e supervision is insufficient to repair
floor addressability. This result does not show that broader coverage cannot
help, nor that the head/TV loss is unnecessary. Preserve useful components;
do not launch another capacity/hinge sweep on this evidence.

The fixed control and all three added-data candidates also completed all 27
coherence cells through `run_full_eval.sh`: M3a = 0.852522 for H control seed
4004 and 0.830485/0.837941/0.837167 for H + rav1e. Those are measurements,
not replacement qualification bars. Complete seed panels and floor checks are
in [the result JSON](cleanup_scientific_controls_2026-09-07.json).

### Data and reproducibility boundaries

Artifacts are under `/mnt/v/output/zensim/cleanup-floor-control-2026-09-07/`.
`ENCODING_INPUTS.json`, `TRAINING_INPUTS.json`, `instruments/CONTROL_MANIFEST.json`,
trainer-embedded input manifests, argv/logs and `INPUTS_AFTER.json` retain actual
identities. Registration commits are `11123107b26a` (control) and `ab14a6787361`
(target sources); training/packing/evaluation binaries are frozen at `9879bbfd53b8`.

All 280 encodes and extractions succeeded (8 sources × 35 settings). Drop the
8 identity rows; append 272 codec rows after the old 4,520, giving 4,792.
Use unclamped `ssim2_gpu / 100` as `human_score`, `ref_basename = image_id`, and
preserve the old Arrow table prefix exactly. The old 204,746-pair TV stream was
reproduced **byte-exactly** with its owner; append 2,010 new pairs, including
900 repeated floor-window pairs, for 206,756. New indices start at 459,671;
old row/pair order is unchanged. No new SVT/JXL data are synthesized.

Source origin splits are train; the recorded nearest CID22 gold dHash distances
are 20–24, beyond the audit's flag threshold of 10. The feature extractor is a
frozen historical binary with **unknown build commit**, pinned by SHA. This
is an explicit historical control, not a newly qualified feature/decoder era.
As a bounded compatibility check, the frozen extractor re-extracted both
retained endpoint settings for each of the four old codecs on reference 7004:
all 8 × 372 stored feature values match exactly. `old-feature-parity/` contains
the keyed pairs, CSV and result. This does not recover the unknown build commit
or establish every historical decoder path.
The A_plain pre-launch manifest alone did not include H-only TV/anchor files:
those are checked against actual trainer manifests, historical known hashes and
the exact TV reconstruction. Do not describe that retrospective check as a
complete external pre-launch pin. The shared post-C versus rev2 decoder warning
still applies. Historical KonJND/KADID overlap remains a guard, not held-out data.
The final hash audit checked 449 path/hash records: all data, frozen binaries
and 360 target bitstreams match. Its sole changed source is the recorded
`bestofall_wave.sh` densify-failure handling/comment update; training and packing
arguments are unchanged.

Each H training fit took 173–233 seconds and peaked at 3.88 GiB RSS under the
16-GiB/eight-job cap. Whole three-seed train/pack/eval waves took 609 and 675
seconds. Packed models are 58,614–62,298 bytes. Runs overlapped other activity;
these are execution costs, **not** evidence of a training speedup.

### Actual target loops and compute decisions

See [the 360-cell target-loop record](cleanup_target_loop_2026-09-07.md).
It measures emitted bitstreams and final reconstructed images through the
Rust surface, with independent judges. It establishes no new qualified common
dial and no encoder-RDO benefit.

The bounded implementation/compute review is closed for this cleanup:

- C/CHdr have a tested canonical read-set planner and explicit IDs; two obsolete
  planners are removed. The activity correction and 984-case census are in the
  [feature-plan record](feature_plan_cleanup_2026-09-07.md).
- Buffered/full-feature and cheap shared accumulations retain real callers,
  geometry/HDR semantics and numerical distinctions. Their wholesale retirement
  lacks replacement/performance evidence. The contended extraction run is
  excluded; coefficient deletion is not a compute saving.
- The H constraint/data control preserves identity but loses rank versus A and
  still fails floors. Removing optional losses, transforms or heads has not
  earned noninferiority, complete floor coverage or independently judged target
  behavior. Keep them; low-value feature ablations stay deferred under the user's
  priority ruling. No unperformed removal experiment is marked as a success.
- Independent Python numerical references and unique live hypotheses remain.
  Obsolete alternate implementations are retired only with the caller/parity
  evidence linked in the execution plan. Canonical Rust ownership alone is not
  proof of correctness.

The cleanup is successful training/instrument maintenance. Product qualification
remains failed/incomplete for these historical candidates and must be earned by
a subsequent source-disjoint model with measured product gates.
