# Start here — one target score, one development path

Reviewed September 8, 2026. [CLAUDE.md](CLAUDE.md) contains current working rules;
[WAVE_PLAYBOOK](docs/WAVE_PLAYBOOK.md) contains the tool map and experiment cycle.
Older operational notes are in [docs/history](docs/history/).

Latest September 8 continuation: [canonical corruption input generation](benchmarks/canonical_corruption_2026-09-08.md)
is complete: 12 train / 8 validation origins, 14,300 retained catalog/anchor
rows plus 760 current-native honest JXL/AVIF feature rows. No new fit or model
qualification. The September 6 nonlinear HGB already superseded the older
linear separability/guard hypothesis; reuse its existing estimator/exporter
and Rust ZCTH serving. Before fitting, deduplicate source/pixel pairs (raw
catalogs still fail C10), complete T0 content audit and public-surface feature
parity, and register exact fit/calibration/evaluation source identities. The
corruption gate's discontinuity and current native adapters' explicit rejection
of corruption companions remain product work. The prior native intervention
and RD findings remain negative/mixed; do not substitute corruption detection
or restoration coherence for useful allocation evidence.

The end user controls one target score. The codec chooses parameters and must
reach useful quality across codecs/content, near-lossless settings and codec
floors with few passes, small output, low latency and low memory. Negative
scores are valid. A rank correlation or research composite cannot establish
that behavior. B remains `codec_target()`; D is the explicit fast profile.
B/C/D may be replaced: there are no consumers calibrated to their current
scores. [The integration guide](docs/CODEC_TARGET_METRIC.md) owns the mapping.

Every new model must execute and serve entirely in Rust through a zensim API,
and evaluation must call that API, including heads, corruption and splines.
`BakeScorer` now provides that dynamic surface. Python remains useful for
invention and independent numerical references.

| Question | Current answer / evidence |
|---|---|
| Can we train after cleanup? | Three A_plain seeds exactly reproduce all measured verdict and ladder fields. Three H_anchorlad seeds also reproduce every earlier verdict and ladder field; three paired rav1e additions complete the floor-data control. [Reproduction](benchmarks/cleanup_training_reproduction_2026-09-07.md); [paired controls](benchmarks/cleanup_scientific_controls_2026-09-07.md). These historical recipes have documented split limits. |
| How does target search behave? | [2,730 witnessed-target cells](benchmarks/target_steering_bounds_2026-09-08.md) compare train-calibrated 1/2/3-shot policies, with exact final-code reproduction. The [360 older loops](benchmarks/cleanup_target_loop_2026-09-07.md) mixed feasible/unwitnessed targets; their median cannot establish model failure. [Native JXL targeting](https://github.com/imazen/jxl-encoder/blob/main/benchmarks/zensim_native_targeting_2026-09-08.md) now measures three arms with shared Rust calibration/search: 810 cases, 45/80 jointly witnessed targets, all arms hit ±1 in three shots on that subset. Coverage and independent RD still prohibit qualification. |
| Does native attribution already exist? | Yes: JXL fused/binned/stale H3, a recovered JPEG `jj` experiment, WebP segments and the later AVIF CQ/H3 experiment. [Fleet/source/chronology audit](benchmarks/diffmap_reuse_audit_2026-09-08.md): reuse them. The binding now exists: `BakeScorer::compute_with_ref_and_attribution` uses complete scoring and declared extraction, with explicit spatial coverage. JXL now consumes it in [commit 9f038d4d](https://github.com/imazen/jxl-encoder/commit/9f038d4dd7f4); 840 ladders and decoded map engagement pass, but D/H3 has no broad RD win. [Native record](https://github.com/imazen/jxl-encoder/blob/main/benchmarks/zensim_candidate_binding_2026-09-08.md). [Serving/coherence evidence](benchmarks/candidate_attribution_serving_2026-09-08.md): exact pixel parity, 15 serving tests and 54 coherence cells. D is the first native baseline; native engagement and product qualification remain incomplete. |
| Is good rank enough? | No. A_plain retains identity and all-codec floor failures. Qualification reports failure/incomplete evidence separately from research selection. No new common-dial winner is established. |
| Can candidates serve their whole model? | Yes, through `BakeScorer`: declared IDs, validated metadata, heads, pin, spline, codec affine, ensemble and corruption composition. [Execution plan](docs/PLAN_CRUFT_PURGE_2026-09-06.md). Pixel luminance revision must still match the process. |
| What happened to C/CHdr? | Both use explicit IDs and canonical activity semantics. The old training/serving mismatch is fixed; two alternate planners are retired. Pixel scores intentionally change. [984-case feature census and HDR/matrix evidence](benchmarks/feature_plan_cleanup_2026-09-07.md). |
| Where do I train and evaluate? | `zensim_mlp_train` → `bake_dial_refit` → `run_full_eval.sh` → `freeze_check`. Admission now checks actual headers, IDs and declarations before work. [Trainer cleanup](benchmarks/trainer_admission_cleanup_2026-09-07.md). |
| What owns calibration? | Rust `bake_dial_refit`, with pre-calibration coordinates and final full-surface evaluation. Four historical writers are retired with recipe/boundary evidence. [Spline record](benchmarks/spline_owner_cleanup_2026-09-07.md). |
| Which data are valid? | [DATA_SPLITS](docs/DATA_SPLITS.md), later [DATASET_HISTORY](docs/DATASET_HISTORY.md) entries, actual manifests and the [shared index](../DATA_PROVENANCE.md). Width is not an era. TID is train-only; historical KonJND/KADID overlaps are guards, not holdouts. |
| Where is the board? | `scripts/v_next/gauntlet.py` renders stored owner verdicts to `/mnt/v/output/zensim/reports/summer_gauntlet{_fair}.html`. Codec-q score charts include negative scores; product qualification appears before composite. |
| Why keep buffered/full features? | Remaining callers and numerical differences prohibit blanket retirement. Free/shared features are not removed on coefficient counts. The contended extraction benchmark was excluded, not presented as a speedup. |

The authorized cleanup is complete: [execution and validation record](benchmarks/cleanup_completion_2026-09-07.md); [original checklist](docs/PLAN_CRUFT_PURGE_2026-09-06.md).
The [transcript/memory audit](benchmarks/science_workflow_audit_2026-09-07.md)
records chronology and bounded confidence in Rust/Python equivalence. Scientific
controls and actual-loop measurements have their own result records; a passing
code migration cannot inherit an unmeasured product gate.

Use the existing capped `run-heavy` owner for heavy work and
`scripts/safe_push.sh` for every push. Data/bake evidence is preserved under
`/mnt/v/output/zensim/cleanup-*-2026-09-07/`; private audit and retired sources
are under `~/tmp/zensim-science-audit-2026-09-07/`. Choose fresh output paths
for reproductions. Do not overwrite historical results or infer completion
from a filename without its content identity.

September 8 repository sync + AVIF binding continuation: all 75 immediate zen
repos plus five external dependency/corpus repos fetched; 52 checkouts advanced
or rebased, local changes rescued and five conflict sets resolved. Exact private
recovery/status: `~/work/zensim-recovery-2026-09-08/sync-all/README.md`. Keep
syncing before new work and use tight local checks; do not wait on CI.

AVIF complete-candidate research binding pushed as `4777a30aa54aaab6534719f6026e4a18e91d60e3`. Exact-bake
`BakeScorer` owns scalar and current spatial scores; first map feeds encode 2,
unsupported terms fail, terminal bytes are decoded/scored independently. On the
one training-family control, legacy gain 10 is inert (all blocks saturate then
normalize to 1); existing zerosum engages and repeats byte/pixel/map-exactly.
Neutral 79.368/18,355 B; active 80.287/19,573 B; three full encodes, three maps,
four scalar comparisons including terminal. Six CLI rejection controls pass.
See sibling `benchmarks/zensim_avif_loop_2026-08-07.md` September 8 addendum and
`zensim_avif_candidate_binding_2026-09-08.json`. Artifacts at
`/mnt/v/output/zensim/avif-candidate-binding-2026-09-08/`; Windows-readable copy
under `~/work/zensim-validation-2026-09-08/avif-candidate-binding/`.
This is engagement only, not RD or competitive-model qualification. Next: AVIF
attained bounds and train-only calibrated actual 1/2/3-encode loops through the
shared Rust search, then matched-quality judging. No new model training or
terminal holdout was performed in this continuation.

Shared native probe owner added after the AVIF binding: unpublished tool feature
`zensim-target/native-probe`, with codec-owned adapter for AVIF. See
`benchmarks/native_probe_instrument_2026-09-08.md`: 51-bound/119-encode train and
validation pilots, 54 target cases, source/model/driver/family rejection checks
and independent decoded-pixel integrity checks pass. AVIF's stateful map uses
the previous complete decode; its first shot has no consumed map. Full 12/8
family matrix is next, then model qualification and native JPEG/WebP work.

Later September 8: the full AVIF 12-train/8-validation matrix and independent
judges are COMPLETE. [Instrument record](benchmarks/native_probe_instrument_2026-09-08.md)
links the codec report: 504 cases, 28/80 joint witnesses; three-shot calibrated
scalar/neutral 28/28 hits ±1 versus active 24/28. Fixed-CQ active maps modestly
help SSIMULACRA2 but worsen Butteraugli in every content class; no spatial
benefit or model qualification. Artifact `RESULT_COMPLETE.json` supersedes
the launch-time pending text in `SOURCE_PINNED.json`. Judge-pair identity,
duplicate/missing controls pass, with JXL/AVIF numerical results preserved.
The September 7 H+rav1e floor-data control already completed and failed all five
floor gates; do not repeat the older September 6 proposal as new work. Continue
with useful native allocation/model qualification and JPEG/WebP integration.

Later September 8 JPEG continuation: complete-candidate binding now runs in the
existing Zq loop (`__zensim-research`, recovered `zq_rd_probe`). The first trace
exposed old AQ defects: unit scales clamped real strengths to 0.20 and the last
strip skipped the controller. Both are repaired, along with a fractional-q
clamp panic near 100. [JPEG record](https://github.com/imazen/zenjpeg/blob/main/benchmarks/zensim_candidate_binding_2026-09-08.md):
one train family, exact D, 444/q80; scalar and neutral 77.438789 / 16,864 B,
active repeat 77.314751 / 16,795 B. All 832 blocks controlled; current maps,
JPEGs and pixels repeat exactly. SSIMULACRA2 and Butteraugli both show the small
quality loss, so no RD benefit is claimed. Eighteen Zq tests pass before/after,
22 AQ tests, two Clippy routes, API snapshot check and ten rejection controls
pass. Artifacts: `/mnt/v/output/zensim/jpeg-candidate-binding-2026-09-08/final/`;
Windows bundle under `~/work/zensim-validation-2026-09-08/jpeg-candidate-binding/`.
This is fixed-seed engagement, not calibrated targeting: two corrections cost
three full encodes, and its existing `targets_met` flag checks only the score
floor. Reuse the shared Rust native owner for actual JPEG 1/2/3-shot bounds and
train calibration; WebP binding and competitive model/spatial qualification
remain incomplete. No training or terminal holdout was performed in this step.


Later September 8 WebP continuation: [complete-candidate binding and accounting
repairs](https://github.com/imazen/zenwebp/blob/main/benchmarks/zensim_candidate_binding_2026-09-08.md)
are pushed in `8aa8a7858b97`. The existing segment loop now accepts the exact Rust
bake through `__zensim-research`; registry 0.2 and the separate recompress A
calibration remain intact. One-pass targeting is measured and strict, encode
counts are actual, ship-band flags are truthful, and finite negatives are valid.
All five old failures were observed in regression tests before repair; hard-floor
selection has its own observed failing control. Local 354 library, 15 targeting,
33 validation tests, two Clippy routes, release/API/format checks pass.

One canonical training family, D/q80/m4: scalar and neutral select 73.673485 /
16,040 B; active selects 72.955315 / 15,728 B. The fixed-q neutral correction is
byte/pixel/map exact; active quantizer 11→14 is consumed and repeats exactly.
All 208 macroblocks, including the partial edge, have explicit map/segment traces.
Both independent judges see quality loss with the smaller active file. Thirteen
full encodes, nine maps, two non-neutral map uses across active+repeat and five
terminal comparisons are counted; ten CLI rejection controls pass. Full result:
`/mnt/v/output/zensim/webp-candidate-binding-2026-09-08/`; Windows copy under
`~/work/zensim-validation-2026-09-08/webp-candidate-binding/`.

All four codec owners now have an exact complete-candidate binding and native
engagement evidence. Useful spatial RD, a competitive qualified model, JPEG/WebP
train-calibrated witnessed bounds and actual 1/2/3-shot validation remain required.
Do not repeat binding screens as model qualification. No training or terminal
holdout was performed in this continuation.

Later September 8 native JXL finite-block continuation:
[intervention record](https://github.com/imazen/jxl-encoder/blob/main/benchmarks/zensim_native_interventions_2026-09-08.md).
Codec instrument/report pushed as `feab1d7fd734c89212b449a1945cf1e6e7bf81e4`.
Four train families, distances 1/3, 16 native transform regions with ±10%
quantizer changes: 272 complete probes reproduce bytes/pixels/scores/maps and
both independent judges exactly. Separate 272-probe 512-long-edge multi-group
coverage passes; 48 total libjxl 0.12 compatibility checks complete. The existing
RD analyzer's new `--interventions` mode passes 19 rejection controls, preserves
older analysis functions, and verifies all source/byte/pixel/quantizer/judge
identities. Six CLI controls and local release/Clippy/format/script checks pass.

Restoration coherence does not establish useful native marginal allocation:
primary map-mass/D-response rank associations range from −0.259 to 0.676.
222/256 interventions affect pixels outside their selected region even though
captured quantizer changes stay local. Seven interventions preserve the final
quantizer field but change pixels; native thresholds also depend on the incoming
field. CfL is frozen here. Do not treat final quantizer identity as a no-op or
these response secants as isolated quantizer gradients. No model/policy change
or qualification. Artifacts: `/mnt/v/output/zensim/jxl-native-interventions-2026-09-08/`;
Windows copy under `~/work/zensim-validation-2026-09-08/jxl-native-interventions/`.
Next spatial hypothesis: preregister a coarse-region/rate-cost intervention
before tuning allocation. Model qualification and JPEG/WebP actual targeting
remain open; no new training or terminal holdout in this packet.
