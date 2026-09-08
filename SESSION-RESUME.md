# Start here — one target score, one development path

Reviewed September 8, 2026. [CLAUDE.md](CLAUDE.md) contains current working rules;
[WAVE_PLAYBOOK](docs/WAVE_PLAYBOOK.md) contains the tool map and experiment cycle.
Older operational notes are in [docs/history](docs/history/).

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
