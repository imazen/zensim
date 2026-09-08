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
| How does target search behave? | [2,730 witnessed-target cells](benchmarks/target_steering_bounds_2026-09-08.md) compare train-calibrated 1/2/3-shot policies, with exact final-code reproduction. The [360 older loops](benchmarks/cleanup_target_loop_2026-09-07.md) mixed feasible/unwitnessed targets; their median cannot establish model failure. Native diffmap qualification remains separate. |
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
