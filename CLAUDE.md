# zensim — current working rules

Start with [SESSION-RESUME.md](SESSION-RESUME.md), then the
[wave playbook](docs/WAVE_PLAYBOOK.md). This file was consolidated on September
7, 2026. The [dated instruction archive](docs/history/CLAUDE-through-2026-09-07.md)
preserves prior findings and recipes; later dated corrections and the current
implementation take precedence over an old status paragraph.

## Product and implementation

The end user controls **one target score**. Quality includes content/codec
consistency, identity and near-lossless behavior, negative tails and distinct
codec floors, target accuracy, bytes, passes, latency and memory. Rank and a
research composite alone cannot establish a useful dial. B is currently
`codec_target()`; the [integration guide](docs/CODEC_TARGET_METRIC.md) owns the
profile mapping. No consumers are calibrated to B/C/D; they may be improved
or replaced, with explicit evidence and versioned experiment identities.

A new model must execute and serve entirely in Rust through a zensim surface
API. Evaluation must call that API, including every head, corruption gate,
spline and ensemble/routing step. `BakeScorer` is the dynamic candidate surface;
`Zensim` owns named profiles. Python is useful for invention and independent
references. Canonical ownership is a maintenance decision, not proof of
correctness. Keep independent numeric references with real parity gates.

Before adding an implementation, find its existing owner in the playbook and
extend it. Do not create a second scorer, calibrator, statistic, report
pipeline or controller under a new filename. A gated mirror must state its
independent contract, test real implementations and include negative controls.
Feature ablation is low priority unless it removes actual extraction work or
settles a named consequential scientific question. Consult `../zenpapers`,
especially `docs/zensim-720-feature-gaps-2026-07-26.md`, and later summaries
before fundamental feature/model changes.

## Workspace and changes

Follow [the parent AGENTS.md](../AGENTS.md). Work in this existing checkout;
no new worktrees, cleanup of another lane, or reverting unexplained dirty files.
Preserve user changes and immutable experiment evidence. Sibling repositories
are separate projects; do not modify them incidentally. Find instructions in
the target project before any separately authorized sibling work.

**Push only through `scripts/safe_push.sh`**, including explicit revisions:
`scripts/safe_push.sh -r <reviewed-revision>`. It fetches, requires ancestry,
checks outgoing identifiers, pushes and verifies. Never bypass its gates with
bare `jj git push`, bookmark pushes or force. Resolve divergence by inspecting
and preserving both histories. Keep private fleet addresses/identifiers in
private configuration; `just lint-scripts` checks tracked files too.

Before new public items or intentional API changes, preregister the concrete
caller and exact delta. Prefer private items; preserve supported APIs unless
the user authorizes a change. Update changelog and API snapshots together.
Run `just api-doc-check` for public/hidden-public edits; its nightly pin is
shared with CI. Run appropriate semver checks for supported API/release work.
`zensim` and `zensim-regress` have separate semver; validation and target tools
are unpublished. Publishing requires explicit authorization, passing checks
and a matching pushed tag; this cleanup does not authorize a release.

## Scientific data and reproducibility

Before training, baking, feature extraction or picker work, read
[`../DATA_PROVENANCE.md`](../DATA_PROVENANCE.md),
[DATA_SPLITS](docs/DATA_SPLITS.md), the relevant later entries in
[DATASET_HISTORY](docs/DATASET_HISTORY.md), and the recipe's file manifests.
Pin original bytes, decoder commits, extraction/formula revision, feature-set
identity, input hashes/order, source splits, seed streams and tool binaries.
Never infer semantic compatibility from column count or a familiar path.
The September 6 safesyn AVIF decoder era is incompatible with R6b inputs.

CID22 human scores and its 49-reference gold set are holdout-only. Metric
labels on the separate training references do not authorize human-label use.
AIC-3/AIC-4 and other T0 content stay holdout-only. Use current reference-level
splits, including KonJND/KADID; historical overlapping rows are integrity or
memorization guards, never held-out evidence. TID is train-only under the
later August 29 ruling. The canonical origin split is owned by
`../zenmetrics/scripts/picker/origin_split.py`; derivatives inherit the origin.
Use the existing near-duplicate audit; dHash flags need contextual review,
not automatic quarantine. Historical replay records limitations explicitly and
cannot qualify a new model.

Feature IDs and arithmetic revisions are explicit. `feature_defs`,
`feature_set_id` and `Plan::for_bake` own feature families/layout/planning;
validation's registry describes historical producers. No silent truncation,
zero fill or width-based family inference. Dense/gapped tables are refused by
loaders that do not support them. A bake's known primary leg cannot hide an
unknown companion or training leg. Pixel scoring currently requires the
process luminance revision to match the bake; do not silently mix revisions.

Use ZNPR v3 via `zenpredict-bake` and the Rust pack/refit owner; do not introduce
v2 writers or hand-serialize model layouts. Keep the pinned git/path dependency
for zenpredict; do not substitute the old crates.io v2-only implementation.
Quantize before calibration; evaluate final packed/declared-ID bytes. Removing
an exactly dead input differs from folding a corpus-constant input; preserve
those distinct guarantees. Emit structured manifests, not TOML string surgery.
Use Parquet for large tables; preserve row keys through joins and extraction.

## Compute and verification

Run heavy builds, tests, training, extraction and benchmarks through
`~/work/zen/scripts/run-heavy --mem 16G --jobs 8 ...` with workload-appropriate
explicit caps. Current LAN/local execution supersedes July Hetzner-first
recipes. Reuse verified binaries when only consuming results; record their
hashes. Do not create duplicate builds or leave failed chains marked complete.
Use the existing harvest/await owners for detached chains; final artifacts
and failure sentinels, not an agent notification, determine completion.

Benchmarks need a quiet machine, pinned geometry/threads/build, before/after
runs and dispersion. Record competing processes. Do not claim a speedup from
contended timing, coefficient counts, a narrower caller vector or a smaller
loop bound that still packs the whole image. Preserve buffered/full-feature
paths until actual callers and discriminating parity/performance gates justify
retirement. Planner evidence is in the September 7 feature-plan record.

Run tests appropriate to changed behavior, CI-exact `just clippy`, and
`just lint-scripts`. Product inference changes require feature-build serving
checks, canonical consumed-feature comparisons, candidate/pixel parity and
relevant HDR coverage. Use actual registered datasets and final bakes for
scientific claims. Full evaluation, selection and qualification are distinct
commands in the playbook. Failed, missing, stale and qualified evidence must
remain distinguishable in JSON and reports.

The authorized cleanup checklist and bounded acceptance evidence are in
[PLAN_CRUFT_PURGE](docs/PLAN_CRUFT_PURGE_2026-09-06.md). Record negative results
and unresolved product limitations; do not declare a model qualified merely
because cleanup tests or a historical training reproduction pass.
