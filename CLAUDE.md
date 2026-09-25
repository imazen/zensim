# zensim — current working rules

## Controlling user priorities — September 15, 2026

Follow [the production execution plan](docs/PRODUCTION_PRIORITIES_2026-09-15.md)
before any conflicting older task list or paused goal: close evidence gaps;
fix attainable targeting tails; recover local ranking on TRAIN; establish JXL
spatial value; train correct native HDR; qualify the integrity head; finish
runtime and product gates. Preserve the fast Rev3 and richer controls and the
completed recovery work. Use the existing owners and tight local checks.

The [Squintly study handoff](https://github.com/imazen/squintly/blob/main/docs/STUDY_READINESS_2026-09-15.md)
owns the separately assignable paid-human task. Consult `../zenpapers` and
primary sources; qualify phone color and unsmoothed pixel zoom, distinguish
normal and magnified judgments, and require verified display behavior for HDR.
Do not start full paid collection from the historical September 1 protocol.
These are the user's current work priorities; they do not weaken data-split,
mathematical-correctness, provenance or registered qualification requirements.

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
references. Model comparisons lead with the registered composite coverage,
full rank/band/within-image panels, scatter geometry and tail/dial/spatial gates.
Raw MAE is an auxiliary calibration diagnostic, never a replacement for that
selection evidence. Full-population scatter diagnostics come from
`zenstats::scatter` through `bake_verdict`/`panel`; plot sampling and rank
normalization must not hide raw density, tails or saturation. Missing evidence
is INCOMPLETE, not a pass. Canonical ownership is a maintenance decision, not proof of
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

**September 14 user clarification:** train-only for training, transforms,
calibration, feature/hyperparameter and checkpoint selection. Use EVAL when
available; when no EVAL exists, published TEST may assess frozen candidates
with recorded exposure and guards against adaptive tuning. CID22's oracle
training references and human test references are distinct. Secret holdouts
remain untouched. Preserve original roles rather than renaming TEST to EVAL.
Use explicit split admission; the legacy checker scans terminal tables and is
not an allowed default for this work. See the current DATA_SPLITS override.

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

## Known Bugs

* **2026-09-25 — bulk sRGB→XYB gives different bits for the same colour in the vector chunks and in the scalar
  remainder. OPEN; the fix needs a formula-revision decision.**
  - `color::srgb_to_positive_xyb_planar_into` converts each band of `rows × width` pixels in one call. Full
    vector chunks (16- and 8-wide) use magetypes' `cbrt_midp` (seed `0x2a508c2d`, unfused Halley steps). The last
    `rows × width mod 8` pixels use `color::cbrtf_fast` (seed `709_958_130`, fused Halley steps). On the scalar tier
    the chunk matrix is also unfused, because magetypes' scalar `mul_add` is `a * b + c`.
  - Measured over all 2^24 sRGB8 colours with the real owner
    (`cargo run --release -p zensim --example xyb_chunk_tail_parity`): the XYB output differs for **84.600 %** of
    colours on x86_64 AVX-512 (per channel X/Y/B 10,857,190 / 8,584,754 / 9,519,698) and **85.766 %** on i686,
    which runs the scalar tier (11,023,236 / 8,768,277 / 9,698,716), by up to **196 ULP**. At most 7 pixels per
    converted band are affected.
  - Consequence: a flat image is not flat after conversion. `feature_v2::tests::gmsbank_constant_chroma_shift_is_visible_without_gradients`
    fails on i686, where the XYB conversion always runs the generic scalar variant because its v3/v4 variants are
    compiled only for x86_64. It passes on x86_64 and aarch64 only because its two test colours shift by the same
    1 ULP in Y, so the reference and distorted artefacts cancel.
  - Any fix (pad the remainder into one more vector chunk, or one cube-root and `mul_add` form on every path)
    changes feature bits for images whose band pixel count is not a multiple of 8. So it is an arithmetic-revision
    decision, not a silent patch.
* **2026-09-25 — the C8 landing left `main` with an unparseable `zensim-bench/Cargo.toml` and two failing C8
  tests; a C1–C4 identity test was also racing. FIXED** (this commit).
  - The landing rebase kept both the E5A and the C8 `gmsd` entries in `[dependencies]`. TOML rejects a duplicate
    key, so nothing in the standalone `zensim-bench` workspace built, and root CI could not see it. One entry now
    remains, with default features off, so the C8 peer scorer keeps the no_std sqrt that produced the stored
    peer columns; `e5a-render` and `gmsd-arm` enable `gmsd/std` themselves.
  - `research_everything_agrees_with_the_production_walk` compared a 1322-wide production walk against the
    1502-wide `research::full_width()`. It now enables `gmsbank` and also checks the research width.
  - `gmsbank_contrast_reduction_has_exact_zero_gain_in_every_tier` still walked the pre-chroma 12 × 15 layout, so
    it read a `cs_dev` slot as a gain slot. It now follows the landed `GmsbankChroma` layout: scale-0 Y, then
    X/Y/B cells plus 10 `cs_*` slots at scales 1–3.
  - `rev4_synthetic_16_pair_identity` (C1–C4, failing on macOS Intel CI before C8 landed) was a test race, not an
    extraction defect. The permutation tests in the same binary disable SIMD tokens process-wide, so its on and
    off walks could run at different tiers. Measured: 3 failures in 4 parallel runs, a different slot each time;
    0 serially. It and `rev4_corpus_toggle_identity` now hold `archmage::testing::lock_token_testing()`; 0
    failures in 12 parallel runs.
* **2026-09-23 — AVIF RGB16→RGB8 in zenmetrics' sRGB-tagged decode route rounds down at near-half levels. OPEN
  (upstream).**
  - The Opus review (`REVIEW_AVIF_DECODE.md`, recorded in `benchmarks/rev4_avif_decode_diff_2026-09-23.md` after its
    correction) measured it on the same zenavif source.
  - The native RGB16 output is identical on both routes.
  - The **untagged route** equals exact `round(v10·255/1023)` on 60,349,731 of 60,349,731 channel values.
  - The **sRGB-tagged `RowConverter` route** (zenpixels-convert, used by the Sept-14 SafeSyn extraction) is −1 on
    96,160 values (0.16%), at near-half levels such as 169.5015→169.
  - **Consequence:** the stored Sept-14 SafeSyn AVIF features and labels sit on the inexact side. Measured on 31
    pairs: SSIMULACRA2 mean +0.0023 (max ±0.085), zensim B mean −0.0007.
  - Separately, zenmetrics drops the signalled transfer tag (a metadata defect).
  - Rescoring or re-extracting AVIF therefore changes values. Pin the decode route per data era, and never mix routes
    in one table.
  - The fix belongs to zenpixels-convert and zenmetrics, not this repo.

* **2026-09-18 — identity disagrees between scoring paths. FIXED** (this
  commit). On a byte-identical pair `Zensim::compute` returned exactly 100
  (identity short-circuit) while `Zensim::compute_with_diffmap` returned
  96.2017 (Profile B, 900×675), `Zensim::compute_streaming_strips` returned
  96.2368 (129×128) and `compute_folded944_score_and_attribution{,_binned}`
  returned 96.2368. All now delegate to `compute`, which owns the single
  identity check (`metric::images_byte_identical` → `identical_result_at`); the
  diffmap is exactly zero. Pinned by
  `diffmap::tests::identity_agrees_across_every_scoring_entry` and
  `one_lsb_difference_is_not_short_circuited`. **Still open by construction:**
  every `*_with_ref*` entry takes an XYB pyramid, not the source pixels, so it
  cannot reach the identity owner and scores a perfect copy through the model.
  `diffmap::tests::identity_is_undecidable_without_the_source` pins that.
* **2026-09-18 — `BakeScorer` scores a Rev3 bake at the process revision without
  refusing. NOT A BUG — the report was wrong; entry corrected, not deleted.**
  The claim was that with `ZENSIM_FORMULA_REV` unset (a Rev1 process) the narrow
  basic/peak plan serves the frozen R915 Rev3 bakes at the wrong arithmetic.
  Measured 2026-09-18 on all ten `/var/tmp/zensim-validation-2026-09-15/recovery/
  calibrated/R915_*.bin` (each of which DOES declare `zentrain.formula_revision
  = 3`; a raw byte grep misses the key, `zenpredict::Model::metadata` is the
  only reliable reader): every score is **bit-identical** with the variable
  unset, `=1` and `=3`. Reconstructed Y60 and basic228 bakes over the same ids
  show the same invariance, and a Rev1-declaring bake over identical ids scores
  differently in the same process — so the narrow route SELECTS the declared
  revision rather than ignoring it. The `metric/bake.rs` early return (~862-884)
  is that selection; the ~885-902 refusal is the wide-family path, which still
  refuses a mismatch. `ZENSIM_FORMULA_REV` is NOT a serving requirement for a
  declared-revision bake. Pinned by
  `metric::bake::revision_contract_tests::narrow_plans_serve_the_declared_revision_in_every_process`
  (Y60 + basic228 + `compute_hdr` + `prepare_steering`, across Rev1/2/3 child
  processes) alongside the pre-existing
  `prepared_workers_honor_model_revision_and_local_subset`. A bake that declares
  NO revision still resolves to `SHIPPED_REVISION` (Rev1) in every process —
  never to the process revision — and an unknown revision is refused at load.
  If the speed-matrix run really saw two different numbers, the cause is
  somewhere other than revision selection; re-open with the two scores and the
  bake sha256.
