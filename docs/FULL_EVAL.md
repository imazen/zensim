# Full-eval — one comprehensive Rust eval per bake → machine-readable JSON

## Strict train/eval feature screens (September 13)

The later [user split instruction](DATA_SPLITS.md#september-13-user-ruling-train--eval-only-never-touch-test)
forbids any test/terminal read, even for final qualification. The historical
v1 screen recipes and their fit/dev/test caches are no longer executable via
`feature_screen.py`. Preserve them as evidence; do not rename their segments.
Other historical commands below are not permission to scan terminal datasets.

Use `schema: zensim-feature-ceiling-recipe-v2` and
`split_policy: train-eval-only-v1`. Keep the registered feature IDs, seeds,
training budget and spatial thresholds explicit. Replace automatic corpus
discovery and `reuse_prepared` with `input_segments`, each containing:

```json
{
  "role": "train",
  "path": "/absolute/path/train-segment.json",
  "sha256": "<segment SHA256>",
  "admission": {
    "path": "/absolute/path/train-admission.json",
    "sha256": "<admission SHA256>"
  }
}
```

Supply train and eval segments for every requested task. An admission file has
`schema: zensim-source-admission-v1`, a named `authority` identifying the
canonical split manifest/rule and its revision/hash, and a `sources` array.
Each source records `corpus`, `origin`, globally consistent `source_family`,
and `split` (`train` or `eval`). Source-only sidecars must be reviewed against
the canonical authority before use. Hash/schema validation is not an independent
proof that a caller's source assignment is correct. Do not create admissions
by relabeling old screen/test segments.

Each segment file has `schema: zensim-feature-segment-v1`, `role`, and `rows`.
Each row contains `corpus`, `origin`, `source_family`, `task` (human/codec/
corruption), distortion `family`, `target`, and absolute `reference`/`distorted`
paths. Every row must match its source admission. Families cannot cross roles,
including across tasks. Protected path components and symlink targets are
rejected before hashing pixels. Unexpected roles or changed bytes fail.

Pin the `spatial_manifest` bytes with recipe field `spatial_manifest_sha256`.
The manifest must carry the same `split_policy`; its cases retain
their existing names/paths/hashes plus `role: eval`, `source_split: eval`, and
`source_family`. Every case must belong to an admitted eval reference. Old
training-origin spatial panels may remain historical diagnostics but cannot
become eval gates by changing their role field.

Stages use the existing Rust owners:

- `--ceiling-stage prepare`: admit explicit segments, extract fresh canonical
  features, write separate train/eval Parquets and an **eval-only** prediction
  buffer. No mixed historical cache is opened.
- `--ceiling-stage fit`: the trainer receives only train tables (or their
  train-only half/class views), with `--no-auto-eval`, no eval group and no
  early stopping. Its checkpoint monitoring falls back to training scores.
  No prediction/evaluation command runs in this stage.
- `--ceiling-stage audit`: freeze/check final bake identities, then run Rust
  prediction, raw-error panels, pixel parity and spatial checks on eval only.
- `--ceiling-stage report`: aggregate the stored eval results, retaining seed
  values, mean, median and spread. No eval-driven checkpoint/capacity refit is
  launched. `all` runs prepare, fit and audit; report remains explicit.

Resume uses the same v2 recipe, input and tool identities. Old cache reuse and
the historical `checkpoints` follow-up are refused. Scalar serving, model
defaults and feature arithmetic are unchanged.

Validation: `python3 scripts/tests/test_feature_screen_splits.py` exercises
thirteen synthetic admission/routing boundaries, including refusal before file
opening. A native Rust smoke with 16 train and 8 eval fixture pairs completed
prepare, train, audit and report; the trainer's only group was `human_train`,
all external panel labels were eval, and consumed-feature pixel parity was
exact. Two eval-source spatial cases ran. The first smoke exposed integer CSV
target inference; targets are now explicitly cast to floating point before
training. Failed and corrected runs remain under
`~/work/zensim-validation-2026-09-13/split-boundary-smoke/`.
These generated software fixtures are not scientific corpus or model-quality
evidence. No corpus test segment was opened for this change.

The additional `run-alltasks-full` smoke covers all three training objectives
and their eval-only audits. The tiny optional half/class fixture failed C6
because some features were constant in that four-row subgroup; those refusals
are retained, and the gate was not relaxed. Preparation now constructs half
tables only for requested half-data controls. Synthetic 0/100 fixture labels
test routing only; they are not corruption severity labels or a fitted product
catcher. The `recipe-alltasks-full.json` file pins the final multi-task smoke.

**September 7 scoring update:** all candidate scores are returned by
`zensim::BakeScorer`. The verdict's `scoring` block records the surface version,
member hashes, blend weights and corruption-head hash/deadband. A supplied
corruption head now affects rank and dial scores as well as its auxiliary
report. Earlier reports remain historical instruments. Cached-row evaluation
requires the recorded extraction/decoder era; byte-format validation is shared
with serving. Pixel verification uses `serve_custom_bake` without static
loaders or leaked models.

`scripts/run_full_eval.sh` runs the **offline evaluation and coherence measurements** through the canonical
Rust owners and emits one machine-readable JSON per bake. No Python touches any
statistic — every number comes from the Rust binaries that already own it
(`bake_verdict` → `zenstats::panel` for the rank+dial+corruption math,
`diffmap_block_coherence` for M3). The JSON is the input the summer-gauntlet
dashboard consumes for its scatter/scorecard panels.

**Scope correction, 2026-09-07:** this wrapper does not run the real-codec
G-RD/G-TARGET legs. Its output also does not establish G-ADDR qualification
unless the compatible ladder, negative-tail and identity measurements are
present and pass. Use [`MODEL_SELECTION_SCORECARD.md`](MODEL_SELECTION_SCORECARD.md)
for the complete product exam. The September board's operative addressability
block is `dial_ladder`; `dial.curves` still describes its canonical grid.

## Usage

```sh
scripts/run_full_eval.sh [--stage all|verdict|coherence|qualify] <bake.bin> <name> [regime=720] [features-root]
```

- `<bake.bin>` — a ZNPR v3 bake; the scorer gathers its declared feature IDs.
- `<name>` — the label embedded in the JSON and used for the output filename.
- `regime` — `720` (default), `372`, `924` or `944`; a legacy evaluation
  preset, not a feature identity. Since September 5 the wrapper resolves the
  feature root through `bake_verdict --print-features-root` from the bake's
  declaration/provenance; an explicit fourth argument or
  `ZENSIM_FEATURES_ROOT` takes precedence. A provenance-free bake needs an
  explicit root. See [`FEATURE_SET_IDS.md`](FEATURE_SET_IDS.md).

Output: `/mnt/v/output/zensim/reports/fulleval/<name>.fulleval.json`
(+ `<name>.verdict.md`, the human `bake_verdict` report, alongside).

Example:

```sh
scripts/run_full_eval.sh \
  /mnt/v/output/zensim/bakes/p1kadis/foldmlp_bigcodec_kadis_720.bin \
  foldmlp_bigcodec_kadis_720 720
```

Stages are independently reusable. `verdict` runs the offline score/statistics
owner; `coherence` requires a current verdict and runs the 27-cell map exam;
`all` runs both. `qualify` reads the assembled evidence through
`freeze_check --qualify`, stores its report, and returns nonzero for failed or
incomplete qualification. G-RD/G-TARGET still need the real codec instruments.

Reuse is automatic only when the owning tools' complete input identities
match: model/member/head hashes, evaluator binary, resolved feature tables and
manifest, probes/truth and settings; coherence additionally pins every fixture
and the sweep executable. `bake_verdict --print-inputs` and
`m3a_sweep.sh --print-inputs` expose those identities without computing scores.
The independent `*.verdict-stage.json` and `*.coherence-stage.json` artifacts
survive interruptions. JSON writes are atomic and output stems are locked.
`harvest_bakes.sh` calls this owner once and copies its verdict for legacy
consumers. It never treats file existence as proof of a completed evaluation.

`feature_set_composition` binds provenance to the complete scoring identity.
Every member and companion needs compatible training admission and evaluation
table declarations; a known primary cannot cover an unknown leg. Historical
replay cannot qualify. The current tree companion format lacks the required
training/decoder admission record, so it remains unqualified even though Rust
can serve it. Root and both supported per-file sidecar locations are hashed,
including their absence, even when malformed/unknown declarations stop admission.
Adding or changing a decoder sidecar therefore invalidates a historical verdict.

Environment overrides: `ZENSIM_M3_FIXTURES`, `ZENSIM_M3_CONTENT` (three names),
`ZENSIM_FULLEVAL_OUT`, `ZENSIM_BAKE_VERDICT` and `ZENSIM_DIFFMAP_BIN`.
`CARGO_TARGET_DIR` is honored. `ZENSIM_M3_ONLY=1` remains an alias for the
coherence stage; `ZENSIM_M3_REUSE=1` cannot bypass identity validation.
`ZENSIM_M3_DIST_Q` was obsolete: the registered grid uses q20/q50/q75.
A missing historical fixture now refuses instead of mixing a newly encoded
file into an old fixture era. Generate a complete new era with the existing
`m3_fixture_gen` owner in a separate directory.

## What it chains (no duplicate stat implementations)

| section | Rust owner | invocation |
|---|---|---|
| rank (Mohammadi 6-stat / corpus) | `bake_verdict` → `zensim_validate::panel` (`zenstats`) | `--fulleval` |
| dial (G1/G3 codec-target) | `bake_verdict::dial_panel` | `--fulleval` (regime dial grid) |
| corruption gate | `bake_verdict` → `eval_report::corruption_gate` | `--fulleval` (regime corruption grid) |
| per_pair (pred vs mos/jnd/ssim2/butter/cvvdp) | `bake_verdict` + `parquet_loader::load_perpair_sample` | `--fulleval` |
| m3_coherence (G-STEER) | `zensim/examples/diffmap_block_coherence.rs --bake` | shell loop, jq-injected |

Each stage builds only its required Rust instrument. The verdict emits the
schema-complete JSON; the sweep supplies M3/M3a means. A coherence stage requires
all 27 cells for both measurements. Partial results retain their logs and the
completed verdict, return failure, and cannot become a valid cache entry.

Qualification evidence uses `product_evidence["G-RANK"|"G-DIAL"|"G-STEER"|
"G-RD"|"G-TARGET"]`: state, candidate `bake_sha256`, `surface` equal to
`zensim::BakeScorer`, instrument identity, positive sample count `n`, and
`artifact: {path, sha256}`. The measuring owner's JSON must name the same bake
and contain that gate/state under `gates`. The qualifier verifies the artifact
bytes before reading the state. An absent or stale record is incomplete; a
known gate failure remains failed even when other evidence is missing. This
extends the existing scorecard and decision owner; it is not another model
registry. The gauntlet displays the stored qualification report without
synthesizing one from rank or badges.

## JSON schema

```jsonc
{
  "bake": "<path>",
  "bake_sha256": "<hex>",           // ties back to the manifest (repro spine)
  "name": "<name>",
  "regime": "720" | "372",
  "n_inputs": 720,                  // the bake's own input width
  "m3_coherence": 0.6456,           // mean M3 over the 27 fixture pairs (null if none)

  "rank": {                         // per held-out corpus (the rank panel)
    "cid22": { "n", "srocc", "plcc", "krocc", "or", "pwrc", "z_rmse" },
    "aic3":  { ... }, ...           // cid22,kadid,tid,csiq,live,konjnd,aic3,aic4,nonphoto,imazen26
  },

  "dial": {                         // codec-target G1/G3, from the regime dial grid
    "mono_pct":      0.977,         // G3 monotonicity = 1 − material inversions
    "tied_pct":      0.0,           // flat/clamp dead-zone rate (the gated tie metric)
    "reach":         19.70,         // full pooled dial span (max − min); G4 cross-codec reach
    "dynamic_range": 12.33,         // robust span p95 − p5 (G1 gate: p5≤25 ∧ p95≥85)
    "p5":  -3.35, "p95": 8.98       // raw percentiles for context
  },

  "corruption": {                   // the bake_verdict corruption gate — see NOTE
    "n_triples":  672,
    "pass_q20":   0.214,            // frac corruptions ranked BELOW an honest q20 encode
    "pass_q10":   0.168,            // same vs a q10 anchor
    "per_family": [ { "family", "pass_rate", "n" }, ... ]
  } | null,                         // null when no matching corruption grid

  "per_pair": {                     // sampled scatter data (≤ 5000 pairs / corpus)
    "cid22":  { "pred": [...], "mos": [...] },   // MOS corpora: cid22,kadid,tid,csiq,live,nonphoto,imazen26
    "aic3":   { "pred": [...], "jnd": [...] },   // JND corpora: aic3,aic4,konjnd
    "kadis":  { "pred": [...], "ssim2": [...], "butter": [...], "cvvdp": [...] }
  }
}
```

`pred` is the bake's dial-space output (the same `score_grid` runtime the rank
panel scores through — transforms + forward + output spline). Each corpus emits
only the reference columns it actually carries ("ONLY the refs that corpus
has"): MOS corpora → `mos`, JND corpora → `jnd`, and the `kadis` block (sampled
from the KADIS-720 metric parquet) → `ssim2` / `butter` / `cvvdp` from
`score_ssim2_gpu` / `score_butteraugli_max_gpu` / `score_cvvdp_cpu_imazen_v0_1_0`.

## Notes / honest deviations

- **corruption field names.** The `bake_verdict` corruption gate is a
  *pass-rate* (`score(corruption) < score(q20)` per `eval_report::CorruptionStats`),
  not a detection-threshold / false-positive ROC. The JSON therefore carries the
  real gate outputs (`pass_q20`, `pass_q10`, `per_family`), not the
  `detection_t50` / `fp_*` names from the original schema sketch — reporting a
  number the tool does not compute would be a fabrication.
- **per_pair sampling.** MOS/JND corpora are even-strided down to the cap (≤5000)
  across the whole corpus. The `kadis` block reads a bounded ≤40k-row window from
  the 2.7 GB metric parquet (projected to features + the 3 metric columns) then
  strides to the cap — bounded memory, source-diverse. Override the cap with
  `bake_verdict --perpair-cap N`; the source with `--perpair-metrics <parquet>`
  (a non-existent path skips the `kadis` block).
- **M3 (G-STEER).** `diffmap_block_coherence --bake` reports M1/M1b/M3/M2; the
  wrapper reads the **M3** line (deployable model-sensitivity map ↔ per-block ΔS)
  and averages it over the 27 fixture image pairs. M3 is per-pair noisy (measured
  city 0.28 / dog 0.75 / girl 0.91 for the fold-MLP), so the mean is the summary;
  the per-pair `<name>.m3.<ref>.log` files are kept for inspection. This holds
  for a nonlinear MLP too — M3 is a rank correlation of per-block ΔS, not an
  additivity assumption (M2 is the linearization ceiling, ≈1.0 for a LeakyReLU
  MLP).
- **dial grid warning.** Under `--regime 720`, `bake_verdict` prints a "NOT the
  canonical grid" warning because the canonical-sha check only knows the 372
  quarantined grid. The 720 dial grid is the regime-matched grid and is the
  correct one for a 720 bake — the warning is expected, not an error.

## Where the pieces live

- `scripts/run_full_eval.sh` — the wrapper (repo-relative binary paths; no
  hardcoded worktree path).
- `zensim-validate/src/bin/bake_verdict.rs` — `--full-json` / `--name` /
  `--perpair-metrics` / `--perpair-cap`; the schema is emitted here.
- `zensim-validate/src/parquet_loader.rs` — `load_perpair_sample` (row-capped
  multi-metric sampler; THE feature-parquet loader owner).
- `zensim/examples/diffmap_block_coherence.rs` — M3 (`--bake`); the v2 fold is
  gated behind `feature-regime-v2`.

Related single-purpose evals that need a second bake / spline internals / an HDR
corpus (`bake_compare`, `bake_dial_refit gate`, the UPIQ panels) are listed in
`bake_verdict`'s own "Related specialized evals" report footer and
`docs/EVAL_PANEL_REQUIREMENT.md`.

---

## 924-era eval slices — REPOINTED at the canonical test views (2026-07-28, user directive)

For models in the folded+append **924** regime, the `imazen26` and `nonphoto`
eval slices come from the **canonical bigcodec 924 TEST views** (held-out
origins {7,9}; exact `encoded_filename`-key joins, match_rate 1.0000):

    /mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/bigcodec/<dataset>/test_924.parquet
    (R2: s3://zentrain/ext924-canonical-2026-07-27/bigcodec/ · Tower mirror sha-verified)

with `score_ssim2` / `score_zensim` targets carried from the canonical picker
datasets. The `nonphoto` slice = the same test views filtered to non-photo
content classes via `/mnt/v/output/imazen-26-features/imazen26_manifest.tsv`.
(Manifest header+split column corrected 2026-08-27 — DATASET_HISTORY §3.24.
Both axes carry the standing annotation
`imazen26-nonphoto-sharing-provenance-2026-08-27`: ~10-32% of rows sit on refs
whose content also feeds synthetic-v2 or a train-split twin; measured effect on
leader SROCCs ≈0. Read `benchmarks/eval_annotations.json` before citing.)

The 720-era `ext_imazen26_720` / `ext_nonphoto_720` tables were built by
NEAREST-NEIGHBOR fingerprint matching against fleet blobs (winning encode
identities never persisted); fingerprint matching cannot cross regimes (the
folded block replaces v1-372), so those tables are **720-legacy only — do NOT
rebuild them for 924**. The eval instruments `corruption_grid_924col` and
`dial_grid_924col` live in `/mnt/v/output/zensim/v2-eval-924-2026-07-27/`.

A repeated stage preserves attached ladder/product evidence only when the aggregate
verdict identity still matches. It clears the previous qualification decision; run
`--stage qualify` again to recheck artifact hashes and complete scorer composition.
Product measurement JSON must carry the same `scoring` block as the verdict.
Changing any verdict input drops these attachments; graft freshly measured evidence
through `promote_fulleval.py` before qualifying again.
# Five-minute feature development screen (September 13, 2026)

Use the existing pipeline's `feature-screen` stage for the small T2-only
experiment recipe. It runs canonical Rust extraction, two H32 fits, final
`BakeScorer` pixel/cache audits and Rust correlation panels under one 300-second
deadline. Input, label, binary, producer/revision and split identities bind the
feature cache. `FAILED_OR_INCOMPLETE` is never a quality pass.

Build once outside the iteration budget:

```bash
../scripts/run-heavy --mem 16G --jobs 8 cargo build --release -p zensim-validate --bin zensim_mlp_train --bin panel
../scripts/run-heavy --mem 16G --jobs 8 cargo build --release --manifest-path zensim-bench/Cargo.toml --example extract_features_372col --features training,zen-decode
```

Run with a fresh output directory (Python requires pyarrow for Parquet I/O):

```bash
../scripts/run-heavy --mem 16G --jobs 8 scripts/run_full_eval.sh --stage feature-screen benchmarks/feature_screen_2026-09-13.json "$HOME/work/feature-screen-run" --cache "$HOME/work/feature-screen-cache"
```

The recipe fits to SSIMULACRA2 proxy labels from 264 admitted JXL pairs, using
eight fit, two checkpoint-selection and two inner-test origins. All twelve
origins remain T2 training content; the test images were examined in previous
experiments. This cannot qualify perceptual quality or a release. It reports
signed/raw correlations separately from the full panel's absolute/logistic
statistics. Logistic-rescaled errors are not end-user target-score errors.
Corruption, HDR, spatial intervention, reachable codec targets and full-size
performance remain explicitly unmeasured in this first packet.

`zensim_mlp_train --no-auto-eval` suppresses its historical automatic protected
holdout evaluation. This stage always passes it and evaluates only its explicit
packet. The existing full-eval and qualification stages remain separate.
See the [measurement and next experiments](../benchmarks/fullres_y_subset_2026-09-12.md).

## Opt-in sampling recipes (September 13 follow-up)

The feature-screen recipe may also declare `spatial_checks` with a `manifest`
path, its `sha256`, and a `block` size of 8, 16 or 32. Build the existing
`zensim` example `diffmap_block_coherence` with
`custom-profiles,feature-regime-v2,threads,training` first. Every case must
inherit an admitted training origin, match its reference path and pass both
PNG hash checks. The stage invokes the Rust block-repair owner for every
final bake and retains scalar scores, measured gains, M2/M3f and unsupported
IDs under the same 300-second deadline. Missing spatial support is
`UNSUPPORTED`, never a passing map. These are fixed development fixtures;
neither a successful run nor a spatial pass establishes perceptual quality.

The [coarse-pool recipe](../benchmarks/coarse_pool_screen_2026-09-13.json)
uses this option for four layouts across three seeds. The box pyramid and
canonical Rev3 feature values are unchanged. Declared IDs now select the
actual v1 masked/IW scales in Rust, while retaining full-resolution Y and
omitting unused finest X/B. Masked and IW share a kernel chain, so selecting
either activates both at that scale. They still lack spatial integrands;
the screen exposes that limitation explicitly.

The same `feature-screen` owner now accepts a `sampling` string:
`v1:{y|xyb}:{triangle|mitchell|robidouxsharp}:{3/2|2|3}`.
`y` retains full-resolution Y and omits finest X/B features; its subsequent
XYB levels are at d, 2d and 4d. `xyb` starts all channels at d, then 2d, 4d
and 8d. These are new feature values and require fresh extraction and fitting.
They use zenresize main's signed floating-point kernels at every transition.

To obtain the producer identity, run the existing extractor with `--sampling`
and an explicit `ZENSIM_FORMULA_REV=3`. Its `.manifest.json` reports the
Rust-generated `feature_set_id` and populated feature IDs; its `.producer.bin`
is a diagnostic extraction model, not a trained quality model. Use those
IDs and identity in the recipe. Optional `arm_seeds` maps arm names to paired
training seeds. The trainer admits one sampling contract per fit and embeds
`zentrain.sampling` in the final bake. The screen checks producer identity,
passes the contract to the final pixel audit, and binds it into cache identity.

Serve the final model through `BakeScorer::compute`, or cache its reference
with that same scorer and call `compute_with_ref_and_attribution`. The existing
`score_features` API consumes feature rows produced under the same contract;
raw slices cannot carry provenance, so cache admission and pixel audits are
required. `research::Request::for_bake_bytes` refuses these models instead of
silently extracting the default pyramid. Mixed sampling ensembles, legacy
reference caches, unmatched corruption companions and HDR sampling are refused.
No named/default model is changed.

Attribution uses squared, normalized resizer tap ownership back to logical
source coordinates, including reflection. This preserves signed map mass;
it is an approximation to finite pixel edits, not a pixel derivative.
`refinement_gain` also includes the existing finite-max correction with the
sampled support. Use `diffmap_block_coherence --bake MODEL --block 8` (also
16 and 32) to compare these predictions against actual reference-pixel
replacements scored through the public API. A successful serving audit does
not imply accurate spatial steering.

The [integrated sampling report](../benchmarks/sampling_serving_2026-09-13.md)
records all 20 layouts, three paired training seeds, scalar/spatial timings,
and intervention results. This supersedes the earlier filter-only status.
# Representative feature/scale capability study (September 13, 2026)

The `feature-screen` owner also accepts `zensim-feature-ceiling-recipe-v1`.
Its representative preparation is separate from the small T2 screen's strict
300-second budget. See the preregistered
[protocol](../benchmarks/feature_ceiling_2026-09-13.md) and
[recipe](../benchmarks/feature_ceiling_2026-09-13.json).

```bash
../scripts/run-heavy --mem 16G --jobs 8 scripts/run_full_eval.sh \
  --stage feature-screen benchmarks/feature_ceiling_2026-09-13.json \
  /absolute/fresh/output --ceiling-stage prepare
../scripts/run-heavy --mem 16G --jobs 8 scripts/run_full_eval.sh \
  --stage feature-screen benchmarks/feature_ceiling_2026-09-13.json \
  /absolute/fresh/output --ceiling-stage fit
../scripts/run-heavy --mem 16G --jobs 8 scripts/run_full_eval.sh \
  --stage feature-screen benchmarks/feature_ceiling_2026-09-13.json \
  /absolute/fresh/output --ceiling-stage checkpoints
../scripts/run-heavy --mem 16G --jobs 8 scripts/run_full_eval.sh \
  --stage feature-screen benchmarks/feature_ceiling_2026-09-13.json \
  /absolute/fresh/output --ceiling-stage audit
scripts/run_full_eval.sh --stage feature-screen \
  benchmarks/feature_ceiling_2026-09-13.json /absolute/fresh/output --ceiling-stage report
```

Build the existing Rust extractor, trainer, cached predictor and `panel` first.
The extractor's `--full-944` option uses an all-live diagnostic bake through
`BakeScorer::compute` and emits an explicit producer manifest; it conflicts with
`--sampling`. It creates no new feature arithmetic. Fresh 944 tables retain v1
peaks/masked/IW slots, unlike historical wide producers that left those empty.
Do not infer compatibility from width. Python preparation/reporting live in the
bounded `feature_screen_ceiling` module; Rust owns features, fitting and scoring.

Preparation pins original bytes, all row IDs, split/family admission, tools,
feature ID and formula revision. Fitting validates tables and supports verified
completed-bake reuse. Eight independent fits run under the campaign's eight-core
cap. Audit uses actual pixels and finite spatial repairs; unsupported features
remain explicit. The extra `--ceiling-panel PATH` selects a separately pinned
raw-error-capable panel binary when preserving an active campaign's original
tool binaries. No stage enters automatic protected full-eval defaults.
The optional `checkpoints` follow-up adds 18 frequent-checkpoint controls and
nine coarse262 fits to the primary 135-fit matrix. Run it before `audit`, which
requires a fresh audits directory. `all` executes preparation, primary fitting
and audit; it does not include this follow-up or the final `report` stage.

`panel --batch jobs.tsv --raw-errors` appends **raw** MAE. The existing `mae`
column is logistic-remapped using the evaluation rows and remains unchanged for
compatibility. Use raw MAE for calibration/error claims. The literal cached
feature API has no pixel-identity override; identity-aware pixel audits are
reported separately. A capacity/data plateau within one MLP family is an
empirical result, not a mathematical feature ceiling or product qualification.

Later September 13: [scale-selective 944 study](../benchmarks/scale_selective_944_2026-09-13.md)
uses the same owner with recipe `log_every: 1` and optional `sampling` metadata.
`--full-944 --sampling v2:xyb:triangle:1,3,5,7` selects direct-from-original
scales; the closed alternatives are `1,2,4,8` and `1,2,3,5`, with Triangle,
Mitchell or RobidouxSharp. This v2 sampling tag is distinct from formula Rev3.
It includes zenresize's binary16 input-row rounding before float filtering;
see the independent precision test and report. Historical v1 sampling keeps
its contract and cannot be combined with `--full-944`.

The native recipe adds masked/IW separately at each scale, for both legacy
and newer weighted families. All 32-epoch fits retain frequent dev checkpoint
selection, fresh/native-verified features, raw Rust error panels and actual
pixel/spatial audits. No capacity-control matrix is repeated in this follow-up.

### Verified pixel identity in cached eval (September 14)

`ensemble_score_rows` and `bake_verdict` corpus scoring accept an optional
`pixels_identical` Float32/Float64 column containing only non-null 0/1 values.
It must come from equality of decoded pixels, pinned to the same row keys and
input hashes. Both owners call `BakeScorer::score_features_with_identity`.
A missing column means unknown identity and retains the historical feature-only
behavior; zero features never prove identity. Do not compare that diagnostic
as pixel-equivalent when the admitted corpus contains identities. Native audit
JSONL from `extract_features_372col` supplies the evidence. The September 13
minimal/wide study's final eval corrects this distinction without retraining.

### Complete scatter and integrity assessment (September 14)

`bake_verdict` now stores `scatter_assessment.<corpus>.<reference>` over the
complete scored population before capping plot rows. `panel --input pairs.tsv
--json --scatter` uses the same `zenstats::scatter` owner. The Python statistics
shim delegates to it; gauntlet consumes stored results and matching normalized
plot coordinates. Missing historical measurements remain missing.

Read robust envelope outlier share, p99/max relative to reference span, raw
absolute and robust-scaled residual tails, raw density, exact extrema mass,
range and slope together with the existing full Mohammadi panels and named
composite floors. Rank normalization preserves geometry but can hide smooth
score compression; raw density is a separate diagnostic. Zero scales are
unmeasured, not epsilon-derived passes. `outlier_gate.py` returns INCOMPLETE
(exit 2) when required statistics, peer bars or declared range are missing.

The new strict corruption fit is `train_corruption_head.py
--strict-train-manifest FIT.json --out-dir NEW`. Its manifest admits only train
fit/calibration, pins inputs/tools, exports ZCTH and checks Rust surface parity
before the calibration advancement gate. Legacy/canonical mixed-table modes
are historical, not approved defaults under the September 13 ruling.
`corruption_gate_eval.py --integrity-admission ADMISSION.json --audit-jsonl
AUDIT.jsonl --out-json NEW.json` reports the actual complete Rust composition.
Unlabelled operations are not negatives; duplicate pixel identities retain all
catalog provenance and conflicting binary labels fail. Prepared steering audits
use `ZENSIM_AUDIT_PREPARED_STEERING=1` in the existing native extractor.

The [registered prototype and results](../benchmarks/steering_integrity_2026-09-14.md)
retain all severe misses, ambiguous activations and missing product gates.
No new head, scatter instrumentation or pipeline-parity result alone qualifies
an all-purpose target dial or native spatial allocator.
