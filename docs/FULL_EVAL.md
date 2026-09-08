# Full-eval — one comprehensive Rust eval per bake → machine-readable JSON

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
