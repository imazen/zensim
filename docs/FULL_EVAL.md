# Full-eval — one comprehensive Rust eval per bake → machine-readable JSON

`scripts/run_full_eval.sh` runs the **entire model exam** through the canonical
Rust owners and emits one machine-readable JSON per bake. No Python touches any
statistic — every number comes from the Rust binaries that already own it
(`bake_verdict` → `zenstats::panel` for the rank+dial+corruption math,
`diffmap_block_coherence` for M3). The JSON is the input the summer-gauntlet
dashboard consumes for its scatter/scorecard panels.

## Usage

```sh
scripts/run_full_eval.sh <bake.bin> <name> [regime=720]
```

- `<bake.bin>` — a ZNPR v3 bake (any width; the scorer reads its own `n_inputs`).
- `<name>` — the label embedded in the JSON and used for the output filename.
- `regime` — `720` (default) or `372`. Selects which pre-extracted corpora,
  dial grid, and corruption grid `bake_verdict` scores against (`--regime`).

Output: `/mnt/v/output/zensim/reports/fulleval/<name>.fulleval.json`
(+ `<name>.verdict.md`, the human `bake_verdict` report, alongside).

Example:

```sh
scripts/run_full_eval.sh \
  /mnt/v/output/zensim/bakes/p1kadis/foldmlp_bigcodec_kadis_720.bin \
  foldmlp_bigcodec_kadis_720 720
```

Env overrides: `ZENSIM_M3_FIXTURES` (default
`/mnt/v/output/zensim/diffmap-coherence-2026-07-18`), `ZENSIM_M3_DIST_Q`
(default `q50`), `ZENSIM_FULLEVAL_OUT` (default the reports dir above),
`ZENSIM_M3_REUSE=1` — carry `m3_*` from the bake's previous fulleval JSON
instead of re-measuring. **Use this for schema re-emits**: the rank/dial/
corruption portion is a cheap rescore over stored feature parquets (numbers
cannot change unless the bake/parquets/estimators changed), but the M3 sweep
is 27 diffmap runs per bake — re-measuring an unchanged value.

## What it chains (no duplicate stat implementations)

| section | Rust owner | invocation |
|---|---|---|
| rank (Mohammadi 6-stat / corpus) | `bake_verdict` → `zensim_validate::panel` (`zenstats`) | `--fulleval` |
| dial (G1/G3 codec-target) | `bake_verdict::dial_panel` | `--fulleval` (regime dial grid) |
| corruption gate | `bake_verdict` → `eval_report::corruption_gate` | `--fulleval` (regime corruption grid) |
| per_pair (pred vs mos/jnd/ssim2/butter/cvvdp) | `bake_verdict` + `parquet_loader::load_perpair_sample` | `--fulleval` |
| m3_coherence (G-STEER) | `zensim/examples/diffmap_block_coherence.rs --bake` | shell loop, jq-injected |

The script builds both binaries release (`bake_verdict`; the example with
`custom-profiles,feature-regime-v2` so a >372 bake's v2 block folds into the M3
map — inert for a ≤372 bake), then jq sets the top-level M3 fields from the
sweep means. Everything else is emitted by `bake_verdict --fulleval` directly
in the target schema — since 2026-08-04 that flag emits the SCHEMA-COMPLETE
file (all five `m3_*`/`m3a_*` slots pre-nulled; `--full-json` remains the
m3_coherence-only legacy form), so the jq step only injects INTO existing keys
and `run_full_eval.sh` adds no statistic of its own.

## JSON schema

```jsonc
{
  "bake": "<path>",
  "bake_sha256": "<hex>",           // ties back to the manifest (repro spine)
  "name": "<name>",
  "regime": "720" | "372",
  "n_inputs": 720,                  // the bake's own input width
  "m3_coherence": 0.6456,           // mean M3 over the 3 fixture pairs (null if none)

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
  and averages it over the 3 fixture image pairs. M3 is per-pair noisy (measured
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

The 720-era `ext_imazen26_720` / `ext_nonphoto_720` tables were built by
NEAREST-NEIGHBOR fingerprint matching against fleet blobs (winning encode
identities never persisted); fingerprint matching cannot cross regimes (the
folded block replaces v1-372), so those tables are **720-legacy only — do NOT
rebuild them for 924**. The eval instruments `corruption_grid_924col` and
`dial_grid_924col` live in `/mnt/v/output/zensim/v2-eval-924-2026-07-27/`.
