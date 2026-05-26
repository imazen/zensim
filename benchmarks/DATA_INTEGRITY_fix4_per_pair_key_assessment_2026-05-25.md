# Fix 4 assessment: carry the per-pair key end-to-end (task #215)

Companion to `DATA_INTEGRITY_root_cause_2026-05-25.md` recommendation #2
("carry a per-pair key or row-provenance end to end"). This documents how
invasive the deeper fix is, and what was done cheaply now.

## The problem

The ssim2_gpu ref-misjoin (Mode B) was only *possible* because the 372-feature
target parquets carry **only `ref_basename`** — they dropped `codec`, `q`,
`knob_tuple_json`, and `dist_path`. With no per-pair distortion identifier, the
only join key available was `ref_basename`, which collapses ~125 distortions of
a reference into one group. The "correct" join was positional (row order ==
dmos.csv order), but nothing enforced or recorded that.

## Where the key is dropped

The feature extractors emit a fixed CSV schema:

- `zensim-bench/examples/extract_features_372col.rs:128-145` — header is
  `ref_basename, human_score, <extra-target columns…>, f0..f<n-1>`. The
  per-row `Pair` struct (`:39`) carries `ref_basename: String` and an optional
  `extra_targets` map — **no `codec` / `q` / `knob_tuple_json` / `dist_path`
  field**. Each corpus loader (`load_kadid`, `load_tid`, `load_konjnd_full`,
  `load_aic3`, …) builds the distorted path internally but discards it after
  decoding; only `ref_basename` survives into the CSV.
- `zensim-bench/examples/extract_features_372col_omni.rs` and
  `zensim-picker-prep/src/bin/extract_features.rs` follow the same shape.

So the drop happens at the *extractor* layer, three corpus generations upstream
of the canonical build. The build scripts never had the key to preserve.

## Cost to fix properly (deferred — not a multi-day item, but not free)

To carry a per-pair key through, the extractors would need to:

1. Add an optional `dist_id: Option<String>` (the distorted-image path or the
   dmos row index) to the `Pair` struct and each loader (`:39`, `:193`, `:240`,
   `:291`, `:347`).
2. Emit it as a `dist_path` (or `pair_index`) column between `ref_basename` and
   `human_score` when present (`:128-145`), guarded so existing 372-col caches
   still parse (the trainer reads by column name, not position — verify against
   `zensim-validate/src/parquet_loader.rs` + `load_csv` at
   `zensim_mlp_train.rs:1056`).
3. Re-extract every corpus parquet under
   `/mnt/v/zen/zensim-training/2026-05-15-full-features/` (~30 min, one-time).
4. Update the canonical builders to use `safe_metric_join` on the full key
   instead of the positional fallback.

This is ~1 focused session, not a multi-day refactor — but it touches the
public CSV schema of three extractors and forces a re-extraction, so it is
sequenced behind the structural guards rather than bundled with them.

## What was done cheaply now (this task)

- `scripts/canonical_corpus/join_safety.py` makes the ref-only join
  **structurally impossible**: `safe_metric_join` raises rather than collapsing,
  and `attach_metric_positional` is the only supported path when the key is
  absent — and it hard-checks row-count equality so a misalignment can't slip
  through silently.
- Both canonical builders validate every metric column before writing
  (`assert_no_leaked_metric_columns` + `assert_metric_not_constant_per_ref`), so
  even without the per-pair key, a re-broadcast or leak fails the build loudly.
- The CI census (`audit_metric_columns.py --fail-on-corruption`) is the
  before-compute backstop.

Net: the *cheap* part (guards that make the bug fail loudly) is done; the
*invasive* part (re-plumb the extractors to carry `dist_path`) is documented
here for a follow-up session and is NOT required for correctness today, because
positional attachment + the guards already prevent the misjoin.
