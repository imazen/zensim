# TRAINERMEM worklog — f32 resident storage + column projection in zensim_mlp_train

Brief: /home/lilith/tmp/zensim-paper/rev4/TRAINERMEM_brief.md (binds with rev4/DEVIN_COMMON.md;
DEVIN_COMMON's literal path /home/lilith/tmp/devin/DEVIN_COMMON.md does not exist — the file lives at
/home/lilith/tmp/zensim-paper/rev4/DEVIN_COMMON.md). Lane = devin-trainermem. Bookmark
quarantine/devin/trainermem. Scratch /var/tmp/trainermem/ (CARGO_TARGET_DIR=/var/tmp/trainermem/target).
Heavy commands: ~/tmp/devin/heavy (flocks /home/lilith/tmp/devin/heavy.lock, then execs
~/work/zen/scripts/run-heavy — satisfies both LANE_PREAMBLE and the brief's run-heavy binding); builds
--jobs 24 via the heavy floor; trainer gate runs HEAVY_KEEP_JOBS=1 --jobs 4 (fleet cells pin 4 threads).
ZENSIM_MAX_TIER=v3 and RAYON_NUM_THREADS=4 on every trainer comparison run.

Fleet trainer: /var/tmp/fitv2/bin-v2/zensim_mlp_train sha256
6b28576f0bf0b315238a063e1a686cda32499fdf37a1799b2820d2286db8a7c9 (per brief, built from zensim
83a205ad = main 424b8b02 + ZENSIM_MAX_TIER cap + optmlp AVX2 restructure).
Profiled cell argv: /home/lilith/tmp/zensim-paper/rev4/argv.json (r0, 2 epochs); keep list
/home/lilith/tmp/zensim-paper/rev4/keep_r0.txt == /home/lilith/tmp/trainer-prof/keep_r0.txt (diff: identical).

## Baseline facts (read-only, verified 2026-10-01)
- /var/tmp/rev4-featpot/v2/wide/{main,aux}/<variant>/ tables: 1825 feature cols ALL Float32; human_score
  Float64; ref_basename LargeUtf8 (pyarrow schema read of main/real/safesyn_fit.parquet: 1827 cols).
- keep_lists.json arms: r0=0..943 (944), minus_basic=228..943 (716), oracle_lo=0..943+946 (945, aux),
  oracle_hi=0..943+947 (945, aux), p3=0..943+944,945+1322..1501 (1126, aux peers), a1 1004, rall 1477.
- Fleet cell = v2_lodo_mlp.py --spec ARM --head N|F --heldout SRC --seed-index K; argv built from
  scripts/rev4_featpot/v2_common.py (EPOCHS=120, PAIRS=50000, HIDDEN=32, WIDTH=1825, seeds from
  seeds(heldout,k), acceptance_weight over fit-leg refs).
- Gate oracle: harvest_fit_cells.weights_sha masks bake bytes 68:72 and 4B before the JSON trailer.

## Command log

### Baseline build (unchanged main)
- START 2026-10-01T07:30:59Z END 07:31:59Z cwd=/home/lilith/work/zen/zensim--trainermem
- `CARGO_TARGET_DIR=/var/tmp/trainermem/target ~/tmp/devin/heavy --mem 16G --jobs 8 -- cargo build --release -p zensim-validate --bin zensim_mlp_train --bin bake_dial_refit` (heavy bumped jobs to 24 floor) → rc=0, 58s, peak-RSS 1.87GiB, min-avail 28470MiB, peak-load 23.64
- outputs: /var/tmp/trainermem/bin-base/zensim_mlp_train.main, bake_dial_refit.main (shas in /var/tmp/trainermem/logs/bin_shas.txt)

### Baseline identity gate (unchanged main build vs fleet binary)
- START 07:35Z END 07:38:48Z; 4 cells × 2 epochs via /var/tmp/trainermem/cell.py (argv-identical replica of
  scripts/rev4_featpot/v2_lodo_mlp.py; ZENSIM_MAX_TIER=v3, RAYON/OMP=4 inside driver), cells:
  r0__N/without_kadid_s0, oracle_hi__F/without_tid2013_s1, rall__N/without_konfig_s0,
  minus_basic__F/without_cid22_a25_s1 — roots /var/tmp/trainermem/cells/{fleet,main}
- compare.py (harvest weights_sha: mask [68:72] + 4B before JSON trailer) → PASS 4/4
  weights_identical=true, curve_identical=true, pred_identical=true on all 4
  (/var/tmp/trainermem/logs/baseline_cmp.json)
- CONCLUSION: current main (parent b1b1a49c) is output-compatible with the fleet binary; base = main.

### Baseline profile (symbolized main build, target-sym, CARGO_PROFILE_RELEASE_DEBUG=true)
- START 07:39:57Z build END 07:41:14Z rc=0 76s; profile runs 07:41:55–07:45:42Z via
  run_argv.sh raw|heap (argv_*_*.txt = NUL-sep argv with --out→profile/out_*.bin)
- /usr/bin/time -v peak RSS (run-heavy report): r0 4.24GiB, oracle_hi 3.72GiB, rall 4.24GiB
- heaptrack r0: peak heap 3.43G; attribution "3.02G over 6 calls" =
  parquet_loader::load_parquet_flat (parquet_loader.rs:410) ← load_group_dispatch ← main:3106
  (~88%, the 6 groups' f64 flat matrices); RSS-incl-heaptrack 4.55G
- artifacts: /var/tmp/trainermem/profile/base_{r0,oracle_hi,rall}.ht.zst (+logs/)

### Coordinator decisions 2026-10-01 01:53 MT (recorded in TRAINERMEM_decisions.md)
- D1 approved; D2 CHANGED → standardize-at-use: resident = compact raw f32 ONLY; consumers evaluate
  the identical f64 (x-mean)/scale expression on expansion (~0.8GiB r0 target). Fallback if gate
  differs: compact f64 std store + free raw (~1.6GiB). D3,D5 approved. D4 approved + byte-check:
  dropped-position standardized value must be +0.0 — verify: dropped raw = +0.0 literal, mean=+0.0,
  var=0 → scale=max(1e-8)→(0-0)/1e-8=+0.0 (matches today's buffer).
- Follow-up commit (separate, same gate): EFFAUDIT D1 — skip per-epoch eval of train-only groups
  (val_w==0); epoch line keeps val(geomean3)= and dev-group values identical.

## Takeover by claude-trainermem (2026-10-01) — bookmark quarantine/claude/trainermem

Devin's loader scaffolding (Emit/Cols/store_cols) did not compile; finished and simplified (no `Cols`; subset + max_width
parameters). Design as approved (D1–D5, coordinator D2 CHANGE): resident = compact raw f32 only, standardize at use.

### Implementation (commit 1)
- `parquet_loader::load_parquet_flat_f32(path,name,target,scale,subset,max_width) -> Result<Option<OwnedLoadedGroupCompact>>`:
  ProjectionMask over the kept columns only, f32 verbatim, pre-reserved from the footer. `Ok(None)` (decline, nothing read) when
  a kept column is not Float32, ids unsorted/duplicated/out of the logical width `min(file width, max_width)`, or empty subset.
- `mlp_train::FeatureRows::Compact(&mut CompactRows)` (+ `LazyStd`, `StdFeatures`, `StdGroup`). Plain head (`train_mlp_strategy`)
  keeps Compact groups lazy: `row()` expands into per-call scratch with `copy_from_slice(template)` + kept-column scatter, where
  `template[d] = (0.0 - mean[d]) / scale[d].max(1e-12)` (the exact value today's zero-masked buffer holds) and kept columns
  evaluate `(f32 as f64 - mean[d]) / scale[d].max(1e-12)` (the in-place pass's expression). `compute_scaler_from_groups` visits kept
  columns only (dropped columns stay +0.0 mean / var, std = 1e-8, as before). Other heads: `standardize_group_releasing_raw`
  expands a Compact table once into the dense buffer.
- Binary: `compact_load` engages only with `--keep-features` and no feature transforms / auto-transforms / TV pairs / GPU /
  pool / hybrid / alpha head; otherwise the dense path is untouched. NiN helper holds rows across a batch -> `row_cow`.
- Behaviour difference (benign): dropped columns are not read, so a null/odd dtype in a dropped column no longer errors.

### Gate, 2 epochs, base = unchanged main binary (bin-base) vs new1, fitbin = base bake_dial_refit for both
8/8 cells PASS (weights, epoch curve, predictions): r0 N kadid s0; oracle_hi F tid2013 s1; rall N konfig s0; minus_basic F cid22_a25 s1;
oracle_lo N kadid s1; p3 F tid2013 s0; a1 N cid22_a25 s0; r0 F konfig s1 (cells/{main,new1}, logs/extra_*.log).

### Gates (final, claude-trainermem, 2026-10-01)
- 2 epochs, 8 cells, base = unchanged main binary: new1 (commit 1) 8/8 and fin (commit 1+2, D1 comparator that ignores the
  train-only segments) 8/8. oracle_hi/oracle_lo baselines were re-run after the R3 aux rebuild (02:18-02:24 MT; pre-R3 cells
  kept in cells/main_pre_r3) so every pair reads one table generation.
- 120 epochs, 4 cells (r0 N kadid s0; oracle_hi F tid2013 s1; rall N konfig s0; p3 F cid22_a25 s1): new1 4/4 and fin 4/4
  identical weights, curve (dev/val fields) and predictions. Wall (loaded box, not a benchmark): r0 394 -> 343 s, p3 362 -> 314 s.
- Profile (2-epoch r0/oracle_hi/rall argv; time -v peak RSS / heaptrack peak heap): base r0 4.24 GiB/3.43G, oracle_hi 3.72/3.34G,
  rall 4.24/3.43G -> final r0 1.36 GiB/1.20G, oracle_hi 1.32 GiB/1.20G, rall 2.14 GiB/1.88G. Attribution after the change:
  compact f32 matrix (532M of the peak, 791M final for r0), a parquet row-group read buffer ~450M and arrow/other ~120M at the
  moment of the peak (during the largest group's load). Before the streaming sha fix the peak also held the whole 1.2 GB
  table read for `sha256_file`.
- tests: `cargo test --release -p zensim-validate --no-fail-fast`: all pass except pre-existing `bake_surface::
  formula_revision_is_selected_per_bake_and_unknown_or_mixed_revisions_refuse` (asserts "4" is an unknown revision; untouched
  by this diff). CI-exact clippy, fmt scoped to zensim-validate, lint-scripts clean.
