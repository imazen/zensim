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
