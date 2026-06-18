# Sweep + training + eval SCRIPT-LAYER dedup inventory — task #223

**Date:** 2026-05-26
**Scope:** the operational/experimental scripting glue — `.sh`, `.py`, and
experiment-tool `examples/*.rs` / `src/bin/*.rs` — across the sweep- and
training-heavy zen repos. This is the layer that breeds silent bugs because
it is copy-pasted, version-forked, and untested.

**Companion to** `benchmarks/dedup_inventory_master_2026-05-26.md` (tasks
#220/#222). That master doc covers **well-factored library dedup** (zenyuv
color math, zencodec builders, GPU-metric backends, reusable CI workflows).
**This doc deliberately does NOT re-cover any of that** — see the "What this
does NOT cover" section at the bottom.

**Motivation (same as the master):** the 2026-05-25 kadid/tid parquet
corruption was a metric-join bug in ad-hoc, untested join logic. Duplicated
operational glue = N homes for one bug; only one home ever gets the fix. The
script layer is *where that incident actually lived* — so it gets its own
focused census.

**Counting rules.** All file:line refs are the **primary checkout** of each
repo. Excluded from counts: sibling worktrees (`zensim--*`, `zenmetrics--*`,
`jxl-encoder--*`, `_build-ctx-*`), `.claude/worktrees/`, `target/`,
`node_modules/`, and per-seed `benchmarks/*/seed_*/` generated snapshots.
The audit universe in primary repos: **507 `.sh`/`.py`** under the
sweep/training script dirs + **~280 `examples/*.rs` + `src/bin/*.rs`** in the
ML/sweep crates.

---

## Executive summary — top 5 script-layer dup clusters (bug-risk × count)

Ranked by silent-divergence risk first (the parquet-corruption bug class:
N copies, no parity, drift ships silently), reach second.

| # | Cluster | Copies | Bug risk | One-line fix |
|---|---|---|---|---|
| 1 | **Parquet build/join Python — only 3 of 38 files use the tested `join_safety.py`; 35 still do raw `pd.merge`/`groupby`** | 35 unguarded | **CRITICAL — this is the exact bug class that caused the 2026-05-25 corruption.** A silent many-to-many join or key-dtype mismatch corrupts a training target and ships. | Mandate `join_safety.safe_merge` in every parquet builder; migrate the 35 callers; add a CI grep-gate that fails on bare `pd.merge(` in `scripts/`. |
| 2 | **IQA-stat function re-implementations (`srocc`/`plcc`/`krocc`/`z_rmse`/`pwrc`/`outlier_ratio`/`mohammadi_panel`)** | 14 files, ≥25 `def`s | **HIGH** — a metric verdict is only as trustworthy as its stat impl. Hand-rolled `spearman` with a tie-handling or NaN-drop difference silently changes ship/no-ship calls. CLAUDE.md mandates the full Mohammadi panel everywhere — 14 forks guarantee they're *not* identical. | One mirrored `zen_stats.py` (Python) — the canonical Mohammadi panel — imported by every eval `.py`. Rust side already centralizes; mirror it. |
| 3 | **R2/S3 endpoint + creds-export + `aws s3 sync` block** (`R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"` verbatim) | 55 primary files | **MEDIUM-HIGH** — a creds-export or endpoint typo in one fork uploads to the wrong bucket or fails silently mid-fleet; the verdict gate (CLAUDE.md "verify artifacts landed") is itself copy-pasted and drifts. | One sourced `zen-r2.sh` exporting `R2_ENDPOINT` + helper fns (`r2_sync`, `r2_cp`, `r2_ls`); a Python `zen_r2.py` for the builders. |
| 4 | **`run_cross_codec_v*_seed.sh` / `run_tuner_v*_seed.sh` / `eval_v*_pjnd_check.py` version-forks** — recipe per experiment-version | 60 recipes in zensim alone; `eval_v6/v7/v8_pjnd_check.py` **byte-identical** mod version-string (same md5) | **MEDIUM-HIGH** — when a trainer flag must change, you must edit N forks; the ones you miss run stale args and produce mislabeled bakes. `janitor_w44_216/219/229.sh` already **drifted** (13/222 lines differ after normalizing sweep-id — not the intended diff). | One parameterized recipe + a per-experiment config file (env block or TOML). Proven model already in-repo: `_picker_lib.py` + thin wrappers. |
| 5 | **`onstart_*.sh` boot-glue boilerplate** (~10 scripts: env-hydrate-from-`/proc/1/environ` + verify-baked-tools + R2-fetch + heartbeat) | ~10 boot scripts | **LOW** (corrected). NOTE: the original entry here ("vast.ai orchestration forked two ways — coefficient Rust vs zenmetrics shell, 2,454 LOC") was a **VERIFIED-FALSE category error**. zenmetrics orchestration is the **4957-LOC tested Rust crate `crates/zenfleet-vastai`** (5 clap subcommands, async tokio worker, 13-test JSON parser); the bash is the DEPRECATED legacy chain being migrated away (`onstart_v2/v3` marked deprecated). The coefficient↔zenmetrics Rust overlap is a vast.ai-only LIB dedup tracked in `dedup_VERIFIED_synthesis_2026-05-26.md` Tier-2 #11, NOT a script-layer item. Only the onstart boot-glue is in-scope here. | `scripts/lib/zen-fleet.sh` for the onstart hydrate/verify boilerplate; the orchestration core is already zenfleet-vastai (finish the bash→Rust migration). |

> **AUTHORITY NOTE:** this doc is a Phase-1 grep-level first pass. The
> verified, deep-read inventory is `dedup_VERIFIED_synthesis_2026-05-26.md`
> — defer to it where they disagree. Several counts here (R2 blocks,
> the cloud "fork", file totals) were corrected by the deep-read ledgers.

**Single highest bug-risk-reduction action:** **Cluster #1** — force every
parquet builder through `join_safety.safe_merge` and gate bare `pd.merge(` in
CI. It is the literal recurrence surface of the incident that triggered this
whole audit, the fix already exists (`scripts/canonical_corpus/join_safety.py`
+ `test_join_safety.py`), and adoption is only 3/38.

---

## Per-class detail table

### Class 1 — Parquet build/join Python (THE corruption surface)

The tested guard exists and is good — it just isn't used.

- **Tested home:** `zensim/scripts/canonical_corpus/join_safety.py` (+
  `test_join_safety.py`). Provides validated-cardinality merges.
- **Adopters (3):** `canonical_corpus/build_canonical_2026_05_21.py`,
  `canonical_corpus/build_canonical_parquets.py`, `test_join_safety.py`.
- **Unguarded raw-`pd.merge`/`groupby` builders (35 files)** across
  `zensim/scripts/{v_next,exp_*,canonical_corpus,picker_data_prep}`,
  `zenmetrics/scripts/sweep/build_per_codec_training*.py`,
  `coefficient/scripts/merge_csvs.py` + `merge_pareto_fit.py`,
  `jxl-encoder/scripts/zenjxl-tuning-sweep/merge_w44_*_cells.py`,
  `zenanalyze/zentrain` joins. Representative high-traffic offenders:
  - `zensim/scripts/v_next/merge_iwssim_into_safesyn.py`
  - `zensim/scripts/merge_safesyn_cvvdp.py`
  - `zenmetrics/scripts/sweep/build_per_codec_training.py:1` (R2-fed)
  - `jxl-encoder/scripts/zenjxl-tuning-sweep/merge_w44_216_cells.py` ×5
- **Risk:** CRITICAL — silent many-to-many / key-dtype / null-target joins.
  Same class as the 2026-05-25 incident.
- **Consolidation:** make `join_safety` a required import; migrate the 35;
  add CI gate `! grep -rn 'pd\.merge(' scripts/ --include='*.py'` (allow only
  via the wrapper). Effort: M.

### Class 2 — IQA-stat re-implementations

| File:line | defs |
|---|---|
| `zensim/scripts/mohammadi_eval.py:39,43,48,56,62,82` | srocc, plcc, krocc, outlier_ratio, pwrc, z_rmse_per_sample |
| `zensim/scripts/exp_ensemble/eval_ensemble_2026-05-18.py:67,72,76,107,120,135,149` | srocc, krocc, pearson_abs, z_rmse, pwrc, outlier_ratio, mohammadi_panel |
| `zenanalyze/zentrain/tools/zensim_metric_train.py:428` | srocc |
| `zensim/scripts/dial_bug_audit/run_dial_audit.py:155` | srocc_sign_tolerant |
| `zensim/scripts/v_next/{ensemble_seeds,per_band_step5,butter_concordance_audit}.py:19/27/27` | spearman ×3 |
| `zensim/scripts/v_next/v0_20b/finetune_head.py:108` | spearman_abs |
| `zensim/scripts/v_next/v0_22_iw_option_c_alpha_sweep.py:76` | z_rmse |
| `zensim/scripts/v_next/aggregate_lr_retune.py:95` | srocc_only |
| `coefficient/scripts/spearman_prune.py:39` | spearman |
| `jxl-encoder/scripts/analyze_smart_fanout.py:80` | spearman |
| `zenanalyze/zentrain/tools/correlation_cleanup.py:158` | spearman_corr_matrix |

- **Count:** 14 distinct primary files, ≥25 `def`s of the same handful of stats.
- **Risk:** HIGH — verdict-gate divergence. `mohammadi_eval.py` and
  `eval_ensemble_2026-05-18.py` are the two most complete; they should be the
  seed for the canonical module.
- **Consolidation:** `zen_stats.py` (mirror the Rust `bake_verdict` panel),
  one import everywhere. Effort: S-M.

### Class 3 — R2/S3 sync block

- **Verbatim line** `R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"`
  in every `finalize_*`/`janitor_*`/`launch_*` in
  `jxl-encoder/scripts/zenjxl-tuning-sweep/` and every
  `zenmetrics/scripts/sweep/*.sh`. Plus the Python builders construct the
  same URL: `coefficient/scripts/selector_corpus/upload_to_r2.py:44`,
  `zenmetrics/scripts/sweep/build_per_codec_training.py`.
- **Count:** 55 primary `.sh`/`.py` files embed the endpoint string
  (288 incl. worktree clones).
- **Risk:** MEDIUM-HIGH — wrong-bucket / failed-upload, and the
  "verify-artifacts-landed" gate is itself copy-pasted.
- **Consolidation:** `scripts/lib/zen-r2.sh` (sourced) exporting
  `R2_ENDPOINT` + `r2_sync`/`r2_cp`/`r2_ls`/`r2_verify`; `zen_r2.py` for
  builders. Effort: S.

### Class 4 — Training-run recipes + eval-check forks

- **zensim recipes (`run_*`/`train_*`/`*_seed*.sh`): 60**, of which 53 call
  the canonical Rust `zensim_mlp_train`. Families:
  `run_cross_codec_v{2,3,4,4b,5,6,7,8,9}_seed.sh` (9),
  `run_tuner_*seed*.sh` (8), `run_persample_*`, `run_metric_inputs_*` (3),
  `v11_*/run_*` (10), `v12/v13_*/run_*`.
- **Byte-identical forks proven:** `eval_v6_pjnd_check.py` ==
  `eval_v7_pjnd_check.py` == `eval_v8_pjnd_check.py` (all 163 lines / 6136 B;
  identical md5 after normalizing the version token). Only the docstring +
  `glob` pattern differ.
- **Shared body:** v5↔v6 cross-codec recipes share the entire trainer
  invocation (`--group safesyn:...:1.0:0.0`, `--target-column mix_cv40_iw60`,
  feature-set, anchor pool); the *only* real diff is which 2-3 flags are
  parameterized vs hard-coded.
- **Drift evidence:** `janitor_w44_216.sh` vs `219.sh` differ by 13/222 lines
  after normalizing the sweep-id — i.e. 94% duplicated but already silently
  diverged.
- **Risk:** MEDIUM-HIGH — stale-arg bakes, mislabeled outputs.
- **Consolidation:** one `run_seed.sh` + per-experiment config (the
  `eval_v*_pjnd_check.py` trio → one `eval_pjnd_check.py --version vN
  --glob ...`). Proven model in-repo: `zenanalyze/zentrain/tools/_picker_lib.py`
  (shared) + thin per-codec wrappers. Effort: M.

### Class 5 — Sweep launchers + cloud orchestration

- **Two parallel stacks for one job:**
  - coefficient Rust: `src/cloud/vastai.rs` (867), `src/cloud/do_droplet.rs`,
    `src/cloud/batch.rs`, `src/bin/{vastai_dispatch,vastai_worker,do_worker,
    cloud_worker,batch_cli}.rs` (≈1,650 LOC orchestration).
  - zenmetrics shell: `scripts/sweep/` — 15 vastai-touching `.sh`,
    `dispatch.sh`, `sweep_janitor.py`, `fleet_status.sh`, plus the
    `zenjxl-tuning-sweep/launch_*_fleet.sh`+`janitor_*`+`finalize_*` mirror in
    jxl-encoder.
- **`onstart_*.sh` boilerplate:** 10 scripts, 2,454 LOC, each re-implementing
  env-hydrate-from-`/proc/1/environ` (14 scripts do this), verify-baked-tools,
  R2-fetch-chunks, heartbeat, worker-loop.
- **Risk:** MEDIUM — chunk-claim-race / heartbeat fixes land in one stack only.
- **Consolidation:** `scripts/lib/zen-fleet.sh` (hydrate+verify+claim+
  heartbeat) sourced by every onstart; medium-term converge on one
  orchestration core (the master doc Tier-2 already flags the
  coefficient↔zenmetrics fleet parallel). Effort: M-L.

### Class 6 — Calibration fitters (logistic / PCHIP / affine)

- **Spline/PCHIP:** `zensim/scripts/fit_output_spline.py`,
  `v_next/calibrate_v9_spline.py`, `v_next/v11_ssim2/calibrate_v11_balanced_spline.py`,
  `v_next/calibrate_balanced_v9_spline.py`.
- **Per-codec affine:** `v_next/fit_per_codec_calibration.py`,
  `v_next/v11_e/fit_per_codec_affine.py`; Rust dup at
  `zensim-validate/src/bin/affine_calibrate.rs`.
- **4-param logistic / curve_fit:** `mohammadi_eval.py`,
  `verify_mohammadi_anchor.py`, `v0_22_iw_option_c_alpha_sweep.py`;
  jxl-encoder `cvvdp_build_calibration_table.py` + `cvvdp_calibration_seed.py`
  + `zensim_calibration_seed.py`.
- **Risk:** MEDIUM — a calibration α/β that ships as a `const` (CLAUDE.md
  source-informing rule) must be reproducible from ONE fitter, not 3 spline
  variants.
- **Consolidation:** `zen_calibrate.py` {affine, logistic4p, pchip} + the
  affine math already in Rust `affine_calibrate.rs` as the canonical home.
  Effort: M.

### Class 7 — Eval harness `.py` + Rust examples overlap

- **Rust bake-forward tools in `zensim-validate/src/bin/` (11)** all load a
  bake and forward through `Predictor` (`bake_verdict` 1597 LOC,
  `bake_compare` 1232, `qsweep_eval` 705, `predict_features_with_bake` 543,
  `score_pair_with_bake` 332, `eval_bake_per_band` 305, `ensemble_score_rows`
  292, `ensemble_mix`, `score_pairs_tuner`, `preview_stats_demo`,
  `zensim_picker_infer`). Each re-does the `has_nontrivial_feature_transforms()
  → predict_transformed` dispatch the CLAUDE.md flags as the recurring
  metadata-loss bug site (Step 7). N homes for one dispatch = N places to get
  it wrong.
- **Two MLP trainers for one job:** Python
  `zenanalyze/zentrain/tools/zensim_metric_train.py` (1039 LOC) vs Rust
  `zensim-validate/src/bin/zensim_mlp_train.rs` (2691 LOC). Cross-codec
  recipes call the Rust one; the Python one is parked but still in tree.
- **Risk:** MEDIUM-HIGH (dispatch) / LOW (parked trainer).
- **Consolidation:** see "Rust examples sprawl" census below. Effort: L.

### Class 8 — Feature-extraction wrappers + corpus assembly

- **Rust feature extractors (overlapping):**
  `zensim-bench/examples/extract_features_372col.rs` +
  `extract_features_372col_omni.rs` (omni = +1 codec axis, near-identical),
  `zensim-validate/src/bin/{extract_ex4_features,extract_pair_features}.rs`,
  `zensim-picker-prep/src/bin/{extract_features,cross_codec_butter_features}.rs`,
  `coefficient/examples/extract_features_{corpus,full_catalog,v3ext}.rs` (3) +
  `extract_zenanalyze_features.rs`.
- **Training-data generators:** `coefficient/examples/generate_zensim_training.rs`
  (canonical, per MEMORY.md) vs Python corpus-assembly in
  `coefficient/scripts/selector_corpus/build_selector_*.py` (5).
- **Risk:** MEDIUM — feature-vector layout drift between extractors silently
  mislabels columns (the 300 vs 372 vs 343 col confusion is already documented
  in CLAUDE.md).
- **Consolidation:** one `extract_features` CLI with `--corpus`/`--cols`/
  `--codecs` flags subsuming the 372col/omni/ex4/pair variants. Effort: M.

### Class 9 — picker-config matrix (already half-consolidated — keep as model)

- **31 `*_picker_config*.py`** in `zenanalyze/zentrain/examples/`:
  zenjpeg 8, zenwebp 8, zenavif 11, zenjxl 4. These are mostly thin config
  dicts (32-119 LOC) that already import the shared
  `zentrain/tools/_picker_lib.py` (9 importers). This is the **target pattern**
  for Classes 4/7/8 — *not* a problem cluster, but the prune-variant
  proliferation (`_aggprune`/`_aggprune20`/`_ultraprune`/`_pruned`/
  `_lightprune`/`_v2`/`_v3_stable`/`_v04`/`_v04full`) is dead-config sprawl
  (CLAUDE.md §6 "delete superseded scripts same-day"). Recommend: retire
  non-`_v3_stable` variants once a winner ships. Effort: S (deletion).

---

## "Recipe explosion" census

| Repo | Recipe-shaped scripts | Could collapse to |
|---|---|---|
| zensim `scripts/` | **60** (`run_*`/`train_*`/`*_seed*.sh`; 53 call `zensim_mlp_train`) | **~4 templates**: `run_seed.sh` (trainer driver, config-file-driven), `run_sweep.sh` (multi-seed fan-out), `eval_check.py` (the pjnd/multi-band checks), `calibrate.py`. + 1 config file per experiment-version. |
| jxl-encoder `zenjxl-tuning-sweep/` | **25** (`build_w44_*`×4, `finalize_*`×5, `janitor_*`×5, `launch_*_fleet`×6, `merge_*_cells`×5) | **~5 templates** parameterized by `$SWEEP_ID` (build/finalize/janitor/launch/merge), 1 sweep-config each. |
| jxl-encoder root `scripts/` | **13 `analyze_*_ab.py` + 9 `bench_*_ab.sh`** | **~2 templates**: `analyze_ab.py --before --after`, `bench_ab.sh --label`. |
| zenmetrics `scripts/sweep/` | **17** (`onstart_*`×10, `launch_*`, `run_*`) | **~3 templates**: `onstart.sh` (sourcing `zen-fleet.sh`, mode flag), `launch_fleet.sh`, `run_local.sh`. |

**Bottom line:** ~115 recipe-shaped scripts across the four repos collapse to
**~14 parameterized templates + per-experiment config files.** The
`_picker_lib.py` pattern already in-repo proves this works (31 configs →
1 shared lib + thin wrappers).

---

## "Rust examples sprawl" census

| Crate / group | examples+bins | Should become |
|---|---|---|
| coefficient `examples/` + `src/bin/` | **148** | The biggest single sprawl. Sub-clusters: `sweep_*`×21, `gpu_butter_*`×11, `knob_eval*`×8, `gpu_{ssim2,zensim}_*`×6, `*_only_v3`/`*_v3`×9, `metrics_v3_rgba*`×3, `extract_features*`×3, `pilot_*`×5, `sweep_xyb_*`×8. → one `coef-sweep` CLI with subcommands (`sweep`, `knob-eval`, `gpu-profile`, `extract-features`, `fit-rd`, `selector`). |
| zensim-validate `src/bin/` | **23** | one `zensim-tool` CLI: `verdict`/`compare`/`score`/`eval-band`/`extract`/`calibrate`/`mlp-train`/`picker`/`holdout-overlap`/`ensemble`. 11 of these forward a bake through `Predictor` and re-do the transform-dispatch — collapsing them removes 11 copies of the bug-prone dispatch. |
| zenmetrics GPU-crate `examples/` | **68** (across 6 GPU crates) | Recurring basenames *per crate*: `bench_t4_warm.rs`×4, `end_to_end.rs`×4, `bench_strip_vs_whole*.rs`×4, `bench_warm_ref`/`blur_parity`/`diffmap_overhead`/`heaptrack_strip_12mp`/`parity_real_image`×2 each. → a shared `zen-metric-bench` harness crate parameterized by metric; per-crate examples shrink to a 5-line `main` calling it. (Borderline lib-bench; flagged here because the *harness body* is copy-pasted glue, not metric API.) |
| zenanalyze `tools/` + `zentrain/` examples+bins | **22** | picker trainers `v06`→`v15` version-forks (`v06_*`, `v07_*`, `v10_*`, `v12_*`, `v14_*`, `v15_*`) → one `metapicker-train --version`. |

**Bottom line:** ~280 ML/sweep `examples`+`bins` in primary repos. The
correctness-relevant collapse is **zensim-validate's 23 → one `zensim-tool`
CLI**, because 11 of them duplicate the `predict_transformed` dispatch that
CLAUDE.md names as a recurring silent-garbage bug.

---

## Prioritized consolidation plan (phased, by bug-risk reduction)

**Phase 1 — kill the recurrence surface (Class 1+2+3). Effort: M, ~1-2 days.**
1. CI grep-gate: fail on bare `pd.merge(`/`pd.concat(` in `*/scripts/*.py`
   outside the `join_safety` wrapper. Migrate the 35 builders. *(Directly
   closes the 2026-05-25 incident class.)*
2. Land `scripts/lib/zen_stats.py` (Mohammadi panel, seeded from
   `mohammadi_eval.py` + `eval_ensemble_2026-05-18.py`); replace the 14
   forks; one parity test vs the Rust `bake_verdict` panel.
3. Land `scripts/lib/zen-r2.sh` + `zen_r2.py`; source it in the 55 files.

**Phase 2 — stop the fork explosion (Class 4+5). Effort: M, ~2-3 days.**
4. Collapse `eval_v{6,7,8}_pjnd_check.py` → one `eval_pjnd_check.py --version`.
5. Parameterize `run_cross_codec_v*_seed.sh` → `run_seed.sh` + config files.
6. Extract `scripts/lib/zen-fleet.sh` (hydrate+verify+claim+heartbeat) sourced
   by the 10 `onstart_*` + the jxl-encoder `worker.sh`/`onstart.sh`.
7. Parameterize the jxl-encoder `zenjxl-tuning-sweep/{build,finalize,janitor,
   launch,merge}_w44_*` by `$SWEEP_ID` (fixes the already-drifted janitors).

**Phase 3 — CLI consolidation (Class 7+8 + examples sprawl). Effort: L.**
8. `zensim-tool` CLI subsuming the 23 zensim-validate bins (single
   `predict_transformed` dispatch home). *(Highest correctness payoff in this
   phase.)*
9. `coef-sweep` CLI subsuming the 148 coefficient examples.
10. Shared `zen-metric-bench` harness for the 68 zenmetrics GPU examples.
11. Retire dead picker-config prune-variants (Class 9) after a winner ships.

**Phase 4 — housekeeping.** Delete the parked Python `zensim_metric_train.py`
(1039 LOC) once the Rust trainer is confirmed canonical for all recipes; per
CLAUDE.md §6 "delete superseded scripts same-day".

---

## What this does NOT cover

This is the **operational script-layer** companion. Out of scope here, fully
covered by `benchmarks/dedup_inventory_master_2026-05-26.md`:

- **Well-factored library dedup** — zenyuv RGB↔YCbCr color math (3 impls),
  zencodec encode/decode builders, pixel types, SIMD kernels, the zenpredict
  runtime.
- **GPU perceptual-metric *backends*** (zenmetrics CubeCL vs coefficient
  cudarse) — that's a *library impl* parity question (master doc Tier-0 #2).
  This doc only flags the *harness glue* around them (Class 7 zenmetrics
  examples).
- **Reusable CI workflows / GitHub Actions** dedup.

Line: *is it a reusable, tested library function?* → master doc. *Is it
operational glue someone pasted into the Nth experiment?* → this doc.
