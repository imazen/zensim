# Cross-repo code / script duplication audit — consolidation plan

**Date:** 2026-05-26
**Task:** #220
**Scope (READ-ONLY):** `~/work/zen/zensim`, `~/work/zen/zenanalyze`, `~/work/zen/zenmetrics`
**Motivation:** the 2026-05-25 parquet corruption (`benchmarks/DATA_INTEGRITY_root_cause_2026-05-25.md`) was a metric-join bug that lived in an ad-hoc, untested corpus builder. Duplicated logic = N homes for the same bug, one of which gets fixed. This audit finds the duplication so it can be consolidated into shared, tested code.

All file:line references are on each repo's main checkout; worktree copies under `.claude/worktrees/` and per-seed benchmark snapshots (`benchmarks/*/seed_0x*`) were excluded from counts.

---

## 1. Executive summary — top 5 duplication clusters by (risk × count)

| # | Cluster | Risk | Count | One-line |
|---|---|---|---|---|
| **1** | **Ad-hoc metric→feature parquet joins** | **CRITICAL** | **36 scripts** read parquet + merge; only **1** uses the post-incident `join_safety.py` | The exact corruption surface. The shared safety module exists but is adopted by a single builder; 35 other join scripts each roll their own `pd.merge`/DuckDB join with their own key set. |
| **2** | Mohammadi stat panel (SROCC/PLCC/KROCC/OR/PWRC/Z-RMSE/DS-AUC) reimplemented | HIGH | Rust: **10× spearman, 7× pearson** + full panel copy-pasted into `bake_verdict.rs` despite a canonical `panel.rs` in the same crate. Python: **23 stat defs across 13 files**. | A verdict-gate metric computed by N implementations can disagree silently; a fix to one leaves the others wrong. |
| **3** | `zentrain.*` ZNPR metadata keys as bare string literals | HIGH | **54** bare literals in zensim Rust + **68** in Python (122 total); shared `zenpredict::metadata::keys` const used at only **2** sites; Python has **no** shared key module. | A one-char typo in any literal silently disables runtime dispatch (per-sample-α, feature-transforms) → garbage/NaN scores. CLAUDE.md documents this exact failure mode (commit `6ad46950`). |
| **4** | R2 creds-export + endpoint + `S3()`/`r2()` wrapper boilerplate | MEDIUM | **26** `.sh` construct the R2 endpoint; **~8** carry the full `source r2-credentials → export AWS_* → R2_ENDPOINT=` block; **5** redefine `S3()` verbatim; **10** copy the `/proc/1/environ` boot-hydration. | Credential/endpoint handling forked across the fleet; an endpoint change or a creds-format change must be applied in ~26 places. |
| **5** | Near-fork script families (eval / anchor-builder / chunk-worker / per-codec-trainer) | MEDIUM | `eval_v4`==`eval_v4b` (byte-identical, 178 LOC); `eval_v6/v7/v8` ~90% shared; `build_v8/v9/v10_anchor_parquet` ~50% shared; 4 `*chunk_worker.sh` ~40% shared; `build_per_codec_training{,_extended}.py` ~80% shared. | "Copy the last one and tweak" propagates whatever bug the donor had into the new variant. |

**Single highest-risk cluster: #1 (ad-hoc metric→feature joins).** That is literally the code shape that produced the corruption. **The one consolidation that most reduces bug surface: make every metric-join script route through the already-written `scripts/canonical_corpus/join_safety.py` (`safe_metric_join` / `attach_metric_positional` / `assert_*`), promoted to an installable shared lib.** It already encodes the two root-caused failure modes as hard errors; today only 1 of 36 join scripts uses it.

---

## 2. Category A — Bash scripts

| Pattern | # copies | Representative files | Consolidation target |
|---|---|---|---|
| R2 endpoint construction `R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"` | **26 `.sh`** | `zenmetrics/scripts/sweep/{onstart_v3,run_local,dispatch,finalize,launch_backfill}.sh`, `zensim/scripts/v_next/{launch_v16_sweep,sync_unified_to_r2}.sh` | A single sourced `zen-r2-lib.sh` that defines `R2_ENDPOINT`, exports `AWS_*` from `~/.config/cloudflare/r2-credentials`, and provides `r2()`/`S3()`/`s5()` wrappers. |
| Full creds block (`source r2-credentials` → `export AWS_ACCESS_KEY_ID/SECRET` → endpoint) | **~8** | `zenmetrics/scripts/sweep/{launch_backfill,launch_single_instance,run_local,run_local_parallel,finalize}.sh`, `cvvdp_backfill/launch_imazen.sh`, `iwssim_backfill/launch.sh`, `cvvdp_goldens/upload_to_r2.sh` | same `zen-r2-lib.sh` |
| `S3() { aws --endpoint-url "$R2_ENDPOINT" "$@"; }` verbatim | **5** | `onstart_v2.sh:90`, `run_local.sh:31`, `finalize.sh:20`, `vastai_zen_metrics_sweep.sh:92`, `cvvdp_goldens/upload_to_r2.sh:29` | `zen-r2-lib.sh` |
| vast.ai boot hydration from `/proc/1/environ` | **10** | all `zenmetrics/scripts/sweep/onstart_*.sh` | `zen-worker-boot.sh` sourced by all onstart scripts (CLAUDE.md already mandates baked images; the onstart should be a thin shared entrypoint). |
| `onstart_*` worker-boot scripts | **10** | `onstart_{cvvdp_backfill,cvvdp_backfill_imazen,feature_backfill,iwssim_backfill,iwssim_backfill_v14,omni_backfill,source_features,unified,v2,v3}.sh` | one parameterized `onstart.sh` taking a `$JOB` arg; v14 is the canonical per CLAUDE.md, others are deprecated patterns. |
| `*_chunk_worker.sh` fetch→run→upload (~40% shared) | **4** | `{cvvdp,iwssim,metric,omni}_backfill_chunk_worker.sh` (271–390 LOC each; cvvdp↔iwssim differ in ~200/349 lines) | shared `chunk_worker_lib.sh` (claim chunk → fetch corpus → run metric binary → upload sidecar → release) parameterized by metric. |
| `launch_*.sh` vast.ai launchers | **6** | `launch_backfill.sh`, `launch_single_instance.sh`, `cvvdp_backfill/launch{,_imazen}.sh`, `v15/launch_gpu.sh`, `iwssim_backfill/launch.sh` | shared `launch_lib.sh` (image select, env injection, instance loop). |

---

## 3. Category B — Python scripts (the corruption-risk surface)

**98** Python files use `.merge`/`.join`/`pd.concat`. **36** of those both read parquet AND merge. The metric-join subset is where the corruption lived.

### Metric-join scripts (CALL-OUT — every parquet builder that attaches a metric/feature column)

| Script | Join mechanism | Uses `join_safety`? | Risk |
|---|---|---|---|
| `zensim/scripts/canonical_corpus/build_canonical_2026_05_21.py` | `safe_metric_join` | **YES** | LOW (the fixed one) |
| `zensim/scripts/canonical_corpus/join_safety.py` | — (the lib) | n/a | n/a — **the consolidation target** |
| `zenmetrics/scripts/sweep/build_per_codec_training.py:182` | DuckDB 3-way SQL join on `{image_path,codec,q,knob_tuple_json}` ∪ `{image_basename,width,height}` | **no** | **HIGH** — own dedupe + join, no leak/constant-per-ref guard |
| `zenmetrics/scripts/sweep/build_per_codec_training_extended.py` | ~80% fork of the above | **no** | **HIGH** |
| `zensim/scripts/v_next/build_unified_parquet.py:145,195` | `tsv.merge(feat,on=key)` then `merge(corpus_features,on="image_basename")` | **no** | **HIGH** — `image_basename` ref-only merge is the exact Mode-B shape |
| `zensim/scripts/v_next/v11_ssim2_v2/build_v11_substrate_v2.py:98` | `omni.merge(feat,on=OMNI_KEY_COLS)` (`OMNI_KEY_COLS=[image_path,codec,q,knob_tuple_json]`) | **no** | MED — full key, but no uniqueness/leak guard |
| `zensim/scripts/v_next/v12_cvvdp/build_v12_cvvdp_substrate.py` | parquet merge | **no** | MED |
| `zensim/scripts/v_next/build_{v8,v9,v10}_anchor_parquet.py`, `build_empirical_anchor_parquet.py` | parquet merge (50% shared family) | **no** | MED |
| `zenanalyze/zentrain/tools/zensim_metric_train.py:222` | `df.merge(sub,on="ref_basename")` | **no** | **HIGH** — `ref_basename`-only merge = Mode-B broadcast shape |
| `zenanalyze/zentrain/tools/train_hybrid.py`, `refresh_features.py` | parquet merge | **no** | MED |
| `zenanalyze/tools/v14_metapicker_train.py`, `v15_metapicker_train.py`, `v15_compare_pickers.py` | parquet merge | **no** | MED |
| `zenmetrics/scripts/sweep/picker_agreement.py` | parquet merge | **no** | MED |
| `zensim/scripts/canonical_corpus/audit_{metric,mix}_columns.py` | merge (audit-only) | **no** (audit tools) | LOW |
| `zensim/scripts/picker_data_prep/join_features_and_split.py` | merge | **no** | MED |
| `zensim/scripts/v_next/eval_v{4,4b,6,7,8}_pjnd_check.py`, `eval_v9_bake.py`, `cross_codec_*`, `v10_per_dataset_eval.py`, `v5_per_codec_q_range.py`, `measure_tuner_v10_cross_codec.py`, `v0_20_feature_transform_greedy_screen{,_scipy}.py`, `exp_ensemble/eval_ensemble_2026-05-18.py`, `baseline_panels_2026-05-18/extract_panels.py` | parquet merge (eval-side) | **no** | MED — eval joins, lower blast radius than corpus builders but still per-pair-key sensitive |

**Total metric-join scripts found: ~36** (read-parquet ∧ merge). **Of these, ~6 are corpus/training-data BUILDERS doing the high-blast-radius join** (`build_per_codec_training{,_extended}`, `build_unified_parquet`, `build_v11_substrate_v2`, `zensim_metric_train`, the anchor-builder family). **Only 1 uses the shared safety guard.**

### Other duplicated Python logic

| Logic | # copies | Files | Risk | Consolidation target |
|---|---|---|---|---|
| Mohammadi stat panel (`spearman`/`pearson`/`plcc`/`pwrc`/`z_rmse`/`logistic`) | **23 defs / 13 files** | `scripts/mohammadi_eval.py`, `scripts/baseline_panels_2026-05-18/panel.py`, `scripts/dial_bug_audit/run_dial_audit.py`, `scripts/v_next/{per_band_step5,ensemble_seeds,aggregate_lr_retune,butter_concordance_audit,v0_22_iw_option_c_alpha_sweep}.py`, `scripts/v_next/v0_20b/finetune_head.py`, `scripts/exp_ensemble/eval_ensemble_2026-05-18.py`, `zenanalyze/zentrain/tools/{correlation_cleanup,zensim_metric_train}.py`, `zenmetrics/benchmarks/iwssim_smallimg/run_validation.py` | HIGH (verdict gates) | `zen_stats.py` shared module (mirror of Rust `panel.rs`); cross-validate the two impls in CI. |
| 4-param logistic fit | **4+** | `mohammadi_eval.py:24`, `verify_mohammadi_anchor.py:24`, `exp_ensemble/eval_ensemble_2026-05-18.py:93`, (Rust mirror `panel.rs:262`) | HIGH | `zen_stats.py` |
| PCHIP spline (`pchip_derivs`/`pchip_endpoint`/`pchip_eval`) | **3 Python files** (9 fns) | `scripts/v_next/calibrate_balanced_v9_spline.py`, `calibrate_v9_spline.py`, `v11_ssim2/calibrate_v11_balanced_spline.py` | HIGH (calibration math feeds shipped weights) | `zen_calibration.py`; the Rust truth is `output_calibration_spline.rs`. |
| `load_features` / `load_corpus` parquet-loader boilerplate | **18 files** roll their own (despite `zentrain/tools/_picker_lib.py:164` `load_features_raw` existing and used by 7) | HIGH-ish | mostly `zenanalyze/tools/*picker*` + `zentrain/tools/*` | promote `_picker_lib.load_features_raw` to the one loader; migrate the 18. |

---

## 4. Category C — Rust code

| Function / logic | Duplicated in (crate:file:line) | Which crate SHOULD own it |
|---|---|---|
| **Full Mohammadi panel** (`spearman`,`pearson`,`kendall_tau`,`outlier_ratio`,`pwrc`,`z_rmse`,`ds_auc`,`logistic_eval`) | **Canonical:** `zensim-validate/src/panel.rs` (lib module, `pub mod panel`). **Byte-identical private copies:** `zensim-validate/src/bin/bake_verdict.rs:83,105,126,162,196,237,302,327` (panel.rs header literally says "Extracted from bake_verdict.rs" — the extraction was never wired back into bake_verdict) | `zensim-validate::panel` (already exists). bake_verdict.rs must `use zensim_validate::panel`. |
| `spearman` / `spearman_correlation` (10 prod impls) | `panel.rs:50`, `bake_verdict.rs:83`, `ensemble_mix.rs:47`, `eval_bake_per_band.rs:42`, `mlp_train/utils.rs:3`, `main.rs:4526`, `zensim-train-core/src/stats.rs:54`, `zensim-bench/examples/profile_compat_report.rs:940`, `zensim-validate/examples/iw_pyramid_ab.rs:223`, `zensim-regress/examples/slice_real_codec_localization.rs:102` (`srocc`) | A workspace-level `zen-stats` crate (or `zensim-train-core::stats`, which already holds `pearson`/`spearman`). |
| `pearson` (7 prod impls) | `panel.rs:72`, `bake_verdict.rs:105`, `eval_bake_per_band.rs:64`, `main.rs:4532`, `zensim-train-core/src/stats.rs:7`, `profile_compat_report.rs:970`, `iw_pyramid_ab.rs:205` | same |
| `kendall_tau` | `panel.rs:93`, `bake_verdict.rs:126`, `profile_compat_report.rs:991` | `zensim-validate::panel` |
| **PCHIP spline** (`pchip_compute_derivs`,`pchip_endpoint`) — identical except comments | `zensim/src/metric.rs:2257,2291` **and** `zensim-validate/src/output_calibration_spline.rs:114,144` | `zensim::metric` is runtime; the validate-side copy should depend on it (or both depend on a shared `zen-calibration`). |
| `logistic_eval` (4-param) | `panel.rs:262`, `bake_verdict.rs:327` | `zensim-validate::panel` |
| ZNPR bake parse/emit | **NOT duplicated** — everything uses `zenpredict::Model::from_bytes` / `zenpredict-bake`. Only `affine_calibrate.rs:60` does raw `b"ZNPR"` byte-poking on the final layer (justified: surgical layer rewrite). | keep as-is; affine rewrite is a candidate for a `zenpredict` helper but low risk. |
| `CodecKind` / `CodecFamily` enum | `zensim-picker-prep/src/bin/picker_sweep.rs:44`, `cross_codec_butter_features.rs:46`, `zensim-target/src/lib.rs:42`, `zenanalyze/zenpicker/src/lib.rs:92` (`CodecFamily`), `zenmetrics/.../sweep/encode.rs:88` | `zenpicker::CodecFamily` is the documented owner (CLAUDE.md). The two zensim-picker-prep binaries should import it. |

---

## 5. Schema / constants duplication

| Constant / schema | Where defined / redefined | Drift mechanism |
|---|---|---|
| **372 / 300 / 228 / 348 feature-count** | magic-number literals across `zensim-picker-prep/.../cross_codec_butter_features.rs:43` (`NUM_FEATURES=372`), `zensim-validate/src/simd_mlp.rs:659,743`, test files, plus hard-coded `*_372col*.parquet` filenames in `bake_compare.rs`, `preview_stats_demo.rs`, `bake_verdict.rs` | No single `const N_FEATURES_372` — each site repeats the integer. A schema bump (e.g., 372→343 extended-feat) requires hunting every literal. |
| **372-feature block layout / names** (`ssim_mean`, `hf_energy_loss`, `BASIC`/`PEAKS`/`MASKED`/`PSYCHO`/`IW` blocks) | **Source of truth:** doc-comment table in `zensim/src/metric.rs` (`FEATURES_PER_CHANNEL_BASIC/_WITH_PEAKS/_EXTENDED/_IW`). **Hand-mirrored copies:** `zenanalyze/zenpredict-viz/web/feature_layout.js` (header: "Names mirror `zensim/src/metric.rs`" — manual sync), `zenanalyze/zenpredict-viz/web/feature_catalog.json` (generated by `zenpredict-viz/src/bin/build_feature_catalog.rs`). | The JS layout is a hand-maintained mirror — explicitly "lookup-of-last-resort." Diverges silently when metric.rs adds/reorders a feature. The `build_feature_catalog.rs` generator should be the single emitter for BOTH the JSON and the JS table. |
| **`zentrain.*` metadata keys** (~20 distinct keys, e.g. `feature_transforms`×81, `profile`×63, `per_sample_alpha_head`×51) | **Canonical const module exists:** `zenanalyze/zenpredict/src/metadata.rs` (`keys::FEATURE_TRANSFORMS = "zentrain.feature_transforms"`, etc.). **But:** zensim Rust uses **bare string literals 54×** (only **2** sites use `keys::*`); Python uses **bare literals 68×** with **no shared module at all**. | A typo in any of 122 bare literals silently breaks runtime dispatch → wrong/NaN scores. CLAUDE.md commit `6ad46950` documents exactly this. |
| **R2 endpoint / bucket names** | 26 `.sh` + `zenmetrics/crates/zenfleet-vastai/src/worker/r2.rs`, `mod.rs`, `main.rs` | endpoint forked across shell + Rust; the Rust fleet crate has its own copy. |
| **Per-pair join key** `(image_path, codec, q, knob_tuple_json)` | redefined as a local set/list in `join_safety.py:40` (`PER_PAIR_KEY_CANDIDATES`), `build_per_codec_training.py:182`, `build_v11_substrate_v2.py:79` (`OMNI_KEY_COLS`), and inline elsewhere | each builder hard-codes the key; a key change (or a codec adding a knob) must be updated per script. Belongs in the shared join lib. |

---

## 6. Prioritized consolidation plan (phased)

Lead with highest risk × reach. Effort is rough dev-time.

### Phase 1 — Kill the corruption surface (HIGHEST PRIORITY)
1. **Promote `join_safety.py` to an installable shared module** (`zensim/scripts/lib/zen_corpus_join.py` or a tiny `zen-corpus-tools` package importable from all three repos). It already encodes Mode A (mock/human-copy leak) + Mode B (ref-misjoin) as hard errors. **Effort: 1h** (move + make importable).
2. **Migrate the ~6 corpus/training BUILDERS to `safe_metric_join` / `attach_metric_positional`** — in priority order: `build_unified_parquet.py` (uses `image_basename` ref-only merge — the exact Mode-B shape), `zensim_metric_train.py` (`ref_basename`-only merge), `build_per_codec_training{,_extended}.py`, `build_v11_substrate_v2.py`, the anchor-builder family. Each: replace the raw `.merge` with `safe_metric_join`; add the `assert_no_leaked_metric_columns` + `assert_metric_not_constant_per_ref` post-checks. **Effort: 0.5–1h each, ~5h total.** **This is the single change that most reduces bug surface.**
3. **Add a CI integrity test** that runs `assert_no_leaked_metric_columns` + `assert_metric_not_constant_per_ref` against every committed canonical/training parquet, so a misjoin can never re-enter the corpus undetected. **Effort: 2h.**
4. Migrate the ~30 eval-side join scripts opportunistically (lower blast radius). **Effort: ongoing.**

### Phase 2 — One stats implementation
5. **Wire `bake_verdict.rs` to `use zensim_validate::panel`** and delete its 8 private stat copies (they are byte-identical). This is the lowest-effort, highest-symbolic dedup — the verdict gate currently runs a copy, not the shared module. **Effort: 1h.**
6. **Migrate the other 8 Rust spearman/pearson sites** to `zensim-train-core::stats` (already has `pearson`/`spearman`) or fold `stats` + `panel` into a `zen-stats` workspace crate. **Effort: 3h.**
7. **Create `zen_stats.py`** mirroring `panel.rs`; migrate the 13 Python stat files. Add a CI cross-check that `zen_stats.py` and `panel.rs` agree on a fixture to ±1e-9. **Effort: 4h.**
8. **Create `zen_calibration.py`** (PCHIP + 4-param logistic); migrate the 3 calibrate scripts. Dedupe the Rust PCHIP between `metric.rs` and `output_calibration_spline.rs`. **Effort: 3h.**

### Phase 3 — Schema/key single-source-of-truth
9. **Replace all bare `"zentrain.*"` literals with `zenpredict::metadata::keys::*`** in zensim Rust (54 sites). **Effort: 2h.**
10. **Emit a `zentrain_keys.py`** generated from `zenpredict/src/metadata.rs` (or a shared TOML); migrate the 68 Python bare literals. **Effort: 3h.**
11. **Make `build_feature_catalog.rs` the sole emitter of `feature_layout.js`** (generate the JS table instead of hand-maintaining it). **Effort: 3h.**
12. **Define `const N_FEATURES_372` etc.** in one place (zensim/src/metric.rs) and import; replace magic-number literals. **Effort: 1h.**
13. **Import `zenpicker::CodecFamily`** into the two `zensim-picker-prep` binaries; retire their local `CodecKind`. **Effort: 1h.**

### Phase 4 — Bash libs
14. **Create `zenmetrics/scripts/sweep/lib/zen-r2-lib.sh`** (endpoint + creds + `r2()`/`S3()`/`s5()`); `source` it from the 26 R2 scripts. **Effort: 3h.**
15. **Collapse the 10 `onstart_*.sh` to one parameterized `onstart.sh`** + shared `zen-worker-boot.sh` (`/proc/1/environ` hydration once). **Effort: 4h.**
16. **Extract `chunk_worker_lib.sh`** (claim→fetch→run→upload) from the 4 `*_chunk_worker.sh`. **Effort: 3h.**
17. **Dedupe `eval_v{4,4b,6,7,8}_pjnd_check.py`** (v4==v4b byte-identical) into one parameterized eval. **Effort: 2h.**

---

## 7. Anti-patterns observed (the shapes that breed bugs)

1. **Ad-hoc parquet joins with locally-redefined keys.** Every builder hard-codes `(image_path,codec,q,knob_tuple_json)` (or worse, `ref_basename` alone) and calls raw `pd.merge`/DuckDB with no uniqueness or constant-per-ref guard. This is the corruption. The fix (`join_safety.py`) exists but is adopted by 1 of 36 scripts.
2. **"Extract a shared module, then don't migrate the original."** `panel.rs` was extracted *from* `bake_verdict.rs`, yet bake_verdict still runs its own byte-identical copy. Extraction without migration doubles maintenance instead of halving it.
3. **Copy-the-last-one script evolution.** `eval_v4→v4b→v6→v7→v8`, `build_v8→v9→v10`, `build_per_codec_training→_extended`, `onstart_v2→v3`. Each fork inherits the donor's bugs and the family never converges.
4. **Reimplemented stats per consumer.** 10 Rust spearmans, 23 Python stat defs. A verdict gate (ship/no-ship) should be computed by exactly one tested function.
5. **Bare magic strings for protocol keys.** 122 bare `"zentrain.*"` literals; a typo silently disables runtime dispatch (no compile error, no runtime error — just wrong scores). A const module exists in Rust and is bypassed; Python has none.
6. **Hand-mirrored schema across languages.** `feature_layout.js` is a manual copy of `metric.rs`'s feature names, kept in sync by discipline. A generator (`build_feature_catalog.rs`) exists but doesn't emit the JS.
7. **Copy-pasted credential/endpoint boilerplate.** The R2 endpoint + AWS export block is forked across 26 shell scripts and the Rust fleet crate.

---

*Generated by a read-only audit. No source was modified. Counts exclude `.claude/worktrees/` and `benchmarks/*/seed_0x*` snapshot copies.*
