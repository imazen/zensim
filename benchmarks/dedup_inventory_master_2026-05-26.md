# Cross-repo deduplication inventory — master doc

**Date:** 2026-05-26
**Task:** #222 (Cluster A) — sibling agents append further clusters
**Motivation:** the 2026-05-25 parquet corruption was a metric-join bug in ad-hoc, untested join logic. Duplicated logic = N homes for one bug; only one home gets the fix. This master doc collects the duplication-class inventory across the zen workspace so each class can be consolidated into shared, tested code with a single owner.

This doc aggregates per-cluster audits. **Cluster A** (below) extends the 3-repo audit at `benchmarks/cross_repo_duplication_audit_2026-05-26.md` (task #220, covered zensim + zenanalyze + zenmetrics) by re-verifying it and adding `coefficient`, `zensally`, `zenpipe`. **Cluster B** (and any later clusters) are appended by sibling agents under their own headings — additive, never clobbering an existing cluster section.

All file:line references are on each repo's main checkout. Worktree copies under `.claude/worktrees/`, `node_modules/`, `target/`, and per-seed benchmark snapshots were excluded from counts.

---

## Cluster A — ML / data / sweep (zensim, zenanalyze, zenmetrics, coefficient, zensally, zenpipe)

### A.0 Scope verdict per repo

| Repo | rs / py / sh | In-cluster duplication? | Verdict |
|---|---|---|---|
| zensim | — | yes (prior audit) | Re-verified below; findings hold |
| zenanalyze | — | yes (prior audit) | Re-verified below; findings hold |
| zenmetrics | — | yes (prior audit; **canonical** sweep + GPU-metric owner) | Re-verified; findings hold |
| **coefficient** | 292 / 54 / 3 | **YES — substantial** | New findings A.6–A.11; second GPU-metric backend, second cloud-orchestration system, third `CodecFamily` order, own ledger/feature-JSON format, 3 per-source metric joins |
| **zensally** | 40 / 11 / 1 | **NO cross-repo overlap** | Self-contained PyTorch saliency trainer; internal `train*/model*` fork family only (A.12). No parquet joins, no IQA stats, no R2 sweep, no `zentrain.*` keys |
| **zenpipe** | 371 / 5 / 16 | **NO cross-repo overlap** | Pure streaming pixel-pipeline. Zero ssim2/butter/zensim/dssim refs in src, zero parquet joins, zero R2-sweep boilerplate, zero IQA stat reimplementations. Consumes `zensim 0.2.5` + `fast-ssim2` as deps (correct). Its 16 `.sh` are playwright/fuzz-seed scripts; its 5 `.py` are docgen/serve/fuzz-corpus/filter-audit — none touch the metric/sweep cluster |

**Net: of the 3 new repos, only `coefficient` adds cross-repo duplication.** zensally and zenpipe are correctly isolated and depend-not-duplicate where they touch metrics.

### A.1 Re-verification of the prior 3-repo audit (task #220)

| Prior finding | Status 2026-05-26 | Note |
|---|---|---|
| Class 1: metric→feature joins, `join_safety.py` adopted by 1 of 36 | **HOLDS (slightly better)** | `join_safety.py` now imported by **2** builders (`build_canonical_2026_05_21.py`, `build_canonical_parquets.py`) + its test. Still 34+ ad-hoc joins unguarded. |
| Class 2: `panel.rs` extracted from `bake_verdict.rs` but bake_verdict still runs its own copy | **HOLDS** | `bake_verdict.rs` has **5** local stat fns, `panel.rs` has **11**; `bake_verdict.rs` has **no** `use ...panel` import. The verdict gate still runs a byte-copy. |
| Class 3: 122 bare `zentrain.*` literals | HOLDS (not re-counted; no contradicting evidence) | — |
| Class 4: 26 `.sh` R2 endpoint boilerplate | HOLDS | coefficient adds 1 more shell + 2 more endpoint constructors (A.8) |
| Class 5: near-fork script families | HOLDS | coefficient adds its own fork families (A.10) |

### A.2 Cross-class summary table (Cluster A, all 6 repos)

| Class | Logic | # copies (repos) | Risk | Proposed single home |
|---|---|---|---|---|
| **1** | **Metric→feature parquet/CSV joins** | **39 total** = 36 (zensim/zenanalyze/zenmetrics, prior) **+ 3 coefficient** (per-`source_hash` / `source_path` joins, Mode-B broadcast shape) | **CRITICAL** | `zen_corpus_join` (promote `join_safety.py`); coefficient joins are CSV-side so need a pandas-compatible `safe_per_source_join` variant |
| **2** | IQA stats (SROCC/PLCC/KROCC/Z-RMSE/PWRC/RMSE/kendall) | Rust **11** (10 prior + 1 coefficient `kendall_tau`); Python **25** (23 prior + coefficient `spearman` + `spearmanr` use) | HIGH | new `zen-iqa-stats` crate + mirrored `zen_stats.py`; coefficient's `examples/ba_ssim2_cross_check.rs:43` + `scripts/spearman_prune.py:39` + `scripts/feature_utility.py:170` join in |
| **3** | `zentrain.*` / `zenpicker.*` ZNPR metadata keys as bare literals | 122 prior + coefficient `scripts/inject_family_order_and_bake.py` (`"zenpicker.family_order"`) | HIGH | `zenpredict::metadata::keys` (Rust) + generated `zentrain_keys.py` |
| **4** | R2/S3 endpoint + creds + `S3()` boilerplate | 26 `.sh` prior + coefficient (`src/store/r2.rs:116`, `scripts/selector_corpus/upload_to_r2.py:44`) | MEDIUM | `zen-r2-lib.sh` + a shared Rust `zen-r2` helper (coefficient's `r2.rs` + zenmetrics `vastai-fleet/src/worker/r2.rs` are two Rust copies) |
| **5** | Parquet/CSV read-write + manifest helpers | prior 18 loaders; coefficient uses CSV/TSV (`load_pareto_rows`, `load_features` in `feature_utility.py`, `fit_*` scripts) | MEDIUM | shared loader lib; coefficient is TSV-first so lower priority |
| **6** | Sweep / cloud orchestration (vast.ai, GCP Batch, DO droplets, chunk workers, Dockerfiles, onstart) | zenmetrics canonical sweep infra **+ coefficient's entirely separate `src/cloud/` (vastai.rs 29k, batch.rs 35k, do_droplet.rs 18k) + `src/bin/vastai_worker.rs` (470 LOC) + 2 launcher `.sh`** | MEDIUM-HIGH | no merge proposed (different providers); but `vastai.rs` overlaps `zenmetrics/crates/vastai-fleet` — candidate for shared `zen-vastai` crate |
| **7** | GPU perceptual-metric scoring (ssim2/butteraugli/dssim/zensim) | **zenmetrics: CubeCL backend** (`butteraugli-gpu`/`ssim2-gpu`/`zensim-gpu`/`dssim-gpu`/`cvvdp-gpu`/`iwssim-gpu`, lilith/cubecl fork). **coefficient: turbo-metrics/cudarse backend** (`ssimulacra2-cuda`/`butteraugli-cuda`/`dssim-cuda` via `../turbo-metrics`, NVIDIA-only). | **HIGH (functional dup)** | Two GPU metric implementations of the *same* metrics. zenmetrics is the documented canonical owner; coefficient's `src/gpu.rs` (299 LOC) should migrate to `zen-metrics` once CubeCL covers its needs |
| **8** | Feature extraction / 372-schema / metric-ledger emission | coefficient `generate_zensim_training.rs` writes own `metric-ledger.jsonl` + per-`source_hash` feature JSON (`extract_zenanalyze_features.rs`); zenmetrics emits 305-col parquet sidecars; zensim canonical-corpus builders emit 372-col parquet | MEDIUM | two data-emission formats for the same (ref, dist) → metric facts; coefficient correctly *calls* `zenanalyze::try_analyze_features_rgb8` (no feature-math dup), but its **output schema** forks |
| **9** | `CodecFamily` enum + family order | **THREE orderings** (see A.7) | **HIGH** | `zenpicker::CodecFamily` is the documented owner |

### A.6 coefficient — what it adds that the prior audit missed (NEW)

coefficient is the training-data generator + codec-RD-experiment repo. It is **not** a thin consumer of the other repos — it carries parallel implementations of four cluster concerns:

1. **A second GPU perceptual-metric backend.** `src/gpu.rs` (`GpuMetrics`, 299 LOC) wraps `ssimulacra2-cuda`, `butteraugli-cuda`, `dssim-cuda`, `cudarse-driver`, `cudarse-npp` from `../turbo-metrics` (`Cargo.toml:109-114`). zenmetrics computes the *same* metrics on a *different* backend (CubeCL / lilith-cubecl fork, `crates/{ssim2-gpu,butteraugli-gpu,dssim-gpu,zensim-gpu}`). Same metric, two GPU codepaths → any metric-definition fix (e.g. a linearization or upsample change) must land twice or the two disagree silently. This is the highest-value *functional* dup in the cluster.

2. **A second cloud/batch orchestration system.** `src/cloud/` carries `vastai.rs` (29 KB), `batch.rs` (35 KB, GCP Batch), `do_droplet.rs` (18 KB, DigitalOcean), `mock_batch.rs`, `quota.rs`, plus `src/bin/vastai_worker.rs` (470 LOC) and `scripts/{vastai_create_workergroup,submit-optimization-job}.sh`. zenmetrics owns the canonical vast.ai sweep infra (`scripts/sweep/*`, `crates/vastai-fleet`). The `vastai.rs` ↔ `vastai-fleet` overlap is real (provision/dispatch/worker-loop); the GCP/DO paths are coefficient-unique.

3. **A third `CodecFamily` enum with a third discriminant order** (see A.7) — genuine drift, not just a copy.

4. **Its own metric-ledger + feature-JSON data format.** `generate_zensim_training.rs` writes an append-only `metric-ledger.jsonl` (line 380) and `extract_zenanalyze_features.rs` writes `source_features/<source_hash>.json`. This is a parallel sidecar schema to zenmetrics' 305-col parquet and zensim's 372-col canonical parquet — three formats for "(ref, dist) → metric + features."

What coefficient does **right** (depend, not duplicate): `extract_zenanalyze_features.rs` calls `zenanalyze::try_analyze_features_rgb8` directly (no feature-math reimplementation); `linear-srgb` reuse noted in `Cargo.toml:168`; it depends on `zenanalyze` by path. The duplication is in *orchestration, GPU scoring, enum order, and output schema*, not in the feature kernels.

### A.7 The three-way `CodecFamily` order divergence (HIGH — bug surface)

| Source | Declared / used order | Discriminant note |
|---|---|---|
| `zenpicker::CodecFamily` (documented canonical, CLAUDE.md) | `Jpeg=0, Webp=1, Jxl=2, Avif=3, Png=4, Gif=5` | the spec |
| `coefficient/src/auto_encode/constraints.rs:30` | enum decl `Jpeg, Avif, Jxl, Webp, Png, Gif`; `ALL` in the same order; `decoder_compat_rank` is a *separate* ranking (`Jpeg=0,Png=1,Gif=2,Webp=3,Avif=4,Jxl=5`) | **enum order differs from zenpicker** (Avif/Webp swapped, Jxl/Webp positions differ) |
| `coefficient/scripts/inject_family_order_and_bake.py:23` | `FAMILY_ORDER_CSV = "jpeg,webp,jxl,avif,png,gif"` baked into ZNPR `zenpicker.family_order` metadata | **matches zenpicker, NOT coefficient's own Rust enum** |

The bake-injection script's hardcoded CSV matches zenpicker's order, but coefficient's own Rust `CodecFamily::ALL` does not. A bake produced with one ordering and consumed with another silently mislabels families. Comment at `constraints.rs` claims it is the dispatch's source of truth, but it disagrees with the documented `zenpicker` owner. **This is exactly the silent-mislabel class the audit is meant to surface.**

### A.8 coefficient R2 boilerplate (adds to Class 4)

- `src/store/r2.rs:116` — `format!("https://{}.r2.cloudflarestorage.com", self.account_id)` (Rust endpoint constructor #3 in cluster, after zenmetrics `vastai-fleet/.../r2.rs`).
- `scripts/selector_corpus/upload_to_r2.py:44` — `f"https://{account}.r2.cloudflarestorage.com"` + `env_or_die` creds pattern (Python endpoint constructor; uses `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY`, boto3-or-aws-CLI fallback).
- No `/proc/1/environ` boot-hydration (coefficient's vast.ai worker hydrates differently).

### A.9 coefficient metric/feature joins (adds 3 to Class 1)

All three are **per-source-key joins** — features are one-row-per-source, joined onto pareto/oracle rows that are many-per-source. This broadcasts each source's features across its (codec, q) rows. Correct by intent, but it is the same `left_on=source / right_index` broadcast shape that produced the corruption when the right side wasn't unique-per-key:

| Script | Line | Join | Key | Risk |
|---|---|---|---|---|
| `scripts/optimal_tree.py` | 49 | `p.merge(fdf[fcols], left_on="source_hash", right_index=True, how="inner")` | `source_hash` | MED — relies on `fdf` being unique-indexed by source_hash; no guard |
| `scripts/fit_selector_model.py` | 53 | `oracle.merge(feat, on="source_path", how="inner")` | `source_path` | MED — no uniqueness/constant-per-source guard; then per-row `apply` picks per-q column |
| `scripts/feature_utility.py` | 327 | `pareto.merge(feat_df[feature_cols], left_on="source_hash", right_index=True)` | `source_hash` | MED — has a post-join empty check (`sys.exit(2)`) but no leak/constant guard |

None use a `join_safety`-equivalent. They are CSV/TSV (pandas) rather than parquet, so the existing `join_safety.py` (DuckDB/pyarrow-shaped) does not import cleanly — a pandas-compatible `safe_per_source_join` is the consolidation need.

### A.10 coefficient near-fork families (adds to Class 5)

- **Knob-eval family:** `examples/knob_eval{,_v2,_v3,_v3_ext,_v3_ext_rgba}.rs`, `knob_eval_round2/3.rs`, `knob_eval_phase2.rs`, `reclassify_knob_eval.rs`, `aggregate_knob_eval.rs` (~11 examples, copy-the-last-one evolution).
- **Selector/oracle family:** `sweep_selector{,_large,_v2,_v3}.rs`, `evaluate_selector_v2.rs`, `selector_validation.rs`, `selector_vs_oracle.rs`, `percentile_selector{,_size_aware}.rs`, `ideal_selector.rs`, `fit_selector_{model,tree}.py`, `fit_oracle_tree.py` (RD/oracle picker training; partially overlaps zentrain's picker work).
- **Sweep XYB/chroma family:** `sweep_xyb_*` (8 examples), `sweep_chroma_*` (3), `sweep_pareto_step5*` (3) — RD-sweep forks.
- **Picker-row builders:** `build_picker_rows.rs` ↔ `build_picker_rows_v3ext.rs` (extended fork).

### A.11 coefficient sweep/orchestration detail (Class 6)

`src/cloud/vastai.rs` (29 KB) implements vast.ai template/workergroup/instance provisioning + dispatch — functionally parallel to `zenmetrics/crates/vastai-fleet`. The two are independent Rust implementations of vast.ai control. GCP Batch (`batch.rs`) + DO droplets (`do_droplet.rs`) are coefficient-unique providers (no zenmetrics equivalent — not a dup, but worth noting the cluster has 3 cloud providers across 2 repos). Onstart hydration differs from zenmetrics' `/proc/1/environ` pattern.

### A.12 zensally — internal fork family (NOT cross-repo)

zensally's `training/` has a clear copy-the-last-one fork chain: `train.py → train_v2.py → train_v3.py → train_v3ds.py → train_v3ds32.py` (247/306/215/296/174 LOC) and `model.py → model_v3.py → model_v3ds.py` (188/183/173 LOC). This is the same anti-pattern as the prior audit's `eval_v4→v4b→…` chain, but it is **internal to zensally and shares no logic with the metric/sweep cluster** — pure PyTorch saliency-net training with ONNX export. Two parallel Rust inference crates (`zensally-tract` via `tract-onnx 0.22` vs `zensally-zentract` via `zentract-api` git dep) are an intentional backend choice (compile-tract vs dlopen-plugin), documented in their descriptions — borderline dup but justified by the tract-compile-cost tradeoff. **No consolidation recommended across repos; the internal train-fork chain could be parameterized but is out of cluster scope.**

### A.13 Highest-value extraction for Cluster A

Same as the prior audit's #1 at the repo level (route every join through `join_safety`), but the **single highest-value *new* cross-repo extraction this audit surfaces is a shared `zen-iqa-stats` crate + mirrored `zen_stats.py`**, because:
- It is now provably reimplemented **11× in Rust and 25× in Python across 4 repos** (zensim, zenanalyze, zenmetrics, coefficient), all computing ship/no-ship verdict gates.
- The CLAUDE.md mandate (full Mohammadi panel: SROCC+PLCC+KROCC+OR+PWRC+Z-RMSE) means every one of those copies must agree to the same numerics or verdicts disagree silently.
- A single crate with a CI cross-check that the Rust impl and `zen_stats.py` agree to ±1e-9 on a fixture eliminates the entire class.

The corruption-surface fix (Class 1, route joins through guarded `safe_metric_join` + add a CI integrity test over committed parquets) remains the **highest-risk** consolidation; `zen-iqa-stats` is the highest-value *cross-repo* extraction newly justified by adding coefficient.

### A.14 Consolidation plan (Cluster A — incremental over prior audit)

Prior audit Phases 1–4 stand. Cluster A adds:

| # | Action | Effort | Risk reduced |
|---|---|---|---|
| A-1 | Promote IQA stats to a workspace `zen-iqa-stats` crate (fold `zensim-train-core::stats` + `panel.rs`); migrate the 11 Rust sites incl. coefficient `ba_ssim2_cross_check.rs:43`. Mirror as `zen_stats.py`; migrate 25 Python sites incl. `spearman_prune.py:39`, `feature_utility.py:170`. CI cross-check ±1e-9. | 6h | HIGH — verdict-gate divergence |
| A-2 | Add pandas-compatible `safe_per_source_join` to `zen_corpus_join`; migrate coefficient's 3 per-source joins (`optimal_tree.py:49`, `fit_selector_model.py:53`, `feature_utility.py:327`). | 2h | CRITICAL — corruption shape |
| A-3 | Import `zenpicker::CodecFamily` into coefficient (replace local `constraints.rs:30` enum); reconcile the 3-way order divergence; assert `inject_family_order_and_bake.py` CSV matches the imported enum. | 3h | HIGH — silent family mislabel in bakes |
| A-4 | Decide GPU-metric ownership: migrate coefficient `src/gpu.rs` (turbo-metrics) to `zen-metrics` (CubeCL) OR document why two backends coexist (NVIDIA-only perf vs cross-platform). Until then, add a numeric-parity test that turbo-metrics ssim2 and CubeCL ssim2 agree on a fixture. | 4h+ (or 1h for parity test) | HIGH — two metric impls drift |
| A-5 | Extract shared `zen-vastai` from coefficient `src/cloud/vastai.rs` + zenmetrics `vastai-fleet` (or document the split). | 4h | MEDIUM |
| A-6 | Replace coefficient `src/store/r2.rs` + `upload_to_r2.py` endpoint constructors with the shared `zen-r2` helper from Phase 4. | 1h | MEDIUM |

### A.15 Anti-patterns (Cluster A additions to prior audit's list)

8. **Parallel GPU-metric backends for identical metrics.** turbo-metrics/cudarse (coefficient) vs CubeCL (zenmetrics) both compute ssim2/butteraugli/dssim. A metric-definition change must land in both or they disagree — and there is no parity test asserting they don't.
9. **Enum order forked three ways.** `CodecFamily` declared in coefficient with a different order than the documented `zenpicker` owner, while a Python bake script hardcodes yet a third CSV that happens to match zenpicker but not coefficient's own Rust. No compile error catches the mismatch.
10. **Parallel data-emission schemas for the same facts.** `(ref, dist) → metric + features` is serialized as coefficient's `metric-ledger.jsonl` + per-source feature JSON, zenmetrics' 305-col parquet, and zensim's 372-col canonical parquet — three formats, three join-key conventions, three places a misjoin can hide.
11. **Per-source broadcast joins without a uniqueness guard.** coefficient joins one-row-per-source features onto many-row-per-source pareto/oracle tables on `source_hash`/`source_path` with no assertion that the feature side is unique-per-key — the same broadcast shape as the corruption.

---

*Cluster A generated by a read-only audit (task #222). No source was modified across any of the 6 repos. Sibling agents append further clusters below this section under their own `## Cluster X — ...` headings.*
