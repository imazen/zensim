# Cross-repo deduplication inventory — master doc

**Date:** 2026-05-26
**Task:** #222 (Cluster A) — sibling agents append further clusters
**Motivation:** the 2026-05-25 parquet corruption was a metric-join bug in ad-hoc, untested join logic. Duplicated logic = N homes for one bug; only one home gets the fix. This master doc collects the duplication-class inventory across the zen workspace so each class can be consolidated into shared, tested code with a single owner.

This doc aggregates per-cluster audits. **Cluster A** (below) extends the 3-repo audit at `benchmarks/cross_repo_duplication_audit_2026-05-26.md` (task #220, covered zensim + zenanalyze + zenmetrics) by re-verifying it and adding `coefficient`, `zensally`, `zenpipe`. **Cluster B** (and any later clusters) are appended by sibling agents under their own headings — additive, never clobbering an existing cluster section.

All file:line references are on each repo's main checkout. Worktree copies under `.claude/worktrees/`, `node_modules/`, `target/`, and per-seed benchmark snapshots were excluded from counts.

---

## Executive summary (cross-cluster synthesis, 2026-05-26)

Synthesized from Cluster A (ML/data/sweep + coefficient) and Cluster B
(codecs + products). Ranked by **correctness risk first, reach second** —
because the incident that triggered this audit (the kadid/tid parquet
corruption) was a *silent divergence* bug, and the highest-priority
duplication clusters are the ones that can silently diverge with no
cross-check.

### Tier 0 — Silent-divergence correctness risks (fix first)

These are the same bug class as the parquet corruption: N
implementations of one thing, no parity test, divergence ships silently.

| Rank | Cluster | What | Risk | Single home |
|---|---|---|---|---|
| 1 | B | **3 independent RGB↔YCbCr/YUV color-math impls** — `zenjpeg/zenyuv` (5363 LOC) + `zenavif/src/yuv_convert*` (3718 LOC, no zenyuv dep) + `zenjxl-decoder/.../ycbcr.rs` | **Color-precision divergence = shipping bug** per CLAUDE.md "ZERO TOLERANCE for image corruption". Three matrix+gamma impls, no cross-check. | promote `zenyuv` to a shared crate; zenavif + zenjxl-decoder depend on it |
| 2 | A | **2 GPU perceptual-metric backends** — zenmetrics CubeCL (`ssim2/butteraugli/dssim-gpu`) vs coefficient cudarse (`src/gpu.rs:92`) | Same metric, two impls, **no parity test** — could score differently and nobody knows. (We just fixed a *join* of ssim2; a 2nd *impl* is the same risk one layer down.) | parity test gate first; then pick one backend or a shared `zen-gpu-metrics` |
| 3 | A | **3-way `CodecFamily` enum order** — zenpicker (canonical) vs coefficient `constraints.rs:30` (different order) vs a CSV that matches zenpicker not its own Rust | **Silent bake-mislabel** — a picker bake tagged with the wrong family. The "safety marker drifted" shape from the iwssim mock leak. | single `CodecFamily` in zenpicker; coefficient depends on it; delete the local enum |
| 4 | A | **IQA stats reimplemented 36×** — 11 Rust + 25 Python across 4 repos, all computing ship-gate verdicts (SROCC/PLCC/Z-RMSE/DS-AUC/KROCC) | A stat bug in one copy gives a wrong ship verdict; copies drift | **`zen-iqa-stats` crate + mirrored `zen_stats.py`** with a ±1e-9 CI cross-check. **Highest-reach extraction in the whole workspace** (spans both clusters). |
| 5 | A | **39 ad-hoc metric→feature joins**, only 2 use `join_safety.py` | The exact corruption shape; 37 unguarded sites remain | route all corpus *builders* through `join_safety.py` (cluster #1 of task #220) |

### Tier 1 — Correctness GAPS found incidentally (cheap, do soon)

| Cluster | Gap | Fix |
|---|---|---|
| B | **`zenraw/ci.yml` missing i686 + cross entirely** — violates CLAUDE.md CI mandate | add the matrix (subsumed by the reusable-workflow extraction below) |
| B | **`fuzz_regression.rs` shipped to only 1 of 10 fuzz-bearing codecs** (zenwebp) | template it to the other 9 |

### Tier 2 — High-reach maintenance (big LOC, low correctness risk)

| Cluster | What | Reach | Single home |
|---|---|---|---|
| B | **Target-quality iterate/rescore loop** — zenjpeg `zq.rs` (1038 LOC) + zenwebp `zensim_target.rs` (1765 LOC, says "mirrors zenjpeg") + zenavif `auto_tune.rs` (504 LOC) | 3 codecs, the user's codec-target work depends on it | **new `zentarget` crate** over a pluggable `Scorer` trait — unify the *control loop* now (picker integration is only 1 prod copy, unify pre-emptively) |
| B | **~17 near-identical `ci.yml`** (win-arm/macos-intel/i686/cross matrix, ~65% identical) | every codec repo | **org reusable workflow** `zen-ci/rust-matrix.yml`; callers shrink to ~20 lines; fixes the zenraw gap by construction |
| B | **zencodec `EncodeJob`/`EncoderConfig` builder boilerplate** — 10-14 mechanical methods × 4 codecs | 4 codecs | shared builder macro or default-impls in `zencodec` |
| A | **R2/S3 boilerplate** — 26 `.sh` build the endpoint, ~8 carry full creds, 5 redefine `S3()` | all sweep scripts | `zen-r2-lib.sh` sourced everywhere |
| A | **2 cloud-orchestration stacks** — coefficient `src/cloud/` (vastai+GCP+DO) vs zenmetrics `zenfleet-vastai` | 2 systems | pick one; out of scope for a quick win |
| A+B | **`panel.rs` vs `bake_verdict.rs` byte-identical stat copy** (subset of Tier-0 #4) | folded into `zen-iqa-stats` | — |

### The one extraction that does the most

**`zen-iqa-stats` (Rust crate + mirrored `zen_stats.py`, CI-cross-checked).**
It is the only item that appears in BOTH clusters (ML verdict gates +
codec target-loop scoring), it's reimplemented 36×, and a bug in any
copy produces a wrong ship decision. It also dissolves the
`panel.rs`/`bake_verdict.rs` duplication for free. Medium effort, highest
blast-radius reduction.

**The one most-urgent for correctness:** promote `zenyuv` to a shared
crate (Tier-0 #1) — three independent color-math implementations is a
latent image-corruption shipping bug per the project's zero-tolerance
rule, and there is no cross-check today.

### Suggested sequencing

1. **Parity-test the two GPU metric backends** (Tier-0 #2) — cheapest way to
   convert a silent risk into a known quantity; may reveal they already agree.
2. **`zen-iqa-stats`** (Tier-0 #4) — highest reach, dissolves several sub-dups.
3. **`zenyuv` shared crate** (Tier-0 #1) — highest correctness urgency.
4. **Single `CodecFamily`** (Tier-0 #3) — small, kills a silent-mislabel risk.
5. **Reusable CI workflow** (Tier-2) — fixes the zenraw gap, big LOC win.
6. Everything else as capacity allows.

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
| **4** | R2/S3 endpoint + creds + `S3()` boilerplate | 26 `.sh` prior + coefficient (`src/store/r2.rs:116`, `scripts/selector_corpus/upload_to_r2.py:44`) | MEDIUM | `zen-r2-lib.sh` + a shared Rust `zen-r2` helper (coefficient's `r2.rs` + zenmetrics `zenfleet-vastai/src/worker/r2.rs` are two Rust copies) |
| **5** | Parquet/CSV read-write + manifest helpers | prior 18 loaders; coefficient uses CSV/TSV (`load_pareto_rows`, `load_features` in `feature_utility.py`, `fit_*` scripts) | MEDIUM | shared loader lib; coefficient is TSV-first so lower priority |
| **6** | Sweep / cloud orchestration (vast.ai, GCP Batch, DO droplets, chunk workers, Dockerfiles, onstart) | zenmetrics canonical sweep infra **+ coefficient's entirely separate `src/cloud/` (vastai.rs 29k, batch.rs 35k, do_droplet.rs 18k) + `src/bin/vastai_worker.rs` (470 LOC) + 2 launcher `.sh`** | MEDIUM-HIGH | no merge proposed (different providers); but `vastai.rs` overlaps `zenmetrics/crates/zenfleet-vastai` — candidate for shared `zen-vastai` crate |
| **7** | GPU perceptual-metric scoring (ssim2/butteraugli/dssim/zensim) | **zenmetrics: CubeCL backend** (`butteraugli-gpu`/`ssim2-gpu`/`zensim-gpu`/`dssim-gpu`/`cvvdp-gpu`/`iwssim-gpu`, lilith/cubecl fork). **coefficient: turbo-metrics/cudarse backend** (`ssimulacra2-cuda`/`butteraugli-cuda`/`dssim-cuda` via `../turbo-metrics`, NVIDIA-only). | **HIGH (functional dup)** | Two GPU metric implementations of the *same* metrics. zenmetrics is the documented canonical owner; coefficient's `src/gpu.rs` (299 LOC) should migrate to `zenmetrics` once CubeCL covers its needs |
| **8** | Feature extraction / 372-schema / metric-ledger emission | coefficient `generate_zensim_training.rs` writes own `metric-ledger.jsonl` + per-`source_hash` feature JSON (`extract_zenanalyze_features.rs`); zenmetrics emits 305-col parquet sidecars; zensim canonical-corpus builders emit 372-col parquet | MEDIUM | two data-emission formats for the same (ref, dist) → metric facts; coefficient correctly *calls* `zenanalyze::try_analyze_features_rgb8` (no feature-math dup), but its **output schema** forks |
| **9** | `CodecFamily` enum + family order | **THREE orderings** (see A.7) | **HIGH** | `zenpicker::CodecFamily` is the documented owner |

### A.6 coefficient — what it adds that the prior audit missed (NEW)

coefficient is the training-data generator + codec-RD-experiment repo. It is **not** a thin consumer of the other repos — it carries parallel implementations of four cluster concerns:

1. **A second GPU perceptual-metric backend.** `src/gpu.rs` (`GpuMetrics`, 299 LOC) wraps `ssimulacra2-cuda`, `butteraugli-cuda`, `dssim-cuda`, `cudarse-driver`, `cudarse-npp` from `../turbo-metrics` (`Cargo.toml:109-114`). zenmetrics computes the *same* metrics on a *different* backend (CubeCL / lilith-cubecl fork, `crates/{ssim2-gpu,butteraugli-gpu,dssim-gpu,zensim-gpu}`). Same metric, two GPU codepaths → any metric-definition fix (e.g. a linearization or upsample change) must land twice or the two disagree silently. This is the highest-value *functional* dup in the cluster.

2. **A second cloud/batch orchestration system.** `src/cloud/` carries `vastai.rs` (29 KB), `batch.rs` (35 KB, GCP Batch), `do_droplet.rs` (18 KB, DigitalOcean), `mock_batch.rs`, `quota.rs`, plus `src/bin/vastai_worker.rs` (470 LOC) and `scripts/{vastai_create_workergroup,submit-optimization-job}.sh`. zenmetrics owns the canonical vast.ai sweep infra (`scripts/sweep/*`, `crates/zenfleet-vastai`). The `vastai.rs` ↔ `zenfleet-vastai` overlap is real (provision/dispatch/worker-loop); the GCP/DO paths are coefficient-unique.

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

- `src/store/r2.rs:116` — `format!("https://{}.r2.cloudflarestorage.com", self.account_id)` (Rust endpoint constructor #3 in cluster, after zenmetrics `zenfleet-vastai/.../r2.rs`).
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

`src/cloud/vastai.rs` (29 KB) implements vast.ai template/workergroup/instance provisioning + dispatch — functionally parallel to `zenmetrics/crates/zenfleet-vastai`. The two are independent Rust implementations of vast.ai control. GCP Batch (`batch.rs`) + DO droplets (`do_droplet.rs`) are coefficient-unique providers (no zenmetrics equivalent — not a dup, but worth noting the cluster has 3 cloud providers across 2 repos). Onstart hydration differs from zenmetrics' `/proc/1/environ` pattern.

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
| A-4 | Decide GPU-metric ownership: migrate coefficient `src/gpu.rs` (turbo-metrics) to `zenmetrics` (CubeCL) OR document why two backends coexist (NVIDIA-only perf vs cross-platform). Until then, add a numeric-parity test that turbo-metrics ssim2 and CubeCL ssim2 agree on a fixture. | 4h+ (or 1h for parity test) | HIGH — two metric impls drift |
| A-5 | Extract shared `zen-vastai` from coefficient `src/cloud/vastai.rs` + zenmetrics `zenfleet-vastai` (or document the split). | 4h | MEDIUM |
| A-6 | Replace coefficient `src/store/r2.rs` + `upload_to_r2.py` endpoint constructors with the shared `zen-r2` helper from Phase 4. | 1h | MEDIUM |

### A.15 Anti-patterns (Cluster A additions to prior audit's list)

8. **Parallel GPU-metric backends for identical metrics.** turbo-metrics/cudarse (coefficient) vs CubeCL (zenmetrics) both compute ssim2/butteraugli/dssim. A metric-definition change must land in both or they disagree — and there is no parity test asserting they don't.
9. **Enum order forked three ways.** `CodecFamily` declared in coefficient with a different order than the documented `zenpicker` owner, while a Python bake script hardcodes yet a third CSV that happens to match zenpicker but not coefficient's own Rust. No compile error catches the mismatch.
10. **Parallel data-emission schemas for the same facts.** `(ref, dist) → metric + features` is serialized as coefficient's `metric-ledger.jsonl` + per-source feature JSON, zenmetrics' 305-col parquet, and zensim's 372-col canonical parquet — three formats, three join-key conventions, three places a misjoin can hide.
11. **Per-source broadcast joins without a uniqueness guard.** coefficient joins one-row-per-source features onto many-row-per-source pareto/oracle tables on `source_hash`/`source_path` with no assertion that the feature side is unique-per-key — the same broadcast shape as the corruption.

---

*Cluster A generated by a read-only audit (task #222). No source was modified across any of the 6 repos. Sibling agents append further clusters below this section under their own `## Cluster X — ...` headings.*

---

## Cluster B — codecs + products (zenjpeg, zenwebp, zenavif, zenjxl*, zengif, zentiff, imageflow, zenpipe, ...)

### Scope notes / corrections to the brief

- `zentiff` is not a top-level repo at `~/work/zen/`; the TIFF crate is
  `~/work/zen/image-tiff` (23 src files). Treated as such below.
- `zencodecs` (the "unified format detection + dispatch registry") does **not
  exist** as a top-level repo. The shared trait crate that *is* present is
  `zencodec` (singular) — traits + ResourceLimits + ImageInfo. No registry.
- `zenpixels-convert` (the transfer-function-aware color crate) is a **workspace
  member of `zenpixels`** (`zenpixels/zenpixels-convert/`), not a top-level
  repo. It owns ICC / gamut / oklab / HDR — **not** YCbCr/YUV codec color.
- `garb` (`~/work/garb`) already owns SIMD byte-level pixel swizzles
  (RGB↔BGR, RGBA↔BGRA, Gray→RGBA, alpha-fill) and is widely adopted
  (zengif, zenbitmaps, zenjpeg, zenwebp, heic, mozjpeg-rs).
- `imageflow` (`~/work/zen/imageflow`) is the legacy v3 product. Its raw "18577
  rs" is dominated by `target/` + vendored `external/` + agent worktrees. It
  carries a **dual codec path**: legacy C-backed (mozjpeg, libpng, lode) +
  new `zen_decoder.rs`/`zen_encoder.rs` bridging the zen* crates via the
  `zencodec` trait. That dual path is a deliberate in-progress migration, not
  cross-repo dup — the zen bridge (1448 LOC) is a single shared adapter.

### Accurate per-repo source footprint (src/*.rs, target+worktrees excluded)

| Repo | src .rs | codec.rs LOC | depends on zencodec | archmage arcane/rite |
|---|--:|--:|:--:|---|
| zenjpeg (`zenjpeg/`) | workspace; `codec.rs` 3609 | 3609 | yes | 66 / 46 |
| zenwebp | 97 | 3410 | yes | 127 / 227 |
| zenavif | 23 | 4247 | yes | 11 / 13 |
| zengif | 22 | 2431 | yes | 0 / 0 |
| zenbitmaps | 32 | (split per format) | yes | 0 / 0 |
| zenraw | 17 | `zencodec_impl.rs` | yes | 0 / 0 |
| zenjxl-decoder | workspace (~75) | — (decoder only) | n/a | 159 / 0 |
| zenrav1e | 122 | — | no | 0 / 0 |
| image-tiff | 23 | — | no | 0 / 0 |

---

### Class 1 — Codec API boilerplate (zencodec trait wiring)

`zencodec` already centralizes the trait surface well: `Encoder`/`Decoder`,
`EncodeJob`/`DecodeJob`, `EncoderConfig`/`DecoderConfig`, the `Dyn*` object-safe
variants, `ResourceLimits`, `ImageInfo`, `EncodeOutput`/`DecodeOutput`, plus
shared `helpers/` (`copy_decode_to_sink`, `parse_exif_orientation`,
`identify_well_known_icc`). So the *contract* is shared.

What is **copy-pasted per codec** is the *builder-method body shape* inside each
codec's `codec.rs`:

- The `EncodeJob` builder quartet `with_stop` / `with_metadata` / `with_limits`
  / `with_policy` is reimplemented verbatim-shaped in every codec.
  - zenavif: `src/codec.rs:512`, `:517`, `:543`, `:548`
  - zenjpeg: `zenjpeg/zenjpeg/src/codec.rs` (11 builder methods)
  - zenwebp: `src/codec.rs` (14 builder methods)
  - zengif: `src/codec.rs` (10 builder methods)
- The `EncoderConfig` generic-knob quartet `with_generic_effort` /
  `generic_effort` / `with_generic_quality` / `generic_quality` /
  `with_lossless` / `is_lossless` — same per-codec.
- Builder-method count per `codec.rs`: zenjpeg 11, zenwebp 14, zenavif 14,
  zengif 10, zenbitmaps/hdr 6 (these are mechanical setters around a `self.x = …; self`).
- Threading-policy translation (`ThreadingPolicy::Parallel` → thread count) is
  hand-rolled — only zenavif has `policy_to_threads` (`src/codec.rs:45`); the
  others inline the same match, so it's *unshared* duplication waiting to spread.

| Logic | #copies | Repos | Risk | Proposed single home |
|---|--:|---|---|---|
| `EncodeJob` builder quartet (`with_stop/metadata/limits/policy`) | 4+ | zenjpeg, zenwebp, zenavif, zengif | Low-Med (drift in policy/limit semantics) | `zencodec` — derive macro or `#[default impl]` blanket + `impl_encode_job_builders!` macro |
| `EncoderConfig` generic-knob setters | 4+ | same | Low | `zencodec` macro `impl_generic_knobs!` |
| `ThreadingPolicy → thread count` | 4 (1 named, 3 inline) | zenavif named; others inline | Med (thread-cap bugs differ per codec) | `zencodec::limits` free fn `resolve_thread_count(policy, limits)` |

**Verdict:** the trait *types* are shared correctly; the boilerplate is the
~10-14 mechanical builder bodies × 4 codecs. A `zencodec` declarative macro
(or `#[zencodec_job]` derive) removes ~40-56 hand-maintained methods. Med-value,
low-risk.

---

### Class 2 — archmage SIMD + color-conversion kernels (HIGHEST-VALUE)

**Top cross-repo dup in the codec group.** Two fully independent RGB↔YCbCr/YUV
SIMD implementations, plus a third decoder-side YCbCr stage:

1. **`zenjpeg/zenyuv/`** — a dedicated workspace crate, **5363 LOC**, with
   per-arch kernels: `avx2_encode.rs`, `avx2_decode.rs`, `neon_encode.rs`,
   `wasm_encode.rs`, `decode_generic.rs`, `encode_generic.rs`, `gamma.rs`,
   `sharp.rs`. Does RGB↔YCbCr + gamma + 4:2:0/4:2:2 sharp downsample.
2. **`zenavif/src/yuv_convert*.rs`** — **3718 LOC across 7 files**:
   `yuv_convert.rs:1295`, `yuv_convert_fast.rs:645`,
   `yuv_convert_libyuv_simd.rs:507`, `yuv_convert_libyuv.rs:281`,
   `yuv_convert_libyuv_autovec.rs:134`, `convert.rs:293`, `strip_convert.rs:563`.
   Does RGB↔YUV (libyuv-port math). **zenavif does NOT depend on zenyuv** —
   confirmed independent reimplementation of the same color math.
3. **`zenjxl-decoder/.../render/stages/ycbcr.rs`** + `xyb.rs` — a third YCbCr
   (and XYB) inverse-transform stage for the JXL decode path.

These three implement overlapping color-space matrix + gamma math for different
codecs, none sharing a kernel. By contrast the *byte-swizzle* layer (channel
reorder, alpha fill) is already shared via `garb` — so the duplication is
specifically the **color-transform math (matrix + transfer function)**, which
no crate currently owns (`zenpixels-convert` owns ICC/gamut/oklab but not the
codec YCbCr fast path).

| Logic | #copies | Repos (file:line) | Risk | Proposed single home |
|---|--:|---|---|---|
| RGB↔YCbCr/YUV SIMD matrix+gamma | 2 full + 1 partial | zenyuv (`zenjpeg/zenyuv/src/*`, 5363 LOC), zenavif (`src/yuv_convert*.rs`, 3718 LOC), zenjxl-decoder (`render/stages/ycbcr.rs`) | **HIGH** — color-precision divergence between codecs is a CLAUDE.md "sacred pixels" class bug; two matrix impls *will* round differently at boundaries | **Promote `zenyuv` to a top-level shared crate** (it's already arch-complete + magetypes-based); migrate zenavif + zenjxl-decoder onto it |
| 4:2:0 / 4:2:2 chroma sharp downsample | 2 | zenyuv `sharp.rs`, zenavif `yuv_convert.rs` | High (same boundary-artifact risk) | `zenyuv` |
| archmage `#[arcane]` entry-point scaffolding | ~370 sites | zenjpeg 66, zenwebp 127, zenjxl-decoder 159, zenavif 11 | Low (idiomatic, not "dup") | leave; shared kernels (above) is the win |

**Verdict:** promoting `zenyuv` to a standalone shared crate and routing zenavif
(and the zenjxl-decoder YCbCr stage) through it is the **single highest-value
extraction in the codec group** — it both deletes ~3700 LOC of parallel color
math and closes a precision-divergence bug surface the user explicitly treats as
a shipping bug. magetypes already backs zenyuv, so it's portable as-is.

---

### Class 3 + 7 — per-codec picker / target-quality loops + metric scoring

The "encode → rescore → adjust" loop is reimplemented **three times** with
shared vocabulary but no shared code:

- **zenjpeg** `zenjpeg/zenjpeg/src/encode/zq.rs` (**1038 LOC**) — struct
  `ZqTarget` with `target` / `_overshoot` / `_undershoot` / `max_passes` /
  `achieved` / clawback. The original.
- **zenwebp** `src/encoder/zensim_target.rs` (**1765 LOC**) — struct
  `ZensimTarget`. Its module doc (line 3) literally says *"Mirrors the design of
  zenjpeg's `target_zq` module (`src/encode/zq.rs`)"* — a self-documented copy.
  Same overshoot/undershoot/max_passes/ship-band semantics.
- **zenavif** `src/auto_tune.rs` (**504 LOC**) — struct `AutoTuneOptions`. A
  *partial* variant: it's LUT-prediction (per-(size-class, target_zq) median-q
  lookup) rather than a closed iterate-and-rescore loop, but shares the
  `target_zq` / target-zensim vocabulary and goal.

Vocabulary-overlap grep (`overshoot|undershoot|max_passes|target|achieved|claw`):
zq.rs 161 hits, zensim_target.rs 240 hits, auto_tune.rs 28 hits.

Picker wiring (zenpredict + zenanalyze): only **zenwebp** has a production picker
module wired in — `src/encoder/picker/{mod,runtime,spec}.rs` (640 LOC),
using `zenpredict::{Predictor, Model, AllowedMask, ScoreTransform}` and
`zenanalyze::analyze_features_rgb8` (`runtime.rs:21`, `:177`, `:229`). zenjpeg
and zenwebp also have dev-only A/B picker scaffolds (`dev/picker_v0_3_holdout_ab.rs`
in both — near-identical filenames, strong copy signal). zenjxl/zenavif have no
wired picker.

Metric scoring (Class 7): each codec wires `zensim` independently as an optional
dep — zenjpeg (`zensim = { workspace = true, optional }`, 2 scoring call sites),
zenwebp (`zensim = "0.2", optional`, 1 site), zenavif (`zensim = "0.2.4"`, scored
inside auto_tune). No shared "score this (ref,dist) pair" helper; each re-derives
the imgref → zensim → score plumbing.

| Logic | #copies | Repos (file:line) | Risk | Proposed single home |
|---|--:|---|---|---|
| Closed-loop target-zensim iterate/rescore (`ZqTarget`/`ZensimTarget`) | 2 near-identical + 1 LUT variant | zenjpeg `encode/zq.rs:1`, zenwebp `encoder/zensim_target.rs:1` (self-documented mirror), zenavif `auto_tune.rs:1` | **HIGH** — 2800+ LOC of parallel control logic; bugfixes (ship-band, clawback, pass-budget) land in one and not others | **New crate `zentarget`** (or `zencodec::target`): generic `TargetLoop<Encoder, Scorer>` parameterized over the codec's encode fn + a `Scorer` trait |
| Encode→score plumbing (imgref→zensim→f32) | 3 | zenjpeg, zenwebp, zenavif | Med | a `zencodec::Scorer` trait impl'd once over zensim |
| zenpredict + zenanalyze picker wiring | 1 prod (+1 dup dev scaffold) | zenwebp `encoder/picker/*`; `dev/picker_v0_3_holdout_ab.rs` in zenjpeg+zenwebp | Med (only 1 prod copy today, but pattern will be pasted into zenjpeg/zenavif as pickers ship) | `zencodec::picker` thin wrapper around `zenpredict::Predictor` + `zenanalyze::analyze_features_rgb8` |

**Verdict:** the target-loop duplication is the **2nd highest-value extraction**
and the most actively dangerous (the zenwebp copy is already drifting from its
zenjpeg parent). A generic `TargetLoop` + `Scorer` trait unifies all three.
**Yes — the per-codec picker loops should be unified**, but the *control loop*
(zentarget) is the urgent half; the picker *integration* is only 1 production
copy today so unify it pre-emptively (cheap) before it's pasted into the other 3.

---

### Class 4 — test harnesses + corpus + fuzz boilerplate

- **`tests/fuzz_regression.rs` template is unfulfilled.** CLAUDE.md says the
  pattern is templated from zenwebp, but **only zenwebp has it**
  (`zenwebp/tests/fuzz_regression.rs`, 90 lines). zenavif (4 fuzz targets),
  zengif (4), zenjxl-decoder (3), image-tiff (2), zenbitmaps (2), zenraw (2),
  zenrav1e (4), zenavif-parse (2), zenflate (2) all have fuzz targets but **no
  regression harness** — a *missing-shared-code* gap, not a dup. The fix is to
  ship the harness everywhere (ideally as a tiny shared crate, not 10 copies).
- **`codec-corpus`** is already the shared corpus crate (adopted by zenwebp,
  zenbitmaps, zengif, zenflate, zenquant, zenpng, zenpipe, zensim). Good — no
  dup here; rather, the codecs *without* it (zenjpeg, zenavif, zenjxl-decoder)
  roll their own `*.testdata` files (zenjpeg has 8 committed `.testdata` blobs).
- **Decode fuzz targets** are small (11-34 LOC) and structurally similar
  (`#![no_main]` + `libfuzzer_sys::fuzz_target!` + decode-with-limits) but
  codec-API-specific — low-value to unify.

| Logic | #copies | Repos | Risk | Proposed single home |
|---|--:|---|---|---|
| `fuzz_regression.rs` walk-`fuzz/regression/*` harness | 1 (should be ~10) | zenwebp only | Low (gap, not dup) | New `zen-fuzz-regress` test-helper crate, or a `codec-corpus`-adjacent `zen-test-harness` crate; `include!`-able template |
| roundtrip/golden-image compare helpers | scattered | per-codec `tests/` | Low | optional `zen-test-harness` |
| ad-hoc `.testdata` blobs vs codec-corpus | 3 holdouts | zenjpeg (8 blobs), zenavif, zenjxl-decoder | Low | migrate to `codec-corpus` |

---

### Class 5 — CI workflows (windows-11-arm + macos-intel + i686/cross matrix)

**~17 near-identical `ci.yml` files** across the codec + shared-infra repos. All
distinct md5 (so they've drifted), but every one carries the same mandated
platform matrix. The most heavily copy-pasted block is the **i686 cross job**,
which is byte-near-identical everywhere:

```
- uses: taiki-e/install-action@cross
  with: { key: cross-i686 }
- run: cross test --target i686-unknown-linux-gnu [--workspace|--all-features|--lib]
```

Platform-matrix coverage grep across the 17 ci.yml files:

| Repo | ci.yml LOC | winarm | macos-intel | i686 | cross |
|---|--:|:--:|:--:|:--:|:--:|
| zenjpeg | 484 | y | y | y | y |
| zenwebp | 230 | y | y | y | y |
| zenavif | 132 | y | y | y | y |
| zengif | 163 | y | y | y | y |
| zenbitmaps | 160 | y | y | y | y |
| **zenraw** | 117 | y | y | **NO** | **NO** |
| zenrav1e | 142 | y | y | y | y |
| zenavif-parse | 162 | y | y | y | y |
| zenavif-serialize | 143 | y | y | y | y |
| zenflate | 209 | y | y | y | y |
| zencodec | 246 | y | y | y | y |
| zenpixels | 210 | y | y | y | y |
| zenresize | 166 | y | y | y | y |
| zenblend | 94 | y | y | y | y |
| zenquant | 165 | y | y | y | y |
| zenjxl | 212 | y | y | y | y |
| zenpipe | 23 (stub) | — | — | — | — |

Structural-overlap spot-check: `zengif/ci.yml` vs `zenbitmaps/ci.yml` (both
~160 LOC) differ in only ~55 lines, mostly the per-feature `cargo test --features X`
step names — i.e. ~65% identical skeleton.

**Gap found:** `zenraw/ci.yml` is **missing the i686 + cross matrix entirely**
(violates the CLAUDE.md "i686 is a primary target" mandate). zenpipe's ci.yml is
a 23-line stub.

| Logic | #copies | Repos | Risk | Proposed single home |
|---|--:|---|---|---|
| Native build/test matrix (ubuntu/macos/win + win-arm + macos-intel) | 17 | all codec + infra repos | Med (drift → inconsistent platform coverage; zenraw already lost i686) | **Shared reusable workflow** in a `imazen/zen-ci` (or `.github` org-default) repo, called via `uses: imazen/zen-ci/.github/workflows/rust-matrix.yml@v1` |
| i686 cross job (`taiki-e/install-action@cross` + `cross test`) | 15 (zenraw + zenpipe missing) | all but zenraw/zenpipe | Med | same reusable workflow |
| fuzz.yml | ~10 | most codecs | Low | same |

**Verdict:** a single org-level reusable workflow (`workflow_call`) parameterized
by `feature-permutations` + crate name collapses ~17 ci.yml × ~150 LOC into
~17 × ~20-line callers + one ~200-line shared definition. **This is the single
highest-value shared-CI extraction** and it also fixes the zenraw i686 gap by
construction.

---

### Class 6 — sweep / benchmark launchers

Largely **not** a codec-repo dup problem: the canonical vast.ai sweep infra
lives in `zenmetrics` (out of Cluster B scope). Codec-local Dockerfiles are few
and mostly per-codec build images (zenjpeg 2, zenrav1e 3, zenavif 1,
zenjxl-decoder 1) — not copy-paste of a sweep launcher. No `scripts/*sweep*` in
the codec repos except zenavif (1).

Bench harness is a *migration* inconsistency, not cross-repo dup: most codecs
have both `zenbench` and leftover `criterion` manifests (zenjpeg 4 zenbench / 2
criterion; zenjxl-decoder 1/2; zenrav1e 0/1). Per CLAUDE.md, criterion should be
ported to zenbench, but that's an in-repo cleanup, not a shared-crate extraction.

justfiles: 6+ codec repos have justfiles (zenjpeg 20 recipes, zenavif 31, zengif
29, zenwebp 14) with overlapping recipe intent (ci/test/fuzz/fmt/clippy/bench/cov)
but varied naming/indent — low-value to unify beyond a documented template.

| Logic | #copies | Repos | Risk | Proposed single home |
|---|--:|---|---|---|
| vast.ai sweep launcher | ~0 in codecs | (canonical in zenmetrics) | Low | leave in zenmetrics |
| criterion→zenbench leftover | several | zenjpeg, zenjxl-decoder, zenrav1e | Low | in-repo cleanup (not dedup) |
| justfile common recipes | ~6 | most codecs | Low | a documented `justfile` template / `just --justfile shared` import |

---

### Prioritized consolidation plan (effort estimates)

| # | Action | Eliminates | Effort | Risk reduced |
|---|---|---|--:|---|
| 1 | **Promote `zenyuv` to a standalone shared crate; route zenavif + zenjxl-decoder YCbCr through it** | ~3700 LOC parallel color math (zenavif `yuv_convert*`) + a 3rd partial copy | High (3-5 days; arch parity + golden-roundtrip tests per codec) | **HIGH** — closes color-precision-divergence shipping-bug surface |
| 2 | **New `zentarget` crate: generic `TargetLoop<Enc, Scorer>` + `Scorer` trait; migrate zenjpeg `zq.rs` + zenwebp `zensim_target.rs` + zenavif `auto_tune.rs`** | ~2800 LOC of parallel + already-drifting control logic | High (3-4 days; the loops have codec-specific RD hooks to abstract) | High — bugfix-in-one-not-others |
| 3 | **Org-level reusable CI workflow (`zen-ci/rust-matrix.yml`); convert 17 ci.yml to callers; fix zenraw i686 gap** | ~17×150 → 17×20 + 1×200 LOC; one missing-coverage bug | Med (1-2 days) | Med — platform-coverage drift |
| 4 | **`zencodec` builder macro (`impl_encode_job_builders!` + `impl_generic_knobs!`) + `resolve_thread_count` free fn** | ~40-56 hand-maintained builder methods × 4 codecs + 3 inline thread-policy matches | Med (1-2 days) | Low-Med |
| 5 | **`zen-fuzz-regress` test-helper crate; ship `fuzz_regression.rs` to all 10 fuzz-bearing codecs; migrate stray `.testdata` to `codec-corpus`** | fills a gap (1→10) + 3 ad-hoc corpus holdouts | Low-Med (1 day) | Low (correctness coverage gain) |
| 6 | **`zencodec::picker` thin wrapper over zenpredict+zenanalyze; adopt in zenwebp; pre-wire for zenjpeg/zenavif/zenjxl** | pre-empts paste of the 640-LOC zenwebp picker into 3 more codecs | Low (0.5 day now vs 3× later) | Med (prevents future dup) |

### Which shared crate each duplicated thing belongs in

- **YCbCr/YUV color math** → promote **`zenyuv`** to top-level shared crate
  (NOT zenpixels-convert, which owns ICC/gamut/oklab; NOT garb, which owns
  byte-swizzle only).
- **Target-quality iterate/rescore loop** → **new `zentarget` crate** (or a
  `zencodec::target` module) with a `Scorer` trait so zensim is pluggable.
- **EncodeJob/EncoderConfig builder boilerplate + thread-policy** → **`zencodec`**
  (macros + a free fn).
- **CI matrix + i686 cross + fuzz.yml** → **new `imazen/zen-ci` reusable-workflow
  repo** (org `.github` default).
- **fuzz_regression harness + roundtrip/golden helpers** → **new `zen-fuzz-regress`
  / `zen-test-harness`** crate (dev-dep); corpus → existing **`codec-corpus`**.
- **Picker integration** → **`zencodec::picker`** wrapping `zenpredict` +
  `zenanalyze`.
