# zensim research guide

> **⚠ STALE — REWRITE PENDING (banner 2026-07-18):** the bakes inventory + Workflow A below are Tuner-trail/228-feature-era fossils (wrong architecture, wrong corpus roots). Use `docs/TOP_MODELS_COOKBOOK.md` for training workflows and `canonical-2026-05-21/` for corpora. Corpus-location tables below are still mostly valid; verify paths before use.

If you just want to compute zensim on an image pair, the
[README](README.md) is enough. **This doc is for researchers,
contributors, and AI agents** who need to train bakes, evaluate
candidates, design experiments, or modify the pipeline.

## Where to start

Pick the row that matches your goal — every cell is a one-line
pointer to the doc + code path with copy-paste-ready commands.

| Goal | Doc | Code |
|---|---|---|
| **Train a new V_X MLP bake** | "Workflow A" below + [CLAUDE.md "Principled experiment workflow"](CLAUDE.md) | `target/release/zensim_mlp_train` |
| **Evaluate an existing bake (full panel)** | "Workflow B" below + [CLAUDE.md "Statistical rigor"](CLAUDE.md) | `target/release/examples/dataset_metric_baseline` |
| **Compute IW-SSIM on a training corpus** | "Workflow C" below | `scripts/v_next/compute_iwssim_on_safesyn.py` (local) or `scripts/v_next/vastai_iwssim/` (parallel) |
| **Inspect a bake's L0 input-column norms** | "Workflow D" below — answers "did GD select these features" | `target/release/inspect_l0_input_norms` |
| **Investigate a falsification** | [CLAUDE.md "Principled experiment workflow"](CLAUDE.md) Step 10 + `benchmarks/falsification_reeval_*` | (no single binary) |
| **Run the runtime cost benchmark** | `benchmarks/extended_iw_runtime_perf*.md` | `target/release/examples/extended_iw_perf` |
| **Add a new training target metric** | "Methodology shift" section below | (trainer source — `--target-column` not yet wired; in flight) |
| **Onboard as an AI agent** | Read [CLAUDE.md](CLAUDE.md) end-to-end, then return here | — |
| **Read the changelog** | [CHANGELOG.md](CHANGELOG.md) | — |
| **Restore session state** | [CONTEXT-HANDOFF.md](CONTEXT-HANDOFF.md) | — |

## Corpus map

The single most-asked question for new researchers is: **"which
dataset can I train on, which is held-out for validation, what's
the anchor unit?"**. Memorize this table.

| Corpus | Type | Path | Anchor | Train? | Val? | Source |
|---|---|---|---|:---:|:---:|---|
| **safesyn** | synthetic | `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv` | ssim2 (legacy) → IW-SSIM (2026-05-15+) | ✓ | — | hex-hashed CLIC + CID22-source tiles (purged via 2026-05-12 contamination audit) |
| **safesyn-clean (recommended)** | synthetic | `/mnt/v/zen/zensim-training/2026-05-14-clean/safe_synth_v19_clean_features.csv` | ssim2 | ✓ | — | 138 872 rows after 2026-05-14 KADID/TID-overlap purge |
| **KADID-10k** | human MOS | `/mnt/v/dataset/kadid10k/images` + `dmos.csv` | DMOS (1–5, lower = better) | ✓ | ✓ | KADID 10k DB |
| **TID2013** | human MOS | `/mnt/v/dataset/tid2013` | MOS (0–9, higher = better) | ✓ | ✓ | Ponomarenko 2013 |
| **CID22** | human MOS | `/mnt/v/dataset/cid22/CID22_validation_set` | MCOS / 100 (0–1, higher = better) | ✗ | **✓ ONLY** | Sneyers / Ben Baruch / Vaxman 2023 (Cloudinary) |
| **KonJND-1k (anchor)** | PJND | `/mnt/v/datasets/KonJND-1k/KonJND-1k/subjective_ratings.csv` | per-source PJND threshold (compression q) | ✓ aux | ✓ | KonJND-1k |
| **KonJND-1k (full)** | metric | `/mnt/v/datasets/KonJND-1k/konjnd_full_scored.csv` | gpu_ssimulacra2 / 100 | ✓ | — | 76 104 (src × codec × q) variants |
| **AIC-3 CTC** | JND | `/mnt/v/dataset/aic3_ctc_epfl/decoded/info.csv` | `score.jnd` (signed JND units) | ✗ | **✓ ONLY** | EPFL JPEG-AIC Common Test Conditions |
| **AIC-4 sample** | JND | `/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset/` | reconstructed JND | ✗ | **✓ ONLY** | EPFL JPEG-AIC-4 |

**CID22 training rule (load-bearing)**: NEVER use CID22 human MOS as
a training target. The 49-reference held-out set is sacred. The
broader CID22 image library (NOT this 49-ref subset) can be used as
training input with ssim2 or CVVDP metric scores as the target —
NEVER human MOS. See [CLAUDE.md "CID22 is VALIDATION-ONLY"](CLAUDE.md).

**Pre-extracted feature CSVs** (300-column, basic+peaks+masked):
- `/mnt/v/zen/zensim-training/2026-05-14-clean/{kadid,tid,konjnd_aligned,safe_synth_v19_clean}_features.csv`
- Schema: `ref_basename, human_score, f0, ..., f299`

**Full 372-column feature CSVs** (basic + peaks + masked + IW pool):
- `/mnt/v/zen/zensim-training/2026-05-15-full-features/{kadid,tid,cid22,konjnd,konjnd_full,aic3}_features_372col_2026-05-15.csv`
- Generated 2026-05-15; provenance manifest in same dir.

## Data storage conventions

### Local layout (`/mnt/v` Windows-mounted drive — reliable, slow)

| Path | What |
|---|---|
| `/mnt/v/input/` | Source data: raw image corpora, image generators' sources |
| `/mnt/v/dataset/` | Human-annotated corpora as distributed (KADID / TID / CID22 / KonJND / AIC) |
| `/mnt/v/output/zensim/` | Synth-pair generation output + per-pair feature CSVs |
| `/mnt/v/zen/zensim-training/<DATE>-clean/` | Curated training corpora (post-contamination-purge) |
| `/mnt/v/zen/zensim-training/<DATE>-full-features/` | 372-column feature CSVs |
| `/mnt/v/fuzzes/<repo>/` | Fuzz corpora + crash artifacts (per CLAUDE.md "Fuzz Corpus") |

### Repository layout

| Path | What |
|---|---|
| `zensim/` | The metric library (public crate). 228-feature input, MLP scoring via `PreviewV0_*` profiles. |
| `zensim-validate/` | Training + bake-construction binaries: `zensim_mlp_train`, `concat_three_way`, `affine_calibrate`, `eval_bake_per_band`, `inspect_l0_input_norms`. Not published. |
| `zensim-bench/` | Benchmark examples: `dataset_metric_baseline` (full Mohammadi panel), `extended_iw_perf` (runtime cost). Not published. |
| `zensim-train-core/` | Pure-Rust WASM-compatible trainer core. WIP. |
| `zensim-regress/` | Regression-testing binary. Published with its own semver. |
| `benchmarks/` | Methodology docs + falsification logs + bake binaries + comparison MDs. Committed; serves as durable record. |
| `docs/` | Long-form methodology + paper notes + literature reviews. Committed. |
| `scripts/v_next/` | Python helpers (corpus prep, screening, baking via JSON pipeline). See `scripts/v_next/README.md`. |
| `zensim/weights/` | **The shipped MLP bake** (`v0_18_zerobiased_lz4_2026-05-13.bin`). Plus archived bakes in `archive/`. |
| `.claude/worktrees/` | Agent worktrees (gitignored). Don't commit anything here. |

### R2 / cloud storage (Cloudflare)

Used by:

- **vast.ai sweeps** (`scripts/v_next/vastai_iwssim/` — adapted from zenmetrics v15 infra). Each chunk's parquet result uploads to R2 keyed by chunk_id; `finalize.sh` merges locally.
- **Interactive comparison site** (planned): public-read R2 bucket hosts the per-corpus parquet files; the gh-pages JS frontend reads them via HTTP-range fetch through DuckDB-WASM.

R2 credentials live at `~/.config/cloudflare/r2-credentials` (3 keys). See [CLAUDE.md "Interactive comparison site"](CLAUDE.md) for the planned web frontend.

### Parquet schemas

| File | Schema |
|---|---|
| Per-pair eval (`--per-pair-output`) | `dataset, reference, distorted, codec, version, human_score, v02_distance, v04_distance, fast_ssim2_score, butter_3norm` |
| IW-SSIM target sidecar (`iwssim_targets_safesyn_*.parquet`) | `source_path, decoded_path, iwssim` |
| Feature CSV → parquet conversion (`build_unified_parquet.py`) | `ref_basename, human_score, codec, quality, f0..f299` (300 cols) or `..f371` (372 cols) |

When a benchmark output exceeds 50 MB, write parquet not TSV/CSV per CLAUDE.md "Parquet vs TSV" — zstd-3 compression, `pq.write_table`.

## Workflow recipes

### A. Train a new V_X MLP bake

```sh
CLEAN=/mnt/v/zen/zensim-training/2026-05-14-clean
DATE=$(date +%F)
./target/release/zensim_mlp_train \
  --group safesyn:$CLEAN/safe_synth_v19_clean_features.csv:1.0:0.0 \
  --group kadid:$CLEAN/kadid_features.csv:0.3:1.0 \
  --group tid:$CLEAN/tid_features.csv:0.3:1.0 \
  --group konjnd:$CLEAN/konjnd_aligned_features.csv:0.5:1.0 \
  --hidden 128 --epochs 300 --seed 1 \
  --max-features 228 --val-policy min \
  --auto-transforms benchmarks/v0_20_feature_transform_greedy_screen_2026-05-15.tsv \
  --out benchmarks/v0_NEW_seed1_${DATE}.bin
```

**Defaults already enforced**: ssim2 target (synth), `--max-features 228` (V_18 ship compatible), per-feature transforms via `--auto-transforms` (recommended for any new bake).

**Notes**:
- CID22 is intentionally **NOT** in the group list — validation only.
- Output is ZNPR v3. ZNPR v2 production is prohibited per CLAUDE.md.
- Multi-seed: rerun with `--seed 42 --seed 7` etc. Concat via `concat_three_way`.
- Methodology doc requirement: every shipped bake needs `benchmarks/v0_X_methodology_$(date +%F).md` before flipping `zensim/weights/`.

### B. Evaluate an existing bake (full Mohammadi panel)

```sh
./target/release/examples/dataset_metric_baseline \
  --cid22 /mnt/v/dataset/cid22/CID22_validation_set \
  --kadid /mnt/v/dataset/kadid10k \
  --tid /mnt/v/dataset/tid2013 \
  --v04-bake <bake.bin> \
  --max-pairs 50000 \
  2>&1 | tee benchmarks/eval_<name>_$(date +%F).log
```

**Output**: aggregate + per-band SROCC / PLCC / KROCC / OR / PWRC / Z-RMSE per corpus. Per CLAUDE.md "SROCC-only verdicts BANNED" — read the panel, not just SROCC.

The harness auto-detects `FeatureRegime` from `Model::n_inputs()`:
- 228 → Standard (basic + peaks)
- 300 → Extended (+ masked features)
- 372 → ExtendedIw (+ IW pool)

so 300- and 372-input bakes Just Work via the same command.

### C. Compute IW-SSIM on the safesyn corpus

```sh
# Local (single-GPU 4090, ~7.5 hr for 196k pairs):
python3 scripts/v_next/compute_iwssim_on_safesyn.py
# Output: /mnt/v/output/zensim/synthetic-v2/iwssim_targets_safesyn_<DATE>.parquet
```

**Parallel via vast.ai** (~1 hr total, ~$5):

```sh
# Deployment plan: benchmarks/iwssim_vastai_deployment_plan_2026-05-15.md
# Scripts: scripts/v_next/vastai_iwssim/{launch,finalize,destroy}.sh
```

Requires R2 creds + vast.ai account + ~$5 budget. The local path is the fallback.

### D. Inspect a bake's L0 input-column norms

```sh
./target/release/inspect_l0_input_norms \
  --bake <bake.bin> \
  --top 20 \
  --regions
```

Reports per-input L2 norm in layer 0 + region-level aggregates (basic / peaks / masked / IW). Answers "did gradient descent actually use these features." Used to falsify the V_20a IW "redundant features" framing — see [`benchmarks/v0_20_l0_norms_2026-05-15.md`](benchmarks/v0_20_l0_norms_2026-05-15.md).

### E. Runtime cost benchmark (4 permutations of extended × IW)

```sh
./target/release/examples/extended_iw_perf --size 1024 --iters 50
```

Reports Standard / Extended-only / IW-only / Both per-pair compute cost. Combined Extended+IW overhead is currently **+12 % at 1024²** post-optimization (down from +25 %). See [`benchmarks/extended_iw_runtime_perf*.md`](benchmarks/).

## Bakes inventory

### Codec-target metric (canonical, 2026-05-24)

`ZensimProfile::codec_target()` → currently
`PreviewV0_5TunerV5` (`v_tuner_v11_2026-05-24.bin`,
rotated from TunerV4/v_tuner_v10 on 2026-05-24 PM after recovery
phase 4 fixed the 0-55 score-floor pathology). This is the
stable alias every zen codec calls when training / dialing /
picking. See [`docs/CODEC_TARGET_METRIC.md`](docs/CODEC_TARGET_METRIC.md)
for the integration guide and
[`benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md`](benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md)
for measured cross-codec consistency (p50 |Δ| = 1.18 overall,
0.6–1.5 in score 60–90 band; score 0–55 is a flat dead zone
pending Tuner v11). Codec target is **rotated** by edit to
`profile.rs::codec_target()`; old variants stay accessible by name
for reproducibility.

### Three-trail production ships (2026-05-20)

| Trail | Profile | Bake | Audience |
|---|---|---|---|
| **Tuner v5** (codec dial — current ship) | `PreviewV0_5TunerV5` | `v_tuner_v11_2026-05-24.bin` | Codec target. Multi-dataset trainer (5 groups) + wider tanh. Full 0-100 dial coverage (p5=28 vs v10's p5=48), JND@60 bit-exact, CID22=0.860, AIC-4=0.929. |
| Tuner v4 (codec dial — prior ship, retained) | `PreviewV0_5TunerV4` | `v_tuner_v10_2026-05-20.bin` | Single-group trainer, 0-55 dial floor pathology. CID22=0.854, AIC-4=0.924. Accessible by explicit name for reproducibility. |
| **Balanced** (general perceptual) | `PreviewV0_5BalancedV3` | `v_balanced_v3_2026-05-20.bin` | KADID/TID/KonJND rank-best (0.967/0.971/0.893). CID22 0.832. |
| **Compression** (codec output rank) | `PreviewV0_5CompressionV3` | `v_compression_v3_2026-05-20.bin` | CID22+AIC-3 rank-best (0.864/0.818). KADID/TID drop 0.03-0.08. |

### Legacy ships (still in profile.rs, not the canonical codec-target)

| Bake | Path | Status | Notes |
|---|---|---|---|
| **PreviewV0_3 (V_18 ship)** | `zensim/weights/v0_18_zerobiased_lz4_2026-05-13.bin` | LIVE | 3-way concat, h=128, V_16 + cycle-14 components. 228-input ZNPR v3 I8. CID22 SROCC 0.8933 (note: inflated by ssim2-target training bias per CLAUDE.md "SROCC-only verdicts BANNED"). |
| **PreviewV0_4 (D2 multi-bake)** | runtime mix | LIVE | V_18 ship + V_20 IS calibrated at α = 0.4 raw space. CID22 B3 +0.080 lift at agg −0.008. |
| V_20 IS calibrated | `zensim/weights/v0_20_is_calibrated_2026-05-15.bin` | secondary | 98-transform input-shaping bake. B3 specialist. |
| V_18 raw | `zensim/weights/v0_18_2026-05-13.bin` | reproduction | Uncompressed source of the LZ4 ship variant. |
| Archive | `zensim/weights/archive/` | historical | V_4 through V_19 + V_18.1 + V_19-overcleaned. Lineage for reproducibility. |

**Bakes-in-flight** (in `benchmarks/`, not yet shipped):
- V_20 extended seed=1 (300-feat)
- V_20a IW k=1/4/8 (372-feat; falsified for CID22 transfer — see [v0_20a_path_a_falsification](benchmarks/v0_20a_path_a_falsification_2026-05-14.md))
- V_20 IW+ext+transforms (372-feat)
- Phase 3 retrains (clean_corpus_transforms / midqboost / lowqboost)

See also: [`benchmarks/v0_20_all_bakes_stat_comparison_2026-05-15.md`](benchmarks/v0_20_all_bakes_stat_comparison_2026-05-15.md) for the full-panel comparison.

## Methodology shift (2026-05-15)

**Two major methodology changes happened today**; they affect every
verdict from this point forward:

1. **SROCC-only verdicts BANNED**: every ship / no-ship call requires
   the full Mohammadi 2025 panel (SROCC + PLCC + KROCC + OR + PWRC +
   Z-RMSE), not just SROCC. Prior "falsified on SROCC" labels in
   `benchmarks/v0_20*` are provisional.

2. **ssim2-target training is the legacy**: training against ssim2
   targets and evaluating via SROCC favors ssim2-shaped surfaces by
   construction. To escape, the safesyn corpus is getting an
   `iwssim` column (in flight 2026-05-15) and the trainer needs a
   `--target-column NAME` flag (queued).

Read [CLAUDE.md "SROCC-only verdicts BANNED + ssim2-target training
bias"](CLAUDE.md) for the full rationale.

## Sibling repos

zensim depends on (via workspace `path = "../zenanalyze/..."` —
never the published crates.io versions per CLAUDE.md):

| Sibling | What it provides |
|---|---|
| **zenanalyze/zenpredict** | ZNPR v3 model parser + Predictor + FeatureTransform + masked-argmin picker. Used by `zensim::mlp` for MLP scoring. |
| **zenanalyze/zenpredict-bake** | Bake serializer (`bake()` for v3) + builder API + JSON CLI. Used by the trainer's bake step. NEVER call `zenpredict::bake::bake_v2` from zensim. |
| **zenmetrics** | GPU metric crates (butteraugli-gpu / ssim2-gpu / dssim-gpu / zensim-gpu via CubeCL) + `zenmetrics` CLI + vast.ai sweep infra. Source of the v15 chunk-claim pattern adapted in `scripts/v_next/vastai_iwssim/`. |
| **zenanalyze (parent)** | Feature extractor (`zenanalyze::analyze_features_rgb8`). Shipped at crate version 0.1.x **forever** — never 0.2.x. |

Don't modify sibling repos from a zensim session. Open issues
against them via `gh issue create -R imazen/zenanalyze ...` instead.

## Pointers to other docs

- [README.md](README.md) — public-facing speed + correlation + quick-start
- [CHANGELOG.md](CHANGELOG.md) — every shipped change
- [CONTEXT-HANDOFF.md](CONTEXT-HANDOFF.md) — session-handoff state
- [CLAUDE.md](CLAUDE.md) — AI-agent operational guide (large; bundles methodology + workflow + gotchas)
- [docs/](docs/) — paper notes, literature reviews, long-form design docs
- [benchmarks/](benchmarks/) — every experiment's methodology + results (76+ docs as of 2026-05-16)
- [scripts/v_next/](scripts/v_next/) — Python helpers for corpus prep, screening, parquet manipulation
