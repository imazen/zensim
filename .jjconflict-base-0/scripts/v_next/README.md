# `scripts/v_next/` — Python helpers for zensim research

Python helpers for corpus preparation, screening, baking via the
ZNPR v3 JSON pipeline, and analysis. All scripts assume the
`/mnt/v` mount + the zensim workspace at `/home/lilith/work/zen/zensim`.

## Where to look first

| If you want to… | Use |
|---|---|
| Compute IW-SSIM on the safesyn corpus | [`compute_iwssim_on_safesyn.py`](compute_iwssim_on_safesyn.py) |
| Run IW-SSIM on vast.ai (parallel) | [`vastai_iwssim/`](vastai_iwssim/) (deployment plan in `benchmarks/iwssim_vastai_deployment_plan_2026-05-15.md`) |
| Screen per-feature transforms (V_20 IS pipeline) | [`v0_20_feature_transform_greedy_screen.py`](v0_20_feature_transform_greedy_screen.py) → [`v0_20_screen_to_trainer_args.py`](v0_20_screen_to_trainer_args.py) |
| Distill V_20b (contrastive pre-train + fine-tune) | [`v0_20b/`](v0_20b/) (subdirectory; see its own README) |
| Affine-calibrate a bake (distance → score) | [`affine_calibrate_bake.py`](affine_calibrate_bake.py) (preferred) or `affine_calibrate_znpr_v2.py` (legacy v2) |
| Build the interactive comparison-site data | [`build_site_data.py`](build_site_data.py) + [`build_scatter_data.py`](build_scatter_data.py) |
| Export a corpus to parquet for the site | [`export_human_corpora_to_parquet.py`](export_human_corpora_to_parquet.py), `export_aic3_to_parquet.py`, `export_aic4_to_parquet.py` |
| Verify a baked MLP reproduces a known SROCC | [`verify_bake_srocc.py`](verify_bake_srocc.py) |
| Sync the unified parquet store to R2 | [`sync_unified_to_r2.sh`](sync_unified_to_r2.sh) |

## Grouped by theme

### IW-SSIM (added 2026-05-15)

| Script | Role |
|---|---|
| `compute_iwssim_on_safesyn.py` | Compute Wang & Li 2011 IW-SSIM per pair via `piq.information_weighted_ssim` (PyTorch, CUDA). Outputs parquet sidecar with `source_path, decoded_path, iwssim`. ~7.3 pairs/sec on a 4090; ~7.5 hr for the 196k-pair safesyn corpus. The `vastai_iwssim/` subdir parallelizes this across N workers (~1 hr / ~$5). |
| `vastai_iwssim/` | vast.ai sweep adaptation. Reads safesyn TSV, splits into 99 chunks, uploads to R2, launches N workers, merges results back. Deployment plan: `benchmarks/iwssim_vastai_deployment_plan_2026-05-15.md`. |

### V_20 input shaping (V_20 IS / V_20a / V_20b / V_20 extended)

| Script | Role |
|---|---|
| `v0_20_feature_transform_greedy_screen.py` | For each feature column, try every `FeatureTransform` and report the one with maximum Pearson lift over identity. Output: TSV with `feat_idx, best_transform, params_csv, lift, ...`. Run against any per-pair features CSV. |
| `v0_20_screen_to_trainer_args.py` | Convert the screen TSV into `--feature-transform TOKEN:IDX[:PARAMS]` flag strings for the trainer. Used to be required; **now redundant** — the trainer has `--auto-transforms <SCREEN.tsv>` (commit `d32ca890`) which loads the TSV directly. |
| `v0_20_parse_reeval_logs.py` | Parse `dataset_metric_baseline` per-corpus eval logs + collate full Mohammadi panel rows into a consolidated comparison markdown. |
| `v0_20_extract_statistical_panels.py` | Same as parse_reeval_logs but reads training-time validation logs + emits the full panel structure. |
| `v0_20_low_n_band_analysis.py` | For (corpus, band) cells with n < 100, compute the empirical SROCC ceiling and rank bakes by mean SROCC. |
| `v0_20b/` | Contrastive pre-train + fine-tune scripts (PyTorch). Su 2023 distortion manifold spike. |

### Corpus prep + parquet

| Script | Role |
|---|---|
| `build_unified_parquet.py` | Merge per-codec CSVs into a single unified parquet store. Mostly used for the interactive site backend. |
| `export_human_corpora_to_parquet.py` | Export KADID / TID / CID22 / KonJND to parquet with consistent schema. |
| `export_aic3_to_parquet.py` / `export_aic4_to_parquet.py` | AIC-3 / AIC-4 EPFL JPEG-AIC corpus → parquet. |
| `aic3_pairs_csv.py` / `aic3_anchor_pairs_tsv.py` / `aic3_jnd_sanity.py` | AIC-3 specific helpers (pair list extraction, sanity-check JND values). |
| `convert_features_bin.py` | Convert the binary `features.bin` cache (from `zensim-validate`'s extractor) to CSV for inspection or alternate consumption. |
| `band_balance_safesyn.py` | Audit / rebalance safesyn rows by quality band. |
| `content_class_explore.py` | Cluster safesyn refs by feature embedding to inform content-class training experiments. |

### Baking + calibration

| Script | Role |
|---|---|
| `affine_calibrate_bake.py` | (Preferred) Apply `y' = α + β·y` calibration to a ZNPR v3 F32 bake's final layer. Used to map V_X bake's raw output onto the MCOS 0..100 scale. |
| `affine_calibrate_znpr_v2.py` | Legacy v2 variant. Don't write new code against this; v2 production is prohibited per CLAUDE.md. Kept for reproducing pre-2026-05-13 bakes. |
| `bake_to_znpr.py` | Pre-`zenpredict-bake` JSON CLI baker. **Legacy**. New bake construction goes through `zenpredict-bake <input.json> <output.bin>` per CLAUDE.md "JSON pipeline mandate" — see `v0_20b/bake_znpr_v3.py` as the canonical template. |
| `ensemble_seeds.py` | Average MLPs across seeds for a single bake. Used in early V_X cycles before the 3-way concat construction landed. |

### Eval + analysis

| Script | Role |
|---|---|
| `verify_bake_srocc.py` | Smoke-test that a bake reproduces a published SROCC on a corpus + features CSV. Doesn't run full Mohammadi panel — use `dataset_metric_baseline` for that. |
| `verify_mohammadi_anchor.py` | Verify our logistic rescale reproduces Mohammadi 2025 Table 2 anchor values (PSNR-Y / IW-SSIM / CVVDP Z-RMSE on AIC-3 CTC). Companion test exists in `dataset_metric_baseline.rs`. |
| `analyze_score_quality.py` | Per-quality-band score-distribution analysis. Inputs: per-pair eval CSV. |
| `apply_butter_filter.py` | Filter pairs by butter-vs-ssim2 concordance. Pre-process step for noise-reduction in the synth corpus. |
| `butter_concordance_audit.py` | Audit how butter and ssim2 disagree across the safesyn corpus. Informed the cycle-7 dssim experiments. |
| `per_band_step5.py` | Bin per-pair scores into step-5 (20 bins of width 0.05) bands and report per-bin SROCC. The granular alternative to 10-band reporting. |
| `score_unified_with_bake.py` | Score the unified parquet store with a given bake. Used for the candlestick chart pipeline. |
| `soft_iso_smooth.py` | Post-hoc smoothing of bake predictions for monotonicity (soft-iso regression). Cycle-11 era. |
| `regen_tv_pairs.py` | Regenerate the TV-regularizer pair indices TSV when the training CSV changes. |

### Sweep / chart / site

| Script | Role |
|---|---|
| `build_site_data.py` / `build_scatter_data.py` | Generate the per-corpus data files for the interactive comparison site at `https://imazen.github.io/zensim/`. |
| `make_v02_v18_candlestick.py` | The V_2 → V_18 candlestick chart used in commit / handoff narratives. |
| `generate_v16_chunks.py` / `launch_v16_sweep.sh` | Vast.ai sweep pre-cursor (V_16 era). Reference only; the canonical sweep infra now lives in zenmetrics + `vastai_iwssim/`. |

### Training (Python trainer — legacy)

| Script | Role |
|---|---|
| `train_v_next_mlp.py` | The Python-side MLP trainer. **Largely retired** — `zensim-validate/src/bin/zensim_mlp_train` (Rust) is the canonical trainer as of V_18. Kept for the Phase 4 trainer reference set. |

### Planning notes

| File | Role |
|---|---|
| `CYCLE_7_DSSIM_COTRAIN_PLAN.md` | Cycle-7 design doc for the dssim co-training experiments. Falsified — see `benchmarks/cycle_7_dssim_outcomes_2026-05-12.md`. Kept for historical record. |

## Conventions

- **Output paths**: scripts that produce data write to `/mnt/v/output/zensim/<theme>/` or to a script-specific path documented in the script's header. Don't write to repo `benchmarks/` from scripts unless the script is generating a permanent artifact.
- **Parquet preferred for >50 MB**: zstd-3 compression per CLAUDE.md "Parquet vs TSV". CSV/TSV is fine for <50 MB human-readable inputs.
- **JSON pipeline for bakes**: ad-hoc Python wire-format emitters are banned per CLAUDE.md "JSON pipeline mandate". All bake-side serialization goes through `zenpredict-bake <input.json> <output.bin>`.
- **Logs to /tmp** for one-shot runs; commit to `benchmarks/<name>_<date>.log` for runs producing ship-relevant data.
