# zensim — context handoff (2026-05-16)

Written for the next session reset. Read this first, then
[RESEARCH.md](RESEARCH.md) and [CLAUDE.md](CLAUDE.md).

## TL;DR — what's live

- **Shipped bake**: `zensim/weights/v0_18_zerobiased_lz4_2026-05-13.bin`
  (ZNPR v3, ~18 KB compressed from 93 KB raw, md5 in
  `benchmarks/v0_18_methodology_2026-05-13.md`). V_18 3-way concat
  (V_16 base + cycle-14 s1 + cycle-14 s42). **CID22 SROCC 0.8933**,
  but per CLAUDE.md "SROCC-only verdicts BANNED" (added 2026-05-15)
  this single number isn't the ship gate — the full Mohammadi 2025
  panel is.
- **Multi-bake runtime live**: `PreviewV0_4` mixes V_18 ship + V_20 IS
  calibrated at α = 0.4 raw space. CID22 B3 +0.080 lift at agg
  −0.008.
- **Crate version**: zensim 0.3.0, never published. Swap bake bytes
  in place; no version bump.
- **Test status**: 68 zensim --lib unit tests pass + 2 regression
  tests (NaN-safe sort, FeatureRegime dispatch).
- **Clippy**: zero zensim-side warnings (zenpredict sibling owns 5
  unrelated warnings at `../zenanalyze`).

## What happened on 2026-05-16 (today's session)

### A. Methodology critique — SROCC banned as primary gate

User directive: *"our srocc gating is bad, wrong, stop it. iwssim
wins the important metrics but we trained older bakes with synth
data and ssim2 as golden and of course we aren't going to win
against that using the ssim2 favoring srocc. also we didn't
calculate iwssim to make it trainable."*

Two new CLAUDE.md sections (commits `ef0ed9a3`, `c81b393f`,
`58e6f8d8`):

1. **"SROCC-only verdicts BANNED + ssim2-target training bias"** —
   every ship/no-ship call requires the full Mohammadi 2025 panel
   (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE). Prior "falsified
   on SROCC" labels in `benchmarks/v0_20*` are provisional.

2. **"CID22 is VALIDATION-ONLY"** — explicit rule that CID22
   human MOS is never a training target. Only metric scores
   (ssim2 / CVVDP) on the training-only subset of the broader
   CID22 image library are permitted.

3. **"ZNPR v2 PROHIBITED"** — every new bake must be v3
   (header byte 4 = `0x03`). `bake_two_layer_znpr_v2` →
   `bake_two_layer_znpr_v3` rename; all `bake_v2` callers
   switched to `bake()` (v3 path).

### B. New infrastructure landed

- **`--auto-transforms <PATH>`** flag on `zensim_mlp_train` (commit
  `d32ca890`): loads winning per-feature transforms from a screen
  TSV. Smoke-tested at min-lift=0.05 → 98 transforms (= V_20 IS
  adopted set exactly).
- **`ProfileParams.extended_features` + `compute_iw_features`**
  (commit `f140776a`): runtime path opt-in for 300- or 372-feature
  regimes. All existing profiles default to `false`.
- **`FeatureRegime` auto-detection** in `dataset_metric_baseline`
  (commits `8baa8e48`, `c8b02b3d`): inspects `Model::n_inputs()`
  and dispatches per-pair compute. 228 → standard, 300 → extended,
  372 → extended + IW. Unit test added.
- **`inspect_l0_input_norms`** binary (commit `bc9e6b60`): per-
  input L2 norm reporter for any ZNPR v3 bake. Used to prove IW +
  masked features ARE selected by GD across 4 bakes (69–96 % of
  basic-block mean L2).
- **`compute_iwssim_on_safesyn.py`** (commit `24986ff3`): IW-SSIM
  target compute script via piq 0.8.0 (PyTorch, CUDA). Full safesyn
  run in flight at PID 4048804 (~7 hr remaining); vast.ai
  parallelization spec at `scripts/v_next/vastai_iwssim/` (~1 hr
  / ~$5).
- **`SteerablePyramidLogGsm`** weight estimator variant (commit
  `f1ad0d6`): paper-faithful Wang & Li 2011 spike. A/B vs spatial
  variance Pearson 0.838 (decorrelated).
- **`info_log_sigma_e_sq`** in `IwWeightConfig` (commit `c23f178c`):
  the paper's `log₂(1 + σ²_p / σ²_e)` weight formula. Off by
  default for back-compat.

### C. Performance optimization — combined Extended+IW overhead

Dropped from **+25 % to +12 % at 1024²** (commit `e5651013` via
worktree perf agent). Fused 2-mask SIMD kernels for SSIM + edge +
MSE+weights paths; V-blur-only on H-blurred sigma buffers; fixed
pre-existing IW-only mu1 swap bug. Full numbers in
[`benchmarks/extended_iw_runtime_perf_optimized_2026-05-15.md`](benchmarks/extended_iw_runtime_perf_optimized_2026-05-15.md).

### D. Tightening sweep

Eight commits clearing warnings + adding regression tests:

- **NaN-safe sort across 17 sites** (`2e5816a1`) —
  `partial_cmp(...).unwrap_or(Equal)` → `total_cmp` to fix the
  Rust 1.81+ sort panic when NaN is present. Closes the per-band
  crash that forced per-corpus eval workarounds.
- **6 clippy fixes** (`02ccc42b`) + **4 warning cleanups**
  (`95c20288`) → zero zensim-side warnings.
- **anchor_csv test env-var gating** (`37c1f397`) — replaces silent
  file-existence skip per CLAUDE.md "NO GRACEFUL SKIPS IN TESTS".
- **FeatureRegime dispatch boundary test** (`c8b02b3d`).
- **Bash readonly variable gotcha** doc (`c8b02b3d`).

### E. Pit-of-success docs (today's chunk)

Three new top-level docs:

- **`RESEARCH.md`** (`ec27122e`) — top-level research-entry guide.
  Corpus map, workflow recipes, bakes inventory, sibling-repo
  map. This is the "first glance" doc for new contributors.
- **`scripts/v_next/README.md`** (`49f8ed1b`) — index of 39
  Python helpers grouped by theme; marks legacy vs current.
- **`benchmarks/INDEX.md`** (`3d14b2bb`) — TOC for 76 methodology
  + falsification docs. Reading-order suggestions for common
  goals.

## V_20a IW + V_20 extended status

Per the new methodology, the V_20a IW falsification verdict
(CID22 SROCC catastrophic) is **provisional**. Why:

- We trained against ssim2 targets → bakes optimize for ssim2-shaped
  output.
- SROCC against ssim2-derived MOS-equivalents on CID22 favors
  ssim2-shaped surfaces.
- IW-SSIM-quality bakes deliberately are NOT ssim2-shaped.
- TID + KADID FULL panel: V_20a iw_k1 wins +0.018 SROCC + better
  Z-RMSE + better PWRC.

Once the IW-SSIM target column lands (in flight at PID 4048804),
retrain against IW-SSIM and re-eval. The expectation is the IW
direction will re-rank.

## Runtime cost ladder (2026-05-16, post-perf-opt)

| Config | 1024² mean ms | × baseline | ns / pixel | When to use |
|---|---:|---:|---:|---|
| Standard (228) | 15.93 | 1.00 | 15.2 | Default fast path; all production bakes today |
| Extended only (300) | ~17.4 | 1.14 | 17.4 | Future PreviewV0_5 candidates (none shipped) |
| IW only (300) | ~17.5 | 1.15 | 17.5 | V_20a-style IW bakes (none shipped) |
| Extended + IW (372) | **~13.8** | **1.12** | **13.8** | V_20 IW+ext+transforms research bakes |

Optimization shrank the combined path; "Extended + IW" is now
cheaper than the user-target +15 %. Compute saving comes from
fused 2-mask SIMD + V-blur-only on H-blurred sigma buffers + IW-
only mu1 fix.

## What's running / pending

- **IW-SSIM compute on safesyn** (local PID 4048804, ~7 hr ETA).
  Output: `iwssim_targets_safesyn_2026-05-15.parquet`. Vast.ai
  alternative at `scripts/v_next/vastai_iwssim/` (other session
  handling that).
- **V_22 CVVDP distillation** (task #45/#49) — highest-conviction
  next direction. Targets the training-objective level which is
  the actual bottleneck per the L0-norm GD-selection analysis.
- **V_20d JND-anchored output calibration** (task #41) — second
  conviction. Replaces the output-scale calibration with PJND units.
- **Trainer `--target-column` flag** — queued. Without it, every
  V_X bake remains structurally an ssim2 predictor.

## Reproduce the V_18 ship from scratch

```sh
CLEAN=/mnt/v/zen/zensim-training/2026-05-14-clean
DATE=$(date -u +%Y-%m-%d)

# Component 1: V_16-equivalent base seed=1 (no TV regularizer)
./target/release/zensim_mlp_train \
  --group safesyn:$CLEAN/safe_synth_v19_clean_features.csv:1.0:0.0 \
  --group kadid:$CLEAN/kadid_features.csv:0.3:1.0 \
  --group tid:$CLEAN/tid_features.csv:0.3:1.0 \
  --group konjnd:$CLEAN/konjnd_aligned_features.csv:0.5:1.0 \
  --hidden 128 --epochs 300 --seed 1 --max-features 228 --val-policy min \
  --out benchmarks/v0_X_base_seed1_${DATE}.bin

# Component 2: cycle-14 TV seed=1 (TV-regularized)
./target/release/zensim_mlp_train \
  --group safesyn:$CLEAN/safe_synth_v19_clean_features.csv:1.0:0.0 \
  --group kadid:$CLEAN/kadid_features.csv:0.3:1.0 \
  --group tid:$CLEAN/tid_features.csv:0.3:1.0 \
  --group konjnd:$CLEAN/konjnd_aligned_features.csv:0.5:1.0 \
  --hidden 128 --epochs 300 --seed 1 --max-features 228 --val-policy min \
  --tv-pairs-file $CLEAN/tv_pairs_bands.tsv \
  --tv-weight 1.0 --tv-band-weights 10,30,10,30 \
  --tv-apply-every 50 --tv-batch 32 \
  --out benchmarks/v0_X_cycle14_s1_${DATE}.bin

# Component 3: cycle-14 TV seed=42 (same flags, --seed 42)

# Concat the three components:
cargo run --release -p zensim-validate --bin concat_three_way -- \
  --base benchmarks/v0_X_base_seed1_${DATE}.bin \
  --s1   benchmarks/v0_X_cycle14_s1_${DATE}.bin \
  --s42  benchmarks/v0_X_cycle14_s42_${DATE}.bin \
  --coeffs 0.65:0.30:0.05 \
  --out  benchmarks/v0_X_concat_3way_${DATE}.bin

# Affine calibrate (V_16-lineage α/β):
cargo run --release -p zensim-validate --bin affine_calibrate -- \
  --in-bake  benchmarks/v0_X_concat_3way_${DATE}.bin \
  --out-bake zensim/weights/v0_X_${DATE}.bin \
  --alpha 28.0366 --beta=-5.0738

# Validate (full Mohammadi panel):
cargo run --release -p zensim-bench --example dataset_metric_baseline -- \
  --cid22 /mnt/v/dataset/cid22/CID22_validation_set \
  --kadid /mnt/v/dataset/kadid10k \
  --tid /mnt/v/dataset/tid2013 \
  --v04-bake zensim/weights/v0_X_${DATE}.bin \
  --max-pairs 50000 > benchmarks/v0_X_panel_${DATE}.md
```

## What to do next session

1. Check the IW-SSIM compute status (PID 4048804 or vast.ai output).
   If done, the parquet sidecar should be at
   `/mnt/v/output/zensim/synthetic-v2/iwssim_targets_safesyn_2026-05-15.parquet`.
2. Add the trainer `--target-column NAME` flag so we can switch
   from ssim2 → IW-SSIM regression target. The merge step (joining
   the parquet sidecar onto the features CSV by `(source_path,
   decoded_path)`) needs a small pre-processing script too.
3. Retrain a V_22-candidate against IW-SSIM target.
4. Run the full Mohammadi panel on it. The expectation per the
   methodology critique: this is the first bake whose CID22 verdict
   isn't structurally rigged.

If any of those steps surface an unexpected blocker, return to
CLAUDE.md "Principled experiment workflow" and write down what was
tried before continuing.
