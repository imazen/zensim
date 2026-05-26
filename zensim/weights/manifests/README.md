# Bake manifests (`.toml`) — per-version reproducibility

Every shipped `.bin` bake under `zensim/weights/` should have a sibling
`.toml` manifest under `zensim/weights/manifests/` with **everything
needed to reproduce it**: the exact training command, every CLI flag,
every input file (path + sha256 + row count), every post-training step
(spline injection, calibration), output bake URL on R2/S3, the build
commit, and the bake_verdict scorecard at ship time.

**Why:** the V32 recipe-archaeology incident — V32's commit message
documented hyperparameters but not the exact CLI; the actual command
lived only in a prior session's chat transcript. A future agent
reconstructing it from documentation alone got `CID22 0.295` (held-out,
~0.59 below V32's documented 0.8879) because the target column and
group structure differed in undocumented ways. The `.toml` manifest
ends that class of incident: every shipped bake is bit-exactly
reproducible from the manifest.

**Naming:** `<bake-filename>.toml` (same stem as the `.bin`). E.g.
`v39_v32plus_spline_seed17_2026-05-25.bin` → manifest
`v39_v32plus_spline_seed17_2026-05-25.toml`.

## Schema

```toml
[bake]
name              = "zensim-a"                  # ZensimProfile external name (the user-facing API)
internal_version  = "v39"                       # internal id; CLIMBS, never rewinds (no v32 reuse)
file              = "../v39_v32plus_spline_seed17_2026-05-25.bin"
sha256            = "72820d66…"                  # of the file
file_bytes        = 257955
n_inputs          = 372
arch              = "372 → 128 → 64 + per-sample-α + tanh-pin(scale=30) + PCHIP spline"
output_disposition = "extrapolate_score"        # spline output passed through; ≤100 upper-clamped runtime-side
date              = "2026-05-25"
ship_commit       = "a109680"                   # git sha when this bake was shipped to A

[training]
# Reproducible command — every flag the trainer accepted.
trainer            = "zensim_mlp_train"
trainer_commit     = "(unknown — pre-session-133ab28d)"
seed               = 17
mse_weight         = 0.6
ranknet_weight     = 0.6
monotonicity_reg   = 1.0
monotonicity_margin = 0.0
tanh_output_head_scale = 30.0
anchor_loss_weight = 0.01   # post-training spline-fit only — NOT a training loss target
anchor_target_score = "PER-ROW (multi-band)"   # not the constant
anchor_step_p      = 0.05
hidden             = 128
n_hidden_layers    = 2
per_sample_alpha_head = true
epochs             = 200
pairs_per_epoch    = 50000
lr                 = 0.001
l2                 = 0.0001
leaky_alpha        = 0.01
target_column      = "human_score"              # per-group normalized to [0,1]
target_scale       = 1.0
max_features       = 372
minibatch_size     = 32
val_aggregate      = "geomean3"
out_dtype          = "f32"

# Group weights as train_w:val_w
groups = [
    { name = "safesyn",      path = "{canonical}/safesyn.parquet",       train_w = 1.0, val_w = 0.5 },
    { name = "cid22_train",  path = "{canonical}/cid22_train_norm.parquet", train_w = 1.5, val_w = 2.0 },
    { name = "kadid",        path = "{canonical}/kadid.parquet",         train_w = 0.5, val_w = 1.0 },
    { name = "tid",          path = "{canonical}/tid.parquet",           train_w = 0.5, val_w = 1.0 },
    { name = "konjnd_dense", path = "{canonical}/konjnd-dense-norm.parquet", train_w = 1.2, val_w = 1.5 },
]

# Auto-transforms applied to features before standardization (Yeo-Johnson + friends).
auto_transforms = "../../../benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv"

# Multi-band anchor for post-training spline fitting.
anchor_parquet = "{canonical}/multiband_anchor_dial100.parquet"

# The exact command (with substituted paths) — paste into a shell to re-run.
command = """
$TRAIN \\
  --group safesyn:$CANON/safesyn.parquet:1.0:0.5 \\
  --group cid22_train:$CANON/cid22_train_norm.parquet:1.5:2.0 \\
  --group kadid:$CANON/kadid.parquet:0.5:1.0 \\
  --group tid:$CANON/tid.parquet:0.5:1.0 \\
  --group konjnd_dense:$CANON/konjnd-dense-norm.parquet:1.2:1.5 \\
  --hidden 128 --n-hidden-layers 2 --per-sample-alpha-head --epochs 200 \\
  --lr 0.001 --l2 0.0001 --seed 17 --target-column human_score --max-features 372 \\
  --auto-transforms $TXFORM \\
  --val-aggregate geomean3 --out-dtype f32 \\
  --mse-weight 0.6 --ranknet-weight 0.6 --monotonicity-reg 1.0 \\
  --tanh-output-head-scale 30.0 \\
  --anchor-parquet $CANON/multiband_anchor_dial100.parquet \\
  --anchor-loss-weight 0.01 --anchor-step-p 0.05 \\
  --minibatch-size 32 --log-every 25 --out $OUT
"""

# Ordered post-training steps applied to produce the final bake.
steps = [
    "1. zensim_mlp_train (above) produces base bake `v32_cidmax_rn06_seed17_2026-05-25.bin`",
    "2. scripts/inject_spline.sh injects `zentrain.output_calibration_spline` metadata (PCHIP 3-knot) producing `v39_v32plus_spline_seed17_2026-05-25.bin`",
]

[inputs.canonical_root]
local = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
r2    = "s3://zentrain/canonical-2026-05-21/train"
tower = "/mnt/tower/output/zensim-archive-2026-05-20/canonical-2026-05-21/train"

[inputs.safesyn]
path   = "{canonical}/safesyn.parquet"
sha256 = "ad15cc79…"
rows   = 196086
columns = 372

# … one [inputs.<name>] block per parquet, each with sha256 + rows.

[urls]
bake_r2     = "s3://zentrain/bakes-2026-05-25/v39_v32plus_spline_seed17_2026-05-25.bin"
base_v32_r2 = "s3://zentrain/bakes-2026-05-25/v32_cidmax_rn06_seed17_2026-05-25.bin"

[eval]
# bake_verdict scorecard at ship time. Re-run with:
#   bake_verdict --bake <file> --corpora cid22,kadid,tid,konjnd,aic3,aic4
verdict_date  = "2026-05-26"
verdict_file  = "../../../benchmarks/v39_goals_and_mohammadi_panel_2026-05-26.md"
cid22_srocc   = 0.8793
kadid_srocc   = 0.9251
tid_srocc     = 0.9317
konjnd_srocc  = 0.4197
aic3_srocc    = 0.8023
aic4_srocc    = 0.9051
goals_weighted_score = 0.714
notes = """
G1 dial: pooled p5=-89.7 (negative-below intended) p95=97.4.
G5 KonJND HF-rank 0.42 — FAILS the 0.70 floor (characterized Pareto limit).
G7 CID22 0.879 — PASSES (≥0.85).
"""

[promotion]
profile          = "ZensimProfile::A"
deprecated_alias = "ZensimProfile::PreviewV0_3"
shipped_in_commit = "a109680"
```

## Reproducing a shipped bake from its manifest

1. `cd zensim/weights/manifests/ && cat <bake>.toml`
2. Verify every `[inputs.*]` parquet's sha256 matches:
   `sha256sum --check <(grep -E "sha256.*[a-f0-9]{32}" <bake>.toml)`.
3. Build the trainer at the manifest's `training.trainer_commit` (if pinned).
4. Run the `training.command` (paste, substitute the env vars defined in
   the manifest header).
5. Apply each entry in `training.steps` in order.
6. Compare the produced `.bin`'s sha256 to `bake.sha256`. They must match.
7. Run `bake_verdict` and compare to the `[eval]` section's numbers
   (tolerance: noise from non-determinism in parallel reductions, typically
   <1e-3 SROCC).

## Generating manifests for new bakes

The training runner script (e.g. `scripts/v_next/run_*.sh`) should
emit the `.toml` manifest alongside the `.bin` bake at finish:

```bash
"$TRAIN" --out "$BAKE" ... 2>&1 | tee "$LOG"
write_manifest "$BAKE" "$LOG" "$@" > "${BAKE%.bin}.toml"
```

`tools/bake_manifest.py` (TODO) generates the manifest from the bake +
log + the `$@` flag set. Until that lands, retrofit existing bakes by
hand (see this directory).
