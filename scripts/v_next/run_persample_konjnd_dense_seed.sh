#!/usr/bin/env bash
# V_24-persample-α-dense — per-sample α head with DENSIFIED konjnd (20k rows
# vs the legacy 1008-row group). Same V_22-LARGE-style recipe otherwise.
#
# Args:
#   <seed>      — RNG seed
#   <konjnd_w>  — train weight for the konjnd group (0.02 = V_22 default; 0.10 = boost)
#   <out_dir>   — output directory for bake + log
set -euo pipefail
SEED="${1:?seed}"
KONJND_W="${2:?konjnd_w}"
OUT_DIR="${3:?out_dir}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/persample_dense_konjnd${KONJND_W}_seed${SEED}.bin"
LOG="$OUT_DIR/persample_dense_konjnd${KONJND_W}_seed${SEED}.log"
TRAINER=/home/lilith/work/zen/zensim/target/release/zensim_mlp_train

# Group paths — safesyn/kadid/tid/cvvdp from V_22-LARGE recipe;
# konjnd uses the DENSIFIED 20k-row parquet (this is the experimental lever).
SAFESYN=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/safesyn_mix_300col.parquet
KADID=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/kadid_mix_300col.parquet
TID=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/tid_mix_300col.parquet
CVVDP=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet
KONJND_DENSE=/mnt/v/zen/zensim-training/2026-05-18-konjnd-dense/konjnd_dense_features_mix_targets_300col.parquet

"$TRAINER" \
  --group "safesyn:${SAFESYN}:1.0:0.0" \
  --group "kadid:${KADID}:0.3:1.0" \
  --group "tid:${TID}:0.3:1.0" \
  --group "konjnd:${KONJND_DENSE}:${KONJND_W}:1.0" \
  --group "cvvdp_iwssim_large:${CVVDP}:0.5:0.0" \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 --target-scale 100.0 \
  --val-policy min --minibatch-size 256 \
  --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED konjnd_w=$KONJND_W bake=$BAKE log=$LOG"
