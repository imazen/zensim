#!/bin/bash
# fastclass2 — G6: grade EVERY landed arm on the ladder A7r + full contract.
# One bake_verdict per bake (~5 s). Computes no statistic; the gate owner does.
# Idempotent: skips a bake whose gaddr json already exists.
set -euo pipefail
BIN="${FC2_BIN:-/mnt/v/zen/cargo-targets/fastclass2/release}"
W="${FC2_OUT:-/mnt/v/output/zensim/fastclass2-2026-09-05}"
ROOT="${FC2_ROOT:-/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01}"
LG=/mnt/v/output/zensim/ladder-2026-09-05/instruments/dial_grid_944col_ladder.parquet
LT=/mnt/v/output/zensim/ladder-2026-09-05/instruments/dialcells_ssim2_ladder.tsv
ID=/mnt/v/output/zensim/dfree-2026-09-05/probes/identity_probe_944pools_2026-09-05.parquet
NT=/mnt/v/output/zensim/dfree-2026-09-05/probes/negtail_probe_944pools_2026-09-05.parquet
mkdir -p "$W/gaddr"
for B in "$W"/bakes/*_packed.bin; do
  [ -f "$B" ] || continue
  N="$(basename "$B" .bin)"
  [ -f "$W/gaddr/gaddr_${N}_ladder.json" ] && continue
  nice -n19 ionice -c3 "$BIN/bake_verdict" --bake "$B" --regime 944 \
    --features-root "$ROOT" --dial-grid "$LG" --gaddr-grid-truth "$LT" \
    --floor-rule resolvable --negtail-probe "$NT" --identity-probe "$ID" \
    --name "${N}@ladder" --gaddr-json "$W/gaddr/gaddr_${N}_ladder.json" \
    --output /dev/null >/dev/null 2>&1 || echo "GRADE FAILED $N"
  echo "graded $N"
done
