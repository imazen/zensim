#!/usr/bin/env bash
# Build the concatenated pooled caches (row order must match the pooled TSVs:
# tidkadid = tid + kadid_train; pooled = cid22a + tid + kadid_train).
set -euo pipefail
OUT=/mnt/v/output/zensim/dvifm-screen2d-2026-09-19
for pl in xyb_y ycbcr_y ycbcr_cb ycbcr_cr; do
  python3 "$OUT/tools/cat_caches.py" "$OUT/cache/tidkadid_${pl}.bin" \
    "$OUT/cache/tid_${pl}.bin" "$OUT/cache/kadid_train_${pl}.bin"
  python3 "$OUT/tools/cat_caches.py" "$OUT/cache/pooled_${pl}.bin" \
    "$OUT/cache/cid22a_${pl}.bin" "$OUT/cache/tid_${pl}.bin" \
    "$OUT/cache/kadid_train_${pl}.bin"
done
echo "concat done"
