#!/bin/bash
# Promote one arm's CSVs into a 944 feature root (owner: promote_ext944_canonical.py).
set -eu
ARM="$1"
WT="${ZR_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export EXT944_RUN=/mnt/v/output/zensim/era2-rank-2026-08-31/run-$ARM
export EXT944_DEST=/mnt/v/zen/zensim-training/era2rank-$ARM-2026-08-31
export EXT944_MODE=folded720append2pools
export EXT944_LEGS="ext_sdr25 ext_aic4 ext_konjnd_jpeg_val ext_aic3 ext_live ext_csiq ext_tid ext_cid22val ext_kadid"
export ZENSIM_COMMIT="${ZENSIM_COMMIT:-$(git -C "$WT" rev-parse HEAD)}"  # as-run: 9e52fb164c28725a6f12d911707b8caaeaac995e
cd "$WT" && python3 scripts/canonical_corpus/promote_ext944_canonical.py
