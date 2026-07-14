#!/usr/bin/env bash
# Reproduce the negatives-unblocked B dial (candidate, NOT yet shipped).
#
# WHY (project_profile_b_hdr / lineage §8.30): shipped B floors at ~1.5 on the
# worst content — 0 negatives — because its dial anchor `target_score` is ssim2
# CLAMPED at 0. ssim2_gpu/BHdr/A all reach −35..−64 on catastrophic pairs; user
# directive 2026-07-14 "negative zensim scores are valid and needed" + "unblock
# negative values on b". Fix = re-anchor the dial to the UNCLAMPED ssim2_gpu
# (the same metric B already predicts), then re-apply the near-lossless extend-top.
#
# RANK-INVARIANT by construction (monotone spline): CID22/KADID/TID SROCC
# byte-identical. Positive operating range (dial≥50) preserved to ≤0.3 pts; only
# the previously-floored dial<20 region spreads into the negative tail.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./target/release/bake_dial_refit
ANCHOR=/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet
SHIP=zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin
OUT=/mnt/v/output/zensim/reports/b_negatives
mkdir -p "$OUT"
# 1) re-anchor dial bottom+body to UNCLAMPED ssim2_gpu (unblocks negatives)
"$BIN" shared-anchor --in "$SHIP" --out "$OUT/_b_ssim2anchored.bin" \
    --anchor "$ANCHOR" --target-col ssim2_gpu
# 2) restore near-lossless dial top (the ssim2 refit tops at 95.9; extend to 100)
"$BIN" extend-top --in "$OUT/_b_ssim2anchored.bin" \
    --out "$OUT/b_sdr_linear_cid80_ssim2anchored_dense_dial_2026-07-14.bin" \
    --anchor "$ANCHOR" --target-col target_score
sha256sum "$OUT/b_sdr_linear_cid80_ssim2anchored_dense_dial_2026-07-14.bin"
