#!/usr/bin/env bash
# Reproduce ZensimProfile::B (SDR) from committed code, byte-for-byte.
#
# B (SDR) = ens-Pline-cid80, winsorized + dense-dial. Full lineage:
#   parquet corpora ──gram──> Gram matrices ──fit──> per-corpus linear heads
#     (cid = hdrmix-lasso0.002-raw : hdr_v3mix 7,410 rows, ssim2+cvvdp-mix target
#      kon = canonhdr15-bvls-raw   : safesyn+cid22_train+kadid+tid+hdr_v3mix)
#   ──ensemble──> ens-Pline-cid80 (raw-space convex blend 0.8*cid + 0.2*kon)
#   ──winsorize_bake──> b_sdr_linear_cid80_winsor (winsor tail guard; weights/archive/)
#   ──dense_dial_refit_b──> b_sdr_linear_cid80_dense_dial_2026-07-05.bin  [SHIPPED]
#     (extends ONLY the winsor bake's dial TOP by the training-fitted concave
#      saturation so near-lossless codec-knob configs resolve toward 100 instead of
#      piling at the top knot; bottom + in-distribution spline kept VERBATIM, so
#      rank is IDENTICAL and both raw tails stay inside the knot domain.)
#
# The linear fits are DETERMINISTIC (Gram-exact full-data solves, no SGD/seed;
# 44/44 refits byte-identical) and dense_dial_refit_b is deterministic (lstsq +
# fixed percentiles), so a same-input re-run reproduces the shipped bytes EXACTLY.
#
# Provenance (pinned input shas, corpus sizes, the winsor-only predecessor now in
# weights/archive/): benchmarks/provenance_best_results_2026-07-04.md
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE=/mnt/v/output/zensim-multicodec-probe
FITS="$PROBE/linear-probe/fits"
OUT="${1:-/tmp/reproduce_b}"
mkdir -p "$OUT"

B_SHA=b78adb15          # b_sdr_linear_cid80_dense_dial_2026-07-05.bin (SHIPPED)
BHDR_SHA=373eac56       # bhdr_linear_shaped_anchored2_2026-07-04.bin (BHdr, informational)

echo "== step 1/2: build the shipped B (extend the winsor bake's dial top), assert byte-identity =="
# bake_dial_refit extend-top extends the COMMITTED archived winsor bake
# (zensim/weights/archive/b_sdr_linear_cid80_winsor_2026-07-05.bin — read for
# weights/scaler/transforms/bottom-spline) + the multiband anchor (top saturation
# fit). The Rust bin reproduces the shipped B BYTE-IDENTICALLY (migrated from
# scripts/v_next/dense_dial_refit_b.py 2026-07-05; see
# benchmarks/bake_refit_rust_migration_2026-07-05.md).
REFIT="$REPO/target/release/bake_dial_refit"
if [ ! -x "$REFIT" ]; then
  echo "  building bake_dial_refit…"
  ( cd "$REPO" && cargo build --release -p zensim-validate --bin bake_dial_refit >/dev/null 2>&1 )
fi
"$REFIT" extend-top \
  --in "$REPO/zensim/weights/archive/b_sdr_linear_cid80_winsor_2026-07-05.bin" \
  --out "$OUT/b_sdr_linear_cid80_dense_dial.bin" \
  --anchor /mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet \
  --target-col target_score
# Python fallback (retired 2026-07-05; kept for provenance):
#   python3 "$REPO/scripts/v_next/dense_dial_refit_b.py" "$OUT/b_sdr_linear_cid80_dense_dial.bin"
got=$(sha256sum "$OUT/b_sdr_linear_cid80_dense_dial.bin" | cut -c1-8)
[ "$got" = "$B_SHA" ] && echo "  sha $got — BYTE-REPRODUCED ✓" || { echo "  !! sha $got != $B_SHA"; exit 1; }
cmp "$REPO/zensim/weights/b_sdr_linear_cid80_dense_dial_2026-07-05.bin" \
    "$OUT/b_sdr_linear_cid80_dense_dial.bin" && echo "  byte-identical to shipped ✓"

echo "== step 2/2: functional verify (rank panel + dial gates + tail gate) =="
VERDICT="$REPO/target/release/bake_verdict"
GRID=/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined.parquet
if [ -x "$VERDICT" ]; then
  "$VERDICT" --bake "$OUT/b_sdr_linear_cid80_dense_dial.bin" --dial-grid "$GRID" --corpora cid22,konjnd \
    2>/dev/null | grep -iE "^\| (CID22|KonJND-1k) |inversions|dead-zone|monotonicity" | grep -iv loaded || true
  echo "  (expect CID22 ~0.8763, KonJND ~0.5474; dial: inversions/dead-zone/mono all PASS)"
else
  echo "  bake_verdict not built (cargo build --release -p zensim-validate --bin bake_verdict); skipping"
fi
"$REFIT" gate --bake "$OUT/b_sdr_linear_cid80_dense_dial.bin" \
  --corpus /mnt/v/output/zensim-multicodec-probe/bigcodec_valdigits_2026-07-02.parquet \
  --ref-col human_score 2>/dev/null | grep -iE "G-RANGE|VERDICT" || true
# Python fallback (retired 2026-07-05):
#   python3 "$REPO/scripts/v_next/bake_outlier_gate.py" --bake "$OUT/..." ...

echo
echo "DONE — B reproduced byte-identically (sha $B_SHA)."
echo "Predecessor (winsor-only, weights/archive/, sha b92b0b7a): bake_dial_refit add-winsor."
echo "BHdr (sha $BHDR_SHA): bake_dial_refit bottom-extend / shared-anchor (+ research densify in scripts/v_next/hdr_anchor_dense_refit.py)."
