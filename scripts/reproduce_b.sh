#!/usr/bin/env bash
# Reproduce ZensimProfile::B (SDR) from committed code, byte-for-byte.
#
# B (SDR) = ens-Pline-cid80, INCLUSIVE-WINSOR + dense-dial. Full lineage:
#   parquet corpora ──gram──> Gram matrices ──fit──> per-corpus linear heads
#     (cid = hdrmix-lasso0.002-raw : hdr_v3mix 7,410 rows, ssim2+cvvdp-mix target
#      kon = canonhdr15-bvls-raw   : safesyn+cid22_train+kadid+tid+hdr_v3mix)
#   ──ensemble──> ens-Pline-cid80 (raw-space convex blend 0.8*cid + 0.2*kon)
#     = b_sdr_linear_cid80_anchored_2026-07-04.bin (the RAW weights/scaler/spline)
#   ──add-winsor (fit corpus = hdr_v3mix + zenjxl near-lossless SDR sweep,
#      near-lossless-INCLUSIVE 2026-07-07; [p0.1,p99.9] per feature) ──> winsor guard
#   ──extend-top (dense dial) ──> b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin [SHIPPED]
#     (extends ONLY the winsor bake's dial TOP by the training-fitted concave
#      saturation so near-lossless codec-knob configs resolve toward 100; bottom +
#      in-distribution spline kept VERBATIM, so rank is IDENTICAL.)
#
# WHY the inclusive corpus (2026-07-07): the predecessor
# (b_sdr_linear_cid80_dense_dial_2026-07-05.bin, sha b78adb15, now in weights/archive/)
# fit its winsor bounds on hdr_v3mix ALONE — its [p0.1,p99.9] bounds sat above the
# SDR near-lossless feature range, clamping 245/372 features CONSTANT there and
# pinning B's near-lossless dial at ~91.5 while ssim2/A climbed to ~96. Adding the
# zenjxl near-lossless SDR sweep to the fit corpus frees those lower bounds (dial
# climbs to ~96.1; near-lossless per-image rank-vs-ssim2 0.657->0.771) at zero
# human-MOS cost (CID22 0.8763->0.8764, KonJND 0.5474->0.5466), f155's upper guard
# unmoved. Details: benchmarks/jxl_nearlossless_dial_2026-07-05.md §7-§8.
#
# The linear fits are DETERMINISTIC (Gram-exact full-data solves, no SGD/seed) and
# add-winsor (fixed percentiles) + extend-top (lstsq + fixed percentiles) are
# deterministic, so a same-input re-run reproduces the shipped bytes EXACTLY.
# Requires the /mnt/v fit inputs (hdr_v3mix, the near-lossless sweep, the anchor).
#
# Provenance (pinned input shas, corpus sizes): benchmarks/provenance_best_results_2026-07-04.md
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE=/mnt/v/output/zensim-multicodec-probe
FITS="$PROBE/linear-probe/fits"
OUT="${1:-/tmp/reproduce_b}"
mkdir -p "$OUT"

B_SHA=b6fe5233          # b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin (SHIPPED)
BHDR_SHA=373eac56       # bhdr_linear_shaped_anchored2_2026-07-04.bin (BHdr, informational)

echo "== step 1/2: rebuild the shipped B (raw -> inclusive-winsor -> dense-dial), assert byte-identity =="
REFIT="$REPO/target/release/bake_dial_refit"
if [ ! -x "$REFIT" ]; then
  echo "  building bake_dial_refit…"
  ( cd "$REPO" && cargo build --release -p zensim-validate --bin bake_dial_refit >/dev/null 2>&1 )
fi
INC=/mnt/v/output/zensim-jxl-nearlossless/inclusive_winsor_corpus.parquet
echo "  rebuilding inclusive winsor fit corpus…"
python3 "$REPO/scripts/v_next/build_inclusive_winsor_corpus.py" >/dev/null
"$REFIT" add-winsor \
  --in "$REPO/zensim/weights/b_sdr_linear_cid80_anchored_2026-07-04.bin" \
  --out "$OUT/b_winsor.bin" --fit-corpus "$INC" --lo-pct 0.1 --hi-pct 99.9
"$REFIT" extend-top \
  --in "$OUT/b_winsor.bin" \
  --out "$OUT/b_sdr_linear_cid80_inclwinsor_dense_dial.bin" \
  --anchor /mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet \
  --target-col target_score
got=$(sha256sum "$OUT/b_sdr_linear_cid80_inclwinsor_dense_dial.bin" | cut -c1-8)
[ "$got" = "$B_SHA" ] && echo "  sha $got — BYTE-REPRODUCED ✓" || { echo "  !! sha $got != $B_SHA"; exit 1; }
cmp "$REPO/zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin" \
    "$OUT/b_sdr_linear_cid80_inclwinsor_dense_dial.bin" && echo "  byte-identical to shipped ✓"

echo "== step 2/2: functional verify (rank panel + dial gates + tail gate) =="
VERDICT="$REPO/target/release/bake_verdict"
GRID=/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined.parquet
if [ -x "$VERDICT" ]; then
  "$VERDICT" --bake "$OUT/b_sdr_linear_cid80_inclwinsor_dense_dial.bin" --dial-grid "$GRID" --corpora cid22,konjnd \
    2>/dev/null | grep -iE "^\| (CID22|KonJND-1k) |inversions|dead-zone|monotonicity" | grep -iv loaded || true
  echo "  (expect CID22 ~0.8764, KonJND ~0.5466; dial: inversions/dead-zone/mono all PASS)"
else
  echo "  bake_verdict not built (cargo build --release -p zensim-validate --bin bake_verdict); skipping"
fi
"$REFIT" gate --bake "$OUT/b_sdr_linear_cid80_inclwinsor_dense_dial.bin" \
  --corpus /mnt/v/output/zensim-multicodec-probe/bigcodec_valdigits_2026-07-02.parquet \
  --ref-col human_score 2>/dev/null | grep -iE "G-RANGE|VERDICT" || true

echo
echo "DONE — B reproduced byte-identically (sha $B_SHA)."
echo "Predecessor (hdr_v3mix-only winsor, weights/archive/, sha b78adb15): near-lossless dial pinned 91.5."
echo "BHdr (sha $BHDR_SHA): bake_dial_refit bottom-extend / shared-anchor (+ research densify in scripts/v_next/hdr_anchor_dense_refit.py)."
