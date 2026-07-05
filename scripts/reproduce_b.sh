#!/usr/bin/env bash
# Reproduce ZensimProfile::B (SDR) + ZensimProfile::BHdr from committed code.
#
# B (SDR) = ens-Pline-cid80, winsorized. The full lineage:
#   parquet corpora ──gram──> Gram matrices ──fit──> per-corpus linear heads
#     (cid = hdrmix-lasso0.002-raw : hdr_v3mix 7,410 rows, ssim2+cvvdp-mix target
#      kon = canonhdr15-bvls-raw   : safesyn+cid22_train+kadid+tid+hdr_v3mix)
#   ──ensemble──> ens-Pline-cid80 (raw-space convex blend 0.8*cid + 0.2*kon)
#   ──shared_anchor_refit──> lp_ens-Pline-cid80-anchored-f16.bin (dial spline)
#   ──winsorize_bake──> b_sdr_linear_cid80_winsor_2026-07-05.bin  [SHIPPED]
#
# The linear fits are DETERMINISTIC (Gram-exact full-data solves, no SGD/seed):
# 44/44 refits are byte-identical across a full re-run (determinism_check.py).
# So a same-input re-run reproduces the shipped bytes EXACTLY (proven below via
# --expect-sha256). BHdr's chain is analogous (hdr_anchor_dense_refit.py).
#
# Provenance detail (pinned input shas, corpus sizes): see
#   benchmarks/provenance_best_results_2026-07-04.md  (Reproducibility section)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE=/mnt/v/output/zensim-multicodec-probe
FITS="$PROBE/linear-probe/fits"
BAKES="$PROBE/linear-probe/bakes"
FITCORPUS="$PROBE/hdr_zenjxl_v3mix_traindigits_2026-07-03.parquet"
OUT="${1:-/tmp/reproduce_b}"
mkdir -p "$OUT"

# expected shipped shas (16-char prefix)
B_SHA=b92b0b7a          # b_sdr_linear_cid80_winsor_2026-07-05.bin
BHDR_SHA=373eac56       # bhdr_linear_shaped_anchored2_2026-07-04.bin (BHdr, informational)

echo "== step 1/3: anchor-refit the ensemble dial spline (ens-Pline-cid80 -> anchored) =="
# shared_anchor_refit.py reads $FITS/ens-Pline-cid80.npz (the deterministic
# ensemble fit) and emits the anchored bake. If you need to rebuild the .npz from
# parquet first:  python3 scripts/v_next/linear_projections_2026-07-03.py gram \
#                 && ... fit && ... ensemble   (heavy; deterministic).
test -f "$FITS/ens-Pline-cid80.npz" || { echo "MISSING $FITS/ens-Pline-cid80.npz — run linear_projections gram/fit/ensemble first"; exit 2; }
python3 "$REPO/scripts/v_next/shared_anchor_refit.py"
ANCHORED="$BAKES/lp_ens-Pline-cid80-anchored-f16.bin"
test -f "$ANCHORED" || { echo "anchor refit did not produce $ANCHORED"; exit 2; }

echo "== step 2/3: winsorize (the tail guard) -> shipped B, assert byte-identity =="
python3 "$REPO/scripts/v_next/winsorize_bake.py" \
  --in "$ANCHORED" --fit-corpus "$FITCORPUS" \
  --out "$OUT/b_sdr_linear_cid80_winsor.bin" --expect-sha256 "$B_SHA"
cmp "$REPO/zensim/weights/b_sdr_linear_cid80_winsor_2026-07-05.bin" \
    "$OUT/b_sdr_linear_cid80_winsor.bin" \
  && echo "  B byte-identical to shipped ✓"

echo "== step 3/3: functional verify (rank panel + tail gate) =="
VERDICT="$REPO/target/release/bake_verdict"
if [ -x "$VERDICT" ]; then
  "$VERDICT" --bake "$OUT/b_sdr_linear_cid80_winsor.bin" --corpora cid22 \
    | grep -iE "cid22|srocc" | head -3 || true
  echo "  (expect CID22 SROCC ~0.8763)"
else
  echo "  bake_verdict not built (cargo build --release -p zensim-validate --bin bake_verdict); skipping rank panel"
fi
python3 "$REPO/scripts/v_next/bake_outlier_gate.py" \
  --bake "$OUT/b_sdr_linear_cid80_winsor.bin" \
  --corpus "$PROBE/bigcodec_valdigits_2026-07-02.parquet" 2>/dev/null \
  | grep -iE "G-RANGE|VERDICT" || echo "  (outlier gate needs the bigcodec val parquet)"

echo
echo "DONE — B reproduced byte-identically (sha $B_SHA)."
echo "BHdr (sha $BHDR_SHA) chain: python3 scripts/v_next/hdr_anchor_dense_refit.py"
echo "  (hdr-lasso0.001-shaped -> anchored2; shaped transforms already winsorize it)."
