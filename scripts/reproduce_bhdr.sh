#!/usr/bin/env bash
# Reproduce ZensimProfile::BHdr from committed code + /mnt/v fit inputs,
# byte-for-byte. Verified 2026-07-14 (this script's first run reproduced the
# shipped sha exactly, incl. against a same-day-regenerated anchor.npz).
#
# Shipped BHdr = `hdrmix-lasso0.0003-shaped`:
#   zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin
#   (11,826 B, sha256 7d7f2123…, 18-knot monotone dial [0.00, 95.77]).
#
# Deterministic chain (NO seed, NO SGD anywhere):
#   grams/hdr_v3mix.npz  (cvvdp-mix corpus Gram: hdr_v3mix, target
#     human_score = 0.5·ssim2norm + 0.5·(JOD−6)/4, 7,410 rows, READ-ONLY)
#   ──lasso(λ=0.0003), shaped feature space──> fits/hdrmix-lasso0.0003-shaped.npz
#     (coordinate descent, fixed sweep order → bit-exact w/bias/mu/sd)
#   ──f16 pack (tau=0) + shared-anchor PCHIP spline on the PACKED forward over
#     val/anchor.npz (built from canonical multiband_anchor_dial100.parquet)──>
#     bakes/lp_hdrmix-lasso0.0003-shaped-tau0-f16.bin == the shipped bake.
#
# The lasso is a full-data Gram solve (deterministic); f16 rounding + fixed
# percentile spline knots are deterministic; the baker (zenpredict) is
# deterministic — so a same-input re-run reproduces the shipped bytes EXACTLY.
#
# ⚠ Provenance caveat (NOT a reproduction issue): the λ=0.0003 PICK was
# scoreboard-selected on UPIQ and is NOT an established in-domain win — see
# benchmarks/bhdr_improvement_split_lineage_2026-07-12.md §7 (post-promotion
# audit). This script reproduces the shipped artifact; it does not re-litigate
# whether that artifact should have been the pick.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE=/mnt/v/output/zensim-multicodec-probe/linear-probe
LP="$REPO/scripts/v_next/linear_projections_2026-07-03.py"
SHIPPED="$REPO/zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin"
BHDR_SHA=7d7f212369f734aa9de072f84ac0e8b97b86deefa3c8bfe26de94de6b49b7ce4
CAND="$PROBE/bakes/lp_hdrmix-lasso0.0003-shaped-tau0-f16.bin"

echo "== inputs =="
for f in "$PROBE/grams/hdr_v3mix.npz" "$PROBE/val/anchor.npz"; do
  [ -f "$f" ] || { echo "MISSING fit input: $f (need the /mnt/v probe artifacts)"; exit 2; }
  echo "  present: $f"
done
[ -x "$HOME/work/zen/zenanalyze/target/release/zenpredict" ] || {
  echo "  building zenpredict baker…"
  ( cd "$HOME/work/zen/zenanalyze" && cargo build --release --bin zenpredict -p zenpredict-bake >/dev/null 2>&1 )
}

echo "== step 1/2: re-fit lasso(0.0003) on the hdr_v3mix cvvdp-mix Gram (deterministic) =="
python3 "$LP" fit --only hdrmix 2>&1 | grep 'hdrmix-lasso0.0003-shaped'

echo "== step 2/2: f16 pack + shared-anchor spline (deterministic) =="
python3 "$LP" finalize --keys hdrmix-lasso0.0003-shaped --taus 0 2>&1 | grep 'hdrmix-lasso0.0003-shaped'

echo "== assert byte-identity to the shipped BHdr bake =="
got="$(sha256sum "$CAND" | awk '{print $1}')"
echo "  reproduced: $got"
echo "  shipped:    $BHDR_SHA"
if [ "$got" = "$BHDR_SHA" ] && cmp -s "$CAND" "$SHIPPED"; then
  echo "  ✅ BHdr reproduces byte-for-byte (gram → lasso0.0003 → f16 bake+spline)."
else
  echo "  ❌ MISMATCH — fit inputs or baker drifted; investigate before trusting the bake."
  exit 1
fi
