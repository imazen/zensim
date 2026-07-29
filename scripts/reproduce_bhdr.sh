#!/usr/bin/env bash
# Reproduce ZensimProfile::BHdr from committed code + /mnt/v fit inputs,
# byte-for-byte, through the PURE-RUST fit chain (task #68) — zero Python
# anywhere between fit and bake. First verified 2026-07-14 via the Python
# chain; the Rust `bake_dial_refit fit-lasso` port reproduced the same sha
# on 2026-07-29 with the lasso weights ADDITIONALLY bit-exact (f64) against
# the Python fit npz (`--parity-fit`).
#
# Shipped BHdr = `hdrmix-lasso0.0003-shaped`:
#   zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin
#   (11,826 B, sha256 7d7f2123…, 18-knot monotone dial [0.00, 95.77]).
#
# Deterministic chain (NO seed, NO SGD anywhere):
#   grams/hdr_v3mix.npz  (cvvdp-mix corpus Gram: hdr_v3mix, target
#     human_score = 0.5·ssim2norm + 0.5·(JOD−6)/4, 7,410 rows, READ-ONLY)
#   ──lasso(λ=0.0003), shaped feature space (coordinate descent, fixed sweep
#     order → bit-exact w/bias/mu/sd)──f16 pack (tau=0)──shared-anchor PCHIP
#     spline on the PACKED forward over val/anchor.npz──> the shipped bake,
#   all inside ONE `bake_dial_refit fit-lasso` invocation (npz reading via
#   zensim_validate::npz, fit math via zensim_validate::gram_lasso, knots via
#   dial_spline::fit_spline_knots, serialization via zenpredict_bake::bake).
#
# Parity note (measured 2026-07-29): the gram standardization + lasso CD are
# BIT-EXACT vs numpy. The anchor forward is not — numpy's BLAS dgemv sums in
# a different order (1371/2000 rows differ, ≤4096 ulp ≈ 2⁻⁴⁰ relative) — but
# the per-bin medians + f32 knot quantization absorb that, and this script's
# whole-file sha assert catches any future drift loudly. Going forward the
# Rust chain is self-consistent (sequential sums, no BLAS), so re-runs are
# exactly deterministic.
#
# ⚠ Provenance caveat (NOT a reproduction issue): the λ=0.0003 PICK was
# scoreboard-selected on UPIQ and is NOT an established in-domain win — see
# benchmarks/bhdr_improvement_split_lineage_2026-07-12.md §7 (post-promotion
# audit). This script reproduces the shipped artifact; it does not re-litigate
# whether that artifact should have been the pick.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE=/mnt/v/output/zensim-multicodec-probe/linear-probe
SHIPPED="$REPO/zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin"
BHDR_SHA=7d7f212369f734aa9de072f84ac0e8b97b86deefa3c8bfe26de94de6b49b7ce4
CAND="$PROBE/bakes/lp_hdrmix-lasso0.0003-shaped-tau0-f16.bin"
SCREEN_TSV="$REPO/benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv"
BDR_BIN="${BDR_BIN:-$REPO/target/release/bake_dial_refit}"

echo "== inputs =="
for f in "$PROBE/grams/hdr_v3mix.npz" "$PROBE/val/anchor.npz"; do
  [ -f "$f" ] || { echo "MISSING fit input: $f (need the /mnt/v probe artifacts)"; exit 2; }
  echo "  present: $f"
done
[ -x "$BDR_BIN" ] || {
  echo "  building bake_dial_refit…"
  ( cd "$REPO" && cargo build --release -p zensim-validate --bin bake_dial_refit >/dev/null 2>&1 )
}

echo "== fit + pack + spline + bake (pure Rust, one invocation) =="
mkdir -p "$PROBE/bakes"
"$BDR_BIN" fit-lasso \
  --gram "$PROBE/grams/hdr_v3mix.npz" \
  --space shaped --target human_score --lam 0.0003 \
  --anchor "$PROBE/val/anchor.npz" \
  --transforms-tsv "$SCREEN_TSV" \
  --out "$CAND"

echo "== assert byte-identity to the shipped BHdr bake =="
got="$(sha256sum "$CAND" | awk '{print $1}')"
echo "  reproduced: $got"
echo "  shipped:    $BHDR_SHA"
if [ "$got" = "$BHDR_SHA" ] && cmp -s "$CAND" "$SHIPPED"; then
  echo "  ✅ BHdr reproduces byte-for-byte (gram → lasso0.0003 → f16 bake+spline, pure Rust)."
else
  echo "  ❌ MISMATCH — fit inputs or baker drifted; investigate before trusting the bake."
  exit 1
fi
