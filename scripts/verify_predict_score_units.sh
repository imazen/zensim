#!/usr/bin/env bash
# Gate for `bake_dial_refit predict --score-units` (added 2026-09-04).
#
# THE CONTRACT: with `--score-units`, a k>=2 `--ensemble` blend must equal
# `bake_verdict --ensemble`'s blend. WITHOUT it, `predict` accumulates RAW
# network output while `bake_verdict` accumulates SCORE units (each member's
# head/tanh-pin/output-spline applied first), so the two disagree on every
# blend whose members have different raw scales. That divergence is the reason
# the flag exists — see the flag's rustdoc and
# benchmarks/fastclass_distill_wave_2026-09-04.md §6f.
#
# It is INVISIBLE at k=1 (a monotone spline is rank-invariant, so every
# single-bake SROCC agrees), which is why it went unnoticed and why this gate
# uses a two-member blend.
#
# Fails loud and nonzero. Fixtures are env-overridable; the defaults are the
# `HYA` pair and the KonJND JPEG-504 ruler at the teacher's own twin-era root.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BDR="${ZL_BDR:-${CARGO_TARGET_DIR:-$REPO/target}/release/bake_dial_refit}"
BV="${ZL_BV:-${CARGO_TARGET_DIR:-$REPO/target}/release/bake_verdict}"
M1="${VPSU_M1:-/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin}"
M2="${VPSU_M2:-/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.2_a0.2_b0.97.bin}"
ROOT="${VPSU_ROOT:-/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30}"
CORPUS="${VPSU_CORPUS:-konjnd}"
PARQ="${VPSU_PARQUET:-$ROOT/ext_konjnd_jpeg_val.parquet}"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
for f in "$BDR" "$BV" "$M1" "$M2" "$PARQ"; do
  [ -e "$f" ] || { echo "verify_predict_score_units: MISSING $f"; exit 2; }
done
rc=0
for wgt in 0.5 0.84; do
  q=$(python3 -c "print(1-$wgt)")
  "$BDR" predict --ensemble "$M1,$M2" --ensemble-weights "$wgt,$q" --score-units \
      --corpus "$PARQ" --out "$W/p.tsv" >/dev/null 2>&1
  "$BV" --ensemble "$M1,$M2" --ensemble-weights "$wgt,$q" --regime 944 \
      --features-root "$ROOT" --corpora "$CORPUS" \
      --per-pair-output "$W/v.tsv" --per-pair-refs --output /dev/null >/dev/null 2>&1
  read -r maxd n < <(python3 - "$W/p.tsv" "$W/v.tsv" <<'PY'
import sys
p=[float(l.split('\t')[1]) for l in open(sys.argv[1]).read().splitlines()[1:]]
rows=[l.split('\t') for l in open(sys.argv[2]).read().splitlines() if l.strip()]
i={n:k for k,n in enumerate(rows[0])}
v=[float(r[i['pred']]) for r in rows[1:]]
assert len(p)==len(v), f"row count {len(p)} != {len(v)}"
print(max(abs(a-b) for a,b in zip(p,v)), len(p))
PY
)
  # 1e-9 is a float-formatting floor, not a tolerance on the math: the two
  # tools run the identical dispatch, so any real divergence is O(1), not 1e-9.
  ok=$(python3 -c "print('PASS' if $maxd < 1e-9 else 'FAIL')")
  echo "w=$wgt  n=$n  max|predict --score-units - bake_verdict --ensemble| = $maxd  -> $ok"
  [ "$ok" = PASS ] || rc=1
done
if [ "$rc" = 0 ]; then echo "verify_predict_score_units: PASS"; else echo "verify_predict_score_units: FAIL"; fi
exit "$rc"
