#!/usr/bin/env bash
#
# wave9_screens.sh — build the THREE wave-9 winsor screens from ONE pooled fit
# (SOTA-944 WAVE 9, amendment 10 §10.2).
#
# The fit corpus, the fit rule and the owner are wave-8's, unchanged: the same
# 7 tables / 408,033 rows of §9.1, percentile_linear at [0.1, 99.9] plus the
# [0,0] -> [0,1e-9] guard, `bake_dial_refit refit-winsor`. Only the SUBSET of
# indices the fit is APPLIED to differs:
#
#   all            -> reproduces benchmarks/wave8/refit_screen_tokens.txt.
#                     Byte-identity with the committed file is the registered
#                     NO-REGRESSION GATE on the selector extension (§10.2) and
#                     an independent re-derivation of wave-8's screen.
#   degenerate     -> w9_degen24_screen_tokens.txt  (the 24 append windows)
#   nondegenerate  -> w9_fold30_screen_tokens.txt   (the 30 fold windows)
#
# Then it checks the registered set identity: the two partial screens differ
# from the base in disjoint line sets whose union is exactly the full refit's.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${ZL_REFIT:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/bake_dial_refit}
E=${SOTA944_E:-/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01}
T=${SOTA944_T:-/mnt/v/zen/zensim-training}
K=${SOTA944_K:-/mnt/v/zen/zensim-training/kadis-944-2026-08-01}
W8=$REPO_ROOT/benchmarks/wave8
W9=$REPO_ROOT/benchmarks/wave9
LOG=${WAVE9_LOG:-$HOME/tmp/wave9}
mkdir -p "$W9" "$LOG"
[[ -x "$BIN" ]] || { echo "missing $BIN" >&2; exit 2; }

# The registered §9.1 pooled fit corpus, in the registered order.
PQ=(
  --parquet "$E/ext_safesyn_full.parquet"
  --parquet "$E/ext_cid22_train201.parquet"
  --parquet "$E/ext_kadid.parquet"
  --parquet "$E/ext_tid.parquet"
  --parquet "$T/tbig_944_200k.parquet"
  --parquet "$K/kadis_944_ssim2_50k.parquet"
  --parquet "$E/konjnd_bpg_train_944.parquet"
)

fit() { # <class> <out-tokens> <out-tsv> <logname>
  "$BIN" refit-winsor "${PQ[@]}" \
    --base-tokens "$W8/base_screen_tokens.txt" \
    --refit-class "$1" --out-tokens "$2" --out-tsv "$3" \
    2>&1 | tee "$LOG/$4.log" >/dev/null
}

echo "[1/4] fit class=all (the no-regression gate)"
fit all "$LOG/w9_all_screen_tokens.txt" "$LOG/w9_all_audit.tsv" fit_all
if cmp -s "$LOG/w9_all_screen_tokens.txt" "$W8/refit_screen_tokens.txt"; then
  echo "  GATE PASS: class=all reproduces benchmarks/wave8/refit_screen_tokens.txt BYTE-IDENTICALLY"
else
  echo "  GATE FAIL: class=all does NOT reproduce the committed wave-8 screen" >&2
  diff <(cat "$W8/refit_screen_tokens.txt") "$LOG/w9_all_screen_tokens.txt" | head -20 >&2
  exit 3
fi

echo "[2/4] fit class=degenerate -> the 24 append windows"
fit degenerate "$W9/w9_degen24_screen_tokens.txt" "$W9/w9_degen24_audit.tsv" fit_degen

echo "[3/4] fit class=nondegenerate -> the 30 fold windows"
fit nondegenerate "$W9/w9_fold30_screen_tokens.txt" "$W9/w9_fold30_audit.tsv" fit_fold

echo "[4/4] registered set identity  B (+) C = A,  disjoint, over the base"
python3 - "$W8/base_screen_tokens.txt" "$W8/refit_screen_tokens.txt" \
           "$W9/w9_degen24_screen_tokens.txt" "$W9/w9_fold30_screen_tokens.txt" <<'PY'
import sys
base, full, deg, fold = (open(p).read().splitlines() for p in sys.argv[1:5])
n = len(base)
assert len({n, len(full), len(deg), len(fold)}) == 1, "screens differ in line count"
D = {i for i in range(n) if deg[i]  != base[i]}
F = {i for i in range(n) if fold[i] != base[i]}
A = {i for i in range(n) if full[i] != base[i]}
assert not (D & F), f"NOT disjoint: {sorted(D & F)}"
assert D | F == A, f"union != full refit: missing {sorted(A - (D | F))} extra {sorted((D | F) - A)}"
# every changed line must equal the full refit's line; every unchanged line the base's
for name, S, arr in (("degen24", D, deg), ("fold30", F, fold)):
    for i in range(n):
        want = full[i] if i in S else base[i]
        assert arr[i] == want, f"{name} line {i}: {arr[i]!r} != {want!r}"
print(f"  IDENTITY PASS: |degen24|={len(D)}  |fold30|={len(F)}  |full|={len(A)}  disjoint, union exact")
print(f"  degen24 indices: {sorted(int(deg[i].split(':')[1]) for i in D)}")
print(f"  fold30  indices: {sorted(int(fold[i].split(':')[1]) for i in F)}")
PY
echo "wave9_screens: OK"
