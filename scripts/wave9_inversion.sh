#!/usr/bin/env bash
#
# wave9_inversion.sh <inverted-bake> <reference-bake> — characterize the KADID
# sign inversion (amendment 10 §10.7, a REQUIRED deliverable, not optional).
#
# Runs the committed owner `bake_contrib` on both bakes over the SAME corpora
# with the SAME invocation, so per-input contributions are directly comparable,
# and adds KADID (which wave 8's contrib runs omitted — it scored csiq/live/
# cid22 only, which is why the inversion was reported but never explained).
#
# The question it answers: on KADID, which inputs carry the ranking, and did
# they flip sign or only change magnitude between the two models? A negative
# ΔSROCC on kadid means ABLATING that input RAISES |SROCC| — i.e. the input is
# pushing the KADID ranking backwards. §10.7 asks specifically whether those
# inputs are the revived append features (ties the inversion to the 24-window
# unsticking) or the fold block (ties it to the un-clipping), which the wave-9
# W9-A/W9-B/W9-C split is what makes attributable.
set -euo pipefail
INV=${1:?usage: wave9_inversion.sh <inverted-bake.bin> <reference-bake.bin>}
REF=${2:?usage: wave9_inversion.sh <inverted-bake.bin> <reference-bake.bin>}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${ZL_CONTRIB:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/bake_contrib}
E=${SOTA944_E:-/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01}
OUTD=${WAVE9_CONTRIB_OUT:-$REPO_ROOT/benchmarks/wave9}
ROWS=${WAVE9_CONTRIB_ROWS:-4000}
mkdir -p "$OUTD"
[[ -x "$BIN" ]] || { echo "missing $BIN" >&2; exit 2; }

CORPORA=(
  --corpus "kadid:$E/ext_kadid.parquet:human_score:100"
  --corpus "csiq:$E/ext_csiq.parquet:human_score:100"
  --corpus "live:$E/ext_live.parquet:human_score:100"
  --corpus "cid22:$E/ext_cid22val.parquet:human_score:100"
)

for b in "$INV" "$REF"; do
  stem=$(basename "$b" .bin)
  echo "== bake_contrib $stem"
  "$BIN" --bake "$b" "${CORPORA[@]}" --rows "$ROWS" --top-movers 320 \
      --out "$OUTD/contrib_${stem}_kadid.tsv" \
      --summary "$OUTD/contrib_${stem}_kadid.md" > "$OUTD/contrib_${stem}_kadid.stdout" 2>&1 \
    || { echo "bake_contrib failed for $stem (see $OUTD/contrib_${stem}_kadid.stdout)" >&2; exit 3; }
done

python3 - "$OUTD/contrib_$(basename "$INV" .bin)_kadid.tsv" \
          "$OUTD/contrib_$(basename "$REF" .bin)_kadid.tsv" \
          "$(basename "$INV" .bin)" "$(basename "$REF" .bin)" <<'PY'
import sys, csv
inv_p, ref_p, inv_n, ref_n = sys.argv[1:5]

def read(p):
    with open(p) as f:
        rows = [r for r in csv.DictReader((l for l in f if not l.startswith('#')), delimiter='\t')]
    return {int(r['idx']): r for r in rows}

I, R = read(inv_p), read(ref_p)
key, mk = 'dsrocc_kadid', 'mean_abs'   # bake_contrib's own column names

def f(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return None

# dsrocc is populated for the top movers only ('-' elsewhere), so a comparison
# is only meaningful where BOTH bakes reported one.
common = sorted(set(I) & set(R))
recs = []
for i in common:
    a, b = f(I[i], key), f(R[i], key)
    if a is None or b is None:
        continue
    recs.append((i, I[i]['family'], a, b, a - b,
                 f(I[i], mk) or 0.0, f(R[i], mk) or 0.0))
print(f"\ninputs with a ΔSROCC in BOTH bakes (top-mover overlap): {len(recs)} "
      f"of {len(common)}")

print(f"\n### KADID ΔSROCC per input — {inv_n} (inverted) vs {ref_n}\n")
print("A NEGATIVE ΔSROCC means ablating the input RAISES |SROCC| on KADID: the "
      "input is pushing that ranking backwards.\n")
print("| idx | family | ΔSROCC kadid (inverted) | ΔSROCC kadid (ref) | diff | mean|Δ| inv | mean|Δ| ref |")
print("|---|---|---:|---:|---:|---:|---:|")
for i, fm, a, b, d, ma, mb in sorted(recs, key=lambda r: r[2])[:20]:
    print(f"| f{i} | {fm} | {a:+.4f} | {b:+.4f} | {d:+.4f} | {ma:.4f} | {mb:.4f} |")

print("\n#### family roll-up of KADID ΔSROCC\n")
print("| family | n | Σ ΔSROCC inverted | Σ ΔSROCC ref | n sign-flipped |")
print("|---|---:|---:|---:|---:|")
for fm in sorted({r[1] for r in recs}):
    g = [r for r in recs if r[1] == fm]
    if not g: continue
    flip = sum(1 for r in g if (r[2] > 0) != (r[3] > 0) and abs(r[2] - r[3]) > 1e-6)
    print(f"| {fm} | {len(g)} | {sum(r[2] for r in g):+.4f} | {sum(r[3] for r in g):+.4f} | {flip} |")
PY
