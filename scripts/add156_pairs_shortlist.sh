#!/usr/bin/env bash
#
# add156_pairs_shortlist.sh — appendix U stage 2.
#
# The grid's full battery gives every cell's scalar axes. Two things it CANNOT
# give, because they need the per-pair predictions:
#
#   * the paired bootstrap (same resampled index sets on both models — marginal
#     per-cell CIs are the wrong test), and
#   * the WIDE-TOP-SLICE high-fidelity axis. Appendix U measured CID22's B9 to
#     be a degenerate sliver (43 pairs, 11 of 49 refs, MOS span 0.0194, marginal
#     bootstrap sd 0.178), on which every sampled board model is INVERTED. The
#     >=0.80 slice (n=1425, span 0.119) is the same high-fidelity question asked
#     where the statistic actually resolves.
#
# So: dump per-pair predictions for a SHORTLIST only, then reduce them through
# the canonical owners (`panel --batch` via wave6_paired_bootstrap.py).
#
#   add156_pairs_shortlist.sh dump  <tag> <cells-file>   # per-pair, cid22
#   add156_pairs_shortlist.sh slice <tag> <cells-file>   # top-slice sweep table
#   add156_pairs_shortlist.sh boot  <tag> <cells-file>   # paired bootstrap
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TGT=${CARGO_TARGET_DIR:-$REPO_ROOT/target}
BV=${ZL_BV:-$TGT/release/bake_verdict}
PANEL=${ZEN_PANEL_BIN:-$TGT/release/panel}
OUT=/mnt/v/output/zensim/bakes/add156pairs
DUMP=$HOME/tmp/add156pairs/shortlist
JOBS=${ZL_JOBS:-4}
mkdir -p "$DUMP"

do_dump() {                      # $1 = "<arm>|<cell_id>" ; $2 = tag
    local arm=${1%%|*} cid=${1##*|} tag=$2
    local st="U_${arm}${tag}_${cid}"
    local o="$DUMP/${st}_cid22.tsv"
    [[ -s $o ]] && return 0
    nice -n19 ionice -c3 "$BV" --bake "$OUT/bakes/${st}.bin" --regime 944 \
        --corpora cid22 --perpair-metrics /nonexistent/x.parquet \
        --per-pair-output "$o" >/dev/null 2>&1 || { echo "DUMP FAIL $st" >&2; return 1; }
}
export -f do_dump
export DUMP BV OUT

cmd_dump() {
    local tag=$1 cells=$2
    { echo "A|BASE"; echo "B|BASE"; awk -F'\t' 'NR>1{print $2"|"$1}' "$cells"; } | sort -u \
      | xargs -P "$JOBS" -n 1 bash -c 'do_dump "$0" '"$tag"''
    echo "[dump] $(ls "$DUMP"/*_cid22.tsv 2>/dev/null | wc -l) dumps"
}

cmd_slice() {
    local tag=$1 cells=$2
    ZEN_PANEL_BIN=$PANEL python3 - "$DUMP" "$tag" "$cells" <<'PY'
import csv, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("scripts/")), "scripts"))
sys.path.insert(0, "scripts")
from lib.zen_stats import panel_batch
dump, tag, cells = sys.argv[1], sys.argv[2], sys.argv[3]
CUT = [0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.75, 0.70]
want = [("A", "BASE"), ("B", "BASE")]
with open(cells) as f:
    want += [(r["arm"], r["cell_id"]) for r in csv.DictReader(f, delimiter="\t")]
rows, seen = [], set()
for arm, cid in want:
    if (arm, cid) in seen:
        continue
    seen.add((arm, cid))
    p = f"{dump}/U_{arm}{tag}_{cid}_cid22.tsv"
    if not os.path.exists(p):
        continue
    a = np.loadtxt(p, delimiter="\t", skiprows=1, usecols=(0, 1))
    h, pr = a[:, 0], a[:, 1]
    res = {r["label"]: float(r["srocc_signed"]) for r in panel_batch(
        [(f"c{i}", pr[h >= c].tolist(), h[h >= c].tolist()) for i, c in enumerate(CUT)],
        stats="srocc")}
    rows.append([arm, cid] + [f"{res[f'c{i}']:+.6f}" for i in range(len(CUT))])
w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
w.writerow(["arm", "cell_id"] + [f"ge{c:.2f}" for c in CUT])
for r in rows:
    w.writerow(r)
PY
}

case "${1:?usage: $0 dump|slice <tag> <cells-file>}" in
    dump)  cmd_dump  "${2:?tag}" "${3:?cells}" ;;
    slice) cmd_slice "${2:?tag}" "${3:?cells}" > "$OUT/shortlist_slice_${2}.tsv"
           echo "wrote $OUT/shortlist_slice_${2}.tsv" ;;
    *) echo "usage: $0 dump|slice <tag> <cells-file>" >&2; exit 2 ;;
esac
