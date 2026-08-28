#!/usr/bin/env bash
# PU21 single-model experiment (registered: balance_campaign NEXT-GEN PROGRAM).
# 800 stratified CID22 pairs -> PQ16 via srgb_to_pq_png (the convention owner)
# -> HDR-route 944 features -> HDR CoR forward -> panel vs MCOS, compared with
# the SDR CoR on the SAME pairs through the SDR route.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
D=$HOME/tmp/pu21exp
HB=$D/heartbeat
mkdir -p "$D/png" "$D/pq"
trap 'echo "$(date -u +%FT%TZ) EXIT rc=$?" >> "$HB"; touch "$HB.done"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HB"; }
PAIRS=$HOME/tmp/twozone/pu21_pairs.tsv
# 1) normalize all images to sRGB PNG (dists are jpg/webp/...)
say "normalize to png"
python3 - "$PAIRS" "$D" <<'PY'
import csv, os, subprocess, sys
pairs, D = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(pairs), delimiter="\t"))
todo = {}
for r in rows:
    for p in (r["ref_path"], r["dist_path"]):
        key = p.replace("/", "__").rsplit(".", 1)[0] + ".png"
        todo[p] = os.path.join(D, "png", key[-160:])
with open(os.path.join(D, "srclist.tsv"), "w") as f:
    for src, dst in todo.items():
        if not os.path.exists(dst):
            subprocess.run(["magick", src, "-strip", dst], check=True)
        f.write(dst + "\n")
json_map = {src: dst for src, dst in todo.items()}
import json
json.dump(json_map, open(os.path.join(D, "srcmap.json"), "w"))
PY
# 2) srgb -> PQ16 (convention owner)
say "srgb->pq"
nice -n19 /home/lilith/.venvs/pytools/bin/python "$REPO/scripts/hdr/srgb_to_pq_png.py" "$D/srclist.tsv" "$D/pq" --jobs 8 >> "$HB" 2>&1
# 3) build pq + sdr pair TSVs
python3 - "$PAIRS" "$D" <<'PY'
import csv, json, os, sys
pairs, D = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(D, "srcmap.json")))
def pq_of(p):
    b = os.path.basename(m[p])
    # srgb_to_pq keeps a name derived from last 3 components of the PNG path
    cands = [os.path.join(D, "pq", f) for f in os.listdir(os.path.join(D, "pq")) if b[:-4] in f]
    return cands[0] if cands else None
rows = list(csv.DictReader(open(pairs), delimiter="\t"))
with open(os.path.join(D, "pq_pairs.tsv"), "w") as fq, open(os.path.join(D, "sdr_pairs.tsv"), "w") as fs, open(os.path.join(D, "mos.tsv"), "w") as fm:
    n = 0
    for r in rows:
        rp, dp = pq_of(r["ref_path"]), pq_of(r["dist_path"])
        if not rp or not dp: continue
        fq.write(f"{rp}\t{dp}\n")
        fs.write(f"{m[r['ref_path']]}\t{m[r['dist_path']]}\n")
        fm.write(r["human_score"] + "\n")
        n += 1
print("pairs ready:", n)
PY
# 4) features both routes
say "extract HDR-route (pq16)"
nice -n19 ionice -c3 "$REPO/zensim-bench/target/release/examples/sdr944_extract" --hdr-pq16 --pairs-tsv "$D/pq_pairs.tsv" --out "$D/pq_feats.tsv" >> "$HB" 2>&1
say "extract SDR-route"
nice -n19 ionice -c3 "$REPO/zensim-bench/target/release/examples/sdr944_extract" --pairs-tsv "$D/sdr_pairs.tsv" --out "$D/sdr_feats.tsv" >> "$HB" 2>&1
say "PU21 pipeline features done"
