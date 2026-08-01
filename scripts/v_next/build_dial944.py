#!/usr/bin/env python3
"""dial_grid_944col builder (PLAN_SOTA944 P1): foldapp2-extract the PERSISTED
dial-grid pairs (the 2026-07-27 re-encode's decoded pixels, persisted from
session scratch to /mnt/v/output/zensim/dial-grid-pixels-2026-07-27/ — NO
re-encode here: current codec revs would produce different pixels and break
the bitwise gate) and join onto the ORIGINAL 4,817-row identity (image_id,
codec, q, codec_param, param_kind) from dial_grid_924col_2026-07-28. Extends
build_dial924.py (mode foldapp -> foldapp2; identity source 720col -> 924col;
scratch input dir -> the persisted pixels dir with path re-prefixing).

Gate after with scripts/canonical_corpus/gate_backfill944.py (f0..f923
bitwise vs the 924 grid).

Env: ZM944_BIN (extractor binary), DIAL944_WORK (scratch, default
~/tmp/backfill944/dial944), DIAL944_PIXELS (persisted inputs dir).
"""
import csv
import json
import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

PIX = os.environ.get(
    "DIAL944_PIXELS", "/mnt/v/output/zensim/dial-grid-pixels-2026-07-27"
)
OLD_PREFIX = "/home/lilith/tmp/dial924"  # as-run path prefix inside pairs_*.tsv
WORK = os.path.expanduser(os.environ.get("DIAL944_WORK", "~/tmp/backfill944/dial944"))
EXTRACT = os.environ.get("ZM944_BIN") or sys.exit("ABORT: ZM944_BIN env required")
OLD = "/mnt/v/output/zensim/v2-eval-924-2026-07-27/dial_grid_924col_2026-07-28.parquet"
OUT = os.environ.get(
    "DIAL944_OUT",
    "/mnt/v/output/zensim/v2-eval-944-2026-08-01/dial_grid_944col_2026-08-01.parquet",
)
FAM = {"zenjpeg": "jpeg", "zenwebp": "webp", "zenavif": "avif", "zenjxl": "jxl"}

os.makedirs(WORK, exist_ok=True)


def repfx(p: str) -> str:
    return PIX + p[len(OLD_PREFIX):] if p.startswith(OLD_PREFIX) else p


feat_by_key = {}
for tag in ("jpeg", "webp", "avif", "jxl"):
    pairs = f"{PIX}/pairs_{tag}.tsv"
    rows = list(csv.DictReader(open(pairs), delimiter="\t"))
    drv = f"{WORK}/drv_{tag}.tsv"
    keys = []
    with open(drv, "w") as f:
        f.write("ref_path\tdist_path\thuman_score\n")
        for i, r in enumerate(rows):
            img = os.path.basename(r["image_path"]).rsplit(".", 1)[0]
            fam = FAM[r["codec"]]
            if fam == "jxl":
                d = json.loads(r["knob_tuple_json"]).get("distance")
                key = (img, fam, round(float(d), 3))
            else:
                key = (img, fam, round(float(r["q"]), 3))
            keys.append(key)
            f.write(f"{repfx(r['ref_path'])}\t{repfx(r['dist_path'])}\t{i}\n")
    csvp = f"{WORK}/feat944_{tag}.csv"
    env = dict(os.environ, ZENSIM_AB_MODE="foldapp2")
    r = subprocess.run([EXTRACT, drv, csvp], env=env)
    if r.returncode != 0:
        sys.exit(f"ABORT: extract {tag} rc={r.returncode}")
    with open(csvp, newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        fi = [hdr.index(f"f{i}") for i in range(944)]
        hs = hdr.index("human_score")
        n = 0
        for row in rd:
            k = keys[int(float(row[hs]))]
            feat_by_key[k] = [float(row[j]) for j in fi]
            n += 1
    print(f"{tag}: {len(rows)} pairs -> {n} feature rows", flush=True)

old = pq.read_table(OLD, columns=["image_id", "codec", "q", "codec_param", "param_kind"])
ident = list(
    zip(*[old.column(c).to_pylist()
          for c in ("image_id", "codec", "q", "codec_param", "param_kind")])
)
missing, out_feats = [], []
for img, codec, q, cp, pk in ident:
    key = (img, codec, round(float(cp), 3)) if pk == "distance" else (img, codec, round(float(q), 3))
    f = feat_by_key.get(key)
    if f is None:
        missing.append(key)
        out_feats.append(None)
    else:
        out_feats.append(f)
print(f"identity rows {len(ident)}; missing {len(missing)}; "
      f"extracted-not-in-identity {len(feat_by_key) - (len(ident) - len(missing))}")
if missing:
    print("first missing:", missing[:6])
    sys.exit(9)
cols = {c: old.column(c) for c in ("image_id", "codec", "q", "codec_param", "param_kind")}
for i in range(944):
    cols[f"f{i}"] = pa.array([r_[i] for r_ in out_feats], pa.float64())
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pq.write_table(pa.table(cols), OUT, compression="zstd")
print(f"G-DIAL: {len(ident)}/{len(ident)} matched -> {OUT}")
