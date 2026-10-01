#!/usr/bin/env python3
"""Raw-RGB8 parity inputs for the signedfeat tier-parity gates (TRAIN-role pixels only, no labels).

Reads label-free pair TSVs (human_score column is never opened) of KADID-train, TID2013 and
KonFiG-train; for 2 distinct non-identical pairs per set writes, per size in 64/256/1024/2112,
`<set>_<row>_<size>_{ref,dst}.rgb` (+ `index.tsv`). 64 and 256 are centre crops of the native
image; 1024 and 2112 are mirror-tiled mosaics of the same pair's pixels (same transform on ref
and dst, so the distortion survives). PIL is a decode/oracle tool for test INPUTS only.
"""
import csv, hashlib, os, sys
import numpy as np
from PIL import Image

PAIRS = "/var/tmp/restore-cuts/pairs"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/signedfeat/parity"
SIZES = [64, 256, 1024, 2112]
PICK = {"kadid_train": [137, 2900], "tid2013": [211, 1700], "konfig_train": [40, 250]}


def load(p):
    return np.asarray(Image.open(p).convert("RGB"))


def crop(a, n):
    h, w = a.shape[:2]
    y, x = (h - n) // 2, (w - n) // 2
    return a[y:y + n, x:x + n]


def mosaic(a, n):
    h, w = a.shape[:2]
    ty, tx = -(-n // h), -(-n // w)
    rows = []
    for j in range(ty):
        row = []
        for i in range(tx):
            t = a
            if i % 2: t = t[:, ::-1]
            if j % 2: t = t[::-1]
            row.append(t)
        rows.append(np.concatenate(row, 1))
    return np.concatenate(rows, 0)[:n, :n]


os.makedirs(OUT, exist_ok=True)
idx = open(os.path.join(OUT, "index.tsv"), "w")
idx.write("name\tset\trow\tsize\tref_path\tdist_path\tsha256_ref\tsha256_dst\n")
for s, rows_i in PICK.items():
    rows = list(csv.reader(open(f"{PAIRS}/{s}.tsv"), delimiter="\t"))[1:]
    for ri in rows_i:
        # advance to the first row whose pair differs at EVERY size (a mild distortion can leave a
        # small centre crop bit-identical)
        while True:
            rp, dp = rows[ri][0], rows[ri][1]
            r, d = load(rp), load(dp)
            if r.shape == d.shape and all(
                not np.array_equal((crop if n <= 256 else mosaic)(r, n), (crop if n <= 256 else mosaic)(d, n))
                for n in SIZES
            ):
                break
            ri += 1
        for n in SIZES:
            f = crop if n <= 256 else mosaic
            a, b = np.ascontiguousarray(f(r, n)), np.ascontiguousarray(f(d, n))
            assert a.shape == (n, n, 3) and not np.array_equal(a, b), (s, ri, n)
            name = f"{s}_{ri}_{n}"
            a.tofile(f"{OUT}/{name}_ref.rgb"); b.tofile(f"{OUT}/{name}_dst.rgb")
            idx.write(f"{name}\t{s}\t{ri}\t{n}\t{rp}\t{dp}\t{hashlib.sha256(a.tobytes()).hexdigest()}\t{hashlib.sha256(b.tobytes()).hexdigest()}\n")
idx.close()
print("wrote", OUT)
