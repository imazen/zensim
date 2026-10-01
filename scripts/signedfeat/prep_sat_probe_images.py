#!/usr/bin/env python3
"""Native-size reference pixels (TRAIN-role sets only, no labels) for the satsign measurements.

Writes `<set>_<ref>_<W>x<H>.rgb` for 8 distinct reference images per set of KADID-train, TID2013,
KonFiG-train into OUT (default /var/tmp/signedfeat/satprobe). PIL decodes test INPUTS only.
"""
import csv, os, sys
import numpy as np
from PIL import Image

PAIRS = "/var/tmp/restore-cuts/pairs"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/signedfeat/satprobe"
os.makedirs(OUT, exist_ok=True)
for s in ["kadid_train", "tid2013", "konfig_train"]:
    rows = list(csv.reader(open(f"{PAIRS}/{s}.tsv"), delimiter="\t"))[1:]
    refs = sorted({r[0] for r in rows})
    step = max(1, len(refs) // 8)
    for p in refs[::step][:8]:
        a = np.asarray(Image.open(p).convert("RGB"))
        h, w = a.shape[:2]
        a.tofile(f"{OUT}/{s}_{os.path.basename(p)[:-4]}_{w}x{h}.rgb")
print("wrote", OUT, len(os.listdir(OUT)))
