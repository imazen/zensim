#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/sihdr-transfer-2026-07-29/rmse_labeled.py
# sha256(source): efc5c2f3b78d89f0ef13950411bb28a4405b385553b70a6163fe596568f208bb
# build_commit:  34cbd9cf03673c48d69127b7c648bc2fd7d95adc
# Protocol doc:  benchmarks/sihdr_transfer_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""Trivial comparator for L1(e): display-nits RMSE on the 324 labeled pairs.

Applies the registered display model (PROTOCOL.md): ref x e x 100,
recon x 100, clamp [0,1000], drtmo center-crop; RMSE over all pixels x 3
channels. Writes rmse_labeled.csv (cid,rmse).
"""
import csv, os, subprocess, sys
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

W = os.path.expanduser("~/tmp/sihdr-work")
Z = "/mnt/tower/input/datasets/si-hdr/reconstructions.zip"
OUT = "/mnt/v/output/zensim/sihdr-transfer-2026-07-29/rmse_labeled.csv"

rows = [r for r in csv.DictReader(open(
    "/mnt/v/datasets/si-hdr/experiment_results/experiment_results.csv"))
    if r["scene"] != "all" and r["method"] not in ("input", "original")]
assert len(rows) == 324

def load(p):
    x = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if x is None:
        return None
    x = x[:, :, :3].astype(np.float64)  # BGR
    x[~np.isfinite(x)] = 0.0
    return x

os.makedirs(f"{W}/rmse_batch", exist_ok=True)
res = []
# group by (scene, clip) so ref percentile is computed once
from collections import defaultdict
by_sc = defaultdict(list)
for r in rows:
    by_sc[(r["image"], r["clip_level"])].append(r["method"])

for n, ((img, clip), methods) in enumerate(sorted(by_sc.items())):
    sid = img[1:]
    ref = load(f"{W}/ref/{sid}.exr")
    assert ref is not None, sid
    e = 1.0 / np.percentile(ref.max(axis=2), float(clip))
    refd = np.clip(ref * e * 100.0, 0, 1000.0)
    for m in methods:
        mem = f"sihdr/reconstructions/{m}/clip_{clip}/{sid}.exr"
        p = f"{W}/rmse_batch/cur.exr"
        with open(p, "wb") as f:
            subprocess.run(["unzip", "-p", Z, mem], check=True, stdout=f)
        d = load(p)
        if d is None:
            print(f"LOAD FAIL {img}-{clip}-{m}")
            continue
        if d.shape != refd.shape:
            oy = (d.shape[0] - refd.shape[0]) // 2
            ox = (d.shape[1] - refd.shape[1]) // 2
            d = d[oy:oy + refd.shape[0], ox:ox + refd.shape[1]]
        dd = np.clip(d * 100.0, 0, 1000.0)
        rmse = float(np.sqrt(np.mean((dd - refd) ** 2)))
        res.append({"cid": f"{img}-{clip}-{m}", "rmse": rmse})
    if (n + 1) % 9 == 0:
        print(f"[{n+1}/54] blocks done", flush=True)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["cid", "rmse"])
    w.writeheader()
    w.writerows(res)
print(f"wrote {OUT}: {len(res)} rows")
