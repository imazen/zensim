#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/avthdr-validation-2026-07-29/chug_scope.py
# sha256(source): 07d6dab0e2f05be09e27f290ba515c0a937274caae30232ac2aba5c1a53c6938
# build_commit:  1f0f92d5075d
# Protocol doc:  benchmarks/avthdr_validation_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""CHUG sampled-leg scoping + registered sampling (NO mos_j contact).

Per PROTOCOL.md: 50 pairs per transcode rung x 6 rungs = 300 pairs.
Eligibility: transcode + its content's ref=1 row both on Tower; csv
framerate equality; both streams PQ (smpte2084), 10-bit, bt2020 matrix,
range tv-or-unset; stream fps equal; packet counts equal. Sampling walks a
default_rng(20260729) permutation per rung, probing candidates in order and
accepting eligible ones until 50 — deterministic given the data; all
reject reasons counted. Output: chug_sample.tsv (+ counts to stderr/log).
"""
import csv
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

import numpy as np

CSV = "/mnt/v/datasets/chug/chug.csv"
VID = "/mnt/tower/input/datasets/chug/videos"
OUTTSV = os.path.expanduser("~/tmp/avthdr-work/chug_sample.tsv")
RUNGS = ["360p_0.2M_", "720p_0.5M_", "720p_2M_", "1080p_0.5M_",
         "1080p_1M_", "1080p_3M_"]
PER_RUNG = 50
SEED = 20260729


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", file=sys.stderr, flush=True)


probe_cache = {}


def probe(vid_id):
    if vid_id in probe_cache:
        return probe_cache[vid_id]
    path = f"{VID}/{vid_id}.mp4"
    cmd = ["nice", "-n19", "ionice", "-c3", "ffprobe", "-v", "error",
           "-select_streams", "v:0", "-count_packets", "-show_entries",
           "stream=codec_name,width,height,pix_fmt,color_range,color_space,"
           "color_transfer,color_primaries,r_frame_rate,nb_read_packets",
           "-of", "default=nw=1", path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        probe_cache[vid_id] = None
        return None
    d = dict(l.split("=", 1) for l in r.stdout.strip().splitlines() if "=" in l)
    probe_cache[vid_id] = d
    return d


def stream_ok(d):
    if d is None:
        return "probe-failed"
    if d.get("color_transfer") != "smpte2084":
        return f"transfer={d.get('color_transfer')}"
    if "10le" not in d.get("pix_fmt", ""):
        return f"pix_fmt={d.get('pix_fmt')}"
    if d.get("color_space") != "bt2020nc":
        return f"matrix={d.get('color_space')}"
    if d.get("color_range") not in ("tv", "unknown", None, "unspecified"):
        return f"range={d.get('color_range')}"
    return None


rows = list(csv.DictReader(open(CSV)))
bycontent = defaultdict(dict)
for r in rows:
    key = "ref" if r["ref"] == "1" else r["bitladder"]
    bycontent[r["content_name"]][key] = r

# structural eligibility (csv-level)
struct = Counter()
cands = defaultdict(list)   # rung -> [row]
have = set(os.listdir(VID))
for content, group in bycontent.items():
    ref = group.get("ref")
    if ref is None:
        struct["no-ref-row"] += 1
        continue
    if f"{ref['Video']}.mp4" not in have:
        struct["ref-file-missing"] += 1
        continue
    for rung in RUNGS:
        t = group.get(rung)
        if t is None:
            struct[f"no-row-{rung}"] += 1
            continue
        if f"{t['Video']}.mp4" not in have:
            struct["transcode-file-missing"] += 1
            continue
        if t["framerate"] != ref["framerate"]:
            struct["csv-framerate-mismatch"] += 1
            continue
        cands[rung].append((content, t, ref))
log(f"structural eligibility: {dict(struct)}")
for rung in RUNGS:
    log(f"  {rung}: {len(cands[rung])} structural candidates")

rng = np.random.default_rng(SEED)
sample = []
reject = Counter()
for rung in RUNGS:
    lst = cands[rung]
    perm = rng.permutation(len(lst))
    taken = 0
    for pi in perm:
        if taken >= PER_RUNG:
            break
        content, t, ref = lst[pi]
        dr = probe(ref["Video"])
        why = stream_ok(dr)
        if why:
            reject[f"ref:{why}"] += 1
            continue
        dt = probe(t["Video"])
        why = stream_ok(dt)
        if why:
            reject[f"tr:{why}"] += 1
            continue
        if dr.get("r_frame_rate") != dt.get("r_frame_rate"):
            reject["stream-fps-mismatch"] += 1
            continue
        if dr.get("nb_read_packets") != dt.get("nb_read_packets"):
            reject["packet-count-mismatch"] += 1
            continue
        sample.append({
            "rung": rung, "content": content,
            "ref_id": ref["Video"], "tr_id": t["Video"],
            "ref_w": dr["width"], "ref_h": dr["height"],
            "tr_w": dt["width"], "tr_h": dt["height"],
            "fps": dr["r_frame_rate"], "n_frames": dr["nb_read_packets"],
        })
        taken += 1
    log(f"{rung}: accepted {taken}/{PER_RUNG}")

log(f"probe rejects: {dict(reject)}")
log(f"total sampled: {len(sample)}; distinct contents "
    f"{len(set(s['content'] for s in sample))}; distinct refs probed "
    f"{len([v for v in probe_cache.values() if v])}")
with open(OUTTSV, "w") as f:
    cols = ["rung", "content", "ref_id", "tr_id", "ref_w", "ref_h",
            "tr_w", "tr_h", "fps", "n_frames"]
    f.write("\t".join(cols) + "\n")
    for s in sample:
        f.write("\t".join(str(s[c]) for c in cols) + "\n")
log(f"wrote {OUTTSV}")
