#!/usr/bin/env python3
"""Eval a corr-lq bake on HELD-OUT gates + the ssim2/CID22 hold-check.
  python3 corr_lq_eval.py <label> <bake.bin>
"""
import sys, subprocess, struct, re, os
import pyarrow.parquet as pq, numpy as np
from collections import defaultdict

label, bake = sys.argv[1], sys.argv[2]
BIN = "./target/release/predict_features_with_bake"
CORRLQ = "/mnt/v/output/zensim/corr-lq"

def score_parquet(path, post="raw"):
    t = pq.read_table(path)
    n = t.num_rows
    feats = np.stack([np.asarray(t[f"f{i}"], dtype=np.float32) for i in range(372)], axis=1)
    # features-file: u32 n_features, u32 n_rows, then row-major f32
    buf = struct.pack("<II", 372, n) + feats.tobytes()
    fp = f"/home/lilith/tmp/_ff_{label}.bin"
    open(fp, "wb").write(buf)
    out = subprocess.run([BIN, "--bake", bake, "--bake-post", post, "--features-file", fp],
                         capture_output=True, text=True)
    scores = np.array([float(x) for x in out.stdout.split()])
    os.unlink(fp)
    return t, scores

# 1) HELD-OUT structural corruption gate (14 unseen types): corruption ranks below q20?
t, sc = score_parquet(f"{CORRLQ}/corruption_gate.parquet", "raw")
refs = [str(x) for x in t["ref_basename"].to_pylist()]
hs = np.asarray(t["human_score"], dtype=float)
grp = defaultdict(dict)
for k, (r, h) in enumerate(zip(refs, hs)):
    lvl = "corruption" if h < 0.05 else ("q20" if h > 0.5 else "q10")
    grp[r][lvl] = k
ok = tot = 0
for r, d in grp.items():
    if "corruption" in d and "q20" in d:
        tot += 1; ok += sc[d["corruption"]] < sc[d["q20"]]
corr_gate = ok / max(tot, 1) * 100

# 2) HELD-OUT kadis_negrich (severe/LQ): does the bake rank it (SROCC vs human_score)?
t2, sc2 = score_parquet(f"{CORRLQ}/kadis_negrich_gate.parquet", "raw")
h2 = np.asarray(t2["human_score"], dtype=float)
def srocc(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]
kn_srocc = srocc(sc2, h2)

# 3) ssim2/CID22 hold-check via bake_verdict
def bv(corp):
    r = subprocess.run(["./target/release/bake_verdict", "--bake", bake, "--corpora", corp,
                        "--output", f"/home/lilith/tmp/_bv_{label}.md"], capture_output=True, text=True)
    md = open(f"/home/lilith/tmp/_bv_{label}.md").read()
    g = {}
    for line in md.splitlines():
        c = line.split("|")
        if len(c) >= 5 and re.search(r"[0-9]\.[0-9]", c[3]) and c[2].strip().isdigit():
            g[c[1].strip()] = c[3].strip()
    return g
g = bv("cid22,imazen26,nonphoto,konjnd,hf_nearlossless")
def gv(pat):
    for k, v in g.items():
        if re.search(pat, k): return v
    return "NA"

print(f"{label:12s} | HELDOUT-corr-gate={corr_gate:5.1f}%  kadis_negrich-SROCC={kn_srocc:+.3f} "
      f"| CID22={gv('CID22')} imazen26={gv('real-codec')} nonphoto={gv('non-photo')} "
      f"KonJND={gv('KonJND')} HF={gv('HF near')}")
