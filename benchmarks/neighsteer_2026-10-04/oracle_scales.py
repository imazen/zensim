#!/usr/bin/env python3
"""Per-scale oracle substitution over every per-feature diagnostic in /var/tmp/neighsteer/diag: replace the map's
predicted per-feature block gains with the observed ones at chosen scales and report the block-rank agreement."""
import glob, json, collections
import numpy as np
from scipy.stats import spearmanr
def scale(i):
    if i < 156: return i // 39
    if 372 <= i < 720: return (i - 372) // 87
    return -1
rows = collections.defaultdict(list)
for f in sorted(glob.glob('/var/tmp/neighsteer/diag/*-b8.json')):
    n = f.split('/')[-1][:-5]; arm, seed, case, _ = n.split('-'); lv = case.split('_')[-1]
    d = json.load(open(f))
    ids = np.array(d['ids']); O = np.array(d['observed']); P = np.array(d['predicted']); ds = np.array(d['score_delta'])
    sc = np.array([scale(i) for i in ids])
    m3 = lambda Q: spearmanr(Q.sum(0), ds)[0]
    out = {'base': m3(P), 'M2': spearmanr(O.sum(0), ds)[0]}
    for name, mask in [('s3', sc == 3), ('s2+3', sc >= 2), ('s1+2+3', sc >= 1), ('s0', sc == 0)]:
        Q = P.copy(); Q[mask] = O[mask]; out[name] = m3(Q)
    rows[(arm, lv)].append(out)
for (arm, lv), v in sorted(rows.items()):
    print(f"{arm:8s} JPEG {lv} n={len(v):2d}  " + "  ".join(f"{k} {np.median([x[k] for x in v]):.3f}/{min(x[k] for x in v):.2f}" for k in ('base', 's3', 's2+3', 's1+2+3', 's0', 'M2')))
