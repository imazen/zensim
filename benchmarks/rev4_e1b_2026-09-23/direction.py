#!/usr/bin/env python3
"""Rev4 E1b — EXPLORATORY (not preregistered): direction of cross-codec errors.

For cross-codec pairs involving the JPEG codec, among pairs the metric orders
wrong while the unit's cross best peer orders right: how often is the JPEG
stimulus the one the metric over-rates (human says JPEG is worse, metric says
better)? Plain tally on the registered pair lists; no statistic.
"""
import json
import sys
from collections import defaultdict

import crosscodec as X

JPEG = {"cid22a": "JPEG", "aic3": "JPEG-1", "aic4crop": "JPEG-1", "aic4full": "JPEG-1", "csiq": "JPEG"}
full = json.load(open("/var/tmp/rev4-e1b/e1b_full.json"))["units"]
out = {}
for u, jp in JPEG.items():
    rows, cols, same_f, cross_f, _ = X.load_unit(u)
    peer = full[u]["classes"]["cross"]["best_peer"]
    fmt = [cross_f(r["stim"]) for r in rows]
    by_ref = defaultdict(list)
    for i, r in enumerate(rows):
        by_ref[r["ref"]].append(i)
    res = {}
    for m in [c for c in cols if c in X.ZENSIM]:
        over_jpeg = under_jpeg = 0
        for ii in by_ref.values():
            for a in range(len(ii)):
                for b in range(a + 1, len(ii)):
                    i, j = ii[a], ii[b]
                    if fmt[i] == fmt[j] or jp not in (fmt[i], fmt[j]) or rows[i]["t"] == rows[j]["t"]:
                        continue
                    ti, tj = float(rows[i]["t"]), float(rows[j]["t"])
                    hw = i if ti < tj else j  # human-worse stimulus
                    def right(c):
                        si, sj = float(rows[i][c]), float(rows[j][c])
                        return si != sj and ((si < sj) == (hw == i))
                    def wrong(c):
                        si, sj = float(rows[i][c]), float(rows[j][c])
                        return si != sj and ((si < sj) != (hw == i))
                    if wrong(m) and right(peer):
                        if fmt[hw] == jp:
                            over_jpeg += 1
                        else:
                            under_jpeg += 1
        res[m] = {"wrong_peer_right": over_jpeg + under_jpeg, "jpeg_overrated": over_jpeg, "jpeg_underrated": under_jpeg}
    out[u] = {"peer": peer, "jpeg": jp, "by_model": res}
    print(u, peer, {m: (v["jpeg_overrated"], v["jpeg_underrated"]) for m, v in res.items()}, flush=True)
json.dump(out, open("/var/tmp/rev4-e1b/direction.json", "w"), indent=1)
