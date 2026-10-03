"""Exploratory (2026-10-03, user question): what if the best seeds are picked and feature sets compared on the same seeds?

  python3 seed_selection_analysis.py --root /var/tmp/rev4-featpot/v2c

Uses v2+basic and the six near-miss additions at seeds 0-9 (head N, H128). Reports: (1) a sets x seeds variance decomposition
of held-out signed SROCC per source and the cross-set seed correlation; (2) gains over v2+basic under five seed selections —
all seeds paired (registered), common top-3 seeds by training-dev score, each set's own best-of-10 by training-dev score, and
two LEAKY held-out selections for contrast; (3) the per-seed correlation of training-dev with held-out; (4) 10-seed ensembles
(mean of the seed models' predictions per fold). Selection by held-out score uses test data and is shown only as a contrast.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from e9_forward import spec_of
from v2_common import SOURCE_ORDER, V2

SETS = {"v2+basic": ["v2", "basic"], "+masked": ["v2", "basic", "masked"], "+c4": ["v2", "basic", "c4"],
        "+c8n": ["v2", "basic", "c8n"], "+iw": ["v2", "basic", "iw"], "+p1": ["v2", "basic", "p1"], "+p3": ["v2", "basic", "p3"]}
SEEDS, K = range(10), 3


def load():
    y = {s: pq.read_table(V2 / "wide" / "main" / "real" / f"{s}.keys.parquet", columns=["target"]).column(0).to_numpy()
         for s in SOURCE_ORDER}
    held, dev, pred = {}, {}, {}
    for name, parts in SETS.items():
        sp = spec_of(parts, ":H128")
        for s in SOURCE_ORDER:
            for i in SEEDS:
                r = json.loads((V2 / "cells" / f"{sp}__N" / f"without_{s}_s{i}" / "result.json").read_text())
                pred[name, s, i] = np.asarray(r["prediction"])
                held[name, s, i] = float(spearmanr(pred[name, s, i], y[s])[0])
                curve = r["dev_geomean3_by_epoch"]
                dev[name, s, i] = curve[max(curve, key=int)]
    return y, held, dev, pred


def main() -> int:
    y, H, D, P = load()
    names = list(SETS)
    print("== variance decomposition of held-out SROCC per source (sets x seeds) ==")
    for s in SOURCE_ORDER:
        m = np.array([[H[n, s, i] for i in SEEDS] for n in names])
        g = m.mean(); se = m.mean(1) - g; sd = m.mean(0) - g; res = m - g - se[:, None] - sd[None, :]
        tot = ((m - g) ** 2).sum()
        corr = np.mean([np.corrcoef(m[a], m[b])[0, 1] for a in range(len(names)) for b in range(a + 1, len(names))])
        print(f"  {s:10s} set {100 * (se ** 2).sum() * len(SEEDS) / tot:5.1f}%  seed {100 * (sd ** 2).sum() * len(names) / tot:5.1f}%  "
              f"residual {100 * (res ** 2).sum() / tot:5.1f}%  cross-set seed corr {corr:+.2f}")
    picks = {
        "A. all 10 seeds, paired (registered)": lambda n, s: list(SEEDS),
        f"B. common top-{K} by training-dev (same seeds every set)": lambda n, s: sorted(SEEDS, key=lambda i: -np.mean([D[x, s, i] for x in names]))[:K],
        "C. own best-of-10 by training-dev": lambda n, s: [max(SEEDS, key=lambda i: D[n, s, i])],
        f"D. LEAKY common top-{K} by held-out": lambda n, s: sorted(SEEDS, key=lambda i: -np.mean([H[x, s, i] for x in names]))[:K],
        "E. LEAKY own best by held-out": lambda n, s: [max(SEEDS, key=lambda i: H[n, s, i])],
    }
    for label, pick in picks.items():
        print(f"== {label} ==")
        for n in names[1:]:
            per = {s: np.mean([H[n, s, i] for i in pick(n, s)]) - np.mean([H["v2+basic", s, i] for i in pick("v2+basic", s)])
                   for s in SOURCE_ORDER}
            print(f"  {n:8s} {np.mean(list(per.values())):+.4f}  pos {sum(v > 0 for v in per.values())}/5")
    cs = [np.corrcoef([D[n, s, i] for i in SEEDS], [H[n, s, i] for i in SEEDS])[0, 1] for n in names for s in SOURCE_ORDER]
    print(f"== corr(training-dev, held-out) across seeds: mean {np.mean(cs):+.2f}, median {np.median(cs):+.2f} ==")
    print("== 10-seed ensembles ==")
    ens = {n: {s: float(spearmanr(np.mean([P[n, s, i] for i in SEEDS], axis=0), y[s])[0]) for s in SOURCE_ORDER} for n in names}
    for n in names:
        single = np.mean([H[n, s, i] for s in SOURCE_ORDER for i in SEEDS])
        e = np.mean(list(ens[n].values()))
        d = np.mean([ens[n][s] - ens["v2+basic"][s] for s in SOURCE_ORDER])
        print(f"  {n:8s} ensemble {e:.4f}  single {single:.4f}  gain {e - single:+.4f}  vs v2+basic ensemble {d:+.4f}")
    return 0


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *sys.argv[1:]]
    sys.exit(main())
