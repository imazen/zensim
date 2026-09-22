#!/usr/bin/env python3
"""Correlation tables for the dvifmish evaluation.

Statistics are the verdict lane's own (`verdict-x1x2/build_verdict.py`):
`metrics` (Spearman, Kendall, PLCC after the 5-parameter logistic) and
`ref_boot_delta` (reference-cluster bootstrap of a paired SROCC difference,
2,000 draws). This script only arranges inputs for them:

* **global** — every stimulus of the set in one bucket (primary number);
* **per-codec** — Spearman/Kendall inside each codec / distortion type,
  averaged over codecs (secondary);
* **per-source** — inside each reference picture, averaged over sources
  (secondary).

Scores are oriented to agree with the label: quality-oriented labels are
compared with `-distortion`; distortion-oriented labels (AIC-4 JND) with
`+distortion`, and that choice is written into every result.

Usage:
  eval_scores.py table --pairs P.tsv --scores DIR --models a,b,c
        [--orient quality|distortion] [--baseline a] --out result.json
  (DIR/<model>.tsv are `dvifmish batch` outputs or two-column `row\tscore`
  peer tables whose score already points the label's way.)
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "verdict-x1x2"))
import build_verdict as bv  # noqa: E402


def load_pairs(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    return rows


def load_model(path, n, orient):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    assert len(rows) == n, (path, len(rows), n)
    if "distortion" in rows[0]:
        e = np.asarray([float(r["distortion"]) for r in rows])
        sc = -e if orient == "quality" else e
        return sc, "dvifmish distortion"
    sc = np.asarray([float(r["score"]) for r in rows])
    return sc, "peer score (label-oriented)"


def grouped(keys, tgt, sc, min_n=3):
    g = defaultdict(list)
    for i, k in enumerate(keys):
        g[k].append(i)
    s_list, k_list, used = [], [], 0
    for k, idx in g.items():
        idx = np.asarray(idx)
        if len(idx) < min_n or np.ptp(tgt[idx]) == 0 or np.ptp(sc[idx]) == 0:
            continue
        s_list.append(scipy.stats.spearmanr(sc[idx], tgt[idx]).statistic)
        k_list.append(scipy.stats.kendalltau(sc[idx], tgt[idx]).statistic)
        used += 1
    return {"srocc_mean": float(np.mean(s_list)) if s_list else None,
            "krocc_mean": float(np.mean(k_list)) if k_list else None,
            "groups": used, "groups_total": len(g)}


def write_verdict_csv(path, pairs, tgt, sc):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ref_path", "dist_path", "target", "score", "E"])
        for r, t, s in zip(pairs, tgt, sc):
            w.writerow([r["source"], r["dist_path"], repr(float(t)), repr(float(s)), repr(float(-s))])


def cmd_table(a):
    pairs = load_pairs(a.pairs)
    n = len(pairs)
    tgt = np.asarray([float(r["human_score"]) for r in pairs])
    src = [r["source"] for r in pairs]
    codec = [r["codec"] for r in pairs]
    models = [m for m in a.models.split(",") if m]
    scratch = Path(a.out).with_suffix("")
    scratch.mkdir(parents=True, exist_ok=True)
    res = {"pairs": a.pairs, "n": n, "sources": len(set(src)), "codecs": len(set(codec)),
           "orient": a.orient, "models": {}}
    for m in models:
        sc, kind = load_model(Path(a.scores) / f"{m}.tsv", n, a.orient)
        d = bv.metrics(np.asarray(src), tgt, sc)
        d["kind"] = kind
        d["per_codec"] = grouped(codec, tgt, sc)
        d["per_source"] = grouped(src, tgt, sc)
        res["models"][m] = d
        write_verdict_csv(scratch / f"{m}.csv", pairs, tgt, sc)
    if a.baseline:
        res["paired_vs_" + a.baseline] = {
            m: bv.ref_boot_delta(scratch / f"{a.baseline}.csv", scratch / f"{m}.csv")
            for m in models if m != a.baseline}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    w = max(len(m) for m in models)
    print(f"{Path(a.pairs).name}: n={n} sources={res['sources']} codecs={res['codecs']} orient={a.orient}")
    for m in models:
        d = res["models"][m]
        pc, ps = d["per_codec"], d["per_source"]
        print(f"  {m:{w}}  SROCC {d['srocc']:.4f}  KROCC {d['krocc']:.4f}  PLCC {d['plcc']:.4f}"
              f"  | per-codec {pc['srocc_mean'] if pc['srocc_mean'] is not None else float('nan'):.4f}"
              f"  per-source {ps['srocc_mean'] if ps['srocc_mean'] is not None else float('nan'):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table"])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--orient", choices=["quality", "distortion"], default="quality")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", required=True)
    cmd_table(ap.parse_args())


if __name__ == "__main__":
    main()
