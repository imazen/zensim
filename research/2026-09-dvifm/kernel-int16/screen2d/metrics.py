#!/usr/bin/env python3
"""Phase 2d Part D evaluation metrics over score CSVs.

Reads `ref_path,dist_path,target,score,E` CSVs emitted by
`fit_standalone.py score`, joins targets from the pairs TSV is NOT needed
(the CSV carries `target = human_score*100`), and reports:

- SROCC (Spearman), KROCC (Kendall tau-b), PLCC after the ITU-T P.1401
  5-parameter logistic map  f(x) = b1*(0.5 - 1/(1+exp(b2*(x-b3)))) + b4*x + b5
  fitted by least squares on (score -> target) of the SAME rows being
  reported (this is the eval-side monotone map; it never touches fitting).
- per-reference SROCC for ref-grouped domains (CID22): pooled + per-ref
  (refs with >=3 distorted rows).
- paired bootstrap over references: resample ref groups with replacement,
  recompute per-scorer SROCC on the resampled rows, report the delta CI.

usage:
  metrics.py table --scores a.csv [--scores b.csv ...] [--tag name]
  metrics.py delta --base ref.csv --cand c1.csv [--cand c2.csv ...]
                  --boots 2000 --seed 0
"""
import argparse
import csv
import json
import sys
from collections import defaultdict

import numpy as np
import scipy.stats
from scipy.optimize import least_squares


def read_scores(path):
    ref, tgt, sc = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            ref.append(r["ref_path"])
            tgt.append(float(r["target"]))
            sc.append(float(r["score"]))
    return np.array(ref), np.asarray(tgt), np.asarray(sc)


def logistic5(b, x):
    return b[0] * (0.5 - 1.0 / (1.0 + np.exp(np.clip(
        b[1] * (x - b[2]), -60, 60)))) + b[3] * x + b[4]


def plcc(sc, tgt):
    sc = np.asarray(sc, float)
    tgt = np.asarray(tgt, float)
    # init: b3 = median score, b4/b5 anchor the linear part on the
    # score->target regression; b1 captures the residual amplitude.
    A = np.stack([sc, np.ones_like(sc)], 1)
    b4, b5 = np.linalg.lstsq(A, tgt, rcond=None)[0]
    b0 = np.array([np.ptp(tgt) * -1.0 if np.ptp(tgt) else -1.0,
                   4.0 / (np.ptp(sc) + 1e-9),
                   float(np.median(sc)), b4, b5])
    lo = [-5 * np.ptp(tgt) - 1, -np.inf, np.min(sc) - 4 * np.ptp(sc) - 1,
          -np.inf, -np.inf]
    hi = [5 * np.ptp(tgt) + 1, np.inf, np.max(sc) + 4 * np.ptp(sc) + 1,
          np.inf, np.inf]
    sol = least_squares(lambda b: logistic5(b, sc) - tgt, b0,
                        bounds=(lo, hi), max_nfev=20000)
    mapped = logistic5(sol.x, sc)
    return float(np.corrcoef(mapped, tgt)[0, 1])


def srocc(sc, tgt):
    return float(scipy.stats.spearmanr(sc, tgt).statistic)


def krocc(sc, tgt):
    return float(scipy.stats.kendalltau(sc, tgt).statistic)


def per_ref_srocc(ref, sc, tgt, min_len=3):
    groups = defaultdict(list)
    for i, r in enumerate(ref):
        groups[r].append(i)
    out = {}
    for r, idx in sorted(groups.items()):
        if len(idx) >= min_len:
            out[r] = srocc(sc[idx], tgt[idx])
    return out


def table(paths, tags):
    res = {}
    for tag, p in zip(tags, paths):
        ref, tgt, sc = read_scores(p)
        pr = per_ref_srocc(ref, sc, tgt)
        res[tag] = {
            "n": len(sc),
            "n_refs_scored": len(pr),
            "srocc": srocc(sc, tgt),
            "krocc": krocc(sc, tgt),
            "plcc_logistic5": plcc(sc, tgt),
            "per_ref_srocc": pr,
            "per_ref_srocc_mean": float(np.mean(list(pr.values())))
            if pr else None,
            "per_ref_srocc_min": float(np.min(list(pr.values())))
            if pr else None,
        }
    return res


def paired_delta(base_csv, cand_csvs, boots=2000, seed=0):
    """Bootstrap the per-reference SROCC delta of each candidate vs base.
    Refs are resampled with replacement; a ref's whole row block moves
    together (cluster bootstrap)."""
    bref, btgt, bsc = read_scores(base_csv)
    out = {}
    for cp in cand_csvs:
        cref, ctgt, csc = read_scores(cp)
        assert np.array_equal(bref, cref), (base_csv, cp)
        assert np.allclose(btgt, ctgt), (base_csv, cp)
        groups = defaultdict(list)
        for i, r in enumerate(bref):
            groups[r].append(i)
        gkeys = np.array(sorted(groups))
        gidx = [np.asarray(groups[k]) for k in gkeys]
        rng = np.random.default_rng(seed)
        deltas = np.empty(boots)
        base_s = np.empty(boots)
        cand_s = np.empty(boots)
        for b in range(boots):
            pick = rng.integers(0, len(gkeys), len(gkeys))
            sel = np.concatenate([gidx[i] for i in pick])
            base_s[b] = srocc(bsc[sel], btgt[sel])
            cand_s[b] = srocc(csc[sel], ctgt[sel])
            deltas[b] = cand_s[b] - base_s[b]
        out[cp] = {
            "n_boots": boots,
            "base_srocc": srocc(bsc, btgt),
            "cand_srocc": srocc(csc, ctgt),
            "delta": float(srocc(csc, ctgt) - srocc(bsc, btgt)),
            "delta_ci95": [float(np.percentile(deltas, 2.5)),
                           float(np.percentile(deltas, 97.5))],
            "p_delta_le_0": float(np.mean(deltas <= 0)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--scores", action="append", required=True)
    t.add_argument("--tag", action="append", required=True)
    d = sub.add_parser("delta")
    d.add_argument("--base", required=True)
    d.add_argument("--cand", action="append", required=True)
    d.add_argument("--boots", type=int, default=2000)
    d.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.cmd == "table":
        res = table(a.scores, a.tag)
    else:
        res = paired_delta(a.base, a.cand, a.boots, a.seed)
    json.dump(res, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
