#!/usr/bin/env python3
"""Rank agreement of dvifmish presets with the organisers' DVIFM columns on
AIC2026 (public tables; no human labels, so this is metric-vs-metric
agreement, never accuracy).

The AIC2026 metric tables carry the submitted DVIFM (`proposal-DVIFM`), a
later version (`proposal-DVIFM-0.2`) and its colour variant
(`proposal-DVIFM-0.2-use_chroma`). All three rise with distortion, as does
dvifmish's pooled distortion E, so agreement is Spearman/Kendall of E
against each column, no fitting. Global (all 9,618 stimuli), then averaged
within codecs and within sources.

Usage:
  aic2026_agreement.py pairs <out_dir>                    # write the pair lists
  aic2026_agreement.py agree <crop|full> <scores_dir> <models,...> <out.json>
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats

AIC = Path("/mnt/v/datasets/aic2026")
X = Path("/var/tmp/dvifmish/datasets/aic2026")
COLS = ["proposal-DVIFM", "proposal-DVIFM-0.2", "proposal-DVIFM-0.2-use_chroma"]


def table(kind):
    return list(csv.DictReader(open(AIC / ("metrics_cropped.csv" if kind == "crop" else "metrics_fullres.csv"))))


def paths(kind, r):
    if kind == "crop":
        base = X / "cropped" / "PTC-crops-all"
        return base / r["source"], base / r["distorted"]
    return X / "sources" / "sources" / r["source"], X / "complete" / "distorted" / r["distorted"]


def write_pairs(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("crop", "full"):
        rows = table(kind)
        with open(out / f"aic2026_{kind}.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["ref_path", "dist_path", "source", "codec"] + COLS)
            miss = 0
            for r in rows:
                rp, dp = paths(kind, r)
                miss += (not rp.exists()) + (not dp.exists())
                w.writerow([rp, dp, r["source"].replace("PTC_", "").replace("_Ref_00.png", ""),
                            r["codec_acronym"]] + [r[c] for c in COLS])
        print(f"aic2026_{kind}.tsv: {len(rows)} rows, missing files {miss}")


def grouped(keys, a, b):
    g = defaultdict(list)
    for i, k in enumerate(keys):
        g[k].append(i)
    s, k_ = [], []
    for idx in g.values():
        idx = np.asarray(idx)
        if len(idx) >= 3 and np.ptp(a[idx]) > 0 and np.ptp(b[idx]) > 0:
            s.append(scipy.stats.spearmanr(a[idx], b[idx]).statistic)
            k_.append(scipy.stats.kendalltau(a[idx], b[idx]).statistic)
    return float(np.mean(s)), float(np.mean(k_)), len(s)


def agree(kind, scores, models, out):
    pairs = list(csv.DictReader(open(Path(scores).parent / f"aic2026_{kind}.tsv"), delimiter="\t"))
    n = len(pairs)
    src = [p["source"] for p in pairs]
    codec = [p["codec"] for p in pairs]
    res = {"set": kind, "n": n, "models": {}}
    # the organisers' columns against each other, for scale
    ref = {c: np.asarray([float(p[c]) for p in pairs]) for c in COLS}
    res["columns_vs_each_other"] = {
        f"{a}~{b}": float(scipy.stats.spearmanr(ref[a], ref[b]).statistic)
        for i, a in enumerate(COLS) for b in COLS[i + 1:]}
    for m in models:
        rows = list(csv.DictReader(open(Path(scores) / f"{m}.tsv"), delimiter="\t"))
        assert len(rows) == n, (m, len(rows), n)
        e = np.asarray([float(r["distortion"]) for r in rows])
        res["models"][m] = {}
        for c in COLS:
            s = float(scipy.stats.spearmanr(e, ref[c]).statistic)
            k = float(scipy.stats.kendalltau(e, ref[c]).statistic)
            pcs, pck, ncodec = grouped(codec, e, ref[c])
            pss, psk, nsrc = grouped(src, e, ref[c])
            res["models"][m][c] = {"srocc": s, "krocc": k, "per_codec_srocc": pcs,
                                   "per_codec_krocc": pck, "codecs": ncodec,
                                   "per_source_srocc": pss, "per_source_krocc": psk,
                                   "sources": nsrc}
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"AIC2026 {kind}: n={n}")
    for pair, v in res["columns_vs_each_other"].items():
        print(f"  organisers' {pair}: SROCC {v:.4f}")
    for m in models:
        line = "  ".join(f"{c.replace('proposal-', '')}: {res['models'][m][c]['srocc']:.4f}/"
                         f"{res['models'][m][c]['krocc']:.4f}" for c in COLS)
        print(f"  {m:28} {line}")


def main():
    if sys.argv[1] == "pairs":
        write_pairs(sys.argv[2])
    else:
        agree(sys.argv[2], sys.argv[3], sys.argv[4].split(","), sys.argv[5])


if __name__ == "__main__":
    main()
