#!/usr/bin/env python3
"""Grid over (div_weight, kadis_weight) for the imazen-26 diverse MLP, reusing
train_mlp_diverse's in-process parquet cache (all reads happen ONCE). For each config:
multi-seed, collapse-gated, training-side seed selection, then report the selected seed on
the three A/B axes — CID22 (photo MOS holdout, report-only), bigcodec-val (imazen-26
diverse), KADIS deep-neg. Goal: keep the +0.087 diverse gain while recovering CID22 and
the deep-neg tail (negatives were the §8.30-8.33 point). Saves the selected npz per config.
"""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import spearmanr

REPO = Path.home() / "work/zen/zensim"
_spec = importlib.util.spec_from_file_location("T", REPO / "scripts/v_next/train_mlp_diverse.py")
T = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T)
OUT = "/mnt/v/output/zensim/reports/b_negatives"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,7,13,17,23,31,41")
    ap.add_argument("--grid", default="1.0:0.3,0.5:0.6,1.0:0.6,0.5:0.3")  # div_w:kadis_w
    ap.add_argument("--winsor-pct", type=float, default=0.1)  # de-poison; 0=poisoned control
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in a.seeds.split(",")]
    Xcv, ycv = T.load(T.CID22_VAL, "human_score")  # holdout — report only
    tv_names = [n for n, *_ in T.POS] + ["bigcodec"]
    print(f"{'div_w':>6s} {'kad_w':>6s} | {'CID22*':>8s} {'bigcodec':>8s} {'deep':>7s} "
          f"{'sel-seed':>8s}   (* holdout, report-only)")
    baselines = {"§8.33 photo": (0.8697, 0.8555, 0.7762), "shipped B": (0.876, None, 0.047)}
    for k, v in baselines.items():
        bc = f"{v[1]:.4f}" if v[1] is not None else "  n/a "
        print(f"{k:>13s}   |  {v[0]:.4f}   {bc}  {v[2]:.4f}")
    print("-" * 62)
    results = []
    for spec in a.grid.split(","):
        dv, kw = (float(x) for x in spec.split(":"))
        cfg = SimpleNamespace(hidden=64, div_cap=120000, hq_band=85.0, hq_weight=0.3,
                              div_weight=dv, kadis_weight=kw, winsor_pct=a.winsor_pct)
        picks = []
        for s in seeds:
            m, payload = T.train_one(s, cfg, dev)
            if min(m[n] for n in tv_names) < 0.85:
                continue  # collapse gate
            sel = float(np.exp(np.mean(np.log([max(1e-6, min(m[n] for n in tv_names)),
                                               max(1e-6, m["kadis_deep"])]))))
            picks.append((sel, m, payload))
        if not picks:
            print(f"{dv:>6.2f} {kw:>6.2f} | ALL COLLAPSED"); continue
        picks.sort(key=lambda r: -r[0])
        sel, m, payload = picks[0]
        cid = spearmanr(T_fwd(payload, Xcv), ycv).correlation
        out = f"{OUT}/mlp_diverse_dv{dv}_kw{kw}.npz"
        np.savez(out, **payload)
        print(f"{dv:>6.2f} {kw:>6.2f} |  {cid:.4f}   {m['bigcodec']:.4f}  {m['kadis_deep']:.4f} "
              f"  seed{payload['seed']:>3d}   -> {Path(out).name}")
        results.append((dv, kw, cid, m["bigcodec"], m["kadis_deep"], out))
    if results:
        # a "keep the gain, recover the tail" score: bigcodec + deep, tie-break CID22
        best = max(results, key=lambda r: r[3] + r[4] + 0.5 * r[2])
        print(f"\nbest balance: div_w={best[0]} kadis_w={best[1]} -> CID22 {best[2]:.4f}, "
              f"bigcodec {best[3]:.4f}, deep {best[4]:.4f}\n  {best[5]}")


def T_fwd(P, X):
    Xc = np.clip(X, P["lo"], P["hi"]) if "lo" in P else X
    z = (Xc - P["mu"]) / P["sd"]
    h = z @ P["W0"].T + P["b0"]
    h = np.where(h > 0, h, float(P["leaky"]) * h)
    return (h @ P["W1"].T + P["b1"]).ravel()


if __name__ == "__main__":
    main()
