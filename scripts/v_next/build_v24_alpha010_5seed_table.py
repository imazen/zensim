#!/usr/bin/env python3
"""Aggregate the 5-seed paired bake_compare results for V_24-α=0.10.

For each seed s in {1..5}, A = v24_alpha010_s${s}, B = v22-mix-LARGE+iwssim_s${s}.
Reports per-seed panel + mean/std across seeds.
"""

import glob
import json
import os
import re

import numpy as np


def collect():
    paths = sorted(glob.glob(
        "/mnt/v/zen/zensim-eval/v24_alpha_2026-05-18/v24_alpha010_s*_vs_v22mixLARGE_s*.json"
    ))
    rows = []
    for p in paths:
        seed = int(re.search(r"s(\d+)_vs", p).group(1))
        with open(p) as fh:
            d = json.load(fh)
        per = {c["name"]: c for c in d["corpora"]}
        rows.append({
            "seed": seed,
            "per": per,
            "counts": d["aggregate_counts"],
            "winner": d["overall_winner"],
        })
    return rows


def main():
    rows = collect()
    if len(rows) != 5:
        print(f"WARN: expected 5 seeds, found {len(rows)}")

    corpora = ["cid22", "kadid", "tid", "konjnd", "aic3"]

    # Per-seed panel
    print("# V_24-α=0.10 5-seed CI: paired bake_compare vs V_22-mix-LARGE+iwssim s_k\n")
    print("| seed | CID22_A | ΔCID22 | KADID_A | ΔKADID | TID_A | ΔTID | KonJND_A | ΔKonJND | AIC-3_A | ΔAIC-3 | Adec/Bdec/pND/tied/noisy | Winner |")
    print("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|")

    arrs = {c: {"A": [], "B": [], "delta": []} for c in corpora}
    winners = []
    counts_a, counts_b = [], []

    for r in rows:
        cols = []
        for c in corpora:
            cc = r["per"].get(c)
            if cc is None:
                cols.extend(["—", "—"])
                continue
            a = cc["aggregate"]["panel_a"]["srocc"]
            b = cc["aggregate"]["panel_b"]["srocc"]
            arrs[c]["A"].append(a)
            arrs[c]["B"].append(b)
            arrs[c]["delta"].append(a - b)
            cols.extend([f"{a:.4f}", f"{a-b:+.4f}"])
        wc = r["counts"]
        adec = wc.get("a_decisively_beats_b", 0)
        bdec = wc.get("b_decisively_beats_a", 0)
        pND = wc.get("promising_not_decisive", 0)
        tied = wc.get("tied", 0)
        noisy = wc.get("noisy", 0)
        counts_a.append(adec)
        counts_b.append(bdec)
        winners.append(r["winner"])
        print(f"| {r['seed']} | {' | '.join(cols)} | {adec}/{bdec}/{pND}/{tied}/{noisy} | {r['winner']} |")

    # Aggregate
    print()
    print("**Mean ± std across 5 seeds:**\n")
    print("| Corpus | A SROCC mean ± std | B SROCC mean ± std | Δ mean ± std |")
    print("|---|---:|---:|---:|")
    for c in corpora:
        a = np.array(arrs[c]["A"])
        b = np.array(arrs[c]["B"])
        d = np.array(arrs[c]["delta"])
        print(f"| {c} | {a.mean():.4f} ± {a.std(ddof=1):.4f} | "
              f"{b.mean():.4f} ± {b.std(ddof=1):.4f} | "
              f"{d.mean():+.4f} ± {d.std(ddof=1):.4f} |")

    print()
    print(f"**Aggregate decisive cells**: A wins {sum(counts_a)} total, "
          f"B wins {sum(counts_b)} total (across 5 seeds).")
    print(f"**Overall winner per seed**: {winners}")
    if all(w == "B" for w in winners):
        print("\n**5-seed verdict: B (V_22-mix-LARGE+iwssim) decisively wins across all 5 seeds.**")
    else:
        print(f"\n**5-seed verdict: mixed.**")


if __name__ == "__main__":
    main()
