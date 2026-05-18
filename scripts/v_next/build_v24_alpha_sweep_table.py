#!/usr/bin/env python3
"""Aggregate V_24-α bake_compare JSONs into a sweep table.

For each α value, pulls SROCC/PWRC/Z-RMSE per corpus from the
bake_compare JSON sidecar (against V_22-mix-LARGE+iwssim seed=3
baseline). Reports a comparison table + per-α decisive-cell totals.

Also computes a weighted-best score:

    score(α) = CID22 + 0.5·(KADID + TID)/2 + 0.5·KonJND + 0.25·AIC-3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


def alpha_pct_from_path(p: str) -> int:
    m = re.search(r"alpha(\d{3})", p)
    return int(m.group(1)) if m else -1


def fmt(x):
    if x is None:
        return "—"
    return f"{x:+.4f}" if isinstance(x, float) and abs(x) < 10 else f"{x:.4f}"


def collect(json_dir: str, pattern: str):
    paths = sorted(glob.glob(os.path.join(json_dir, pattern)))
    rows = []
    for p in paths:
        pct = alpha_pct_from_path(p)
        with open(p) as fh:
            d = json.load(fh)
        per_corp = {c["name"]: c for c in d["corpora"]}
        counts = d.get("aggregate_counts", {})
        winner = d.get("overall_winner", "")
        rows.append({
            "alpha_pct": pct,
            "alpha": pct / 100.0,
            "corpora": per_corp,
            "counts": counts,
            "winner": winner,
            "json_path": p,
        })
    return rows


def score(row, baseline_row=None):
    """Weighted score relative to a baseline (or absolute if no baseline).

    score(α) = CID22 + 0.5·mean(KADID,TID) + 0.5·KonJND + 0.25·AIC-3
    using A's SROCC (the α bake). For deltas vs baseline, subtract B
    (the V_22 baseline SROCC) from each component.
    """
    def srocc(corp, side="a"):
        c = row["corpora"].get(corp)
        if c is None:
            return None
        return c["aggregate"][f"panel_{side}"]["srocc"]

    a_cid = srocc("cid22", "a")
    a_kad = srocc("kadid", "a")
    a_tid = srocc("tid", "a")
    a_kon = srocc("konjnd", "a")
    a_aic = srocc("aic3", "a")
    if None in (a_cid, a_kad, a_tid, a_kon, a_aic):
        return None
    return a_cid + 0.5 * (a_kad + a_tid) / 2.0 + 0.5 * a_kon + 0.25 * a_aic


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json-dir", default="/mnt/v/zen/zensim-eval/v24_alpha_2026-05-18")
    p.add_argument("--pattern", default="v24_alpha*_vs_v22mixLARGE.json")
    p.add_argument("--baseline-srocc", action="store_true",
                   help="Print B SROCC (V_22 baseline) alongside.")
    args = p.parse_args()

    rows = collect(args.json_dir, args.pattern)
    if not rows:
        print("No bake_compare JSONs found in", args.json_dir, "matching", args.pattern, file=sys.stderr)
        return 1

    # Pick baseline SROCC from the first row's panel_b (it's identical across α
    # since baseline doesn't change).
    base = rows[0]["corpora"]
    base_srocc = {}
    for corp in ("cid22", "kadid", "tid", "konjnd", "aic3"):
        if corp in base:
            base_srocc[corp] = base[corp]["aggregate"]["panel_b"]["srocc"]
        else:
            base_srocc[corp] = None

    # Baseline row score (V_22-mix-LARGE) — use panel_b SROCCs.
    def base_score():
        v = base_srocc
        if None in v.values():
            return None
        return v["cid22"] + 0.5 * (v["kadid"] + v["tid"]) / 2.0 + 0.5 * v["konjnd"] + 0.25 * v["aic3"]

    print(f"# V_24-α sweep table (vs V_22-mix-LARGE+iwssim seed=3)\n")
    print(f"**Baseline (B) SROCC**: CID22={base_srocc['cid22']:.4f}, "
          f"KADID={base_srocc['kadid']:.4f}, TID={base_srocc['tid']:.4f}, "
          f"KonJND={base_srocc['konjnd']:.4f}, AIC-3={base_srocc['aic3']:.4f}, "
          f"score={base_score():.4f}\n")

    header = ("| α | CID22_A | ΔCID22 | KADID_A | ΔKADID | TID_A | ΔTID "
              "| KonJND_A | ΔKonJND | AIC-3_A | ΔAIC-3 | score_A | Δscore "
              "| Adec / Bdec / promA / promB / tied / noisy | Winner |")
    sep = "|" + "|".join(["---"] * (header.count("|") - 1)) + "|"
    print(header)
    print(sep)

    bs = base_score()

    for row in rows:
        c = row["corpora"]
        winner = row["winner"]
        counts = row["counts"]

        def srocc(corp):
            cc = c.get(corp)
            return cc["aggregate"]["panel_a"]["srocc"] if cc else None

        a_cid = srocc("cid22")
        a_kad = srocc("kadid")
        a_tid = srocc("tid")
        a_kon = srocc("konjnd")
        a_aic = srocc("aic3")
        sa = score(row)
        ds = (sa - bs) if (sa is not None and bs is not None) else None

        # Decisive cell counts — pull from aggregate_counts dict (keys per § A.9).
        adec = counts.get("a_decisively_beats_b", 0)
        bdec = counts.get("b_decisively_beats_a", 0)
        # Promising can be split or unified — try both.
        proma = counts.get("promising_a", 0)
        promb = counts.get("promising_b", 0)
        if "promising_not_decisive" in counts:
            # Older single-bucket counter — split into A/B by panel_a vs panel_b SROCC.
            proma = counts.get("promising_not_decisive", 0)
            promb = 0  # unknown
        tied = counts.get("tied", 0)
        noisy = counts.get("noisy", 0)

        row_str = (
            f"| {row['alpha_pct']/100:.2f} | "
            f"{a_cid:.4f} | {a_cid - base_srocc['cid22']:+.4f} | "
            f"{a_kad:.4f} | {a_kad - base_srocc['kadid']:+.4f} | "
            f"{a_tid:.4f} | {a_tid - base_srocc['tid']:+.4f} | "
            f"{a_kon:.4f} | {a_kon - base_srocc['konjnd']:+.4f} | "
            f"{a_aic:.4f} | {a_aic - base_srocc['aic3']:+.4f} | "
            f"{sa:.4f} | {ds:+.4f} | "
            f"{adec}/{bdec}/{proma}/{promb}/{tied}/{noisy} | {winner} |"
        )
        print(row_str)

    print()
    print("**Score formula**: `CID22 + 0.5·mean(KADID,TID) + 0.5·KonJND + 0.25·AIC-3`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
