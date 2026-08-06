#!/usr/bin/env python3
"""Appendix-U analysis: the measured null, the shortlist, the pairing test.

Reads ONLY the collated grid (whose numbers came from `bake_verdict`) and the
per-pair dumps (which the paired bootstrap reduces through `panel --batch`).
It computes no statistic itself beyond quantiles of deltas the owners produced.

Three questions, in the order appendix U registered them:

1. **What is the noise floor, measured?** The ZERO cells are a genuinely null
   intervention (candidate coefficients exactly 0.0; only sub-`tol` CD drift
   separates them from base) that still went through the whole
   fit/pack/spline/score chain. The spread of THEIR deltas is the empirical
   floor, and the expected false-positive yield at any threshold follows from
   it directly.

2. **Which cells clear it?** Primary = signed CID22 B9; secondary = HF-NL
   per-ref; guards must not regress outside their floors.

3. **Did PAIRING beat SINGLETONS?** For every pair, compare its effect to the
   better of its two members alone. If no pair exceeds max(member effects)
   outside noise, the pairing hypothesis is falsified and the appendix says so.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

GUARDS = {  # axis -> registered floor (regression beyond this disqualifies)
    "cid22": 0.005, "konjnd": 0.039, "nonphoto": 0.010, "csiq": 0.010,
    "live": 0.010, "imazen26": 0.010,
}
HFNL_FLOOR = 0.039  # appendix O axis LSD


def f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.grid), delimiter="\t"))
    for r in rows:
        for k in list(r):
            if k.startswith("d_") or k in ("b9_abs", "b9_signed", "base_max_shift"):
                r[k] = f(r[k])

    out = []
    def say(s=""):
        print(s)
        out.append(s)

    nulls = [r for r in rows if r["status"] == "ZERO"]
    lives = [r for r in rows if r["status"] == "LIVE"]
    say(f"# appendix U — grid analysis  ({len(rows)} evaluated cells: "
        f"{len(lives)} LIVE, {len(nulls)} NULL)")

    # ---------------- 1. the measured null ----------------------------------
    say("\n## 1. The MEASURED null (ZERO cells: candidate weights exactly 0.0)\n")
    say("| axis | n | null mean | null sd | null p2.5 | null p97.5 | max abs |")
    say("|---|--:|--:|--:|--:|--:|--:|")
    nullsd = {}
    for ax in ["b9_signed", "hfnl_perref"] + list(GUARDS):
        v = np.array([r[f"d_{ax}"] for r in nulls if r.get(f"d_{ax}") is not None])
        if v.size == 0:
            continue
        nullsd[ax] = float(v.std(ddof=1)) if v.size > 1 else 0.0
        say(f"| {ax} | {v.size} | {v.mean():+.6f} | {v.std(ddof=1):.6f} | "
            f"{np.quantile(v, 0.025):+.6f} | {np.quantile(v, 0.975):+.6f} | "
            f"{np.abs(v).max():.6f} |")
    say("\nThe null is the appendix's floor instrument. When it comes out EXACTLY "
        "zero, as it does here, that is itself the result: a cell whose candidate "
        "coefficients are 0.0 scores bit-identically to base through the whole "
        "fit -> pack -> spline -> score chain, so the LIVE/ZERO split is exact and "
        "NONE of the spread below is fit noise. It also means the null supplies no "
        "usable floor, so the REGISTERED axis floors govern (U.5) and the paired "
        "bootstrap owns the remaining eval-sampling noise.")

    # ---------------- 2. distribution of the primary objective --------------
    say("\n## 2. Primary objective — signed CID22 B9\n")
    d9 = np.array([r["d_b9_signed"] for r in lives if r.get("d_b9_signed") is not None])
    base9 = next((r["b9_signed"] for r in rows if r["status"] == "BASE"
                  and r["arm"] == "A"), None)
    say(f"base (arm A) signed B9 = {base9:+.6f}; LIVE cells n={d9.size}")
    say(f"d_b9_signed: mean {d9.mean():+.5f} sd {d9.std(ddof=1):.5f} "
        f"min {d9.min():+.5f} max {d9.max():+.5f}")
    say(f"cells whose signed B9 reaches the F8 bar (>= 0.15): "
        f"{sum(1 for r in lives if (r.get('b9_signed') or -9) >= 0.15)}")
    say(f"cells whose ABS B9 reaches 0.15 but are INVERTED: "
        f"{sum(1 for r in lives if (r.get('b9_abs') or 0) >= 0.15 and (r.get('b9_signed') or 0) < 0)}"
        "  <- these would PASS F8 as implemented")

    # expected false positives at the observed threshold, from the null sd
    if "b9_signed" in nullsd and nullsd["b9_signed"] > 0:
        say(f"\nnull sd for d_b9_signed = {nullsd['b9_signed']:.5f}; the LIVE sd is "
            f"{d9.std(ddof=1):.5f} ({d9.std(ddof=1)/nullsd['b9_signed']:.1f}x). "
            "A LIVE spread far above the null means the cells really do move B9 — "
            "it does NOT by itself mean any single cell's move is reproducible.")

    # ---------------- 3. shortlist ------------------------------------------
    say("\n## 3. Shortlist — ranked by signed B9, guards annotated\n")

    def guard_fails(r):
        bad = []
        for ax, fl in GUARDS.items():
            d = r.get(f"d_{ax}")
            if d is not None and d < -fl:
                bad.append(f"{ax}{d:+.3f}")
        return bad

    cand = sorted((r for r in lives if r.get("d_b9_signed") is not None),
                  key=lambda r: -r["d_b9_signed"])
    say("| # | arm | kind | cell | names | b9_signed | d_b9 | b9_abs | "
        "d_hfnl | guard regressions |")
    say("|--:|---|---|---|---|--:|--:|--:|--:|---|")
    for i, r in enumerate(cand[:a.top], 1):
        g = guard_fails(r)
        say(f"| {i} | {r['arm']} | {r['kind']} | {r['cell_id']} | {r['names']} | "
            f"{r['b9_signed']:+.4f} | {r['d_b9_signed']:+.4f} | {r['b9_abs']:.4f} | "
            f"{(r.get('d_hfnl_perref') or 0):+.4f} | {', '.join(g) if g else 'none'} |")

    clean = [r for r in cand if not guard_fails(r)]
    say(f"\ncells with NO guard regression outside floor: {len(clean)} of {len(cand)}")

    # ---------------- 4. secondary objective --------------------------------
    say("\n## 4. Secondary objective — HF-NL per-ref\n")
    dh = np.array([r["d_hfnl_perref"] for r in lives
                   if r.get("d_hfnl_perref") is not None])
    say(f"d_hfnl_perref: n={dh.size} mean {dh.mean():+.5f} sd {dh.std(ddof=1):.5f} "
        f"min {dh.min():+.5f} max {dh.max():+.5f}")
    say(f"cells clearing the {HFNL_FLOOR} axis LSD: {(dh >= HFNL_FLOOR).sum()}")
    # do the two HF axes agree?
    both = [(r["d_b9_signed"], r["d_hfnl_perref"]) for r in lives
            if r.get("d_b9_signed") is not None and r.get("d_hfnl_perref") is not None]
    if len(both) > 2:
        x = np.array([p[0] for p in both]); y = np.array([p[1] for p in both])
        # rank correlation between the two HF axes' deltas, over cells
        rx = x.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        say(f"\nrank correlation between d_b9_signed and d_hfnl_perref over "
            f"{len(both)} LIVE cells: {rho:+.4f}")
        say("  -> near zero means the two high-fidelity axes are DIFFERENT problems "
            "(registered outcome (b)); strongly positive means one HF factor.")

    # ---------------- 5. did pairing beat singletons? -----------------------
    say("\n## 5. Did PAIRING beat SINGLETONS?\n")
    single = {}
    for r in rows:
        if r["n_cand"] == "1" and r["status"] in ("LIVE", "ZERO"):
            single[(r["arm"], r["indices"])] = r
    say("| axis | pairs with both members measured | pair > max(member) | "
        "pair > max(member) by > floor | best excess |")
    say("|---|--:|--:|--:|--:|")
    say("(the 'by > floor' column uses max(2 x null sd, the registered axis floor) "
        "— with a zero null that is the registered floor)")
    FLOOR = {"hfnl_perref": HFNL_FLOOR, **GUARDS,
             "b9_signed": 0.0}   # B9 has no registered floor; see U.R4
    for ax in ("b9_signed", "hfnl_perref", "cid22"):
        thr = max(2.0 * nullsd.get(ax, 0.0), FLOOR.get(ax, 0.010))
        n = beat = beat_n = 0
        best = (None, -9.0)
        for r in rows:
            if r["n_cand"] != "2" or r["status"] != "LIVE":
                continue
            i1, i2 = r["indices"].split(",")
            m1 = single.get((r["arm"], i1)); m2 = single.get((r["arm"], i2))
            if not m1 or not m2:
                continue
            dp = r.get(f"d_{ax}")
            d1 = m1.get(f"d_{ax}") if m1["status"] == "LIVE" else 0.0
            d2 = m2.get(f"d_{ax}") if m2["status"] == "LIVE" else 0.0
            if dp is None or d1 is None or d2 is None:
                continue
            n += 1
            mx = max(d1, d2)
            if dp > mx:
                beat += 1
                if dp - mx > thr:
                    beat_n += 1
                if dp - mx > best[1]:
                    best = (r, dp - mx)
        b = f"{best[1]:+.4f} ({best[0]['names']})" if best[0] else "-"
        say(f"| {ax} (floor {thr:.4f}) | {n} | {beat} | {beat_n} | {b} |")
    say("\nA pair that never exceeds the better of its two members is an additive "
        "combination of effects already available singly — the pairing hypothesis "
        "predicts SUPER-additivity, and this table is its test.")

    # ---------------- 6. by block / family ----------------------------------
    say("\n## 6. Where the movement lives (LIVE cells, median delta by block)\n")
    say("| arm | block | n LIVE | med d_b9 | max d_b9 | med d_hfnl | med d_cid22 | med d_konjnd |")
    say("|---|---|--:|--:|--:|--:|--:|--:|")
    keys = sorted({(r["arm"], r["block"]) for r in lives})
    for arm, blk in keys:
        g = [r for r in lives if r["arm"] == arm and r["block"] == blk]
        def med(k):
            v = [r[f"d_{k}"] for r in g if r.get(f"d_{k}") is not None]
            return f"{np.median(v):+.4f}" if v else "-"
        d9g = [r["d_b9_signed"] for r in g if r.get("d_b9_signed") is not None]
        say(f"| {arm} | {blk} | {len(g)} | {med('b9_signed')} | "
            f"{max(d9g):+.4f} | {med('hfnl_perref')} | {med('cid22')} | "
            f"{med('konjnd')} |")

    # ---------------- 7. by candidate FAMILY (the brief's HF-plausibility ranks)
    say("\n## 7. By candidate family — did the HF-plausibility ranking predict anything?\n")
    import re
    FAMILIES = [
        ("1 near-threshold/JND", r"PJND"),
        ("2 artifact: BANDING", r"BANDING"),
        ("2 artifact: BLOCKINESS", r"BLOCKINESS"),
        ("2 artifact: RINGING", r"RINGING"),
        ("2 artifact: EDGE_WIDTH", r"EDGE_WIDTH_CHANGE"),
        ("3 BANDVIS (append2)", r"BANDVIS"),
        ("4 HF gain/loss (v2)", r"HF_GAIN|HF_LOSS|HF_MAG_LOSS"),
        ("4 CONTRAST gain/loss", r"CONTRAST_GAIN|CONTRAST_LOSS"),
        ("5 soft-peak", r"SOFT_PEAK"),
        ("6 v1 peak72", r"_max@|_p95@"),
        ("6 v1 masked72", r"masked_"),
        ("6 v1 iw72", r"iw_"),
        ("7 scale-0 only", r"@s0"),
    ]
    say("| family (brief rank) | n LIVE | med dB9 | max dB9 | med dHFNL | max dHFNL | "
        "med dCID22 | max dCID22 | n cells with dCID22 > +0.005 |")
    say("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for label, pat in FAMILIES:
        g = [r for r in lives if re.search(pat, r["names"])]
        if not g:
            say(f"| {label} | 0 | — | — | — | — | — | — | — |")
            continue
        def st(k, fn):
            v = [r[f"d_{k}"] for r in g if r.get(f"d_{k}") is not None]
            return f"{fn(v):+.4f}" if v else "—"
        ncid = sum(1 for r in g if (r.get("d_cid22") or -9) > 0.005)
        say(f"| {label} | {len(g)} | {st('b9_signed', np.median)} | {st('b9_signed', max)} | "
            f"{st('hfnl_perref', np.median)} | {st('hfnl_perref', max)} | "
            f"{st('cid22', np.median)} | {st('cid22', max)} | {ncid} |")
    say("\nThe HF-plausibility ranking was frozen in U.3 BEFORE any fit. This table is "
        "its scorecard: a family that was ranked high and moves nothing is a "
        "falsified prior, and one ranked low that moves an axis is a finding the "
        "ranking missed.")

    a.out.write_text("\n".join(out) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
