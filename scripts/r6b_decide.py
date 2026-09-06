#!/usr/bin/env python3
"""R6b: read every F17 gate off the artifacts and apply the PRE-REGISTERED rule.

Reads only what the owning tools produced — `bake_verdict`'s per-pair dumps and
dial JSON, `r6b_arm_delta.py`'s cell-exact JSON, and the paired bootstrap in
`wave6_paired_bootstrap.py`. It computes no statistic of its own; `srocc_of` is
imported from `r6_decide` so both lanes read a per-pair dump the same way.

The rule is `docs/PLAN_FEATURE_REV2_2026-09-05.md` §11.9, written and pushed
before any table was extracted. This file implements it; it does not get to
change it. In particular §11.8's H6 is a RANKING, not a gate — printing it with
a PASS/FAIL column would quietly reinstate a bar the pre-registration removed on
the record, so it prints a rank.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r6_decide import srocc_of  # noqa: E402 — one owner for the per-pair read

BASE = "ratio"
PRIMARY = ("cid22", "konjnd", "aic3")
SECONDARY = ("csiq", "live", "tid", "kadid")
# §11.9 rule 5: the registered prior, applied only on an exact H6 tie.
PRIOR = "satexcess"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/v/output/zensim/rev2-2026-09-05/r6b")
    ap.add_argument("--arms", default="ratio,bexcess,log1p,satexcess,cap")
    ap.add_argument("--variants", default="s156_lasso,s156_bvls,s228_lasso,s228_bvls")
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    arms = a.arms.split(",")
    cands = [x for x in arms if x != BASE]
    variants = a.variants.split(",")
    rep: dict = {"decision_rule": "docs/PLAN_FEATURE_REV2_2026-09-05.md §11.9",
                 "base_arm": BASE, "prior": PRIOR, "rank": {}, "bootstrap": {}, "gates": {}}

    # ---- H1 / H2 point estimates -------------------------------------------
    print("== RANK (|SROCC| as bake_verdict reports it; KonJND is |·|) ==")
    hdr = f"{'variant':12s} {'arm':10s}" + "".join(f"{c:>10s}" for c in PRIMARY + SECONDARY)
    for v in variants:
        print(f"\n{hdr}")
        for arm in arms:
            row, cells = {}, ""
            for c in PRIMARY + SECONDARY:
                p = root / "perpair" / f"{arm}_{v}_{c}.tsv"
                if not p.exists():
                    cells += f"{'--':>10s}"
                    continue
                s, ss, n = srocc_of(p)
                row[c] = {"srocc": s, "srocc_signed": ss, "n": n}
                cells += f"{s:10.5f}"
            rep["rank"].setdefault(v, {})[arm] = row
            print(f"{v:12s} {arm:10s}{cells}")

    # ---- H1 paired bootstrap ------------------------------------------------
    print("\n== H1 PAIRED BOOTSTRAP vs the revision-1 arm (95 % CI on the DELTA) ==")
    boot = Path(__file__).resolve().parent / "wave6_paired_bootstrap.py"
    wins: dict = {arm: {v: [] for v in variants} for arm in cands}
    for v in variants:
        for c in PRIMARY:
            series = [f"{arm}_{v}" for arm in arms
                      if (root / "perpair" / f"{arm}_{v}_{c}.tsv").exists()]
            if len(series) < 2:
                continue
            r = subprocess.run(
                [sys.executable, str(boot), "--dir", str(root / "perpair"),
                 "--corpus", c, "--series", *series, "--ref", f"{BASE}_{v}",
                 "--b", str(a.b), "--seed", str(a.seed)],
                capture_output=True, text=True)
            rep["bootstrap"].setdefault(v, {})[c] = r.stdout
            print(f"\n-- {v} / {c}")
            for line in r.stdout.splitlines():
                if " - " in line or line.startswith(("paired bootstrap", "comparison")):
                    print("   " + line)
                # §11.8: a win counts only if the 95 % CI on the DELTA excludes 0.
                # `wave6_paired_bootstrap.py` prints
                #   "<series> - <ref>   median  2.5%  97.5%  P(d>0)"
                # so the test is on its 2.5 % column, read positionally rather
                # than by grepping for a word the owner never prints.
                for arm in cands:
                    tok = f"{arm}_{v} - {BASE}_{v}"
                    if not line.strip().startswith(tok):
                        continue
                    parts = line.strip()[len(tok):].split()
                    if len(parts) < 4:
                        continue
                    try:
                        med, lo, hi = (float(parts[0]), float(parts[1]), float(parts[2]))
                    except ValueError:
                        continue
                    if lo > 0.0:
                        wins[arm][v].append((c, {"median": med, "lo": lo, "hi": hi}))
                    rep.setdefault("deltas", {}).setdefault(v, {}).setdefault(arm, {})[c] = {
                        "median": med, "ci_lo": lo, "ci_hi": hi,
                        "win": bool(lo > 0.0), "loss": bool(hi < 0.0)}
    rep["gates"]["h1_ci_excluding_wins"] = {
        k: {v: [{"corpus": c, **d} for c, d in x] for v, x in dd.items()}
        for k, dd in wins.items()
    }

    # ---- CB4 / CB2 / H3 / H5 / H6 -------------------------------------------
    dp = root / "arm_delta_all.json"
    agg: dict = {}
    if dp.exists():
        d = json.load(open(dp))
        print("\n== CB4 containment / CB2 pathology / H3 bound / H5 order / H6 healthy ==")
        agg = {arm: {"cb4": True, "h3_max": 0.0, "h3_bound": d["arm_bound"].get(arm),
                     "h5_inv": 0, "h5_ties": 0, "h6_cells": 0, "h6_max": 0.0,
                     "h6_over_bar": 0, "h6_worst_corpus": None} for arm in cands}
        rev1_max = 0.0
        for corpus, cd in d["corpora"].items():
            rev1_max = max(rev1_max, cd["rev1_max_over_f17"])
            for arm, r in cd["arms"].items():
                g = agg[arm]
                g["cb4"] &= r["cb4_only_f17_slots"]
                g["h3_max"] = max(g["h3_max"], r["h3_arm_max"])
                g["h5_inv"] += r["h5_inversions"]
                g["h5_ties"] += r["h5_new_ties"]
                g["h6_cells"] += r["h6_moved_cells"]
                g["h6_over_bar"] += r["h6_cells_over_bar"]
                if r["h6_max_abs_delta"] > g["h6_max"]:
                    g["h6_max"] = r["h6_max_abs_delta"]
                    g["h6_worst_corpus"] = corpus
        print(f"  revision 1 max over the twelve F17 slots, all legs: {rev1_max:.6g}")
        print(f"\n  {'arm':10s} {'CB4':>5s} {'H3 max':>10s} {'bound':>7s} {'H3':>5s} "
              f"{'H5 inv':>8s} {'H5 tie':>9s} {'H5':>5s} {'H6 cells':>10s} "
              f"{'H6 max|d|':>10s} {'H6 >1e-4':>10s}")
        for arm in cands:
            g = agg[arm]
            b = g["h3_bound"]
            g["h3_pass"] = b is not None and g["h3_max"] <= b + 1e-12
            g["h5_pass"] = g["h5_inv"] == 0 and g["h5_ties"] == 0
            print(f"  {arm:10s} {'ok' if g['cb4'] else 'BAD':>5s} {g['h3_max']:10.5g} "
                  f"{str(b):>7s} {'PASS' if g['h3_pass'] else 'FAIL':>5s} "
                  f"{g['h5_inv']:8d} {g['h5_ties']:9d} "
                  f"{'PASS' if g['h5_pass'] else 'FAIL':>5s} {g['h6_cells']:10d} "
                  f"{g['h6_max']:10.4g} {g['h6_over_bar']:10d}")
        # H6 is a RANK, per §11.8.
        order = sorted(cands, key=lambda x: (agg[x]["h6_cells"], agg[x]["h6_max"]))
        print("\n  H6 RANK (smallest healthy-cell perturbation first; NOT a gate): "
              + " < ".join(order))
        rep["gates"]["cell"] = agg
        rep["gates"]["h6_rank"] = order

    # ---- H7 dial -------------------------------------------------------------
    dd = root / "dial"
    if dd.exists():
        print("\n== H7 DIAL (each arm's own in-era ladder + probes) ==")
        print(f"  {'bake':28s} {'mono%':>8s} {'tied%':>7s} {'reach':>9s} {'min':>9s} "
              f"{'max':>8s} {'p5':>8s} {'ident':>9s} {'>ident':>7s} {'negfrac':>8s}")
        for v in variants:
            for arm in arms:
                p, g = dd / f"verdict_{arm}_{v}.json", dd / f"gaddr_{arm}_{v}.json"
                if not p.exists() and not g.exists():
                    continue
                dl = json.load(open(p)).get("dial", {}) if p.exists() else {}
                addr = dl.get("addressability") or (json.load(open(g)) if g.exists() else {})
                m = (addr or {}).get("measured", {})
                ident, neg = m.get("identity") or {}, m.get("negtail") or {}
                rep["gates"].setdefault("h7", {})[f"{arm}_{v}"] = {
                    "mono_pct": dl.get("mono_pct"), "tied_pct": dl.get("tied_pct"),
                    "reach": dl.get("reach"), "min": dl.get("min"), "max": dl.get("max"),
                    "p5": dl.get("p5"), "identity_dial_max": ident.get("dial_max"),
                    "n_above_identity": ident.get("n_above_identity"),
                    "negtail_frac_below_zero": neg.get("frac_below_zero")}
                def f(x, w=9, dg=4):
                    return f"{x:{w}.{dg}f}" if isinstance(x, (int, float)) else f"{'--':>{w}s}"
                print(f"  {arm + '_' + v:28s} {f(dl.get('mono_pct'),8)} {f(dl.get('tied_pct'),7)} "
                      f"{f(dl.get('reach'))} {f(dl.get('min'))} {f(dl.get('max'),8,3)} "
                      f"{f(dl.get('p5'),8,3)} {f(ident.get('dial_max'))} "
                      f"{str(ident.get('n_above_identity')):>7s} {f(neg.get('frac_below_zero'),8)}")

    # ---- §11.9, applied step by step ---------------------------------------
    if agg:
        print("\n== §11.9 DECISION RULE, step by step ==")
        surviving = []
        for arm in cands:
            g = agg[arm]
            bad = [n for n, ok in (("H3", g["h3_pass"]), ("H5", g["h5_pass"]),
                                   ("CB4", g["cb4"])) if not ok]
            print(f"  1. {arm:10s} {'OUT (' + ','.join(bad) + ')' if bad else 'survives'}")
            if not bad:
                surviving.append(arm)
        # §11.9 rule 2 says "a strict majority (>= 2 of 3) of {CID22, KonJND,
        # AIC-3}" without pinning how the four (slice x solver) variants
        # aggregate. Read LITERALLY — a majority in at least one variant — and
        # the stricter reading (a majority in most variants) is printed beside
        # it so the choice is visible rather than convenient. R6's precedent is
        # the strict one: `c1` won a majority in 1 of 6 variants there and the
        # lane still took rule 4.
        per_variant = {arm: {v: sum(1 for _ in wins[arm][v]) for v in variants}
                       for arm in cands}
        print(f"\n  primary-corpus CI-excluding wins per (arm, variant), of 3:")
        print(f"  {'arm':10s}" + "".join(f"{v:>14s}" for v in variants))
        for arm in cands:
            print(f"  {arm:10s}" + "".join(f"{per_variant[arm][v]:>14d}" for v in variants))
        maj = [arm for arm in surviving
               if any(per_variant[arm][v] >= 2 for v in variants)]
        maj_strict = [arm for arm in surviving
                      if sum(per_variant[arm][v] >= 2 for v in variants) * 2 > len(variants)]
        rep["gates"]["h1_wins_per_variant"] = per_variant
        rep["gates"]["h1_majority_strict"] = maj_strict
        print(f"  2. rank-majority survivors (literal, >=1 variant): {maj or 'NONE'}")
        print(f"     ... under the strict reading (majority of variants):    "
              f"{maj_strict or 'NONE'}")
        if len(maj) == 1:
            pick, why = maj[0], "rule 2 — sole survivor with a rank majority"
        elif len(maj) > 1:
            pick = max(maj, key=lambda x: sum(len(wins[x][v]) for v in variants))
            why = "rule 3 — most CI-excluding wins"
        elif surviving:
            best = sorted(surviving,
                          key=lambda x: (agg[x]["h6_cells"], agg[x]["h6_max"]))
            tie = (len(best) > 1
                   and (agg[best[0]]["h6_cells"], agg[best[0]]["h6_max"])
                   == (agg[best[1]]["h6_cells"], agg[best[1]]["h6_max"]))
            pick = PRIOR if tie and PRIOR in best else best[0]
            why = ("rule 5 — exact H6 tie, registered prior"
                   if tie and PRIOR in best
                   else "rule 4 — no rank majority; smallest healthy-cell perturbation")
        else:
            pick, why = None, "NO ARM SURVIVED H3/H5/CB4"
        print(f"\n  ⇒ REV2_HFGAIN = {pick}   ({why})")
        rep["decision"] = {"arm": pick, "why": why, "surviving": surviving,
                           "rank_majority": maj}

    out = a.out or str(root / "decide.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
