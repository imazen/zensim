#!/usr/bin/env python3
"""R6: read every gate off the artifacts and apply the PRE-REGISTERED rule.

Reads only what the owning tools produced — `bake_verdict`'s per-pair dumps and
G-ADDR JSON, `r6_arm_delta.py`'s cell-exact JSON, and (for G1) the paired
bootstrap in `wave6_paired_bootstrap.py`. It computes no statistic of its own:
every SROCC comes from `panel` through `scripts/lib/zen_stats`.

The decision rule is `docs/PLAN_FEATURE_REV2_2026-09-05.md` §7.6, written and
pushed before any table was extracted. This file implements it; it does not get
to change it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.zen_stats import panel  # noqa: E402

BASE = "ssim2"
PRIMARY = ("cid22", "konjnd", "aic3")
SECONDARY = ("csiq", "live", "tid", "kadid")
G4_BAR = 1e-4  # pre-registered §7.5


def srocc_of(path: Path) -> tuple[float, float, int]:
    a = np.loadtxt(path, delimiter="\t", skiprows=1)
    human, pred = a[:, 0], a[:, 1]
    r = panel(pred.tolist(), human.tolist())
    return float(r["srocc"]), float(r.get("srocc_signed", r["srocc"])), len(human)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/v/output/zensim/rev2-2026-09-05/r6")
    ap.add_argument("--arms", default="ssim2,c1,lorentz,clamp")
    ap.add_argument("--variants", default="s156_lasso,s156_bvls,s228_lasso,s228_bvls,s372_lasso,s372_bvls")
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    arms = a.arms.split(",")
    variants = a.variants.split(",")
    rep: dict = {"decision_rule": "docs/PLAN_FEATURE_REV2_2026-09-05.md §7.6",
                 "base_arm": BASE, "g4_bar": G4_BAR, "rank": {}, "bootstrap": {},
                 "gates": {}}

    # ---- G1 / G2 point estimates -------------------------------------------
    print("== RANK (|SROCC|; KonJND is |.| exactly as bake_verdict reports it) ==")
    hdr = f"{'variant':12s} {'arm':8s}" + "".join(f"{c:>10s}" for c in PRIMARY + SECONDARY)
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
            print(f"{v:12s} {arm:8s}{cells}")

    # ---- G1 paired bootstrap ------------------------------------------------
    print("\n== G1 PAIRED BOOTSTRAP vs the rev1 arm (95 % CI on the DELTA) ==")
    boot_script = Path(__file__).resolve().parent / "wave6_paired_bootstrap.py"
    for v in variants:
        for c in PRIMARY:
            series = [f"{arm}_{v}" for arm in arms
                      if (root / "perpair" / f"{arm}_{v}_{c}.tsv").exists()]
            if len(series) < 2:
                continue
            r = subprocess.run(
                [sys.executable, str(boot_script), "--dir", str(root / "perpair"),
                 "--corpus", c, "--series", *series, "--ref", f"{BASE}_{v}",
                 "--b", str(a.b), "--seed", str(a.seed)],
                capture_output=True, text=True)
            out = r.stdout
            rep["bootstrap"].setdefault(v, {})[c] = out
            keep = False
            for line in out.splitlines():
                if line.startswith("paired bootstrap"):
                    keep = True
                    print(f"\n-- {v} / {c}")
                if keep and (" - " in line or line.startswith("comparison")):
                    print("   " + line)

    # ---- C2 / G3 / G4 -------------------------------------------------------
    dp = root / "arm_delta_all.json"
    if dp.exists():
        d = json.load(open(dp))
        print("\n== C2 pathology / G3 outlier removal / G4 healthy perturbation ==")
        print(f"{'corpus':10s} {'rows':>7s} {'C2 path rows':>13s} {'arm':8s} "
              f"{'G3 rev1 max':>13s} {'G3 arm max':>12s} {'G4 cells':>10s} {'G4 max|Δ|':>11s} G4")
        g4 = {arm: {"worst": 0.0, "worst_corpus": None, "cells": 0} for arm in arms}
        g3 = {arm: 0.0 for arm in arms}
        for corpus, cd in d["corpora"].items():
            for arm, r in cd["arms"].items():
                g3[arm] = max(g3[arm], r["g3_arm_max_abs_on_moved_slots"])
                cells = r["g4_healthy_moved_cells"] or 0
                mx = r["g4_healthy_max_abs_delta"] or 0.0
                g4[arm]["cells"] += cells
                if mx > g4[arm]["worst"]:
                    g4[arm].update(worst=mx, worst_corpus=corpus)
                print(f"{corpus:10s} {cd['rows']:7d} {str(cd['c2_pathological_rows']):>13s} "
                      f"{arm:8s} {r['g3_rev1_max_abs_on_moved_slots']:13.6g} "
                      f"{r['g3_arm_max_abs_on_moved_slots']:12.6g} {cells:10d} "
                      f"{mx:11.4g} {'PASS' if mx <= G4_BAR else 'FAIL'}")
        rep["gates"]["g3_arm_max"] = g3
        rep["gates"]["g4"] = g4
        print("\n  G3 (max |f| over moved slots, all corpora):")
        rev1 = max(cd["rev1_max_abs_over_all_slots"] for cd in d["corpora"].values())
        print(f"    rev1 (ssim2) over ALL slots: {rev1:.6g}")
        for arm in arms:
            if arm == BASE:
                continue
            print(f"    {arm:8s} {g3[arm]:12.6g}   "
                  f"{'PASS (<=2 structural bound)' if g3[arm] <= 2.0 else 'FAIL'}")
        print("\n  G4 (healthy-cell perturbation, bar 1e-4):")
        for arm in arms:
            if arm == BASE:
                continue
            w = g4[arm]
            print(f"    {arm:8s} cells={w['cells']:9d} max|Δ|={w['worst']:.6g} "
                  f"({w['worst_corpus']}) "
                  f"{'PASS' if w['worst'] <= G4_BAR else 'FAIL'}")

    # ---- G5 ------------------------------------------------------------------
    dd = root / "dial"
    if dd.exists():
        print("\n== G5 DIAL (each arm's own in-era ladder + probes) ==")
        print(f"{'bake':26s} {'mono%':>8s} {'tied%':>7s} {'reach':>9s} {'min':>9s} "
              f"{'max':>8s} {'p5':>8s} {'ident':>9s} {'>ident':>7s} {'negfrac':>8s}")
        for v in variants:
            for arm in arms:
                # `--gaddr-json` writes the ADDRESSABILITY block alone (it
                # carries `measured`/`checks`, but not `mono_pct`/`reach`/the
                # percentiles, which live in `--full-json`'s `dial` block).
                # Read the full verdict and fall back to the gaddr file.
                p = dd / f"verdict_{arm}_{v}.json"
                g = dd / f"gaddr_{arm}_{v}.json"
                if not p.exists() and not g.exists():
                    continue
                dl = json.load(open(p)).get("dial", {}) if p.exists() else {}
                addr = dl.get("addressability")
                if addr is None and g.exists():
                    addr = json.load(open(g))
                m = (addr or {}).get("measured", {})
                ident = (m.get("identity") or {})
                neg = (m.get("negtail") or {})
                rep["gates"].setdefault("g5", {})[f"{arm}_{v}"] = {
                    "mono_pct": dl.get("mono_pct"), "tied_pct": dl.get("tied_pct"),
                    "reach": dl.get("reach"), "min": dl.get("min"), "max": dl.get("max"),
                    "p5": dl.get("p5"),
                    "identity_dial_max": ident.get("dial_max"),
                    "n_above_identity": ident.get("n_above_identity"),
                    "negtail_frac_below_zero": neg.get("frac_below_zero"),
                    "codec_floor": m.get("codec_floor"),
                }
                f = lambda x, w=9, d=4: (f"{x:{w}.{d}f}" if isinstance(x, (int, float)) else f"{'--':>{w}s}")
                print(f"{arm + '_' + v:26s} {f(dl.get('mono_pct'),8)} {f(dl.get('tied_pct'),7)} "
                      f"{f(dl.get('reach'))} {f(dl.get('min'))} {f(dl.get('max'),8,3)} "
                      f"{f(dl.get('p5'),8,3)} {f(ident.get('dial_max'))} "
                      f"{str(ident.get('n_above_identity')):>7s} {f(neg.get('frac_below_zero'),8)}")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=1)
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
