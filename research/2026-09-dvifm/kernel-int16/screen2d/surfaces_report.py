#!/usr/bin/env python3
"""Aggregate fit artefacts + grid surfaces into the Part-D summary tables
the DONE file leads with:

- per (variant, plane, level): committed c0/beta, refit MSE, SROCC/KROCC,
  edge findings
- sharpness: MSE increase at ±1 grid step on each axis (from the LAST
  sweep's surface CSV)
- neighbouring-level agreement: |Δlog c0|, |Δlog beta| between adjacent
  levels of the same plane (final committed constants)
- luma-vs-chroma knee comparison per variant

usage: surfaces_report.py --out-root /mnt/v/output/zensim/dvifm-screen2d-2026-09-19
emits tables/done_tables.md + tables/done_tables.json
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_grid(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    c0s = sorted({float(r["c0"]) for r in rows})
    betas = sorted({float(r["beta"]) for r in rows})
    m = {}
    for r in rows:
        m[(float(r["c0"]), float(r["beta"]))] = float(r["mse_refit"])
    return c0s, betas, m


def sharpness(grid):
    """MSE increase at +/-1 grid step around the argmin, per axis.
    Returns dict of deltas (None where the optimum is on the edge)."""
    c0s, betas, m = grid
    bi, bj, bv = None, None, math.inf
    for i, c0 in enumerate(c0s):
        for j, b in enumerate(betas):
            if m[(c0, b)] < bv:
                bi, bj, bv = i, j, m[(c0, b)]
    def d(di, dj):
        i, j = bi + di, bj + dj
        if 0 <= i < len(c0s) and 0 <= j < len(betas):
            return m[(c0s[i], betas[j])] - bv
        return None
    return {"c0": c0s[bi], "beta": betas[bj], "mse": bv,
            "d_c0_lo": d(-1, 0), "d_c0_hi": d(1, 0),
            "d_beta_lo": d(0, -1), "d_beta_hi": d(0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root",
                    default="/mnt/v/output/zensim/dvifm-screen2d-2026-09-19")
    a = ap.parse_args()
    out = Path(a.out_root)
    fits = sorted(out.glob("fits/*.json"))
    report = {}
    md = []
    for fp in fits:
        fit = json.loads(fp.read_text())
        if "level_report" not in fit:
            continue
        tag = fp.stem
        ent = {"variant": fit.get("variant"), "fit_mse": fit["fit_mse"],
               "var_y": fit.get("var_y"),
               "srocc_negE_fit": fit.get("srocc_negE_fit"),
               "krocc_negE_fit": fit.get("krocc_negE_fit"),
               "map": fit.get("map"), "sanity": fit.get("sanity"),
               "history": fit.get("history"), "wall_s": fit.get("wall_s"),
               "levels": []}
        # final committed constants per (plane, level)
        for p in fit["planes"]:
            for l, lv in enumerate(fit["levels"][p]):
                # last sweep's surface for this (p,l) gives sharpness
                reps = [r for r in fit["level_report"]
                        if r["plane"] == p and r["level"] == l]
                last = reps[-1]
                csvp = out / "surfaces" / tag / last["surface_csv"]
                sh = sharpness(load_grid(csvp)) if csvp.exists() else {}
                ent["levels"].append({
                    "plane": p, "level": l,
                    "g": lv["g"], "p_exp": lv["p"], "c0": lv["c0"],
                    "beta": lv["beta"], "sharp": lv["sharp"],
                    "grid_edge": last.get("grid_edge"),
                    "grid_extended": last.get("grid_extended"),
                    "grid_best_mse": last.get("best_mse"),
                    "grid_best_c0": last.get("best_c0"),
                    "grid_best_beta": last.get("best_beta"),
                    "grid_srocc": last.get("best_srocc"),
                    "sharpness": sh})
        # neighbour agreement on committed constants
        agree = []
        for p in fit["planes"]:
            lv = fit["levels"][p]
            for l in range(len(lv) - 1):
                agree.append({"plane": p, "levels": [l, l + 1],
                              "dlog_c0": abs(math.log(lv[l]["c0"])
                                             - math.log(lv[l + 1]["c0"])),
                              "dlog_beta": abs(math.log(lv[l]["beta"])
                                               - math.log(lv[l + 1]["beta"]))})
        ent["neighbour_agreement"] = agree
        # luma vs chroma knees
        if "ycbcr_y" in fit["levels"]:
            ent["knee_luma_mean_logc0"] = float(np.mean(
                [math.log(v["c0"]) for v in fit["levels"]["ycbcr_y"]]))
            for cp in ("ycbcr_cb", "ycbcr_cr"):
                if cp in fit["levels"]:
                    ent[f"knee_{cp}_mean_logc0"] = float(np.mean(
                        [math.log(v["c0"]) for v in fit["levels"][cp]]))
        report[tag] = ent
        ent["level_weights"] = fit.get("level_weights")
        ent["channel_weights"] = fit.get("channel_weights")

        init_mse = (fit.get("sanity") or {}).get("init_mse")
        md.append(f"## {tag} — fit_mse {fit['fit_mse']:.3f} "
                  f"(var_y {fit.get('var_y', 0):.2f}, "
                  f"init_mse {init_mse if init_mse is not None else float('nan'):.3f}, "
                  f"SROCC {fit.get('srocc_negE_fit', 0):.4f}, "
                  f"KROCC {fit.get('krocc_negE_fit', 0):.4f}, "
                  f"wall {fit.get('wall_s', 0)/60.0:.0f}min)")
        md.append("| plane | l | c0 | beta | g | p | sharp | edge |"
                  " dMSE c0±1 | dMSE beta±1 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for e in ent["levels"]:
            s = e["sharpness"] or {}
            dC = " / ".join(
                "edge" if s.get(k) is None else f"{s[k]:+.3f}"
                for k in ("d_c0_lo", "d_c0_hi"))
            dB = " / ".join(
                "edge" if s.get(k) is None else f"{s[k]:+.3f}"
                for k in ("d_beta_lo", "d_beta_hi"))
            md.append(f"| {e['plane']} | {e['level']} | {e['c0']:.4g} | "
                      f"{e['beta']:.4g} | {e['g']:.3f} | {e['p_exp']:.3g} | "
                      f"{e['sharp']:.3g} | {e['grid_edge'] or '-'} | "
                      f"{dC} | {dB} |")
        md.append("")
        lw = fit.get("level_weights") or {}
        for p in fit.get("planes", []):
            if p in lw:
                md.append(f"level_weights[{p}] = "
                          + ", ".join(f"{w:.3f}" for w in lw[p]))
        cw = fit.get("channel_weights")
        if cw:
            md.append(f"channel_weights (Y, C-per-chroma) = "
                      f"{cw[0]:.4f}, {cw[1]:.4f}")
        mp = fit.get("map") or {}
        md.append(f"map: A={mp.get('A', 0):.4g} B={mp.get('B', 0):.4g} "
                  f"lam={mp.get('lambda', 0):.4g}")
        md.append("")
    (out / "tables").mkdir(exist_ok=True)
    (out / "tables" / "done_tables.json").write_text(
        json.dumps(report, indent=1) + "\n")
    (out / "tables" / "done_tables.md").write_text("\n".join(md) + "\n")
    print(f"{len(report)} fits -> tables/done_tables.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
