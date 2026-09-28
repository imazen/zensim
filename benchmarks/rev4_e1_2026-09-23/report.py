#!/usr/bin/env python3
"""Rev4 E1 — decision rule + figure-ready tables from analyze.py / fc.py output.

Pure bookkeeping over the owner's numbers (no statistic is computed here):
reads /var/tmp/rev4-e1/{e1_full,fc_full}.json, applies prereg §6, writes the
compact committed JSON and markdown tables.
"""
from __future__ import annotations

import argparse
import json

ZORDER = ["B", "C", "D", "R915_fast", "R915_rich", "V0_2"]
UNITS = {"cid22a": "CID22-A", "csiq": "CSIQ", "konjnd404": "KonJND-404",
         "aic3": "JPEG-AIC", "aic4crop": "JPEG-AIC", "aic4full": "JPEG-AIC", "sdr25": "JPEG-AIC", "fc": "JPEG-AIC"}
CNAME = {"cid22a": "CID22-A(25)", "csiq": "CSIQ", "konjnd404": "KonJND-404", "aic3": "AIC-3 CTC",
         "aic4crop": "AIC-4 crop", "aic4full": "AIC-4 full-res", "sdr25": "SDR25", "fc": "JPEG-AIC FC (btc_native)"}
FC_MAP = {"C": "C", "B": "B"}


def r4(x):
    return None if x is None or x != x else round(x, 4)


def primary(c):
    return "srocc" if c == "konjnd404" else "pairwise"


def verdict_of(ci):
    if ci[1] < 0:
        return "LOSS"
    if ci[0] > 0:
        return "WIN"
    return "TIE"


def cells(E, F, stat_mode, best_mode):
    """Yield (corpus, band, role, zmodel, best_peer, delta_point, ci) for every model/band."""
    out = []
    for c, cr in E["corpora"].items():
        st = primary(c) if stat_mode == "primary" else "srocc"
        glob_best = cr["bands"]["ALL"]["srocc"]["best_peer"]
        for b, e in cr["bands"].items():
            if b == "ALL":
                continue
            blk = e.get(st)
            if not blk:
                continue
            bp = blk["best_peer"] if best_mode == "band" else glob_best
            for z in cr["zensim"]:
                d = blk["deltas"][f"{z}-{bp}"]
                out.append((c, b, e["role"], z, bp, d["point"], d["ci"]))
    # forced choice: btc_native, question type `all`, per fidelity band, vs ssim2, image-clustered
    arm = F["arms"]["btc_native"]
    for fb, role in (("HF", "NT"), ("MF", "MID"), ("LF", "LOW")):
        cell = arm[f"{fb}/all"]
        for z in ("C", "B"):
            d = cell["deltas_vs_ssim2"][z]
            out.append(("fc", fb, role, z, "ssim2", d["point"], d["ci_img"]))
    return out


def decide(cs, z, family_once=True):
    T, L, detail = set(), set(), []
    for c, b, role, zz, bp, p, ci in cs:
        if zz != z:
            continue
        unit = UNITS[c] if family_once else f"{UNITS[c]}:{c}"
        v = verdict_of(ci)
        detail.append({"corpus": c, "band": b, "role": role, "best_peer": bp, "delta": r4(p),
                       "ci": [r4(ci[0]), r4(ci[1])], "verdict": v})
        if v == "LOSS":
            (T if role == "NT" else L).add(unit)
    units = sorted({(UNITS[c] if family_once else f"{UNITS[c]}:{c}") for c, _, _, zz, *_ in cs if zz == z})
    if len(T) >= 2 and not L:
        out = "CONFIRMED"
    elif L and len(L) >= len(T):
        out = "REFUTED"
    else:
        out = "UNRESOLVED"
    return {"outcome": out, "T_nt_deficit_units": sorted(T), "L_mid_low_deficit_units": sorted(L),
            "units_with_rows": units, "cells": detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="/var/tmp/rev4-e1/e1_full.json")
    ap.add_argument("--fc", default="/var/tmp/rev4-e1/fc_full.json")
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--md-out", required=True)
    a = ap.parse_args()
    E, F = json.load(open(a.e1)), json.load(open(a.fc))

    decisions = {}
    for mode, (sm, bm, fam) in {"primary": ("primary", "band", True),
                               "sens_bandsrocc": ("srocc", "band", True),
                               "sens_family_per_member": ("primary", "band", False),
                               "sens_bestpeer_fixed_per_corpus": ("primary", "corpus", True)}.items():
        cs = cells(E, F, sm, bm)
        decisions[mode] = {z: decide(cs, z, fam) for z in ZORDER}

    # compact per-band table (committed): primary + band SROCC, points and CIs, zensim deltas vs best & ssim2
    table = {}
    for c, cr in E["corpora"].items():
        tc = {"n_rows": cr["n_rows"], "n_refs": cr["n_refs"], "band_def": cr["band_def"], "bands": {}}
        for b, e in cr["bands"].items():
            if c == "konjnd404" and b == "ALL":
                continue
            be = {"role": e["role"], "n_stim": e["n_stim"], "n_refs": e["n_refs"], "n_pairs": e.get("n_pairs"),
                  "n_pair_refs": e.get("n_pair_refs"), "t_range": [r4(x) for x in e["t_range"]]}
            for st in ("pairwise", "srocc"):
                blk = e.get(st)
                if not blk:
                    continue
                keep_ci = set(cr["zensim"]) | {"ssim2", blk["best_peer"]}
                be[st] = {"best_peer": blk["best_peer"],
                          "m": {m: ([r4(v["point"]), r4(v["ci"][0]), r4(v["ci"][1])] if m in keep_ci else r4(v["point"]))
                                for m, v in blk["metrics"].items()},
                          "d_best": {z: [r4(blk["deltas"][f"{z}-{blk['best_peer']}"]["point"]),
                                         [r4(x) for x in blk["deltas"][f"{z}-{blk['best_peer']}"]["ci"]]]
                                     for z in cr["zensim"]},
                          "d_ssim2": {z: [r4(blk["deltas"][f"{z}-ssim2"]["point"]),
                                          [r4(x) for x in blk["deltas"][f"{z}-ssim2"]["ci"]]] for z in cr["zensim"]}}
            tc["bands"][b] = be
        table[c] = tc
    fc = {}
    for arm, cellsd in F["arms"].items():
        fa = {}
        for k, cell in cellsd.items():
            if arm != "btc_native" and not k.endswith("/all") and k != "ALLF/cross_codec":
                continue
            fa[k] = {"n_triplets": cell["n_triplets"], "n_images": cell["n_images"], "n_responses": cell["n_responses"],
                     "ceiling": r4(cell["ceiling"]),
                     "acc": {s: r4(v["acc"]) for s, v in cell["scorers"].items()},
                     "tie": {s: r4(v["tie_rate"]) for s, v in cell["scorers"].items() if v["tie_rate"] > 0},
                     "d_ssim2": {s: [r4(v["point"]), [r4(x) for x in v["ci_img"]], [r4(x) for x in v["ci_q"]]]
                                 for s, v in cell["deltas_vs_ssim2"].items() if s in ("B", "C") or k.startswith("ALLF")}}
        fc[arm] = fa
    out = {"schema": "rev4-e1-regime-v1", "prereg": "benchmarks/rev4_e1_prereg_2026-09-23.md",
           "seed": E["seed"], "B": E["B"], "decisions": {k: {z: {kk: vv for kk, vv in d.items() if kk != "cells"}
                                                            for z, d in v.items()} for k, v in decisions.items()},
           "decision_cells_primary": {z: [[c["corpus"], c["band"], c["role"], c["best_peer"], c["delta"], c["ci"], c["verdict"]] for c in decisions["primary"][z]["cells"]] for z in ZORDER},
           "decision_cells_columns": ["corpus", "band", "role", "best_peer", "delta", "ci", "verdict"],
           "bands": table, "forced_choice": fc}
    # Split so every committed file stays under 30 KB: main = decisions + primary pairwise bands;
    # _srocc = band SROCC; _fc = forced choice + per-cell decision table.
    base = a.json_out[:-len(".json")]
    srocc = {c: {b: be.pop("srocc") for b, be in tc["bands"].items() if "srocc" in be} for c, tc in table.items()}
    extra = {"decision_cells_primary": out.pop("decision_cells_primary"),
             "decision_cells_columns": out.pop("decision_cells_columns"), "forced_choice": out.pop("forced_choice")}
    out["companions"] = [base + "_srocc.json", base + "_fc.json"]
    json.dump(out, open(a.json_out, "w"), separators=(",", ":"))
    json.dump({"schema": "rev4-e1-regime-srocc-v1", "band_srocc": srocc}, open(base + "_srocc.json", "w"), separators=(",", ":"))
    json.dump({"schema": "rev4-e1-regime-fc-v1", **extra}, open(base + "_fc.json", "w"), separators=(",", ":"))

    # markdown: figure-ready table (metric x corpus x band -> primary stat; zensim deltas vs best)
    L = []
    L.append("| corpus | band (role) | n stim / pairs / refs | best peer (acc) | SSIMULACRA2 | " + " | ".join(ZORDER) + " |")
    L.append("|---|---|---|---|---|" + "---|" * len(ZORDER))
    for c, cr in E["corpora"].items():
        st = primary(c)
        for b, e in cr["bands"].items():
            if c == "konjnd404" and b == "ALL":
                continue
            blk = e.get(st)
            if not blk:
                continue
            M, bp = blk["metrics"], blk["best_peer"]
            row = [CNAME[c], f"{b} ({e['role']})", f"{e['n_stim']} / {e.get('n_pairs', '—')} / {e['n_refs']}",
                   f"{bp} {M[bp]['point']:.4f}", f"{M['ssim2']['point']:.4f}"]
            for z in ZORDER:
                if z not in cr["zensim"]:
                    row.append("—")
                    continue
                d = blk["deltas"][f"{z}-{bp}"]
                v = verdict_of(d["ci"])
                mark = {"LOSS": "**", "WIN": "*", "TIE": ""}[v]
                row.append(f"{M[z]['point']:.4f} {mark}{d['point']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]{mark}")
            L.append("| " + " | ".join(row) + " |")
    arm = F["arms"]["btc_native"]
    for fb, role in (("ALLF", "GLOBAL"), ("HF", "NT"), ("MF", "MID"), ("LF", "LOW")):
        cell = arm[f"{fb}/all"]
        row = [CNAME["fc"], f"{fb} ({role})", f"{cell['n_triplets']} triplets / {cell['n_responses']:.0f} resp / {cell['n_images']} img",
               f"ssim2 {cell['scorers']['ssim2']['acc']:.4f}", f"{cell['scorers']['ssim2']['acc']:.4f}"]
        for z in ZORDER:
            if z not in ("B", "C"):
                row.append("—")
                continue
            d = cell["deltas_vs_ssim2"][z]
            v = verdict_of(d["ci_img"])
            mark = {"LOSS": "**", "WIN": "*", "TIE": ""}[v]
            row.append(f"{cell['scorers'][z]['acc']:.4f} {mark}{d['point']:+.4f} [{d['ci_img'][0]:+.4f}, {d['ci_img'][1]:+.4f}]{mark}")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("| model | primary | band SROCC | family per member | best peer fixed per corpus |")
    L.append("|---|---|---|---|---|")
    for z in ZORDER:
        L.append(f"| {z} | " + " | ".join(
            f"{decisions[k][z]['outcome']} (T={len(decisions[k][z]['T_nt_deficit_units'])}, L={len(decisions[k][z]['L_mid_low_deficit_units'])})"
            for k in ("primary", "sens_bandsrocc", "sens_family_per_member", "sens_bestpeer_fixed_per_corpus")) + " |")
    open(a.md_out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    for z in ZORDER:
        d = decisions["primary"][z]
        print(z, d["outcome"], "T=", d["T_nt_deficit_units"], "L=", d["L_mid_low_deficit_units"], "units=", d["units_with_rows"])


if __name__ == "__main__":
    main()
