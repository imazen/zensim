#!/usr/bin/env python3
"""Collate the appendix-U grid: one row per cell, every axis as a DELTA against
that arm's own base cell.

Reads only what the owners produced (`bake_verdict --full-json`, the fit npzs
via `add156_pairs_live.py`'s TSV). No statistic is computed here beyond
differencing two numbers the panel owner already emitted.

The PRIMARY objective column is `d_b9_signed` — `rank.cid22.bands[B9].srocc_signed`.
The absolute `srocc` the board and `freeze_check` F8 consume is carried
alongside as `b9_abs`, because those two can disagree in sign and the appendix
reports both (appendix U P5).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

# rank axes pulled straight from the verdict, plus how they are read
RANK = ["cid22", "konjnd", "nonphoto", "imazen26", "csiq", "live",
        "aic3", "aic4", "sdr25", "tid", "kadid"]


def band(v: dict, b: str, key: str):
    bands = (v.get("rank", {}).get("cid22") or {}).get("bands") or []
    for r in bands:
        if r.get("band") == b:
            return r.get(key), r.get("n")
    return None, None


def read(p: Path) -> dict | None:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def axes(v: dict) -> dict:
    out = {}
    for k in RANK:
        r = v.get("rank", {}).get(k) or {}
        out[k] = r.get("srocc")
    out["kadid_signed"] = ((v.get("rank", {}).get("kadid") or {}).get("srocc_signed"))
    out["hfnl_perref"] = ((v.get("rank", {}).get("hfnlproxy") or {}).get("per_ref_mean"))
    out["b9_signed"], out["b9_n"] = band(v, "B9", "srocc_signed")
    out["b9_abs"], _ = band(v, "B9", "srocc")
    out["b3_signed"], out["b3_n"] = band(v, "B3", "srocc_signed")
    out["b8_signed"], _ = band(v, "B8", "srocc_signed")
    d = v.get("dial") or {}
    out["dial_mono"] = d.get("mono_pct")
    out["dial_range"] = d.get("dynamic_range")
    return out


DELTA = ["b9_signed", "hfnl_perref", "cid22", "konjnd", "nonphoto", "imazen26",
         "csiq", "live", "aic3", "aic4", "sdr25", "tid", "b3_signed",
         "b8_signed", "dial_mono", "dial_range"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--verdicts", required=True, type=Path)
    ap.add_argument("--live", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    base = {}
    for arm in ("A", "B"):
        v = read(a.verdicts / f"U_{arm}{a.tag}_BASE.full.json")
        if v is None:
            raise SystemExit(f"FATAL: missing base verdict for arm {arm}")
        base[arm] = axes(v)

    live = {(r["arm"], r["cell_id"]): r
            for r in csv.DictReader(open(a.live), delimiter="\t")}
    man = list(csv.DictReader(open(a.manifest), delimiter="\t"))

    cols = (["cell_id", "arm", "kind", "block", "indices", "n_cand", "names",
             "status", "cand_w", "n_cand_live", "base_max_shift", "n_active",
             "b9_n", "b9_abs"]
            + [f"d_{k}" for k in DELTA]
            + [k for k in DELTA])
    rows = []
    for m in man:
        arm, cid = m["arm"], m["cell_id"]
        lr = live.get((arm, cid))
        if lr is None:
            continue
        v = read(a.verdicts / f"U_{arm}{a.tag}_{cid}.full.json")
        if v is None:
            continue                                     # not evaluated
        ax = axes(v)
        b = base[arm]
        row = {"cell_id": cid, "arm": arm, "kind": m["kind"], "block": m["block"],
               "indices": m["indices"], "n_cand": m["n_cand"], "names": m["names"],
               "status": lr["status"], "cand_w": lr["cand_w"],
               "n_cand_live": lr["n_cand_live"],
               "base_max_shift": lr["base_max_shift"], "n_active": lr["n_active"],
               "b9_n": ax["b9_n"], "b9_abs": ax["b9_abs"]}
        for k in DELTA:
            x, y = ax.get(k), b.get(k)
            row[k] = x
            row[f"d_{k}"] = (x - y) if (isinstance(x, (int, float))
                                        and isinstance(y, (int, float))
                                        and math.isfinite(x) and math.isfinite(y)) else None
        rows.append(row)

    # base rows, for the record (their deltas are 0 by construction)
    for arm in ("A", "B"):
        r = {"cell_id": "BASE", "arm": arm, "kind": "BASE", "block": "-",
             "indices": "-", "n_cand": 0, "names": "-", "status": "BASE",
             "cand_w": "", "n_cand_live": 0, "base_max_shift": "0",
             "n_active": "", "b9_n": base[arm]["b9_n"], "b9_abs": base[arm]["b9_abs"]}
        for k in DELTA:
            r[k] = base[arm].get(k)
            r[f"d_{k}"] = 0.0
        rows.append(r)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in cols])
    print(f"wrote {a.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
