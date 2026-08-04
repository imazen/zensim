#!/usr/bin/env python3
"""wave9_tables.py — build the wave-9 endpoint tables from the fulleval JSONs.

Reads ONLY what the owning tools produced (`bake_verdict` -> fulleval JSON,
`freeze_check --profile/--select`) and recomputes NOTHING. Every number in the
results section comes from here so the tables cannot drift from the artifacts.

usage: wave9_tables.py <fulleval-dir> <stem> [<stem> ...]
"""
import json
import subprocess
import sys
from pathlib import Path

FE = Path(sys.argv[1])
STEMS = sys.argv[2:]
FREEZE = Path.home() / "tmp/zensimw9-target/release/freeze_check"
ANN = Path(__file__).resolve().parent.parent / "benchmarks/eval_annotations.json"


def load(stem):
    p = FE / f"{stem}.fulleval.json"
    return json.load(open(p)) if p.exists() else None


def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def floors(stem):
    p = FE / f"{stem}.fulleval.json"
    if not p.exists():
        return None, None
    cmd = [str(FREEZE), "--fulleval", str(p), "--profile", "balanced-2026-08-04"]
    if ANN.exists():
        cmd += ["--annotations", str(ANN)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    n = comp = None
    for line in out.splitlines():
        ls = line.strip()
        # "5 of 8 floors pass"
        if " of 8 floors" in ls:
            n = f"{ls.split()[0]}/8"
        if "balanced_composite" in ls:
            for tok in ls.replace("**", " ").replace("=", " ").split():
                try:
                    comp = float(tok)
                except ValueError:
                    pass
    return n, comp


def fnum(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


rows = []
for s in STEMS:
    d = load(s)
    if d is None:
        rows.append({"stem": s, "missing": True})
        continue
    r = d.get("rank", {})
    n8, comp = floors(s)
    rows.append({
        "stem": s, "missing": False,
        "csiq": g(r, "csiq", "srocc"), "live": g(r, "live", "srocc"),
        "cid22": g(r, "cid22", "srocc"),
        "kadid_signed": g(r, "kadid", "srocc_signed"),
        "kadid_teq": g(r, "kadid", "train_eq_val"),
        "konjnd": g(r, "konjnd", "srocc") or g(r, "konjnd_jpeg_val", "srocc"),
        "nonphoto": g(r, "nonphoto", "srocc"),
        "imazen26": g(r, "imazen26", "srocc"),
        "hfnl": g(r, "hfnlproxy", "per_ref_mean"),
        "m3a": d.get("m3a_coherence"),
        "mono": g(d, "dial", "mono_pct"), "tied": g(d, "dial", "tied_pct"),
        "composite": d.get("composite"), "best_val": g(d, "repro", "best_val"),
        "floors": n8, "bal_comp": comp,
    })

hdr = ("| cell | CSIQ(E1) | LIVE(E1) | CID22(E2) | KADID signed† | KonJND | "
       "nonphoto | imazen26 | HF-NL per-ref | M3a | mono/tied | composite | floors |")
print(hdr)
print("|" + "---|" * 13)
for x in rows:
    if x["missing"]:
        print(f"| {x['stem']} | *(fulleval absent)* |" + " |" * 12)
        continue
    mono = "—" if x["mono"] is None else f"{x['mono']:.4f}"
    tied = "—" if x["tied"] is None else f"{x['tied']:g}"
    dag = "†" if x["kadid_teq"] else ""
    print(f"| {x['stem']} | {fnum(x['csiq'],5)} | {fnum(x['live'],5)} | "
          f"{fnum(x['cid22'],5)} | {fnum(x['kadid_signed'],4)}{dag} | "
          f"{fnum(x['konjnd'],4)} | {fnum(x['nonphoto'],4)} | "
          f"{fnum(x['imazen26'],4)} | {fnum(x['hfnl'],4)} | {fnum(x['m3a'],4)} | "
          f"{mono} / {tied} | {fnum(x['composite'],4)} | {x['floors'] or '—'} |")
print("\n† KADID is train==val in every wave-9 arm (all train on the kadid leg): "
      "a FIT/integrity number, never skill. E1 gates on CSIQ+LIVE, which no arm "
      "trains on; CID22 is held out of training in every arm.")

print("\nE1 (CSIQ ≥ 0.85 AND LIVE ≥ 0.85) / E2 (CID22 ≥ 0.875, DIAGNOSTIC — the "
      "campaign floor stays 0.885):")
for x in rows:
    if x["missing"]:
        continue
    e1 = (x["csiq"] or 0) >= 0.85 and (x["live"] or 0) >= 0.85
    e2 = (x["cid22"] or 0) >= 0.875
    print(f"  {x['stem']:<14} E1 {'PASS' if e1 else 'FAIL'}   "
          f"E2 {'PASS' if e2 else 'FAIL'}")
for arm in ("W9A", "W9B", "W9C"):
    cells = [x for x in rows if x["stem"].startswith(arm) and not x["missing"]]
    if not cells:
        continue
    n1 = sum(1 for x in cells if (x["csiq"] or 0) >= 0.85 and (x["live"] or 0) >= 0.85)
    print(f"  {arm}: E1 holds in {n1} of {len(cells)} seeds"
          f"  (registered bar: ≥ 2 of 3)")
