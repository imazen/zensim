#!/usr/bin/env python3
"""Appendix-R results tables (sota944 campaign).

Reads `bake_verdict --regime 944` full-JSONs + fulleval JSONs (the owners of
every statistic) and does NOTHING but select fields and apply the frozen
appendix-R bars:

  one-artifact bar (judged PACKED): hfnl/ref >= 0.75  AND  cid22 >= 0.875
      AND dial mono >= 0.93 AND tied <= 0.05 AND dynamic_range >= 30
  R1 "dial recovered" = the dial part of the bar
  R1 "HF-NL holds"    = packed hfnl >= 0.75 AND |raw->packed delta| <= 0.02

Emits:
  benchmarks/sparsehf/r1_dial_recovery_<date>.tsv   (raw -> dial -> packed)
  benchmarks/sparsehf/r2_ladder_<date>.tsv          (CS cells vs matched GL siblings + bar)
No statistic is recomputed; every number is read from a stored JSON.
"""

import json
import os
import subprocess
import sys
from datetime import date

VD = "/mnt/v/output/zensim/bakes/sota944/verdicts"
FE = "/mnt/v/output/zensim/reports/fulleval"
BK = "/mnt/v/output/zensim/bakes/sparsehf"
FB = "/mnt/v/output/zensim/bakes/featsub"
HOME = os.path.expanduser("~")
INSTR = f"{HOME}/tmp/sparsehf"  # instrument (_dial) verdicts, per R.2 step 4

R1_CELLS = ["GL0p3_s2503", "GL1_s2503", "GL2_s2503", "PILOT1_s2501"]
R2_LAMBDAS = ["0p3", "1", "2", "4"]
SEEDS = ["2501", "2503"]


def j(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def pick(d, *path):
    for k in path:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def row(d, fe=None, spec=None, bake=None):
    if d is None:
        return None
    r = {
        "cid22": pick(d, "rank", "cid22", "srocc"),
        "konjnd": pick(d, "rank", "konjnd", "srocc"),
        "nonphoto": pick(d, "rank", "nonphoto", "srocc"),
        "hfnl_ref": pick(d, "rank", "hfnlproxy", "per_ref_mean"),
        "kadid": pick(d, "rank", "kadid", "srocc"),
        "csiq": pick(d, "rank", "csiq", "srocc"),
        "live": pick(d, "rank", "live", "srocc"),
        "tid": pick(d, "rank", "tid", "srocc"),
        "sdr25": pick(d, "rank", "sdr25", "srocc"),
        "aic3": pick(d, "rank", "aic3", "srocc"),
        "aic4": pick(d, "rank", "aic4", "srocc"),
        "imazen26": pick(d, "rank", "imazen26", "srocc"),
        "mono": pick(d, "dial", "mono_pct"),
        "tied": pick(d, "dial", "tied_pct"),
        "range": pick(d, "dial", "dynamic_range"),
        "p5": pick(d, "dial", "p5"),
        "p95": pick(d, "dial", "p95"),
        "composite": pick(d, "composite"),
    }
    if fe is not None:
        r["m3a"] = pick(fe, "m3a_coherence")
    if spec is not None:
        r["live_l0"] = pick(spec, "live_l0_rows")
    if bake is not None and os.path.isfile(bake):
        r["bytes"] = os.path.getsize(bake)
    return r


def bar(r):
    """The frozen one-artifact bar on a packed row -> (pass, reasons)."""
    if r is None:
        return None, ["missing"]
    checks = [
        ("hfnl>=0.75", r["hfnl_ref"] is not None and r["hfnl_ref"] >= 0.75),
        ("cid22>=0.875", r["cid22"] is not None and r["cid22"] >= 0.875),
        ("mono>=0.93", r["mono"] is not None and r["mono"] >= 0.93),
        ("tied<=0.05", r["tied"] is not None and r["tied"] <= 0.05),
        ("range>=30", r["range"] is not None and r["range"] >= 30.0),
    ]
    return all(ok for _, ok in checks), [name for name, ok in checks if not ok]


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def emit(path, header, rows, meta_cmd):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(fmt(x) for x in r) + "\n")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    ).stdout.strip()
    with open(path.replace(".tsv", ".meta"), "w") as fh:
        fh.write(
            f"git_commit: {commit}\ncommand: {meta_cmd}\n"
            f"inputs: verdicts {VD}/(FS_*|R1_*|CS*).full.json, fullevals {FE}/*.fulleval.json,"
            f" instrument dial verdicts {INSTR}/R1_*_dial.full.json (not campaign cells)\n"
            "stats: none recomputed — all fields read from bake_verdict/run_full_eval outputs\n"
        )
    print(f"wrote {path}")


def main():
    today = date.today().isoformat()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "sparsehf")
    out_dir = os.path.normpath(out_dir)

    # ---- R1: raw -> dial -> packed ----
    cols = ["cell", "stage", "cid22", "konjnd", "nonphoto", "hfnl_ref", "mono", "tied",
            "range", "p5", "p95", "composite", "m3a", "bytes", "bar", "bar_miss"]
    rows = []
    for c in R1_CELLS:
        stages = [
            ("raw", j(f"{VD}/FS_{c}.full.json"), None, None),
            ("dial", j(f"{INSTR}/R1_{c}_dial.full.json"), None, f"{BK}/R1_{c}_dial.bin"),
            ("packed", j(f"{VD}/R1_{c}_packed.full.json"), j(f"{FE}/R1_{c}_packed.fulleval.json"),
             f"{BK}/R1_{c}_packed.bin"),
        ]
        for name, d, fe, bake in stages:
            r = row(d, fe, None, bake)
            if r is None:
                rows.append([c, name] + ["—"] * (len(cols) - 2))
                continue
            ok, miss = bar(r) if name == "packed" else (None, [])
            rows.append([c, name, r["cid22"], r["konjnd"], r["nonphoto"], r["hfnl_ref"],
                         r["mono"], r["tied"], r["range"], r["p5"], r["p95"], r["composite"],
                         r.get("m3a"), r.get("bytes"),
                         ("PASS" if ok else "FAIL") if ok is not None else "—",
                         ",".join(miss) if miss else "—"])
    emit(f"{out_dir}/r1_dial_recovery_{today}.tsv", cols, rows, "sparsehf_tables.py (R1)")

    # ---- R2: CS cells + matched GL siblings ----
    cols2 = ["cell", "stage", "cid22", "konjnd", "nonphoto", "hfnl_ref", "kadid", "csiq",
             "live", "tid", "sdr25", "mono", "tied", "range", "composite", "live_l0",
             "m3a", "bytes", "bar", "bar_miss",
             "d_cid22_vs_GL", "d_hfnl_vs_GL", "d_konjnd_vs_GL", "d_nonphoto_vs_GL", "d_composite_vs_GL"]
    rows2 = []
    for lam in R2_LAMBDAS:
        for s in SEEDS:
            cs = f"CS{lam}_s{s}"
            gl = j(f"{VD}/FS_GL{lam}_s{s}.full.json")
            glr = row(gl)
            for name, d, fe, spec, bake in [
                ("raw", j(f"{VD}/{cs}.full.json"), j(f"{FE}/{cs}.fulleval.json"),
                 j(f"{BK}/{cs}.bin.spec.json"), f"{BK}/{cs}.bin"),
                ("packed", j(f"{VD}/R1_{cs}_packed.full.json"), j(f"{FE}/R1_{cs}_packed.fulleval.json"),
                 None, f"{BK}/R1_{cs}_packed.bin"),
            ]:
                r = row(d, fe, spec, bake)
                if r is None:
                    rows2.append([cs, name] + ["—"] * (len(cols2) - 2))
                    continue
                ok, miss = bar(r) if name == "packed" else (None, [])
                deltas = ["—"] * 5
                if glr is not None and name == "raw":
                    deltas = [r["cid22"] - glr["cid22"], r["hfnl_ref"] - glr["hfnl_ref"],
                              r["konjnd"] - glr["konjnd"], r["nonphoto"] - glr["nonphoto"],
                              r["composite"] - glr["composite"]]
                rows2.append([cs, name, r["cid22"], r["konjnd"], r["nonphoto"], r["hfnl_ref"],
                              r["kadid"], r["csiq"], r["live"], r["tid"], r["sdr25"],
                              r["mono"], r["tied"], r["range"], r["composite"], r.get("live_l0"),
                              r.get("m3a"), r.get("bytes"),
                              ("PASS" if ok else "FAIL") if ok is not None else "—",
                              ",".join(miss) if miss else "—"] + deltas)
    emit(f"{out_dir}/r2_ladder_{today}.tsv", cols2, rows2, "sparsehf_tables.py (R2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
