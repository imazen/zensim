#!/usr/bin/env python3
"""Tabulate the era-2 rank-preservation roster: per-corpus signed SROCC on a
control arm and one or more candidate arms, the delta, and the composite,
against the registered bar (era-2 §21.1, registered 2026-08-31 BEFORE any
candidate existed):

    PASS iff no corpus loses more than 0.005 SROCC and the composite does
    not fall.

This script computes NO statistic. Every number it prints is read verbatim
from a `bake_verdict --full-json` fulleval (whose stats come from
`zensim_validate::panel` -> `zenstats`). It only arranges them.

Several corpora carry a distortion-oriented target and are canonically
NEGATIVE (aic4, kadid, konjnd, sdr25), so deltas are taken in MAGNITUDE, and
a SIGN FLIP is reported as a failure regardless of magnitude — the same
convention the blur/radius lane's analyze_quality.py used.

usage: era2_rank_table.py <verdict-root> <control-arm> <cand-arm>[,<cand-arm>...]
       [--models a,b,c] [--tsv out.tsv]
"""
import json, os, sys

BAR = 0.005
MODELS_DEFAULT = ["B", "C944", "WLIN7b_g020", "WLIN7b_g025", "ADD156", "BHdr"]


def load(root, arm, m):
    p = os.path.join(root, f"verdicts-{arm}", f"{m}.fulleval.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    root, ctrl, cands = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    models = MODELS_DEFAULT
    tsv = None
    a = sys.argv[4:]
    while a:
        if a[0] == "--models":
            models = a[1].split(","); a = a[2:]
        elif a[0] == "--tsv":
            tsv = a[1]; a = a[2:]
        else:
            sys.exit(f"unknown arg {a[0]}")
    rows = []
    summary = []
    for m in models:
        b = load(root, ctrl, m)
        if not b:
            print(f"\n### {m}: ABSENT on control arm {ctrl} — not measured\n"); continue
        corpora = sorted(b["rank"])
        nin = b.get("n_inputs") or b.get("model", {}).get("n_inputs")
        print(f"\n### {m}   n_inputs={nin}   control={ctrl}\n")
        hdr = f"{'corpus':>8} {ctrl:>11}" + "".join(f"{c:>26}" for c in cands)
        print(hdr)
        worst = {c: 0.0 for c in cands}
        flips = {c: [] for c in cands}
        for cp in corpora:
            bv = b["rank"][cp].get("srocc_signed")
            if bv is None:
                continue
            cells = []
            for c in cands:
                d = load(root, c, m)
                v = d["rank"][cp].get("srocc_signed") if d and cp in d["rank"] else None
                if v is None:
                    cells.append(f"{'--':>26}"); continue
                flip = (v < 0) != (bv < 0)
                delta = abs(v) - abs(bv)      # positive = gained magnitude
                worst[c] = min(worst[c], delta)
                if flip:
                    flips[c].append(cp)
                cells.append(f"{v:>+11.5f} ({delta:+.5f}){'!FLIP' if flip else '':>5}")
                rows.append((m, cp, c, bv, v, delta, flip))
            print(f"{cp:>8} {bv:>+11.5f}" + "".join(cells))
        cb = b.get("composite")
        cells = []
        for c in cands:
            d = load(root, c, m)
            cv = d.get("composite") if d else None
            cells.append(f"{cv:>+11.5f} ({cv - cb:+.5f})     " if cv is not None else f"{'--':>26}")
        print(f"{'COMPOSITE':>8}"[:8].rjust(8) + f" {cb:>+11.5f}" + "".join(cells))
        cells = []
        for c in cands:
            d = load(root, c, m)
            if not d:
                cells.append(f"{'--':>26}"); continue
            ok = worst[c] >= -BAR and d.get("composite", 0.0) >= cb and not flips[c]
            cells.append(f"{'PASS' if ok else 'FAIL':>13} (worst {worst[c]:+.5f})")
            summary.append((m, c, ok, worst[c], d.get("composite") - cb, ",".join(flips[c])))
        print(f"{'BAR':>8} {'':>11}" + "".join(cells))
    print("\n### Verdict against the registered bar "
          "(no corpus loses > 0.005 in magnitude; composite does not fall; no sign flip)\n")
    print(f"{'model':>13} {'arm':>10} {'worst corpus delta':>20} {'composite delta':>17} {'flips':>8} {'':>6}")
    for m, c, ok, w, cd, fl in summary:
        print(f"{m:>13} {c:>10} {w:>+20.5f} {cd:>+17.5f} {fl if fl else '-':>8} {'PASS' if ok else 'FAIL':>6}")
    if tsv:
        with open(tsv, "w") as fh:
            fh.write("model\tcorpus\tarm\tctrl_srocc_signed\tarm_srocc_signed\tdelta_magnitude\tsign_flip\n")
            for r in rows:
                fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.10f}\t{r[4]:.10f}\t{r[5]:.10f}\t{int(r[6])}\n")
        print(f"\nwrote {tsv}")


if __name__ == "__main__":
    main()
