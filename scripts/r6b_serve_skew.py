#!/usr/bin/env python3
"""R6b: what a rev2 flip costs a bake that is NOT refitted (train/serve skew).

The R6b fits answer "what rank would we get if we shipped arm X and refitted".
This answers the different question a fleet actually faces first: **the
extractor flips, the bakes do not** — until the recalculation lands, every
deployed bake is a revision-1 fit being served revision-2 features.

Two numbers per corpus, both from `bake_verdict` (nothing is computed here):

* **rank** — SROCC on the arm's own eval root vs the revision-1 root;
* **dial** — the per-pair predicted-score shift, reported against this repo's
  own 0.5-point materiality bar and beside the two era shifts already on record
  (extractor era -4.98, decoder era -3.658 points for shipped B).

Scope, stated because it does not generalise: a bake's exposure is its READ SET
intersected with the twelve F17 slots (section 11.3 of the rev2 record). Profile
D reads two of them; Profile CHdr reads all twelve at `identity` and is a 944
bake, so a D-shaped answer says nothing about it.

Usage: r6b_serve_skew.py --bake <b.bin> [--arms bexcess,satexcess] [--out J]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = "/mnt/v/output/zensim/rev2-2026-09-05/r6b"
BASE = "ratio"
CORPORA = ("cid22", "konjnd", "aic3", "csiq", "live", "tid", "kadid")
MATERIALITY_PT = 0.5


def run(bake, arm, corpus, tmp):
    pp = os.path.join(tmp, f"{arm}_{corpus}.tsv")
    js = os.path.join(tmp, f"{arm}_{corpus}.json")
    r = subprocess.run(
        [os.path.join(REPO, "target/release/bake_verdict"), "--bake", bake,
         "--corpora", corpus, "--features-root", os.path.join(ROOT, "evalroot", arm),
         "--per-pair-output", pp, "--json", js, "--output", os.devnull],
        capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(pp):
        return None, None
    a = np.loadtxt(pp, delimiter="\t", skiprows=1)
    srocc = None
    if os.path.exists(js):
        d = json.load(open(js))
        if d.get("corpora"):
            srocc = d["corpora"][0].get("srocc")
    return a, srocc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True)
    ap.add_argument("--arms", default="satexcess")
    ap.add_argument("--corpora", default=",".join(CORPORA))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = {"bake": a.bake, "base_arm": BASE, "materiality_pt": MATERIALITY_PT, "corpora": {}}
    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/tmp")) as tmp:
        print(f"{'corpus':8s} {'arm':10s} {'n':>6s} {'SROCC':>9s} {'dSROCC':>9s} "
              f"{'mean dpt':>10s} {'median':>10s} {'min':>10s} {'max':>10s} {'frac>0.5pt':>11s}")
        for c in a.corpora.split(","):
            base, s0 = run(a.bake, BASE, c, tmp)
            if base is None:
                print(f"{c:8s} -- revision-1 root has no {c}")
                continue
            rep["corpora"][c] = {"n": int(base.shape[0]), "rev1_srocc": s0, "arms": {}}
            print(f"{c:8s} {BASE:10s} {base.shape[0]:6d} "
                  f"{(s0 if s0 is not None else float('nan')):9.5f} {'—':>9s} "
                  f"{'—':>10s} {'—':>10s} {'—':>10s} {'—':>10s} {'—':>11s}")
            for arm in a.arms.split(","):
                got, s1 = run(a.bake, arm, c, tmp)
                if got is None:
                    continue
                d = got[:, 1] - base[:, 1]
                e = {"srocc": s1,
                     "d_srocc": (s1 - s0) if (s1 is not None and s0 is not None) else None,
                     "mean": float(d.mean()), "median": float(np.median(d)),
                     "min": float(d.min()), "max": float(d.max()),
                     "frac_over_materiality": float(np.mean(np.abs(d) > MATERIALITY_PT))}
                rep["corpora"][c]["arms"][arm] = e
                print(f"{'':8s} {arm:10s} {len(d):6d} "
                      f"{(s1 if s1 is not None else float('nan')):9.5f} "
                      f"{(e['d_srocc'] if e['d_srocc'] is not None else float('nan')):+9.5f} "
                      f"{e['mean']:+10.5f} {e['median']:+10.5f} {e['min']:+10.5f} "
                      f"{e['max']:+10.5f} {e['frac_over_materiality']:11.4f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=1)
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
