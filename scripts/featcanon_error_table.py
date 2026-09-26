#!/usr/bin/env python3
"""featcanon error-vs-exact tables, recomputed from the lane's dumped vectors.

The reproduction the featcanon lane never committed (REVIEW_FEATCANON.md D9);
logic identical to the review's `errtab.py` / `accsplit.py`, which reproduce the
lane's `errtable_family.tsv` exactly. Inputs are the `tier_audit_features`
dumps (`ZENSIM_FEATCANON_DUMP=<dir>`, one `<mode>_<tier>_<pair>.f64bin` of 1825
little-endian f64 per run plus `names_<pair>.tsv`), for the modes
off/c32/c64/neum/exact. Join key: (pair label, slot index).

  table  FAMILIES   per family and candidate: max/median relative error and
                    max absolute error vs the exact arm (`errtab`)
  split             per family: accumulation error A32 = |c32-c64|/|exact| and
                    Av3 = |off_v3-c64|/|exact| against element error
                    E = |c64-exact|/|exact| (`accsplit`)

Example: featcanon_error_table.py --vecs /var/tmp/featcanon/lane-scratch/vecs table basic,csfw,tailhist,append
"""

import argparse
import glob
import os

import numpy as np

WIDTH = 1825


def load_names(vecs):
    pairs = sorted({os.path.basename(p)[len("names_") : -4] for p in glob.glob(f"{vecs}/names_*.tsv")})
    assert pairs, f"no names_*.tsv under {vecs}"
    fam = None
    for p in pairs:
        rows = [line.rstrip("\n").split("\t") for line in open(f"{vecs}/names_{p}.tsv")]
        assert [int(r[0]) for r in rows] == list(range(WIDTH)), p
        f = [r[1] for r in rows]
        fam = fam or f
        assert fam == f, f"family map differs for {p}"
    return pairs, np.array(fam)


def load(vecs, mode, tier, pair):
    a = np.fromfile(f"{vecs}/{mode}_{tier}_{pair}.f64bin", dtype="<f8")
    assert a.size == WIDTH, (mode, tier, pair, a.size)
    return a


def table(vecs, pairs, fam, want):
    cands = [("off", "v3"), ("off", "v4"), ("off", "scalar"), ("c32", "v3"), ("c64", "v3"), ("neum", "v3")]
    for F in want:
        m = fam == F
        for mode, tier in cands:
            rel, ab = [], []
            for p in pairs:
                x = load(vecs, mode, tier, p)[m]
                e = load(vecs, "exact", "v3", p)[m]
                d = np.abs(x - e)
                ab.append(d)
                rel.append(d / np.maximum(np.abs(e), 1e-300))
            rel, ab = np.concatenate(rel), np.concatenate(ab)
            print(
                f"{F:10s} {mode}_{tier:7s} n={rel.size:5d} max_rel={rel.max():.3e} "
                f"med_rel={np.median(rel):.3e} max_abs={ab.max():.3e}"
            )


def split(vecs, pairs, fam):
    print(
        f"{'family':10s} {'med A32':>9s} {'max A32':>9s} {'med Av3':>9s} {'max Av3':>9s} "
        f"{'med E':>9s} {'max E':>9s}  frac(A32>E) frac(Av3>E)"
    )
    for F in dict.fromkeys(fam):
        m = fam == F
        a32, av3, el = [], [], []
        for p in pairs:
            e = load(vecs, "exact", "v3", p)[m]
            den = np.maximum(np.abs(e), 1e-300)
            c64 = load(vecs, "c64", "v3", p)[m]
            a32.append(np.abs(load(vecs, "c32", "v3", p)[m] - c64) / den)
            av3.append(np.abs(load(vecs, "off", "v3", p)[m] - c64) / den)
            el.append(np.abs(c64 - e) / den)
        a32, av3, el = map(np.concatenate, (a32, av3, el))
        ok = el < 1e-3  # drop degenerate / boundary slots from the maxima
        print(
            f"{F:10s} {np.median(a32):9.2e} {a32[ok].max():9.2e} {np.median(av3):9.2e} "
            f"{av3[ok].max():9.2e} {np.median(el):9.2e} {el[ok].max():9.2e}  "
            f"{np.mean(a32 > el):.3f} {np.mean(av3 > el):.3f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vecs", required=True, help="dump directory (ZENSIM_FEATCANON_DUMP)")
    ap.add_argument("--pairs", type=int, default=11, help="expected pair count (the lane's audit: 11)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("families", nargs="?", default="", help="comma list; default every family")
    sub.add_parser("split")
    a = ap.parse_args()
    pairs, fam = load_names(a.vecs)
    assert len(pairs) == a.pairs, pairs
    if a.cmd == "table":
        table(a.vecs, pairs, fam, a.families.split(",") if a.families else list(dict.fromkeys(fam)))
    else:
        split(a.vecs, pairs, fam)


if __name__ == "__main__":
    main()
