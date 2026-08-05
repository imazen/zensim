#!/usr/bin/env python3
"""Appendix-J cross-phase check (J.3 step 5): overlap between Phase A's
contribution-ranked top-K and Phase B's learned/stability-selected subset,
at matched size. Pure set arithmetic over committed index files."""

import argparse


def read_idx(path):
    for line in open(path):
        line = line.split("#")[0].strip()
        if line:
            return sorted(int(x) for x in line.split(","))
    raise ValueError(f"{path}: no index line")


def fam(i):
    if i < 156:
        return "v1fold156"
    if i < 372:
        return "zeros"
    if i < 720:
        return "v2-348"
    if i < 924:
        return "append204"
    return "tail20"


def mix(s):
    out = {}
    for i in s:
        out[fam(i)] = out.get(fam(i), 0) + 1
    return " ".join(f"{k}:{v}" for k, v in sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked", required=True, help="idx/ranked.tsv (frozen order)")
    ap.add_argument("--selected", required=True, help="stability-selected .idx")
    args = ap.parse_args()

    order = []
    with open(args.ranked) as fh:
        next(fh)
        for line in fh:
            order.append(int(line.split("\t")[1]))
    sel = set(read_idx(args.selected))
    k = len(sel)
    top = set(order[:k])
    inter = top & sel
    union = top | sel
    print(f"selected |S|={k}  top-{k} by contribution")
    print(f"|A∩B|={len(inter)}  |A∪B|={len(union)}  Jaccard={len(inter)/len(union):.4f}"
          f"  overlap@k={len(inter)/k:.4f}")
    print(f"top-{k} family mix : {mix(top)}")
    print(f"selected family mix: {mix(sel)}")
    print(f"only-in-topK   : {mix(top - sel)}")
    print(f"only-in-selected: {mix(sel - top)}")


if __name__ == "__main__":
    main()
