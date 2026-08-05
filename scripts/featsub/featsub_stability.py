#!/usr/bin/env python3
"""Appendix-J Phase B: the stability-selected input subset over the
lambda x seed grid.

Input is the concatenation of `bake_contrib --live-mask` TSVs — the Rust owner
emits, per bake, whether each layer-0 input row is exactly zero. This script
only counts and thresholds:

    an input is SELECTED  <=>  it is live in >= --min-frac of the runs

which is the robust answer given the measured seed noise (a single lambda x
seed run's live set is one draw; the intersection-ish rule is what survives).
No statistic is computed here, and no bake is parsed here.
"""

import argparse
import os
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-tsv", nargs="+", required=True,
                    help="one or more `bake_contrib --live-mask` TSVs")
    ap.add_argument("--min-frac", type=float, default=0.8)
    ap.add_argument("--out", required=True, help="index file for --keep-features")
    ap.add_argument("--report", help="optional per-run TSV summary")
    args = ap.parse_args()

    per_bake = {}
    width = 0
    for path in args.mask_tsv:
        with open(path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                bake, idx, live = line.split("\t")[:3]
                per_bake.setdefault(bake, set())
                width = max(width, int(idx) + 1)
                if live.strip() == "1":
                    per_bake[bake].add(int(idx))
    if not per_bake:
        print("FATAL: no rows in the mask TSVs", file=sys.stderr)
        return 1

    counts = Counter()
    for s in per_bake.values():
        counts.update(s)
    n = len(per_bake)
    thresh = args.min_frac * n
    selected = sorted(i for i, c in counts.items() if c >= thresh)
    if not selected:
        print(f"FATAL: 0 inputs cleared {thresh:.1f}/{n} runs", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(f"# stability selection over {n} runs, min_frac {args.min_frac} "
                 f"(>= {thresh:.1f} runs), width {width}\n")
        fh.write(f"# runs: {', '.join(sorted(per_bake))}\n")
        fh.write(f"# selected {len(selected)} of {width}\n")
        fh.write(",".join(str(i) for i in selected) + "\n")
    print(f"runs={n} width={width} selected={len(selected)} -> {args.out}")

    lines = ["run\tn_live"] + [f"{b}\t{len(per_bake[b])}" for b in sorted(per_bake)]
    # how many inputs are live in exactly k runs — the shape of the agreement
    hist = Counter(counts.values())
    lines.append("#")
    lines.append("live_in_k_runs\tn_inputs")
    for k in sorted(hist):
        lines.append(f"{k}\t{hist[k]}")
    lines.append(f"0\t{width - len(counts)}")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.report:
        open(args.report, "w").write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
