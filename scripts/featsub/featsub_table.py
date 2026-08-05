#!/usr/bin/env python3
"""Appendix-J results table: the K sweep (or the lambda sweep) against the
frozen noise band.

Reads `bake_verdict --regime 944` full-JSONs (the owner of every statistic
here) and does NOTHING but select, average over seeds, and apply the frozen
band rule from appendix J.2:

    sd_944  = sample sd over the k=3 full-944 baseline seeds, per axis
    band    = +- 2 * sd_944
    OUTSIDE <=> |mean_arm - mean_944| > band  AND  both arm seeds fall on the
                same side of the baseline mean
    anything else is INSIDE and is not a finding.

No statistic is recomputed: SROCC/dial/etc. come straight from the verdict.
"""

import argparse
import glob
import json
import os
import statistics
import sys

# axis -> (json path, higher-is-better)
AXES = [
    ("cid22", ("rank", "cid22", "srocc"), True),
    ("konjnd", ("rank", "konjnd", "srocc"), True),
    ("nonphoto", ("rank", "nonphoto", "srocc"), True),
    ("imazen26", ("rank", "imazen26", "srocc"), True),
    ("kadid", ("rank", "kadid", "srocc"), True),
    ("csiq", ("rank", "csiq", "srocc"), True),
    ("live", ("rank", "live", "srocc"), True),
    ("tid", ("rank", "tid", "srocc"), True),
    ("sdr25", ("rank", "sdr25", "srocc"), True),
    ("hfnlproxy", ("rank", "hfnlproxy", "per_ref_mean"), True),
    ("dial_mono", ("dial", "mono_pct"), True),
    ("dial_tied", ("dial", "tied_pct"), False),
    ("composite", ("composite",), True),
    ("best_val", ("repro", "best_val"), True),
    ("n_live", ("_live",), None),
]


def dig(d, path):
    for k in path:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def live_rows(v):
    """Exactly-zero layer-1 input rows are what `pack` prunes; the spec
    sidecar records the count the trainer measured on the produced bake."""
    spec = (v.get("bake") or "") + ".spec.json"
    if os.path.exists(spec):
        try:
            n = json.load(open(spec)).get("live_l0_rows")
            if n is not None:
                return float(n)
        except (OSError, ValueError):
            pass
    return float(v.get("n_inputs") or 0)


def load(verdict_dir, arm):
    out = []
    for p in sorted(glob.glob(os.path.join(verdict_dir, f"FS_{arm}_s*.full.json"))):
        if p.endswith("_packed.full.json"):
            continue
        v = json.load(open(p))
        row = {"seed": os.path.basename(p).split("_s")[-1].split(".")[0]}
        for name, path, _ in AXES:
            row[name] = live_rows(v) if path == ("_live",) else dig(v, path)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", default="/mnt/v/output/zensim/bakes/sota944/verdicts")
    ap.add_argument("--baseline", default="K944")
    ap.add_argument("--arms", required=True, help="comma-separated arm names")
    ap.add_argument("--out", help="TSV output path")
    args = ap.parse_args()

    base = load(args.verdicts, args.baseline)
    if len(base) < 2:
        print(f"FATAL: baseline {args.baseline} has {len(base)} seeds (need >= 2 for a band)",
              file=sys.stderr)
        return 1
    band, bmean = {}, {}
    for name, _, _ in AXES:
        vals = [r[name] for r in base if r[name] is not None]
        if not vals:
            continue
        bmean[name] = statistics.fmean(vals)
        band[name] = 2.0 * (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    lines = []
    hdr = ["arm", "n_seeds"] + [f"{n}" for n, _, _ in AXES] 
    lines.append("\t".join(hdr))
    detail = [f"# baseline {args.baseline}: n={len(base)} seeds " +
              " ".join(f"{n}={bmean.get(n, float('nan')):.4f}+-{band.get(n, 0)/2:.4f}(sd)"
                       for n, _, _ in AXES if n in bmean)]

    calls = {}
    for arm in [args.baseline] + args.arms.split(","):
        rows = load(args.verdicts, arm)
        if not rows:
            continue
        cells = [arm, str(len(rows))]
        for name, _, higher in AXES:
            vals = [r[name] for r in rows if r[name] is not None]
            if not vals:
                cells.append("-")
                continue
            m = statistics.fmean(vals)
            if arm == args.baseline or name not in bmean or higher is None:
                cells.append(f"{m:.4f}")
                continue
            d = m - bmean[name]
            same_side = all((v - bmean[name]) * d > 0 for v in vals) and len(vals) > 1
            outside = abs(d) > band[name] and same_side and band[name] > 0
            if outside:
                good = (d > 0) == bool(higher)
                calls.setdefault(arm, []).append((name, d, "BETTER" if good else "WORSE"))
            cells.append(f"{m:.4f}{'*' if outside else ''}")
        lines.append("\t".join(cells))

    detail.append("# '*' = OUTSIDE the +-2*sd_944 band AND both seeds on the same side")
    for arm, cs in calls.items():
        detail.append("# " + arm + " OUTSIDE: " +
                      ", ".join(f"{n} {d:+.4f} {w}" for n, d, w in cs))
    text = "\n".join(detail + lines) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
