#!/usr/bin/env python3
"""Severity-ramp monotonicity instrument (§8.19b): for each (ref, dist_type)
ramp in kadis-hdr (levels 1..5, q = type*10 + level), a correct dial must be
non-increasing as severity rises. Reports, per bake: %monotone ramps
(non-increasing with ε-slack), %strict, mean worst inversion (dial pts),
per-type worst offenders.

  usage: severity_ramp_monotonicity.py BAKE [BAKE ...]
         [--features .../zensim_features_pulinear.parquet] [--eps 0.5]

Signed types (7/18/25, dist_param signed) are U-shaped BY DESIGN — excluded
from the monotone denominator and reported separately.
"""
import argparse
import os
import numpy as np
import pyarrow.parquet as pq
import importlib.util

spec = importlib.util.spec_from_file_location(
    "xdi", os.path.join(os.path.dirname(__file__), "upiq_crossdomain_instrument.py"))
src = open(os.path.join(os.path.dirname(__file__), "upiq_crossdomain_instrument.py")).read()
ns = {}
import sys
_argv = sys.argv
sys.argv = ["xdi"]  # the exec'd head runs its own argparse — feed it nothing
exec(compile(src.split("# ---- SDR half")[0], "xdi_head", "exec"), ns)
sys.argv = _argv
score_bake = ns["score_bake"]

ap = argparse.ArgumentParser()
ap.add_argument("bakes", nargs="+")
ap.add_argument("--features", default="/mnt/v/output/zenmetrics/datagen-2026-07-12-hdr-kadis/sidecars/kadis-hdr/zensim_features_pulinear.parquet")
ap.add_argument("--eps", type=float, default=0.5, help="tie slack in dial points")
a = ap.parse_args()

SIGNED = {7, 18, 25}
# Signed U-shaped types: two half-ramps as `level` sequences from the identity
# (min |dist_param|) outward, plus the identity level. Matches
# eval_report.rs::signed_fold_arms (kadis-distort per-level params are fixed):
#   d7  color_saturate_hsv : +.40 +.20 +.10  .00 −.40  (identity L4)
#   d18 mean_shift         : +.15 +.08  .00 −.08 −.15  (identity L3)
#   d25 contrast           : +.30 +.15  .00 −.40 −.60  (identity L3)
SIGNED_FOLD = {7: ([4, 3, 2, 1], [4, 5], 4), 18: ([3, 2, 1], [3, 4, 5], 3),
               25: ([3, 2, 1], [3, 4, 5], 3)}
t = pq.read_table(a.features)
refs = [os.path.basename(x) for x in t["image_path"].to_pylist()]
qs = [int(float(x)) for x in t["q"].to_pylist()]
types = [q // 10 for q in qs]
levels = [q % 10 for q in qs]

for bake in a.bakes:
    dial = score_bake(bake, t)
    ramps = {}
    for i, (r, ty, lv) in enumerate(zip(refs, types, levels)):
        ramps.setdefault((r, ty), {})[lv] = dial[i]
    mono = strict = tot = 0
    inv_mags = []
    per_type = {}
    n_signed = 0
    sig_mono = sig_tot = 0
    sig_per_type = {}
    sig_ident = {}
    for (r, ty), lv in ramps.items():
        if len(lv) < 5:
            continue
        if ty in SIGNED:
            # Fold the U at its identity: each arm must be non-increasing as
            # |dist_param| rises (see SIGNED_FOLD). Signed arms are reported
            # separately so the unsigned headline stays comparable to history.
            n_signed += 1
            pos, neg, idlv = SIGNED_FOLD[ty]
            for arm in (pos, neg):
                seq = [lv[k] for k in arm if k in lv]
                if len(seq) < 2:
                    continue
                ok = bool((np.diff(seq) <= a.eps).all())
                sig_tot += 1
                sig_mono += ok
                d = sig_per_type.setdefault(ty, [0, 0])
                d[0] += ok
                d[1] += 1
            if idlv in lv:
                sig_ident.setdefault(ty, []).append(lv[idlv])
            continue
        seq = [lv[k] for k in sorted(lv)]
        diffs = np.diff(seq)
        tot += 1
        ok = bool((diffs <= a.eps).all())
        st = bool((diffs < 0).all())
        mono += ok
        strict += st
        d = per_type.setdefault(ty, [0, 0])
        d[0] += ok
        d[1] += 1
        if not ok:
            inv_mags.append(float(diffs.max()))
    worst = sorted(((v[0] / v[1], ty, v[1]) for ty, v in per_type.items()))[:5]
    print(f"{os.path.basename(bake)}: unsigned ramps={tot}; signed folded={n_signed} ({sig_tot} half-ramps)")
    print(f"  monotone(eps={a.eps}): {100*mono/tot:.1f}%   strict: {100*strict/tot:.1f}%   "
          f"mean worst-inversion: {np.mean(inv_mags) if inv_mags else 0:.1f} dial pts")
    print("  worst unsigned types: " + ", ".join(f"d{ty}={100*frac:.0f}% (n={n})" for frac, ty, n in worst))
    if sig_tot:
        print(f"  signed folded monotone: {100*sig_mono/sig_tot:.1f}% ({sig_tot} half-ramps)")
        for ty in sorted(sig_per_type):
            c = sig_per_type[ty]
            idm = np.mean(sig_ident.get(ty, [float('nan')]))
            print(f"    d{ty}: {100*c[0]/c[1]:.0f}% ({c[1]} arms)  identity dial={idm:.1f} (→100)")
