#!/usr/bin/env python3
"""Post-pipeline LYB analysis for the dst-activity adjudication:
(1) drift check fresh-OFF vs the July master (ffmpeg determinism + toggle-off
    byte-stability on 960 real 1080p pairs);
(2) paired GAIN-lane deltas ON vs OFF (same frames);
(3) hand off SROCC computation to the committed eval script per arm.
"""
import csv, os, subprocess, sys

BD = os.path.expanduser("~/tmp/bandvis-dst")
JULY = os.path.expanduser("~/tmp/lyb-out/lyb_foldapp2_master.csv")
OFF = os.path.join(BD, "lyb-off/lyb_foldapp2_master.csv")
ON = os.path.join(BD, "lyb-on/lyb_foldapp2_master.csv")
EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bandvis_lyb_eval_2026-07-28.py")

# (1) drift check vs July (byte-level, then value-level fallback detail)
july = open(JULY).read()
off = open(OFF).read()
if july == off:
    print("DRIFT CHECK: fresh-OFF master BYTE-IDENTICAL to the July master "
          "(ffmpeg frame extraction deterministic AND toggle-off math "
          "byte-stable on 960 real 1080p pairs)")
else:
    ja = july.splitlines(); oa = off.splitlines()
    print(f"DRIFT CHECK: NOT byte-identical (july {len(ja)} lines, fresh {len(oa)})")
    ndiff = sum(1 for a, b in zip(ja, oa) if a != b)
    print(f"  differing lines: {ndiff}/{min(len(ja), len(oa))}")
    # column-level triage on the first few differing rows
    shown = 0
    for a, b in zip(ja[1:], oa[1:]):
        if a != b and shown < 3:
            ca, cb = a.split(","), b.split(",")
            cols = [i for i, (x, y) in enumerate(zip(ca, cb)) if x != y]
            print(f"  row {ca[1]}: {len(cols)} cols differ, first {cols[:6]}")
            shown += 1

# (2) paired per-row GAIN deltas ON vs OFF
def load(path):
    with open(path) as f:
        rd = csv.reader(f)
        hdr = next(rd)
        return hdr, {int(float(r[1])): r for r in rd}

hdr, offr = load(OFF)
_, onr = load(ON)
gain_cols = [hdr.index(f"f{924 + s * 5}") for s in range(4)]
loss_cols = [hdr.index(f"f{924 + s * 5 + 1}") for s in range(4)]
other = [i for i in range(2, len(hdr)) if i not in gain_cols]
n_other_diff = 0
n_gain_diff = 0
for rid, ro in offr.items():
    rn = onr[rid]
    for i in other:
        if ro[i] != rn[i]:
            n_other_diff += 1
            break
    if any(ro[i] != rn[i] for i in gain_cols):
        n_gain_diff += 1
print(f"PAIRED ARMS: rows with any non-GAIN column differing: {n_other_diff}/960 "
      f"(must be 0); rows with GAIN differing: {n_gain_diff}/960")

import statistics as st
for s, c in enumerate(gain_cols):
    do = [float(offr[r][c]) for r in offr]
    dn = [float(onr[r][c]) for r in offr]
    print(f"  GAIN s{s}: OFF med {st.median(do):.5f} ON med {st.median(dn):.5f} "
          f"med ratio {st.median([b / a if a > 1e-12 else float('nan') for a, b in zip(do, dn) if a > 1e-12]):.3f}")

# (3) committed eval per arm
for arm, d in (("OFF", os.path.join(BD, "lyb-off")), ("ON", os.path.join(BD, "lyb-on"))):
    print(f"\n================ EVAL ARM {arm} ================")
    sys.stdout.flush()
    subprocess.run(["python3", EVAL, "--out-dir", d], check=True)
