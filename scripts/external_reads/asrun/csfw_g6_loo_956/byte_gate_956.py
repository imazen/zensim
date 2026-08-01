#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/csfw-g6-loo-2026-07-29/harness/byte_gate_956.py
# sha256(source): 215bed2587f5fdde3a0876a860b80b59cea3943869f62090af9a66fc69314b66
# build_commit:  7bfd511de78f85e8fcd618df15716ca56575bb60
# Protocol doc:  benchmarks/csfw_g6_loo_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""Sanity gate before ANY 956 analysis: for EVERY row of EVERY leg, the first
944 feature columns of the ext956-instrument parquets must be BYTE-EQUAL
(f64 bit-pattern identical) to ext944-instrument-2026-07-28, and
ref_basename/human_score must match. The csfw tier-1 V1/V2 gates
(`csfw_layout_identity_and_first944_bit_stable`) guarantee first-944
stability with the toggle ON; a mismatch here means regime drift between the
944-instrument rev (b1d4bc25) and the current tip (7bfd511d) — STOP and
report, do not analyze.

Also checks: >=3 rows per leg; f156..f371 structurally-zero spot-stride;
csfw block f944..f955 all finite in [0,1]; per-lane std + fire-rate stats
persisted (R-class rare-fire flagging: CGAIN std ~1e-5 on SDR is documented
structural behavior, not neutrality)."""

import csv
import json
import sys

import numpy as np
import pyarrow.parquet as pq

D944 = "/mnt/v/zen/zensim-training/ext944-instrument-2026-07-28"
D956 = "/mnt/v/zen/zensim-training/ext956-instrument-2026-07-29"
OUTDIR = "/mnt/v/output/zensim/csfw-g6-loo-2026-07-29"

LANES = {0: "W_GLOBAL_DMEAN", 1: "W_GLOBAL_CGAIN", 2: "W_GLOBAL_CLOSS"}

CORPORA = [e["corpus"] for e in json.load(open(f"{D944}/_MANIFEST.json"))["entries"]]

fail = False
report = []
lane_rows = []
for c in CORPORA:
    t9 = pq.read_table(f"{D944}/{c}.parquet")
    t5 = pq.read_table(f"{D956}/{c}.parquet")
    n9, n5 = t9.num_rows, t5.num_rows
    line = {"corpus": c, "rows_944": n9, "rows_956": n5}
    if n9 != n5 or n5 < 3:
        line["verdict"] = "FAIL rows"
        fail = True
        report.append(line)
        continue
    rb_eq = t9["ref_basename"].to_pylist() == t5["ref_basename"].to_pylist()
    hs9 = np.asarray(t9["human_score"].combine_chunks().to_numpy(), np.float64)
    hs5 = np.asarray(t5["human_score"].combine_chunks().to_numpy(), np.float64)
    hs_eq = bool(np.array_equal(hs9.view(np.uint64), hs5.view(np.uint64)))
    ndiff_cols = 0
    first_bad = None
    for i in range(944):
        a = np.asarray(t9[f"f{i}"].combine_chunks().to_numpy(), np.float64)
        b = np.asarray(t5[f"f{i}"].combine_chunks().to_numpy(), np.float64)
        if not np.array_equal(a.view(np.uint64), b.view(np.uint64)):
            ndiff_cols += 1
            if first_bad is None:
                j = int(np.nonzero(a.view(np.uint64) != b.view(np.uint64))[0][0])
                first_bad = (i, j, float(a[j]), float(b[j]))
    zero_ok = all(
        not np.any(np.asarray(t5[f"f{i}"].combine_chunks().to_numpy(), np.float64))
        for i in range(156, 372, 27)  # spot-stride the structurally-zero band
    )
    cw = np.column_stack(
        [np.asarray(t5[f"f{i}"].combine_chunks().to_numpy(), np.float64)
         for i in range(944, 956)]
    )
    cw_finite = bool(np.all(np.isfinite(cw)))
    cw_bounded = bool((cw.min() >= 0.0) and (cw.max() <= 1.0))
    for s in range(4):
        for l, nm in LANES.items():
            col = cw[:, s * 3 + l]
            med = float(np.median(col))
            fire = float(np.mean(np.abs(col - med) > 1e-9))
            lane_rows.append(
                dict(corpus=c, feat=944 + s * 3 + l, lane=nm, scale=s, n=n5,
                     mean=float(col.mean()), std=float(col.std()),
                     minv=float(col.min()), maxv=float(col.max()),
                     median=med, fire_rate_vs_median_1e9=round(fire, 6))
            )
    line.update(
        ref_eq=rb_eq, human_eq=hs_eq, n_f_cols_differing=ndiff_cols,
        first_diff=first_bad, foldslot_zero_spot=zero_ok,
        csfw_finite=cw_finite, csfw_bounded=cw_bounded,
        csfw_max=float(cw.max()),
    )
    ok = rb_eq and hs_eq and ndiff_cols == 0 and zero_ok and cw_finite and cw_bounded
    line["verdict"] = "PASS" if ok else "FAIL"
    fail |= not ok
    report.append(line)
    print(f"{c}: {line['verdict']} rows={n9} diff_cols={ndiff_cols} "
          f"csfw_max={line['csfw_max']:.4f}", flush=True)

json.dump(report, open(f"{OUTDIR}/byte_gate_956_report.json", "w"), indent=1)
with open(f"{OUTDIR}/csfw_lane_stats_956.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(lane_rows[0]))
    w.writeheader()
    w.writerows(lane_rows)
print(f"wrote {OUTDIR}/byte_gate_956_report.json + csfw_lane_stats_956.csv")
if fail:
    print("BYTE GATE: FAIL — regime drift; DO NOT ANALYZE")
    sys.exit(1)
print("BYTE GATE: PASS (all legs, all rows, first-944 bit-identical)")
