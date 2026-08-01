#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/bandvis-loo-2026-07-28/harness/byte_gate_944.py
# sha256(source): a6e093322f323e978196ebbcbd594f037313cc6e270211952f2f13e17f6e54e6
# build_commit:  b1d4bc257e57f7c3215ec8a237e9f87cdad8e35f
# Protocol doc:  benchmarks/bandvis_loo_944_2026-07-28.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""Sanity gate before ANY 944 analysis: for EVERY row of EVERY leg, the first
924 feature columns of the ext944-instrument parquets must be BYTE-EQUAL
(f64 bit-pattern identical) to ext924-canonical-2026-07-27, and
ref_basename/human_score must match. append2 is first-924-stable by its V2
gate (`append2_layout_identity_and_first924_bit_stable`); a mismatch here
means regime drift between the W1 rev (0b3d16b04b19) and the current tip —
STOP and report, do not analyze.

Also checks: f156..f371 structurally zero; f924..f943 all finite in [0,1];
HL bins (locals 3,4) exactly 0 on these SDR legs.
"""

import json
import sys

import numpy as np
import pyarrow.parquet as pq

D924 = "/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27"
D944 = "/mnt/v/zen/zensim-training/ext944-instrument-2026-07-28"

CORPORA = [e["corpus"] for e in json.load(open(f"{D924}/_MANIFEST.json"))["entries"]]

fail = False
report = []
for c in CORPORA:
    t9 = pq.read_table(f"{D924}/{c}.parquet")
    t4 = pq.read_table(f"{D944}/{c}.parquet")
    n9, n4 = t9.num_rows, t4.num_rows
    line = {"corpus": c, "rows_924": n9, "rows_944": n4}
    if n9 != n4:
        line["verdict"] = "FAIL rows"
        fail = True
        report.append(line)
        continue
    rb_eq = t9["ref_basename"].to_pylist() == t4["ref_basename"].to_pylist()
    hs9 = np.asarray(t9["human_score"].combine_chunks().to_numpy(), np.float64)
    hs4 = np.asarray(t4["human_score"].combine_chunks().to_numpy(), np.float64)
    hs_eq = bool(np.array_equal(hs9.view(np.uint64), hs4.view(np.uint64)))
    ndiff_cols = 0
    first_bad = None
    for i in range(924):
        a = np.asarray(t9[f"f{i}"].combine_chunks().to_numpy(), np.float64)
        b = np.asarray(t4[f"f{i}"].combine_chunks().to_numpy(), np.float64)
        if not np.array_equal(a.view(np.uint64), b.view(np.uint64)):
            ndiff_cols += 1
            if first_bad is None:
                j = int(np.nonzero(a.view(np.uint64) != b.view(np.uint64))[0][0])
                first_bad = (i, j, float(a[j]), float(b[j]))
    # fold-slot zeros + append2 block checks on the 944 side
    zero_ok = all(
        not np.any(np.asarray(t4[f"f{i}"].combine_chunks().to_numpy(), np.float64))
        for i in range(156, 372, 27)  # spot-stride the structurally-zero band
    )
    a2 = np.column_stack(
        [np.asarray(t4[f"f{i}"].combine_chunks().to_numpy(), np.float64)
         for i in range(924, 944)]
    )
    a2_finite = bool(np.all(np.isfinite(a2)))
    a2_bounded = bool((a2.min() >= 0.0) and (a2.max() <= 1.0))
    hl_cols = [s * 5 + local for s in range(4) for local in (3, 4)]
    hl_zero = bool(not np.any(a2[:, hl_cols]))
    line.update(
        ref_eq=rb_eq, human_eq=hs_eq, n_f_cols_differing=ndiff_cols,
        first_diff=first_bad, foldslot_zero_spot=zero_ok,
        a2_finite=a2_finite, a2_bounded=a2_bounded, hl_zero_on_sdr=hl_zero,
        a2_max=float(a2.max()),
    )
    ok = rb_eq and hs_eq and ndiff_cols == 0 and zero_ok and a2_finite and a2_bounded and hl_zero
    line["verdict"] = "PASS" if ok else "FAIL"
    fail |= not ok
    report.append(line)
    print(f"{c}: {line['verdict']} rows={n9} diff_cols={ndiff_cols} "
          f"a2_max={line['a2_max']:.4f} hl_zero={hl_zero}", flush=True)

out = "/mnt/v/output/zensim/bandvis-loo-2026-07-28/byte_gate_944_report.json"
json.dump(report, open(out, "w"), indent=1)
print(f"wrote {out}")
if fail:
    print("BYTE GATE: FAIL — regime drift; DO NOT ANALYZE")
    sys.exit(1)
print("BYTE GATE: PASS (all legs, all rows, first-924 bit-identical)")
