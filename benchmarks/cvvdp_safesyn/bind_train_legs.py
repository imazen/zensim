#!/usr/bin/env python3
"""cvvdp-safesyn E2b — bind TRAIN-role label rows to their pairs-TSV image rows.

The ext944 train parquets carry `ref_basename` + labels but NO distorted-image
key, so the row->image binding is established by order-within-reference and
PROVEN by label equality: for every reference, the parquet's `human_score`
sequence (in parquet order) must equal the pairs TSV's own label column
sequence (in TSV order) for the same reference, element for element. Any
count or value mismatch is a hard failure — never a nearest-match bind.

Outputs one pairs file per leg (ref_path<TAB>dist_path<TAB>label columns
carried for audit), plus a JSON binding report with row counts, per-ref
counts, and a sha256 over the joined key sequence (the value the worklog
and report cite as the binding fingerprint).

Usage: bind_train_legs.py <out_dir>
"""
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pyarrow.parquet as pq

LEGS = OrderedDict([
    ("kadid", dict(
        pairs="/mnt/v/output/zensim/reports/refmetrics/kadid_pairs.tsv",
        parquet="/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_kadid_train_2026-08-29.parquet",
        tsv_label="DMOS",
        parquet_label="human_score",
        # KADID TSV carries raw DMOS (1..5, higher better); the ext944 view
        # stores human_score = (DMOS - 1) / 4. Verified elementwise.
        tsv_map=lambda v: (v - 1.0) / 4.0,
        extra=[],
        expect_rows=5000,
    )),
    ("tid", dict(
        pairs="/mnt/v/dataset/tid2013/tid_pairs_ab.tsv",
        parquet="/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_tid_train_2026-08-29.parquet",
        tsv_label="human_score",
        parquet_label="human_score",
        tsv_map=lambda v: v,
        extra=[],
        expect_rows=1440,
    )),
    ("konfig", dict(
        pairs="/mnt/v/output/zensim/konfig944/build/konfig_pairs.tsv",
        parquet="/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/konfig_originsplit_train_944.parquet",
        tsv_label="human_score",
        parquet_label="human_score",
        tsv_map=lambda v: v,
        extra=["q_jnd"],
        expect_rows=327,
    )),
])


def read_tsv(path):
    lines = Path(path).read_text().splitlines()
    head = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        rows.append(dict(zip(head, parts)))
    return head, rows


def bind_leg(name, spec, out_dir):
    head, tsv = read_tsv(spec["pairs"])
    want = ["ref_basename", spec["parquet_label"]] + spec["extra"]
    t = pq.read_table(spec["parquet"], columns=want)
    parq = t.to_pydict()
    n_parq = len(parq["ref_basename"])

    # group TSV rows by ref stem, preserving TSV order (parquet ref_basename
    # has no extension: 'I02' vs the TSV's 'I02.png')
    def bname(p):
        return Path(p).stem

    tsv_by_ref = OrderedDict()
    for i, r in enumerate(tsv):
        tsv_by_ref.setdefault(bname(r["ref_path"]), []).append(i)
    parq_by_ref = OrderedDict()
    for i, rb in enumerate(parq["ref_basename"]):
        parq_by_ref.setdefault(rb, []).append(i)

    missing = sorted(set(parq_by_ref) - set(tsv_by_ref))
    if missing:
        raise SystemExit(f"{name}: {len(missing)} parquet refs absent from pairs TSV: {missing[:5]}")

    bound = []  # (tsv_idx, parq_idx)
    per_ref = {}
    for rb, pidxs in parq_by_ref.items():
        tidxs = tsv_by_ref[rb]
        if len(tidxs) != len(pidxs):
            raise SystemExit(
                f"{name}: ref {rb}: TSV has {len(tidxs)} rows, parquet {len(pidxs)}")
        lab_t = [spec["tsv_map"](float(tsv[i][spec["tsv_label"]])) for i in tidxs]
        lab_p = [parq[spec["parquet_label"]][i] for i in pidxs]
        if lab_t != lab_p:
            # allow last-ulp float render differences; flag anything larger
            diffs = [abs(a - b) for a, b in zip(lab_t, lab_p)]
            if max(diffs) > 1e-6:
                k = diffs.index(max(diffs))
                raise SystemExit(
                    f"{name}: ref {rb}: label mismatch at pair {k}: "
                    f"tsv={lab_t[k]} parquet={lab_p[k]} (max|d|={max(diffs)})")
        for ex in spec["extra"]:
            ev_t = [float(tsv[i][ex]) for i in tidxs]
            ev_p = [float(parq[ex][i]) for i in pidxs]
            if ev_t != ev_p:
                raise SystemExit(f"{name}: ref {rb}: {ex} sequences differ")
        for ti, pi in zip(tidxs, pidxs):
            bound.append((ti, pi))
        per_ref[rb] = len(pidxs)

    if len(bound) != n_parq:
        raise SystemExit(f"{name}: bound {len(bound)} != parquet rows {n_parq}")
    if spec["expect_rows"] and len(bound) != spec["expect_rows"]:
        raise SystemExit(f"{name}: expected {spec['expect_rows']} rows, bound {len(bound)}")

    # Emit pairs file in PARQUET row order (the leg's canonical row order):
    # ref_path \t dist_path \t human_score [+ extra label cols].
    out = Path(out_dir) / f"{name}_train_pairs.tsv"
    cols = ["ref_path", "dist_path", spec["parquet_label"]] + spec["extra"]
    h = hashlib.sha256()
    key_h = hashlib.sha256()
    with out.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for ti, pi in bound:
            r = tsv[ti]
            line = [r["ref_path"], r["dist_path"], repr(parq[spec["parquet_label"]][pi])]
            line += [repr(float(parq[ex][pi])) for ex in spec["extra"]]
            s = "\t".join(line) + "\n"
            f.write(s)
            key_h.update(f"{r['ref_path']}|{r['dist_path']}\n".encode())
    h.update(out.read_bytes())
    return {
        "leg": name,
        "rows": len(bound),
        "refs": len(parq_by_ref),
        "per_ref_rows": per_ref,
        "pairs_out": str(out),
        "pairs_sha256": h.hexdigest(),
        "keyseq_sha256": key_h.hexdigest(),
        "label_check": f"tsv[{spec['tsv_label']}] == parquet[{spec['parquet_label']}] elementwise per ref",
        "extra_checked": spec["extra"],
    }


def main():
    out_dir = sys.argv[1]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    report = OrderedDict()
    for name, spec in LEGS.items():
        report[name] = bind_leg(name, spec, out_dir)
        r = report[name]
        print(f"{name}: {r['rows']} rows / {r['refs']} refs -> {r['pairs_out']}",
              file=sys.stderr)
        print(f"  pairs sha256 {r['pairs_sha256']}", file=sys.stderr)
        print(f"  keyseq sha256 {r['keyseq_sha256']}", file=sys.stderr)
    rep_path = Path(out_dir) / "binding_report.json"
    rep_path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"binding report -> {rep_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
