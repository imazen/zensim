#!/usr/bin/env python3
"""Baseline index views — a filtered .index.jsonl over a shared .bin.

The baseline cell's records are constants-independent (bin121 + local +
n5): the joint-core caches ARE its cache. A view = a symlink to the
source .bin plus an index file holding only the leg's rows, in pairs-TSV
order, so `PlaneCacheF16`/`RowView` see a leg-shaped cache with zero
re-extraction.

  make_view.py <pairs.tsv> <src_bin> <dst_bin> [--note NAME]

<dst_bin> is created as a symlink to <src_bin>; <dst_bin>.index.jsonl
carries the filtered entries with `subset_of`/`subset_seed` provenance.
"""
import csv, json, sys
from pathlib import Path

def main():
    pairs_tsv, src_bin, dst_bin = sys.argv[1:4]
    note = "baseline_view"
    if "--note" in sys.argv:
        note = sys.argv[sys.argv.index("--note") + 1]
    src = Path(src_bin)
    idx_src = src.with_name(src.name + ".index.jsonl")
    dst = Path(dst_bin)
    idx_dst = dst.with_name(dst.name + ".index.jsonl")

    pairs = list(csv.DictReader(open(pairs_tsv), delimiter="\t"))
    # (ref_path, dist_path) -> pairs row order
    want = [(r["ref_path"], r["dist_path"]) for r in pairs]
    entries = {}
    for line in open(idx_src):
        e = json.loads(line)
        entries.setdefault((e["ref_path"], e["dist_path"]), e)
    out_rows = []
    for i, key in enumerate(want):
        e = entries.get(key)
        if e is None:
            raise SystemExit(f"row {i}: {key} not in {idx_src}")
        e = dict(e)
        e["row_index"] = i
        e["subset_of"] = note
        e["subset_seed"] = 20260921
        out_rows.append(e)
    missing = len(want) - len(out_rows)
    assert missing == 0
    if not dst.exists():
        dst.symlink_to(src)
    with open(idx_dst, "w") as w:
        for e in out_rows:
            w.write(json.dumps(e, sort_keys=True) + "\n")
    print(f"{dst.name}: {len(out_rows)} rows -> {src}")

if __name__ == "__main__":
    main()
