#!/usr/bin/env python3
"""Build the id100 dial anchor for a 944-POOLS fast-class bake.

`bake_dial_refit pack` takes exactly ONE `--anchor` parquet, so the id100 chain
(benchmarks/d_id100_2026-09-04.md) -- which pins the identity dial at 100 by
putting ref==dist rows in the anchor -- needs the identity rows CONCATENATED
into the anchor rather than passed as a second `--anchor-parquet` (which only
`fit-lasso` accepts).

n_id = 21 is the value d_id100 selected and registered; it is REUSED, not
re-swept. At 2020 anchor rows that is 1.04 % of the mass, against the 1.03 %
that lane measured at 2035 rows.

Identity rows come from the D+free lane's 944-pools probe (39 refs, one
ref==dist row each). The 21 are the first by sorted `entry` -- deterministic
and stated, not selected on any outcome.

Usage: build_id100_anchor.py <base_anchor.parquet> <identity_probe.parquet> <out.parquet> [n_id]
"""
import sys
import pyarrow as pa
import pyarrow.parquet as pq

def main() -> int:
    base_p, probe_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
    n_id = int(sys.argv[4]) if len(sys.argv) > 4 else 21
    base = pq.read_table(base_p)
    probe = pq.read_table(probe_p)
    feats = [c for c in base.column_names if c.startswith("f") and c[1:].isdigit()]
    missing = [c for c in feats if c not in probe.column_names]
    if missing:
        print(f"ABORT: probe lacks {len(missing)} feature columns, first {missing[:5]}")
        return 2
    order = sorted(range(probe.num_rows), key=lambda i: probe.column("entry")[i].as_py())[:n_id]
    if len(order) < n_id:
        print(f"ABORT: probe has {probe.num_rows} rows, need {n_id}")
        return 2
    sel = probe.take(pa.array(order))
    cols = {}
    for name in base.column_names:
        t = base.schema.field(name).type
        if name in feats:
            cols[name] = sel.column(name).cast(t)
        elif name == "ref_basename":
            cols[name] = pa.array(
                [f"identity::{sel.column('entry')[i].as_py()}" for i in range(sel.num_rows)], t
            )
        elif name == "human_score":
            cols[name] = pa.array([1.0] * sel.num_rows, t)
        elif name == "target_score":
            cols[name] = pa.array([100.0] * sel.num_rows, t)
        elif name == "regime":
            cols[name] = pa.array([base.column(name)[0].as_py()] * sel.num_rows, t)
        else:
            print(f"ABORT: unhandled anchor column {name!r} -- refusing to invent a value")
            return 2
    idt = pa.table(cols, schema=base.schema)
    out = pa.concat_tables([base, idt])
    pq.write_table(out, out_p, compression="zstd")
    print(f"wrote {out_p}: {base.num_rows} anchor + {idt.num_rows} identity = {out.num_rows} rows, "
          f"{out.num_columns} cols; identity mass {idt.num_rows / out.num_rows:.4%}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
