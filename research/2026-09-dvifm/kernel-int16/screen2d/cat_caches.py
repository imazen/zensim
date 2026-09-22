#!/usr/bin/env python3
"""Concatenate dvifm-block-records-v1 caches, rebasing .index.jsonl offsets.

Usage: cat_caches.py out.bin in1.bin in2.bin ...

Each input must have a sibling <in>.index.jsonl. Output rows preserve input
order; row_index is reassigned 0..N-1 and `offset` rebased to byte offsets in
the concatenated payload. Writes <out>.index.jsonl and verifies the total
payload byte count.
"""
import json
import sys
from pathlib import Path


def main():
    out = Path(sys.argv[1])
    ins = [Path(p) for p in sys.argv[2:]]
    assert len(ins) >= 1
    offset = 0
    row_index = 0
    n_idx = 0
    with open(out, "wb") as fo, open(str(out) + ".index.jsonl", "w") as fi:
        for p in ins:
            idx_path = Path(str(p) + ".index.jsonl")
            entries = [json.loads(l) for l in idx_path.open() if l.strip()]
            blob = p.read_bytes()
            cursor = 0
            for e in entries:
                n = sum(e["level_records"]) * 18 * 4
                assert e["offset"] + n <= len(blob), (p, e["row_index"])
                payload = blob[e["offset"]:e["offset"] + n]
                fo.write(payload)
                e2 = dict(e)
                e2["offset"] = offset
                e2["row_index"] = row_index
                fi.write(json.dumps(e2, sort_keys=True) + "\n")
                offset += n
                cursor = max(cursor, e["offset"] + n)
                row_index += 1
                n_idx += 1
            # payload must be exactly the concatenation of row records
            assert cursor == len(blob), (
                f"{p}: {cursor} bytes covered but file is {len(blob)} — "
                "non-record payload present, refusing to guess")
    print(f"wrote {out} rows={n_idx} bytes={offset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
