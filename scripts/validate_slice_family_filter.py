#!/usr/bin/env python3
"""Family-filter the VALIDATE eval slices (validate-selection migration,
balance_campaign methodology audit 2026-08-28).

Keep rows whose origin's FAMILY bucket == validate (split_map_family.tsv —
families were re-bucketed atomically, 0 spanning), and conservatively drop
any row whose family contains a known sharing id (the union of the D1
CERTAIN/UPPER sets — parity with the test-slice treatment; channel-A never
entered training per the measured safesyn no-op, so this is belt-and-
suspenders comparability, not leak repair)."""
import re, sys
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

VS = Path(sys.argv[1])
FAM_MAP = Path.home() / "work/imazen-26/manifests/split_map_family.tsv"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "canonical_corpus"))
from apply_d1_exclusion import CERTAIN, UPPER  # the committed sharing-id sets

fam, bucket = {}, {}
for ln in open(FAM_MAP):
    p = ln.rstrip("\n").split("\t")
    if len(p) >= 4 and p[0].isdigit():
        fam[p[0]] = p[2] if p[2] else f"solo-{p[0]}"
        bucket[p[0]] = p[1]
share_ids = set().union(*CERTAIN.values(), *UPPER.values())
share_fams = {fam[i] for i in share_ids if i in fam}

def origin_of(name):
    m = re.match(r"^o?_?(\d{4})", name)
    return m.group(1) if m else None

for slc in ("imazen26", "nonphoto", "hfnlproxy"):
    p = VS / f"ext_{slc}.parquet"
    t = pq.read_table(p)
    refs = t["ref_basename"].to_pylist()
    keep = []
    for i, r in enumerate(refs):
        o = origin_of(r)
        if o is None or bucket.get(o) != "validate":
            continue
        if fam.get(o) in share_fams:
            continue
        keep.append(i)
    out = t.take(pa.array(keep))
    pq.write_table(out, p, compression="zstd", compression_level=7)
    n_refs = len({origin_of(r) for r in out["ref_basename"].to_pylist()})
    print(f"ext_{slc}: {t.num_rows} -> {out.num_rows} rows ({n_refs} origins, validate-family, share-fams dropped)")
