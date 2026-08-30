#!/usr/bin/env python3
"""Family-filter the VALIDATE eval slices (validate-selection migration,
balance_campaign methodology audit 2026-08-28).

Keep rows whose origin's FAMILY bucket == validate (split_map_family.tsv —
families were re-bucketed atomically, 0 spanning), and conservatively drop
any row whose family contains a known sharing id (the union of the D1
CERTAIN/UPPER sets — parity with the test-slice treatment; channel-A never
entered training per the measured safesyn no-op, so this is belt-and-
suspenders comparability, not leak repair).

Usage: validate_slice_family_filter.py <slice-dir> [--verify-against <dir>]

Operates on whichever of `ext_<slc>.parquet` / `keys_<slc>.parquet` exist in
<slice-dir>, applying ONE keep-index list per slice so the feature table and its
R1b key sidecar (build_eval_slices_944.py --emit-keys) can never disagree about
which rows survived. With --verify-against, the surviving `ref_basename`
sequence is asserted row-for-row equal to the stored canonical
`ext_<slc>.parquet` in that dir -- the proof that a key sidecar names exactly
the rows the shipped slice holds."""
import re, sys
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

argv = sys.argv[1:]
VERIFY = None
if "--verify-against" in argv:
    i = argv.index("--verify-against")
    VERIFY = Path(argv[i + 1])
    del argv[i:i + 2]
VS = Path(argv[0])
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

rc = 0
for slc in ("imazen26", "nonphoto", "hfnlproxy"):
    targets = [VS / f"{pre}_{slc}.parquet" for pre in ("ext", "keys")]
    targets = [t for t in targets if t.exists()]
    if not targets:
        print(f"{slc}: no ext_/keys_ table in {VS} — skipped")
        continue
    keep = None
    for p in targets:
        t = pq.read_table(p)
        refs = t["ref_basename"].to_pylist()
        k = []
        for i, r in enumerate(refs):
            o = origin_of(r)
            if o is None or bucket.get(o) != "validate":
                continue
            if fam.get(o) in share_fams:
                continue
            k.append(i)
        if keep is None:
            keep = k
        elif k != keep:
            sys.exit(f"REFUSING {slc}: ext_/keys_ tables disagree on the keep set "
                     f"({len(k)} vs {len(keep)}) — they are not the same rows")
        out = t.take(pa.array(k))
        pq.write_table(out, p, compression="zstd", compression_level=7)
        n_refs = len({origin_of(r) for r in out["ref_basename"].to_pylist()})
        print(f"{p.name}: {t.num_rows} -> {out.num_rows} rows "
              f"({n_refs} origins, validate-family, share-fams dropped)")
        if VERIFY is not None:
            want = pq.read_table(VERIFY / f"ext_{slc}.parquet",
                                 columns=["ref_basename"])["ref_basename"].to_pylist()
            got = out["ref_basename"].to_pylist()
            if got == want:
                print(f"    G-KEY OK: row identity == {VERIFY}/ext_{slc}.parquet ({len(want)} rows)")
            else:
                first = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), "len")
                print(f"    G-KEY FAIL {p.name}: {len(got)} vs {len(want)} rows, "
                      f"first mismatch at {first}")
                rc = 1
sys.exit(rc)
