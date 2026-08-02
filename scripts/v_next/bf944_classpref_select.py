#!/usr/bin/env python3
"""Build the bf944 matched_ledger with EXACT-WORKER-then-VENDOR-CLASS selection.

The bf944 wave's G-BF1 (f0..f923 bitwise vs the frozen 924 rows) requires each
cell's re-extraction to come from the same CPU-VENDOR x SIMD-tier class as its
bf924 extractor (22 append slots use vendor-specific approximation-instruction
tables — AMD vs Intel rsqrt/rcp; measured 2026-08-02, zensim
benchmarks/backfill944_bigcodec_2026-08-02.md). The wave's multi-worker
re-scoring + the vendor repair runs mean most cells have several candidate
blobs; the assembler's naive keep-first picked arbitrarily. This selector
prefers, per cell:
  1. a done row from the EXACT bf924 worker (strongest — covers any
     within-class heterogeneity not yet measured),
  2. else a done row from the same vendor class,
  3. else FAIL loudly (the cell needs a repair extraction).

Ledger job-ids appear in TWO encodings (serde_json preserve_order feature
unification flipped JobId::of's json! key order in the docker-built worker —
sorted-keys in ctl/mac/bf924, insertion-order in the bf944 docker fleet); the
id map carries both (id944 / id944po).

Output: matched_ledger.parquet (pool, job_id, output_sha, image_path, codec)
compatible with fleet_blob_assemble_944.py's fetch stage.
"""
import collections
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

CLASS = {
    'lilith-lianli': 'amdv4x', 'wsl-smoke': 'amdv4x', 'wsl-944': 'amdv4x',
    'tower-unraid': 'amdv4', 'zen-node-3': 'amdv4',
    'zen-node-2': 'intelv4', 'i265': 'intelv4',
    'lilith-mac': 'neon', 'mac-login-test': 'neon', 'mac-debug': 'neon',
    'lilith-mac-gapfix': 'neon', 'lilith-mac-manual': 'neon',
    'lilith-mac-manual2': 'neon', 'lilith-mac-manual3': 'neon',
}
# bf924 worker -> the bf944-era worker(s) that are the SAME physical box.
SAME_BOX = {
    'tower-unraid': {'tower-unraid'},
    'zen-node-2': {'zen-node-2'},
    'zen-node-3': {'zen-node-3'},
    'i265': {'i265'},
    'lilith-lianli': {'lilith-lianli'},
    'wsl-smoke': {'wsl-944'},
    'lilith-mac': {'lilith-mac', 'lilith-mac-gapfix'},
    'mac-login-test': {'lilith-mac', 'lilith-mac-gapfix'},
    'mac-debug': {'lilith-mac', 'lilith-mac-gapfix'},
}


def main() -> int:
    id_map = sys.argv[1] if len(sys.argv) > 1 else '/home/lilith/tmp/bigcodec944/id_map2.parquet'
    ledgers = Path(sys.argv[2] if len(sys.argv) > 2 else '/home/lilith/tmp/bf944_join/ledgers')
    out = sys.argv[3] if len(sys.argv) > 3 else '/home/lilith/tmp/bf944_join/matched_ledger.parquet'

    cand = collections.defaultdict(list)  # job_id -> [(run, worker, output_sha, image_path, codec)]
    for run_dir in sorted(ledgers.iterdir()):
        if not run_dir.is_dir():
            continue
        t = ds.dataset(str(run_dir), format='parquet').to_table(
            columns=['job_id', 'status', 'worker', 'output_sha', 'image_path', 'codec'])
        for j, s, w, o, ip, cod in zip(
            t.column('job_id').to_pylist(), t.column('status').to_pylist(),
            t.column('worker').to_pylist(), t.column('output_sha').to_pylist(),
            t.column('image_path').to_pylist(), t.column('codec').to_pylist(),
        ):
            if s == 'done' and o:
                cand[j].append((run_dir.name, w, o, ip, cod, j))

    m = pq.read_table(id_map)
    rows = {k: [] for k in ('pool', 'job_id', 'output_sha', 'image_path', 'codec')}
    stats = collections.Counter()
    missing = []
    for i4, i4p, w924 in zip(
        m.column('id944').to_pylist(), m.column('id944po').to_pylist(),
        m.column('worker924').to_pylist(),
    ):
        cands = cand.get(i4, []) + cand.get(i4p, [])
        pick = None
        boxes = SAME_BOX.get(w924, set())
        for c in cands:
            if c[1] in boxes:
                pick = c
                stats['exact_worker'] += 1
                break
        if pick is None:
            cls = CLASS[w924]
            for c in cands:
                if CLASS.get(c[1]) == cls:
                    pick = c
                    stats['vendor_class'] += 1
                    break
        if pick is None:
            stats['MISSING'] += 1
            missing.append((w924, i4))
            continue
        run, w, o, ip, cod, ledger_jid = pick
        rows['pool'].append(run)
        rows['job_id'].append(ledger_jid)
        rows['output_sha'].append(o)
        rows['image_path'].append(ip)
        rows['codec'].append(cod)
    print('selection stats:', dict(stats))
    if missing:
        print(f'FAIL: {len(missing)} cells have no class-matched blob; first: {missing[:3]}')
        return 1
    pq.write_table(pa.table(rows), out, compression='zstd')
    print(f'wrote {out}: {len(rows["job_id"])} rows')
    return 0


if __name__ == '__main__':
    sys.exit(main())
