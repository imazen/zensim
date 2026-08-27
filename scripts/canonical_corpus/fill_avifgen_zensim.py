#!/usr/bin/env python3
"""Fill the avifgen harvest's all-null zensim column: forward the current-regime
C bake (c_sdr_mlp944_corrmix_2026-08-05, caller width 944) over the stored
944-feature parquet via the CANONICAL `predict_features_with_bake` binary
(chunked wire blobs; no model math here). Emits an encode_sha-keyed sidecar
`zensim_c944_sidecar.parquet` + sha-stamped manifest fragment — the harvest
scores.parquet is NOT rewritten (append-only discipline; consumers join).
"""
import hashlib
import json
import struct
import subprocess
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/mnt/v/output/avifgen-2026-08-06/harvest-2026-08-26/features_folded720append2.parquet"
OUT = "/mnt/v/output/avifgen-2026-08-06/harvest-2026-08-26/zensim_c944_sidecar.parquet"
BAKE = "zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin"
FWD = "target/release/predict_features_with_bake"
BLOB = "/tmp/claude-1000/-home-lilith-work-zen-zensim/9d242656-d636-45a6-9468-565163baed2d/scratchpad/fill_feats.blob"

f = pq.ParquetFile(SRC)
featn = [f"feat_{i}" for i in range(944)]
shas, scores = [], []
for g in range(f.num_row_groups):
    t = f.read_row_group(g, columns=["encode_sha"] + featn)
    shas.extend(t.column("encode_sha").to_pylist())
    n = t.num_rows
    cols = [t.column(c).to_numpy(zero_copy_only=False) for c in featn]
    with open(BLOB, "wb") as bf:
        bf.write(struct.pack("<II", 944, n))
        import numpy as np
        m = np.column_stack(cols).astype("<f4")
        bf.write(m.tobytes())
    r = subprocess.run([FWD, "--bake", BAKE, "--features-file", BLOB],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"forward failed rg{g}: {r.stderr[-300:]}")
    part = [float(x) for x in r.stdout.split()]
    assert len(part) == n, (g, len(part), n)
    scores.extend(part)
    print(f"rg {g+1}/{f.num_row_groups} done ({len(shas)} rows)", flush=True)

bake_sha = hashlib.sha256(open(BAKE, "rb").read()).hexdigest()
# Dedupe to per-sha: identical bytes => identical features => identical
# score. Assert equality (never assume) — a mismatch would mean the features
# were NOT a pure function of the bytes and must fail loud.
per_sha = {}
for k, v in zip(shas, scores):
    if k in per_sha:
        assert abs(per_sha[k] - v) < 1e-9, f"sha {k}: {per_sha[k]} != {v}"
    else:
        per_sha[k] = v
print(f"dedupe: {len(shas)} cell rows -> {len(per_sha)} distinct shas (equality-asserted)")
shas, scores = list(per_sha.keys()), list(per_sha.values())
tbl = pa.table({"encode_sha": shas, "zensim_c944": scores})
pq.write_table(tbl, OUT, compression="zstd")
mf = {
    "what": "zensim scores for the avifgen harvest, C-bake forward over stored 944 features",
    "bake": BAKE, "bake_sha256": bake_sha,
    "src": SRC, "rows": len(shas),
    "forward": "predict_features_with_bake (canonical)",
    "note": "harvest scores.parquet zensim_score column is 100% null (zensim was never fleet-scored); this sidecar is the current-regime fill — join on encode_sha",
}
json.dump(mf, open(OUT + ".manifest.json", "w"), indent=1)
print("sidecar:", OUT, len(shas), "rows; bake", bake_sha[:12])
