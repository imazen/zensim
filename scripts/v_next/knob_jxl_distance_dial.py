#!/usr/bin/env python3
"""What B (and A, ssim2) value corresponds to a JXL encode at a given distance?
Uses the near-lossless full sweep (200 images x 11 distances, real JXL bitstreams),
current-B (b6fe5233) + A (v47) forwarded via predict_features_with_bake, ssim2 joined
from pareto.tsv. Reports per-distance dial distribution across the 200 images."""
import struct, subprocess, os, json, csv
import numpy as np, pyarrow.parquet as pq

FULL = "/mnt/v/output/zensim-jxl-nearlossless/full"
B = "zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin"
A = "zensim/weights/v47_strict_qat_native_2026-05-27.bin"

t = pq.read_table(f"{FULL}/features.parquet")
X = np.column_stack([t.column(f"feat_{i}").to_numpy().astype(np.float32) for i in range(372)])
dist = np.array([round(json.loads(k)["distance"], 4) for k in t.column("knob_tuple_json").to_pylist()])
img = t.column("image_path").to_pylist()
n = X.shape[0]
wire = "/mnt/v/output/zensim-multicodec-probe/nl_full.feats.bin"
with open(wire, "wb") as f:
    f.write(struct.pack("<II", 372, n)); X.tofile(f)
def fwd(bake):
    r = subprocess.run(["./target/release/predict_features_with_bake", "--bake", bake, "--features-file", wire],
                       capture_output=True, text=True); assert r.returncode == 0, r.stderr[:500]
    return np.array([float(x) for x in r.stdout.split()])
b = fwd(B); a = fwd(A); os.remove(wire)

# ssim2 from pareto.tsv, join by (image_path, distance)
ss = {}
try:
    for r in csv.DictReader(open(f"{FULL}/pareto.tsv"), delimiter="\t"):
        d = round(json.loads(r["knob_tuple_json"])["distance"], 4)
        ss[(r["image_path"], d)] = float(r["score_ssim2"])
    s2 = np.array([ss.get((img[i], dist[i]), np.nan) for i in range(n)])
except Exception as e:
    print("ssim2 join skipped:", e); s2 = np.full(n, np.nan)

print(f"{'JXL dist':>9} {'n':>4} | {'B med':>6} {'B p10-p90':>13} | {'A med':>6} {'A p10-p90':>13} | {'ssim2 med':>9}")
for d in sorted(set(dist)):
    m = dist == d
    def stats(v): v = v[m][~np.isnan(v[m])]; return (np.median(v), np.percentile(v, 10), np.percentile(v, 90)) if len(v) else (np.nan,)*3
    bm, bl, bh = stats(b); am, al, ah = stats(a); sm, _, _ = stats(s2)
    mark = "   <=== d0.04" if abs(d - 0.04) < 1e-6 else ""
    print(f"{d:>9} {m.sum():>4} | {bm:6.1f} {f'{bl:.1f}-{bh:.1f}':>13} | {am:6.1f} {f'{al:.1f}-{ah:.1f}':>13} | {sm:9.1f}{mark}")
