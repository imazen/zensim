#!/usr/bin/env python3
"""Build the HDR training parquets from the datagen sidecars — the HDR
counterpart of the SDR corpus builds (PLAN_HDR step 2).

Joins, per (image_path, codec, q):
  - zensim_features.parquet  (372 PU21 feat cols + zensim_score)
  - the omni's inline ssim2  (or ssim2-gpu sidecar when present)
  - cvvdp.parquet            (JOD)
into train/val digit-split parquets:
  ref_basename, human_score (= clamp(ssim2/100)), score_cvvdp, zensim_score,
  f0..f371
LSD origin rule on the leading numeric stem (origin_split.py — the imazen-26
convention; HDR stems like `1064_general_...` lead with the origin id).
Validates with validate_parquet (contracts declared inline) and prints the
sha256s for manifest pinning.

  usage: build_hdr_train_parquets.py [--datagen DIR] [--out-prefix P]
"""
import argparse, hashlib, os, subprocess, sys
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.expanduser("~/work/zen/zenmetrics/scripts/picker"))
from origin_split import split_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--datagen", default="/mnt/v/output/zenmetrics/datagen-2026-06-23-hdr")
ap.add_argument("--extra-datagen", action="append", default=[],
                help="additional datagen dirs (e.g. the q90-100 top-up) merged in")
ap.add_argument("--out-prefix", default="/mnt/v/output/zensim-multicodec-probe/hdr_zenjxl")
ap.add_argument("--date", default="2026-07-03")
a = ap.parse_args()

def key(t):
    return list(zip((os.path.basename(x) for x in t["image_path"].to_pylist()),
                    t["codec"].to_pylist(),
                    [float(x) for x in t["q"].to_pylist()]))

def load_scores(d):
    """(key -> (ssim2, cvvdp)) from a datagen dir's omni + cvvdp sidecar."""
    out = {}
    omni = os.path.join(d, "omni", "zenjxl.tsv")
    if os.path.exists(omni):
        import csv
        for r in csv.DictReader(open(omni), delimiter="\t"):
            s2 = r.get("score_ssim2") or r.get("score_ssim2_gpu") or ""
            if s2:
                out[(os.path.basename(r["image_path"]), r["codec"], float(r["q"]))] = [float(s2), None]
    cv = os.path.join(d, "sidecars", "zenjxl", "cvvdp.parquet")
    if os.path.exists(cv):
        t = pq.read_table(cv)
        col = [c for c in t.schema.names if c not in ("image_path", "codec", "q", "knob_tuple_json")][0]
        for k, v in zip(key(t), np.asarray(t[col], dtype=float)):
            out.setdefault(k, [None, None])[1] = float(v)
    return out

rows = {"ref_basename": [], "human_score": [], "score_cvvdp": [], "zensim_score": []}
feats = []
n_miss_scores = 0
for d in [a.datagen] + a.extra_datagen:
    fp = os.path.join(d, "sidecars", "zenjxl", "zensim_features.parquet")
    if not os.path.exists(fp):
        print(f"NOTE: no features sidecar in {d} — skipped")
        continue
    t = pq.read_table(fp)
    md = pq.read_metadata(fp)
    assert md.num_rows > 0, f"IMPL BUG guard: features sidecar {fp} is EMPTY"
    fcols = sorted((c for c in t.schema.names if c.startswith("feat_")), key=lambda c: int(c.split("_")[1]))
    assert len(fcols) == 372, f"{fp}: {len(fcols)} feature cols"
    scores = load_scores(d)
    F = np.column_stack([np.asarray(t[c], dtype=float) for c in fcols])
    zs = np.asarray(t["zensim_score"], dtype=float)
    for i, k in enumerate(key(t)):
        sc = scores.get(k)
        if not sc or sc[0] is None:
            n_miss_scores += 1
            continue
        rows["ref_basename"].append(k[0])
        rows["human_score"].append(min(max(sc[0] / 100.0, 0.0), 1.0))
        rows["score_cvvdp"].append(sc[1] if sc[1] is not None else float("nan"))
        rows["zensim_score"].append(float(zs[i]))
        feats.append(F[i])
print(f"joined rows: {len(feats):,} (score-missing skipped: {n_miss_scores})")
assert feats, "no rows joined"
F = np.vstack(feats)
data = {k: pa.array(v) for k, v in rows.items()}
for j in range(372):
    data[f"f{j}"] = pa.array(F[:, j])
full = pa.table(data)

buckets = [split_of(n) for n in rows["ref_basename"]]
out = {}
for name, want in (("train", "train"), ("val", "val")):
    idx = [i for i, b in enumerate(buckets) if b == want]
    t = full.take(idx)
    p = f"{a.out_prefix}_{name}digits_{a.date}.parquet"
    pq.write_table(t, p, compression="zstd")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    out[name] = (p, t.num_rows, h)
    print(f"{name}: {t.num_rows:,} rows -> {p}\n  sha256 {h}")

rc = subprocess.run([sys.executable,
                     os.path.join(os.path.dirname(__file__), "..", "v_next", "validate_parquet.py"),
                     out["train"][0], out["val"][0], "--kind", "train"]).returncode
sys.exit(rc)
