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
With --mix-target, human_score = 0.5*clamp(ssim2/100,0,1) +
0.5*clamp((cvvdp-6)/4,0,1) (JOD 6..10 → 0..1; both higher=better) and rows
with missing cvvdp are DROPPED (counted + printed).
LSD origin rule on the leading numeric stem (origin_split.py — the imazen-26
convention; HDR stems like `1064_general_...` lead with the origin id).
Validates with validate_parquet (contracts declared inline) and prints the
sha256s for manifest pinning.

  usage: build_hdr_train_parquets.py [--datagen DIR] [--out-prefix P] [--mix-target]
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
ap.add_argument("--mix-target", action="store_true",
                help="human_score = 0.5*clamp(ssim2/100,0,1) + 0.5*clamp((cvvdp-6)/4,0,1) "
                     "(JOD 6..10 -> 0..1); rows with missing cvvdp are DROPPED")
ap.add_argument("--iw-target", choices=["mix", "pure"], default=None,
                help="iwssim-teacher targets (teacher-ceiling probe, "
                     "bhdr_improvement_split_lineage_2026-07-12.md §8.3). "
                     "iw_logn = clamp(-log10(clamp(1-iw,1e-6,1))/4, 0, 1) spreads the "
                     "near-1 saturation. mix: human_score = 0.5*s2n + 0.5*iw_logn; "
                     "pure: human_score = iw_logn. iwssim-missing rows are DROPPED.")
ap.add_argument("--iwssim-sidecar", action="append", default=[],
                help="iwssim.parquet path(s) to join by (basename, codec, q) — the "
                     "v3 pu-linear datagens carry no iwssim sidecar of their own; the "
                     "scores live in the old-feature datagens over the SAME encodes "
                     "(key overlap verified 17,100/17,100 on 2026-07-12).")
ap.add_argument("--codec", default="zenjxl",
                help="sidecar/omni codec dir name inside each datagen "
                     "(sidecars/<codec>/, omni/<codec>.tsv). Non-zenjxl HDR "
                     "families (kadis-hdr synthetic, zenavif) use their own dir.")
a = ap.parse_args()
assert not (a.mix_target and a.iw_target), "--mix-target and --iw-target are exclusive"
assert not a.iw_target or a.iwssim_sidecar, "--iw-target requires --iwssim-sidecar"

def key(t):
    return list(zip((os.path.basename(x) for x in t["image_path"].to_pylist()),
                    t["codec"].to_pylist(),
                    [float(x) for x in t["q"].to_pylist()]))

def load_scores(d):
    """(key -> (ssim2, cvvdp)) from a datagen dir's omni + sidecars.

    ssim2 comes from the omni's inline score when present, else from the
    `ssim2-gpu.parquet` sidecar (non-encode families like kadis-hdr have no
    inline encode-time score — their ssim2 is a score-pairs pass).
    """
    out = {}
    omni = os.path.join(d, "omni", f"{a.codec}.tsv")
    if os.path.exists(omni):
        import csv
        for r in csv.DictReader(open(omni), delimiter="\t"):
            s2 = r.get("score_ssim2") or r.get("score_ssim2_gpu") or ""
            if s2:
                out[(os.path.basename(r["image_path"]), r["codec"], float(r["q"]))] = [float(s2), None]
    s2p = os.path.join(d, "sidecars", a.codec, "ssim2-gpu.parquet")
    if os.path.exists(s2p):
        t = pq.read_table(s2p)
        col = [c for c in t.schema.names if c not in ("image_path", "codec", "q", "knob_tuple_json")][0]
        for k, v in zip(key(t), np.asarray(t[col], dtype=float)):
            if np.isfinite(v):
                out.setdefault(k, [None, None])
                if out[k][0] is None:
                    out[k][0] = float(v)
    cv = os.path.join(d, "sidecars", a.codec, "cvvdp.parquet")
    if os.path.exists(cv):
        t = pq.read_table(cv)
        col = [c for c in t.schema.names if c not in ("image_path", "codec", "q", "knob_tuple_json")][0]
        for k, v in zip(key(t), np.asarray(t[col], dtype=float)):
            out.setdefault(k, [None, None])[1] = float(v)
    return out

# Global iwssim map — keys are (basename, codec, q), so dir-agnostic; the scores
# were computed on the same encodes the v3 features were re-extracted from.
IW = {}
for p in a.iwssim_sidecar:
    t = pq.read_table(p)
    col = [c for c in t.schema.names if c not in ("image_path", "codec", "q", "knob_tuple_json")][0]
    for k, v in zip(key(t), np.asarray(t[col], dtype=float)):
        IW[k] = float(v)
if a.iwssim_sidecar:
    print(f"iwssim map: {len(IW):,} keys from {len(a.iwssim_sidecar)} sidecar(s)")

def iw_logn(iw: float) -> float:
    d = min(max(1.0 - iw, 1e-6), 1.0)
    return min(max(-np.log10(d) / 4.0, 0.0), 1.0)

rows = {"ref_basename": [], "human_score": [], "score_cvvdp": [], "zensim_score": []}
if a.iwssim_sidecar:
    rows["score_iwssim"] = []
feats = []
n_miss_scores = 0
n_miss_cvvdp = 0
n_miss_iw = 0
for d in [a.datagen] + a.extra_datagen:
    fp = os.path.join(d, "sidecars", a.codec, "zensim_features.parquet")
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
    seen_content = set()
    for i, k in enumerate(key(t)):
        sc = scores.get(k)
        if not sc or sc[0] is None:
            n_miss_scores += 1
            continue
        # mix mode: cvvdp is load-bearing — drop rows without it. Checked
        # BEFORE dedup registration so a later identical-content row that
        # DOES have cvvdp can still join.
        if a.mix_target and (sc[1] is None or not np.isfinite(sc[1])):
            n_miss_cvvdp += 1
            continue
        iw = IW.get(k)
        if a.iw_target and (iw is None or not np.isfinite(iw)):
            n_miss_iw += 1
            continue
        # dedup-by-content (DATA_SPLITS policy): zenjxl --hdr floors q<15, so
        # q5==q15 byte-identical on every rendition (verified 2026-07-03,
        # 1,140/7,980 cells). Identical features+target = zero information.
        ck = (k[0], hash(F[i].tobytes()), sc[0])
        if ck in seen_content:
            continue
        seen_content.add(ck)
        rows["ref_basename"].append(k[0])
        s2n = min(max(sc[0] / 100.0, 0.0), 1.0)
        if a.mix_target:
            cvn = min(max((sc[1] - 6.0) / 4.0, 0.0), 1.0)
            rows["human_score"].append(0.5 * s2n + 0.5 * cvn)
        elif a.iw_target == "mix":
            rows["human_score"].append(0.5 * s2n + 0.5 * iw_logn(iw))
        elif a.iw_target == "pure":
            rows["human_score"].append(iw_logn(iw))
        else:
            rows["human_score"].append(s2n)
        rows["score_cvvdp"].append(sc[1] if sc[1] is not None else float("nan"))
        if a.iwssim_sidecar:
            rows["score_iwssim"].append(iw if iw is not None else float("nan"))
        rows["zensim_score"].append(float(zs[i]))
        feats.append(F[i])
print(f"joined rows: {len(feats):,} (score-missing skipped: {n_miss_scores})")
if a.mix_target:
    print(f"mix-target: cvvdp-missing rows DROPPED: {n_miss_cvvdp}")
if a.iw_target:
    print(f"iw-target({a.iw_target}): iwssim-missing rows DROPPED: {n_miss_iw}")
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
                     out["train"][0], out["val"][0], "--kind", "train", "--allow-const-cols", "2"]).returncode
sys.exit(rc)
