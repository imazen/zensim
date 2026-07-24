#!/usr/bin/env python3
"""720-feature severe-HONEST corpus — negrich methodology, matched content.

negrich (severe-but-honest KADIS degradations) is the corruption detector's hard
negatives, but it is native-372 and the shared dial+diffmap subset needs v2. So
regenerate it at 720 the RIGHT way: run the same kadis-distort 25 IQA distortion
types (blur/noise/compress/color/pixelate...) at severe levels {3,4,5} on the SAME
imazen-26 sources the structural corruptions use — matched content (no confound) at
720. Deterministic (kadis seed_for), streaming generate→extract→discard.

Output rows: f0..f719 + is_corruption=0 + neg_subclass=severe_honest + dist_type +
level + content_class + ref_id — so it drops straight into train_corruption_head.py.
"""
import argparse, os, subprocess, shutil, sys, tempfile
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.expanduser("~/work/kadis-distort"))
from kadis_distort.io import load_ref, save_image, rng_for
from kadis_distort.generator import imdist_generator, LEVELS

EXTRACT = "./target/release/examples/v2_ab_extract"
FEATCOLS = [f"f{i}" for i in range(720)]
MAX_DIM = 1024
SEVERE_LEVELS = [3, 4, 5]           # negative-rich (score p50 14 / -16 / -57)
SKIP_TYPES = {15}                   # 15 = denoise_dncnn (needs the DnCNN weights)


def process_ref(ref_path, ref_id, cclass, tmpdir):
    try:
        im = Image.open(ref_path).convert("RGB")
        if max(im.size) > MAX_DIM:
            s = MAX_DIM / max(im.size)
            im = im.resize((round(im.size[0]*s), round(im.size[1]*s)), Image.LANCZOS)
        small = os.path.join(tmpdir, "ref.png"); im.save(small)
        ref = load_ref(small)               # float [0,1]
    except Exception as e:
        print(f"  SKIP {ref_id}: load ({e})", flush=True); return None
    gen = os.path.join(tmpdir, "g"); shutil.rmtree(gen, ignore_errors=True); os.makedirs(gen)
    labels = []
    with open(os.path.join(tmpdir, "pairs.tsv"), "w") as f:
        f.write("ref_path\tdist_path\thuman_score\n")
        i = 0
        for dt in sorted(LEVELS):
            if dt in SKIP_TYPES:
                continue
            for lv in SEVERE_LEVELS:
                try:
                    d = imdist_generator(ref, dt, lv, rng=rng_for(ref_id, dt, lv))
                    dp = os.path.join(gen, f"d_{dt}_{lv}.png"); save_image(d, dp)
                except Exception:
                    continue
                f.write(f"{small}\t{dp}\t{i}\n"); labels.append((dt, lv)); i += 1
    if not labels:
        shutil.rmtree(gen, ignore_errors=True); return None
    fcsv = os.path.join(tmpdir, "f.csv")
    r = subprocess.run([EXTRACT, os.path.join(tmpdir, "pairs.tsv"), fcsv],
                       capture_output=True, text=True)
    shutil.rmtree(gen, ignore_errors=True)
    if r.returncode != 0 or not os.path.exists(fcsv):
        print(f"  SKIP {ref_id}: extract ({r.stderr.strip()[:100]})", flush=True); return None
    import csv as _csv
    with open(fcsv) as fh:
        rd = _csv.reader(fh); hdr = next(rd)
        if any(c not in hdr for c in FEATCOLS) or "human_score" not in hdr:
            print(f"  SKIP {ref_id}: truncated CSV", flush=True); return None
        fi = [hdr.index(c) for c in FEATCOLS]; hj = hdr.index("human_score")
        feats = {int(float(row[hj])): np.array([float(row[j]) for j in fi], np.float32) for row in rd}
    cols = {c: [] for c in FEATCOLS}
    meta = dict(is_corruption=[], neg_subclass=[], dist_type=[], level=[], content_class=[], ref_id=[])
    for k, (dt, lv) in enumerate(labels):
        v = feats.get(k)
        if v is None or not np.all(np.isfinite(v)):
            continue
        for j, c in enumerate(FEATCOLS):
            cols[c].append(float(v[j]))
        meta["is_corruption"].append(0); meta["neg_subclass"].append("severe_honest")
        meta["dist_type"].append(int(dt)); meta["level"].append(int(lv))
        meta["content_class"].append(cclass); meta["ref_id"].append(ref_id)
    if not meta["is_corruption"]:
        return None
    return pa.table({**{c: pa.array(cols[c], pa.float32()) for c in FEATCOLS},
                     **{k: pa.array(v) for k, v in meta.items()}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    srcs = [l.rstrip("\n").split("\t") for l in open(a.sources) if l.strip()]
    writer = None; n = 0
    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/tmp")) as tmp:
        for k, (rp, rid, cc) in enumerate(srcs):
            if not os.path.exists(rp):
                continue
            t = process_ref(rp, rid, cc, tmp)
            if t is None:
                continue
            if writer is None:
                writer = pq.ParquetWriter(a.out, t.schema, compression="zstd")
            writer.write_table(t); n += t.num_rows
            print(f"[{k+1}/{len(srcs)}] {rid}: +{t.num_rows} severe-honest (tot {n})", flush=True)
    if writer:
        writer.close()
    print(f"DONE: {a.out}  {n} severe-honest rows", flush=True)


if __name__ == "__main__":
    main()
