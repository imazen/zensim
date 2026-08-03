#!/usr/bin/env python3
"""build_hdr944_sidecars.py — package the hdr944_extract TSV into per-datagen
feature sidecars shaped for build_hdr_train_parquets.py (SOTA-944 hdr_v3mix
amendment; benchmarks/sota944_campaign_2026-08-03.md "B-gap resolution").

Per datagen: sidecars/zenjxl/zensim_features.parquet with
  image_path / codec / q / knob_tuple_json / zensim_score / feat_0..feat_943
where image_path/q come from the datagen's pairs TSV (keyed by dist basename),
zensim_score is CARRIED from the v3 sidecar by (basename, codec, q) — the v3
zensim scalar is informational, never a target — and features are the fresh
944 HDR-PQ extraction. cvvdp sidecar + omni are symlinked from the v3/original
dirs so load_scores() resolves targets identically.

FRONT-END NOTE (the leg's manifest carries this): features are the CURRENT
chunk-2 HDR route at 944 (foldapp2hdrpq class), NOT the v3 pu-linear-372
front-end — a NEW-REGIME leg whose carried asset is the TARGET.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-tsv", required=True, help="hdr944_extract output")
    ap.add_argument("--datagen", action="append", required=True,
                    help="original datagen dir (repeatable; pairs/ + omni/ live here)")
    ap.add_argument("--v3-sidecar-dir", action="append", required=True,
                    help="v3 sidecar dir per datagen (zensim_features + cvvdp.parquet)")
    ap.add_argument("--out-datagen", action="append", required=True,
                    help="output 944 datagen dir per input datagen")
    a = ap.parse_args()
    assert len(a.datagen) == len(a.out_datagen) == len(a.v3_sidecar_dir)

    # features TSV -> dist_basename -> (q, feats[944])
    import csv
    feat_by_dist = {}
    with open(a.features_tsv) as f:
        rdr = csv.reader(f, delimiter="\t")
        header = next(rdr)
        assert header[0] == "dist_basename" and header[1] == "q"
        n_feat = len(header) - 2
        assert n_feat == 944, f"width {n_feat}"
        for row in rdr:
            feat_by_dist[row[0]] = (row[1], np.array(row[2:], dtype=np.float64))
    print(f"features TSV: {len(feat_by_dist):,} cells x {n_feat}")

    for dg, v3sc, outdg in zip(a.datagen, a.v3_sidecar_dir, a.out_datagen):
        pairs = Path(dg) / "pairs" / "zenjxl.pairs.tsv"
        # (image_path, q, dist_basename) per cell, pairs order
        cells = []
        with open(pairs) as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for r in rdr:
                cells.append((r["image_path"], r["q"],
                              os.path.basename(r["dist_path"])))
        # zensim_score carry from the v3 sidecar by (basename, codec, q)
        v3 = pq.read_table(Path(v3sc) / "zensim_features.parquet",
                           columns=["image_path", "codec", "q", "zensim_score"])
        zmap = {}
        for ip, cd, qv, zs in zip(v3["image_path"].to_pylist(), v3["codec"].to_pylist(),
                                  v3["q"].to_pylist(), v3["zensim_score"].to_pylist()):
            zmap[(os.path.basename(ip), cd, float(qv))] = zs
        rows = {"image_path": [], "codec": [], "q": [], "knob_tuple_json": [],
                "zensim_score": []}
        feats = []
        miss = 0
        for ip, qv, db in cells:
            got = feat_by_dist.get(db)
            if got is None:
                miss += 1
                continue
            tq, fv = got
            assert float(tq) == float(qv), f"q mismatch {db}: {tq} vs {qv}"
            rows["image_path"].append(ip)
            rows["codec"].append("zenjxl")
            rows["q"].append(float(qv))
            rows["knob_tuple_json"].append("{}")
            z = zmap.get((os.path.basename(ip), "zenjxl", float(qv)))
            rows["zensim_score"].append(float(z) if z is not None else float("nan"))
            feats.append(fv)
        assert miss == 0, f"{dg}: {miss} pairs missing from features TSV"
        F = np.vstack(feats)
        data = {k: pa.array(v) for k, v in rows.items()}
        for j in range(944):
            data[f"feat_{j}"] = pa.array(F[:, j])
        out_sc = Path(outdg) / "sidecars" / "zenjxl"
        out_sc.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(data), out_sc / "zensim_features.parquet",
                       compression="zstd")
        # cvvdp sidecar + omni: symlink from the v3/original dirs.
        cv_src = Path(v3sc) / "cvvdp.parquet"
        cv_dst = out_sc / "cvvdp.parquet"
        if cv_src.exists() and not cv_dst.exists():
            os.symlink(cv_src, cv_dst)
        omni_dst = Path(outdg) / "omni"
        if not omni_dst.exists():
            os.symlink(Path(dg).resolve() / "omni", omni_dst)
        print(f"{outdg}: {len(feats):,} cells -> sidecars/zenjxl/zensim_features.parquet")
        manifest = {
            "leg": "hdr_v3mix-944 feature sidecar",
            "front_end": "foldapp2hdrpq (chunk-2 HDR route @944) — NEW REGIME vs the "
                         "v3 pu-linear-372 front-end; targets are the carried asset",
            "source_pairs": str(pairs),
            "features_tsv": a.features_tsv,
            "v3_zensim_score_carried_from": str(Path(v3sc) / "zensim_features.parquet"),
        }
        (Path(outdg) / "_MANIFEST_944.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
