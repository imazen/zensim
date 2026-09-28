#!/usr/bin/env python3
"""Pair lists for scoring DVIFM (and peers) on zensim's canonical corruption
packet of 2026-09-08 (`docs/CANONICAL_CORRUPTION_2026-09-08.md`).

Per split (validate = 8 origins, train = 12), every catalog row of the
accepted table (713 corruption attempts + the q10/q20 native JPEG anchors per
origin; inert attempts keep `is_corruption = 0`) plus the accepted honest
native supplement (JPEG XL / AVIF scalar-bound encodes, `honest-<split>.tsv`),
whose bitstreams are decoded once by zensim's decode owner
(`verify_bitstream_decode --decode-list`) so every metric sees the same pixels.

Nothing here reads a label beyond the packet's own `is_corruption` flags; the
packet (imazen-26 origins at longest side 256) carries no human labels.

Usage: corruption_prep.py decode|lists <out_dir> <decoded_root>
"""
import csv
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

PKT = Path("/mnt/v/output/zensim/canonical-corruption-2026-09-08")
SPLITS = ("validate", "train")


def sources(split):
    name = "val-sources.json" if split == "validate" else "train-sources.json"
    out = {}
    for s in json.load(open(PKT / name))["sources"]:
        p = Path(s["path"])
        if not p.exists():                       # the packet keeps a copy
            p = PKT / "source-inputs" / p.name
        assert p.exists(), p
        out[s["origin"]] = (str(p), s["content_class"])
    return out


def honest_link(root, split, dist):
    d = Path(dist)
    codec = d.suffix.lstrip(".")
    return Path(root) / "links" / f"corr_{split}" / f"{codec}__{d.stem}{d.suffix}", \
        Path(root) / f"corr_{split}" / f"{codec}__{d.stem}.png"


def main():
    mode, out, root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        src = sources(split)
        honest = list(csv.DictReader(open(PKT / f"honest-{split}.tsv"), delimiter="\t"))
        if mode == "decode":
            ldir = root / "links" / f"corr_{split}"
            ldir.mkdir(parents=True, exist_ok=True)
            lst = root / f"decode_corr_{split}.tsv"
            with open(lst, "w") as f:
                f.write("dist_path\n")
                for r in honest:
                    link, _ = honest_link(root, split, r["dist_path"])
                    if not link.exists():
                        os.symlink(r["dist_path"], link)
                    f.write(f"{link}\n")
            print(f"{split}: {len(honest)} honest bitstreams -> {lst} (out {root / f'corr_{split}'})")
            continue
        rows = []
        t = pq.read_table(PKT / f"{split}.parquet", columns=[
            "ref_id", "content_class", "is_corruption", "kind", "family", "region", "severity",
            "inert", "filename", "row_id"]).to_pylist()
        for r in t:
            ref, _cls = src[r["ref_id"]]
            kind = r["kind"]
            if kind == "honest_anchor":
                kind = f"anchor_q{r['severity']}"
            rows.append({"ref_path": ref, "dist_path": r["filename"],
                         "human_score": int(r["is_corruption"]), "source": r["ref_id"],
                         "codec": r["family"], "orig_dist_path": r["filename"],
                         "catalog_row": r["row_id"], "kind": kind, "inert": int(bool(r["inert"])),
                         "content_class": r["content_class"]})
        for r in honest:
            _, png = honest_link(root, split, r["dist_path"])
            origin = str(r["ref_id"])
            rows.append({"ref_path": src[origin][0], "dist_path": str(png), "human_score": 0,
                         "source": origin, "codec": Path(r["dist_path"]).suffix.lstrip("."),
                         "orig_dist_path": r["dist_path"], "catalog_row": f"honest:{r['row_id']}",
                         "kind": "honest_native", "inert": 0, "content_class": src[origin][1]})
        cols = ["ref_path", "dist_path", "human_score", "source", "codec", "orig_dist_path",
                "catalog_row", "kind", "inert", "content_class"]
        p = out / f"corruption_{split}.tsv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        missing = sum(1 for r in rows if not Path(r["dist_path"]).exists())
        kinds = {}
        for r in rows:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(f"{p.name}: {len(rows)} rows {kinds}; missing dist files {missing}")


if __name__ == "__main__":
    main()
