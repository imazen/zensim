#!/usr/bin/env python3
"""Rebuild the 372-wide G-ADDR instruments (dial grid + identity + negative-tail
probe) at a chosen extractor era, from surviving PIXELS.

WHY THIS EXISTS (measured 2026-09-05, `benchmarks/d_peaks_372_postC_2026-09-05.md`).
The shipped runtime is one extraction era ahead of every stored 372 artifact
(option C, `56bbcda2`), so a G-ADDR reading taken on the stored instruments does
not describe the dial a user gets. Rebuilding needs the PIXELS, and the three
instruments are in three different states:

* **dial grid** — the canonical grid's OWN pixels are GONE. Its build list
  `/mnt/v/output/zensim/eval_panels_2026-05-29/qsweep_372_grid.tsv` points every
  distorted cell at `/mnt/v/input/zensim/images/<ref>/<codec>/q<N>.png`, the
  decode cache DELETED 2026-06-22 (`CLAUDE.md`), and **0 of 2,560 of those paths
  exist**. The surviving `dial-grid-pixels-2026-07-27/` set is a DIFFERENT
  encode of the same (image, codec, q) lattice — `zenjpeg` where the 2026-05-29
  jpeg leg was `mozjpeg-rs-420-e4`. So a rebuild here is a NEW instrument on the
  same key set, never a re-read of the old one; it is registered separately and
  the old grid is not touched.
  **It is the better-matched instrument for the bars**: the registered
  `peer_ssim2` grid pins ARE ssim2 on the 2026-07-27 pixels — measured, in
  `dialcells_ssim2_qv2grid.tsv`, which is `dialcells_ssim2_944grid.tsv`
  restricted to these 4,424 keys (4,424 of 4,424 exactly equal) and whose
  min/max/p5/p95 reproduce the registry row to the last digit.
* **identity probe** — era-INVARIANT at w372 and measured so: a fresh HEAD
  extraction of the same 38 `ref == dist` pairs is the zero vector on all
  14,136 cells, 0 differing from the stored probe. Rebuilt anyway so the
  property stays gated rather than assumed.
* **negative-tail probe** — rebuildable. Its 2,000 rows join uniquely (2,000 of
  2,000, 0 ambiguous) to `kadis700k_canonical_gpu_2026-07-01.parquet` on
  `(feat_0, feat_1, feat_2, score_ssim2_gpu)`, which supplies `distorted_url`
  (R2-persisted PNG) and `source_filename` (local under
  `/mnt/v/datasets/kadis700k/refs/`, 2,000 of 2,000 present).

Extraction is `extract_features_372col --corpus pairs-tsv` — THE owner. Nothing
here computes a feature.

Usage:
  build_dial372_instruments.py --extractor <bin> --out-dir <dir> --tag postC \\
      [--negtail-pixels <dir>] [--what grid,identity,negtail]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

CANON_GRID = ("/mnt/v/output/zensim/eval_panels_2026-05-29/"
              "dial_grid_372col_2026-05-29_quarantined_v2.parquet")
PAIRS_TSV = "/mnt/v/output/zensim/reports/refmetrics/dialgrid_pairs.tsv"
IDENTITY_PAIRS = "/mnt/v/output/zensim/dialgate-2026-09-04/build/identity_pairs.tsv"
NEGTAIL_PROBE = ("/mnt/v/output/zensim/dialgate-2026-09-04/"
                 "negtail_probe_372_2026-09-04.parquet")
KADIS_CANON = ("/mnt/v/datasets/kadis700k/canonical/"
               "kadis700k_canonical_gpu_2026-07-01.parquet")
KADIS_REFS = "/mnt/v/datasets/kadis700k/refs"
NF = 372
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".avif", ".jxl")


def stem(name: str) -> str:
    low = name.lower()
    for e in EXTS:
        if low.endswith(e):
            return name[: -len(e)]
    return name


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_pairs(path: str, cols: list[str], rows: list[list]) -> None:
    """Write a pairs TSV carrying a `row_id` the extractor hands back.

    THE ORDER TRAP (found the hard way 2026-09-05, see `extract`)."""
    with open(path, "w") as f:
        f.write("\t".join(cols + ["row_id"]) + "\n")
        for i, r in enumerate(rows):
            f.write("\t".join([str(x) for x in r] + [str(i)]) + "\n")


def extract(extractor: str, pairs_tsv: str, out_csv: str) -> list[list[float]]:
    """Run THE extractor over a pairs TSV and return features IN INPUT ORDER.

    **`extract_features_372col` does NOT emit rows in input order.** It ends
    with `rows.sort_by(|a, b| a.0.cmp(&b.0))` — a stable sort on
    `ref_basename` — so groups come out alphabetically by reference. Attaching
    key columns positionally after it is therefore WRONG for any input not
    already sorted that way, and it fails SILENTLY: the row count is right,
    every key is present exactly once, and only the pairing is scrambled.

    MEASURED cost of not knowing this: the first build of the 2026-09-05 dial
    grid attached (image_id, codec, q) positionally, which shuffled features
    across the 38 references. Every scorer's ladder monotonicity collapsed
    (shipped D 0.9847 -> 0.5611) and jpeg q0 read a dial of 75.6 on an image
    whose ssim2 truth is -8.03. Nothing errored.

    The fix needs no change to the extractor (whose sort other callers'
    positional comparisons against stored tables depend on): the pairs TSV
    carries a numeric `row_id`, `load_pairs_tsv` forwards every numeric extra
    column, and we sort back by it. The permutation is then CHECKED, not
    assumed."""
    r = subprocess.run([extractor, "--corpus", "pairs-tsv", "--path", pairs_tsv,
                        "--out", out_csv], capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"extractor failed rc={r.returncode}")
    rows = list(csv.DictReader(open(out_csv)))
    n = sum(1 for _ in open(pairs_tsv)) - 1
    if len(rows) != n:
        raise SystemExit(f"extractor emitted {len(rows)} rows for {n} pairs")
    if "row_id" not in rows[0]:
        raise SystemExit("the extractor did not carry `row_id` back — refusing to "
                         "attach key columns positionally (see this function's docstring)")
    out: list[list[float] | None] = [None] * n
    for row in rows:
        i = int(round(float(row["row_id"])))
        if not 0 <= i < n or out[i] is not None:
            raise SystemExit(f"row_id {i} is out of range or duplicated — refusing")
        out[i] = [float(row[f"f{j}"]) for j in range(NF)]
    if any(x is None for x in out):
        raise SystemExit("some row_id never came back — refusing")
    return out  # type: ignore[return-value]


def grid_pair_order() -> tuple[list[dict], dict]:
    """Map every canonical-grid row to its 2026-07-27 pixel pair. Refuses on any
    miss — a partially-mapped grid is a different instrument, not a smaller one."""
    t = pq.read_table(CANON_GRID, columns=["image_id", "codec", "q", "codec_param",
                                           "param_kind"]).to_pydict()
    rows = list(csv.DictReader(open(PAIRS_TSV), delimiter="\t"))
    idx = {}
    for i, r in enumerate(rows):
        knob = json.loads(r["knob_tuple_json"] or "{}")
        p = float(knob["distance"]) if "distance" in knob else float(r["q"])
        k = (stem(os.path.basename(r["ref_path"])), r["codec"].replace("zen", "", 1),
             round(p, 6))
        idx.setdefault(k, i)
    out, miss = [], []
    for i in range(len(t["image_id"])):
        k = (t["image_id"][i], t["codec"][i], round(float(t["codec_param"][i]), 6))
        j = idx.get(k)
        if j is None:
            miss.append(k)
        else:
            out.append(rows[j])
    if miss:
        raise SystemExit(f"{len(miss)} canonical grid cells have no surviving pixel pair, "
                         f"first {miss[0]} — refusing")
    return out, t


def build_grid(args, work: str) -> dict:
    pairs, meta = grid_pair_order()
    tsv = os.path.join(work, "grid_pairs.tsv")
    write_pairs(tsv, ["ref_path", "dist_path", "human_score"],
                [[r["ref_path"], r["dist_path"], f"{float(p):.6f}"]
                 for r, p in zip(pairs, meta["codec_param"])])
    feats = extract(args.extractor, tsv, os.path.join(work, "grid.csv"))
    assert len(feats) == len(pairs)
    cols = {"image_id": meta["image_id"], "codec": meta["codec"], "q": meta["q"],
            "codec_param": meta["codec_param"], "param_kind": meta["param_kind"]}
    for i in range(NF):
        cols[f"f{i}"] = [row[i] for row in feats]
    dst = os.path.join(args.out_dir, f"dial_grid_372col_{args.tag}_2026-09-05.parquet")
    pq.write_table(pa.table(cols), dst, compression="zstd")
    return {"file": dst, "rows": len(feats), "sha256": sha256(dst),
            "pairs_tsv": tsv, "refs": len({r["ref_path"] for r in pairs})}


def build_identity(args, work: str) -> dict:
    src = list(csv.DictReader(open(IDENTITY_PAIRS), delimiter="\t"))
    entries = [os.path.basename(r["ref_path"]) for r in src]
    tsv = os.path.join(work, "identity_pairs.tsv")
    write_pairs(tsv, ["ref_path", "dist_path", "human_score"],
                [[r["ref_path"], r["dist_path"], r["human_score"]] for r in src])
    feats = extract(args.extractor, tsv, os.path.join(work, "identity.csv"))
    nonzero = sum(1 for row in feats for v in row if v != 0.0)
    cols = {"entry": entries}
    for i in range(NF):
        cols[f"f{i}"] = [row[i] for row in feats]
    dst = os.path.join(args.out_dir, f"identity_probe_372_{args.tag}_2026-09-05.parquet")
    pq.write_table(pa.table(cols), dst, compression="zstd")
    return {"file": dst, "rows": len(feats), "sha256": sha256(dst),
            "nonzero_cells": nonzero,
            "note": "the identity feature vector is the ZERO vector at w372; nonzero_cells "
                    "MUST be 0 (measured, not assumed — the 944 layout is NOT zero here)"}


def negtail_rows() -> list[dict]:
    """Join the stored probe's 2,000 rows back to KADIS pixels. Refuses on any
    miss or ambiguity — a probe rebuilt from a subset is a different probe."""
    import numpy as np  # noqa: F401  (kept local; only this path needs it)
    kc = pq.read_table(KADIS_CANON, columns=["feat_0", "feat_1", "feat_2",
                                             "score_ssim2_gpu", "distorted_url",
                                             "source_filename", "source_id"]).to_pydict()
    key: dict = {}
    for i in range(len(kc["feat_0"])):
        key.setdefault((kc["feat_0"][i], kc["feat_1"][i], kc["feat_2"][i],
                        kc["score_ssim2_gpu"][i]), []).append(i)
    pr = pq.read_table(NEGTAIL_PROBE, columns=["f0", "f1", "f2", "ssim2_gpu",
                                               "human_score", "entry"]).to_pydict()
    out, amb, miss = [], 0, 0
    for k in range(len(pr["f0"])):
        v = key.get((pr["f0"][k], pr["f1"][k], pr["f2"][k], pr["ssim2_gpu"][k]))
        if v is None:
            miss += 1
            continue
        if len(v) > 1:
            amb += 1
        i = v[0]
        out.append({"entry": pr["entry"][k], "ssim2_gpu": pr["ssim2_gpu"][k],
                    "human_score": pr["human_score"][k],
                    "url": kc["distorted_url"][i],
                    "ref": os.path.join(KADIS_REFS, kc["source_filename"][i]),
                    "source_id": kc["source_id"][i]})
    if miss or amb:
        raise SystemExit(f"negtail join: {miss} unmatched, {amb} ambiguous — refusing")
    return out


def build_negtail(args, work: str) -> dict:
    rows = negtail_rows()
    px = args.negtail_pixels
    missing = [r for r in rows if not os.path.exists(os.path.join(px, os.path.basename(r["url"])))]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(rows)} negtail PNGs absent under {px}. "
                         "Fetch them first:\n"
                         "  set -a; . ~/.config/cloudflare/r2-credentials; set +a\n"
                         "  s5cmd --endpoint-url https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com "
                         "run <cp commands built from distorted_url>")
    tsv = os.path.join(work, "negtail_pairs.tsv")
    write_pairs(tsv, ["ref_path", "dist_path", "human_score", "ssim2_gpu"],
                [[r["ref"], os.path.join(px, os.path.basename(r["url"])),
                  repr(r["human_score"]), repr(r["ssim2_gpu"])] for r in rows])
    feats = extract(args.extractor, tsv, os.path.join(work, "negtail.csv"))
    cols = {"entry": [r["entry"] for r in rows],
            "ssim2_gpu": [r["ssim2_gpu"] for r in rows],
            "human_score": [r["human_score"] for r in rows]}
    for i in range(NF):
        cols[f"f{i}"] = [row[i] for row in feats]
    dst = os.path.join(args.out_dir, f"negtail_probe_372_{args.tag}_2026-09-05.parquet")
    pq.write_table(pa.table(cols), dst, compression="zstd")
    return {"file": dst, "rows": len(feats), "sha256": sha256(dst),
            "sources": len({r["source_id"] for r in rows}),
            "pixels_dir": px}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", required=True,
                    help="path to extract_features_372col built at the target era")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True, help="era tag, e.g. postC / preC")
    ap.add_argument("--build-commit", default="", help="the extractor's tree commit")
    ap.add_argument("--negtail-pixels",
                    default="/mnt/v/output/zensim/dpeaks372-2026-09-05/negtail_pixels")
    ap.add_argument("--what", default="grid,identity,negtail")
    ap.add_argument("--work", default=os.path.expanduser("~/tmp/dial372_instruments"))
    args = ap.parse_args()
    work = os.path.join(args.work, args.tag)
    os.makedirs(work, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    want = set(args.what.split(","))
    man = {"built_utc": datetime.now(timezone.utc).isoformat(),
           "extractor": args.extractor, "build_commit": args.build_commit,
           "tag": args.tag, "instruments": {}}
    if "grid" in want:
        man["instruments"]["dial_grid"] = build_grid(args, work)
    if "identity" in want:
        man["instruments"]["identity_probe"] = build_identity(args, work)
    if "negtail" in want:
        man["instruments"]["negtail_probe"] = build_negtail(args, work)
    mp = os.path.join(args.out_dir, f"_MANIFEST_{args.tag}.json")
    with open(mp, "w") as f:
        json.dump(man, f, indent=1)
    for k, v in man["instruments"].items():
        print(f"{k:16s} rows={v['rows']:6d} sha={v['sha256'][:16]} {v['file']}")
    print("wrote", mp)


if __name__ == "__main__":
    main()
