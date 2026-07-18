#!/usr/bin/env python3
"""Build ref/dist/human_score pairs manifests for FR-IQA corpora → feed
`extract_features_372col --corpus pairs-tsv`. One reproducible builder per dataset so the
FR-corpus expansion (CSIQ/LIVE/TID2008/...) isn't amnesiac.

Convention (MUST match kadid/tid): human_score is QUALITY-oriented in [0,1] (higher = better).
Datasets whose native label is a distortion score (DMOS higher=worse) are flipped to 1−norm.
Output TSV columns: ref_path, dist_path, human_score  (the tool derives ref_basename).

Usage: python3 scripts/canonical_corpus/build_fr_corpus_pairs.py <csiq|live|tid2008>
Then:  extract_features_372col --corpus pairs-tsv --path <out.tsv> --out <corpus>_features_372col.csv
       (convert csv→parquet, add a Corpus entry to bake_verdict CORPORA, rebuild)
"""
import sys, csv
from pathlib import Path


def build_csiq():
    """CSIQ: 30 refs × 6 distortions. DMOS in [0,1] (0=best). human = 1 − DMOS."""
    import openpyxl
    SRC = "/mnt/v/dataset/csiq"
    DST = "/mnt/v/datasets/csiq/dst_imgs"
    OUT = "/mnt/v/datasets/csiq/csiq_pairs.tsv"
    # dst_type (xlsx) -> (folder, filename_token)
    M = {"noise": ("awgn", "AWGN"), "blur": ("blur", "BLUR"), "contrast": ("contrast", "contrast"),
         "fnoise": ("fnoise", "fnoise"), "jpeg": ("jpeg", "JPEG"), "jpeg 2000": ("jpeg2000", "jpeg2000")}
    ws = openpyxl.load_workbook("/mnt/v/datasets/csiq/csiq.DMOS.xlsx")["all_by_image"]
    rows = [r for r in ws.iter_rows(values_only=True)]
    hi = next(i for i, r in enumerate(rows) if r and "image" in [str(x) for x in r])
    hdr = [str(x) for x in rows[hi]]
    I = hdr.index
    out, miss = [], 0
    for r in rows[hi + 1:]:
        if not r or r[I("image")] is None or r[I("dmos")] is None:
            continue
        img, dt = str(r[I("image")]), str(r[I("dst_type")])
        lev = str(r[I("dst_lev")]).split(".")[0]
        if dt not in M:
            continue
        folder, tok = M[dt]
        ref = f"{SRC}/{img}.png"
        dist = f"{DST}/{folder}/{img}.{tok}.{lev}.png"
        if not (Path(ref).exists() and Path(dist).exists()):
            miss += 1
            continue
        out.append((ref, dist, 1.0 - float(r[I("dmos")])))  # 1 − DMOS → quality-oriented
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        w.writerows(out)
    print(f"CSIQ: {len(out)} pairs → {OUT}  (skipped {miss} missing)")


BUILDERS = {"csiq": build_csiq}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in BUILDERS:
        print(f"builders: {list(BUILDERS)}")
        sys.exit(2)
    BUILDERS[name]()
