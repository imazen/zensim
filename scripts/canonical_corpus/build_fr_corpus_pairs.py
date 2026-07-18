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


def build_live():
    """LIVE IQA Release 2: 29 refs × {jp2k,jpeg,wn,gblur,fastfading}, 779 real distortions.

    readme.txt fixes the concat order EXACTLY:
      dmos=[jp2k(1:227) jpeg(1:233) wn(1:174) gblur(1:174) fastfading(1:174)]  (982 total)
    orgs(i)==1 marks a reference-copy placed in-sequence (dmos~0) -> skip (982-779=203).
    We use the REALIGNED dmos (dmos_new, Sheikh 2006 recommended) + refnames_all for the
    ref join. human_score = 1 - dmos_new/100  (quality-oriented [0,1], higher=better, to
    match kadid/tid/csiq). dmos_std (per-sample sigma) is emitted as a 4th column for
    later Z-RMSE; the feature extractor reads only the first 3 (ref/dist/human_score)."""
    import scipy.io as sio
    import numpy as np
    BASE = "/mnt/v/datasets/LIVE/databaserelease2"
    OUT = "/mnt/v/datasets/LIVE/live_r2_pairs.tsv"
    # (folder, global offset, count) — offsets are the readme concat order.
    SEG = [("jp2k", 0, 227), ("jpeg", 227, 233), ("wn", 460, 174),
           ("gblur", 634, 174), ("fastfading", 808, 174)]
    rea = sio.loadmat(f"{BASE}/dmos_realigned.mat")
    dmos = np.asarray(rea["dmos_new"]).flatten()          # realigned DMOS, ~[-3,112]
    std = np.asarray(rea["dmos_std"]).flatten()           # per-sample sigma
    orgs = np.asarray(rea["orgs"]).flatten().astype(int)  # 1 == reference-copy -> skip
    refs = [str(x[0]) for x in np.asarray(sio.loadmat(f"{BASE}/refnames_all.mat")["refnames_all"]).flatten()]
    out, miss = [], 0
    for folder, off, cnt in SEG:
        for k in range(1, cnt + 1):
            gi = off + (k - 1)
            if orgs[gi] == 1:            # reference-copy in-sequence, not a real distortion
                continue
            ref = f"{BASE}/refimgs/{refs[gi]}"
            dist = f"{BASE}/{folder}/img{k}.bmp"
            if not (Path(ref).exists() and Path(dist).exists()):
                miss += 1
                continue
            out.append((ref, dist, 1.0 - float(dmos[gi]) / 100.0, float(std[gi])))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score", "sigma"])
        w.writerows(out)
    print(f"LIVE R2: {len(out)} pairs -> {OUT}  (skipped {miss} missing)")


BUILDERS = {"csiq": build_csiq, "live": build_live}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in BUILDERS:
        print(f"builders: {list(BUILDERS)}")
        sys.exit(2)
    BUILDERS[name]()
