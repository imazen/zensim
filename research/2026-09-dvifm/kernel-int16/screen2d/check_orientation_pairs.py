#!/usr/bin/env python3
"""Phase-2d Part D — target-orientation gate over the PAIRS TSVs.

Companion to scripts/canonical_corpus/check_target_orientation.py (which
checks parquet feature tables): this applies the SAME sign test to the
ref/dist/human_score TSVs the standalone fitter consumes — same ground
truths (KADID raw DCR / dmos.csv, TID mos_with_names.txt), same verdict
vocabulary, verdicts written into pairs/ORIENTATION.json and each file's
manifest `target_orientation` field.

The mapping every leg shares after correction: quality_0_100 =
human_score * 100. Per-file source conventions:
  cid22a/b   MCOS/100                     (quality by definition)
  tid        MOS/9                        (published MOS, quality)
  kadid_*    (dmos-1)/4  — RE-DERIVED from dmos.csv; the upstream
             (5-dmos)/4 convention was exactly inverted (Amendment 1)
  imazen26   clip01(score_ssim2/100) — the wlin7 parquet's canonical
             human_score column; the first TSV erroneously carried raw
             ssim2 in 0-100 units (Amendment-1 units fix)
  konfig_val 1 - q/3.2                    (already quality, Appendix L)

usage: check_orientation_pairs.py   (writes pairs/ORIENTATION.json)
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/lilith/work/zen/zensim")
from scripts.lib.zen_stats import panel  # noqa: E402
from scripts.canonical_corpus import check_target_orientation as cto  # noqa

PAIRS = Path("/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/pairs")


def signed_srocc(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mag = panel(list(x), list(y))["srocc"]
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(mag) * (1.0 if np.cov(rx, ry)[0, 1] >= 0 else -1.0)


def load_hs(name):
    with open(PAIRS / name) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def check_kadid(name):
    dmos = {}
    with open(cto.KADID_DMOS) as f:
        for r in csv.DictReader(f):
            dmos[r["dist_img"]] = float(r["dmos"])
    rows = load_hs(name)
    xs, ys = [], []
    for r in rows:
        b = os.path.basename(r["dist_path"])
        if b in dmos:
            xs.append(float(r["human_score"]))
            ys.append(dmos[b])
    s = signed_srocc(xs, ys)
    return {"joined": len(xs), "rows": len(rows), "signed_srocc": s,
            "ground_truth": "kadid dmos.csv (quality-oriented mean DCR)",
            "verdict": "OK" if s > 0 else "INVERTED"}


def check_tid(name):
    mos = {}
    for line in open(cto.TID_MOS):
        p = line.split()
        if len(p) == 2:
            mos[p[1].lower().replace(".bmp", ".png")] = float(p[0])
    rows = load_hs(name)
    xs, ys = [], []
    for r in rows:
        b = os.path.basename(r["dist_path"]).lower()
        if b in mos:
            xs.append(float(r["human_score"]))
            ys.append(mos[b])
    s = signed_srocc(xs, ys)
    return {"joined": len(xs), "rows": len(rows), "signed_srocc": s,
            "ground_truth": "tid2013 mos_with_names.txt (published MOS)",
            "verdict": "OK" if s > 0 else "INVERTED"}


def declared(name, lo, hi, note):
    rows = load_hs(name)
    hs = np.array([float(r["human_score"]) for r in rows])
    return {"rows": len(rows), "range_0_1": [float(hs.min()),
            float(hs.max())], "range_0_100": [float(hs.min()) * 100,
            float(hs.max()) * 100], "verdict": "DECLARED", "note": note}


def main():
    out = {}
    out["kadid_train"] = check_kadid("kadid_train.tsv")
    out["kadid_dev"] = check_kadid("kadid_dev.tsv")
    out["tid_jp2kjpeg"] = check_tid("tid_jp2kjpeg.tsv")
    # pooled files inherit their components' verdicts; check the KADID
    # sub-rows inside them too (basename-join still applies)
    out["tidkadid"] = {
        "rows": len(load_hs("tidkadid.tsv")),
        "composition": "tid_jp2kjpeg(250) + kadid_train(400)",
        "kadid_rows": check_kadid("tidkadid.tsv"),
        "verdict": "OK" if check_kadid("tidkadid.tsv")["verdict"] == "OK"
                   else "INVERTED"}
    out["cid22a"] = declared("cid22a.tsv", 0, 100,
        "MCOS/100 — quality-oriented by definition; no raw-vote ground "
        "truth exists for CID22 (checker convention: cid22 = QUALITY)")
    out["imazen26"] = declared("imazen26_pairs.tsv", 0, 100,
        "clip01(score_ssim2/100) metric oracle — quality-oriented by "
        "construction; canonical wlin7 human_score column")
    out["konfig_val"] = declared("konfig_val.tsv", 0, 100,
        "1 - q/3.2 already quality-oriented (campaign Appendix L)")
    out["_schema"] = "dvifm2d-target-orientation-v1"
    out["_mapping"] = ("quality_0_100 = human_score * 100 for every leg; "
                       "kadid corrected to (dmos-1)/4 per Amendment 1")
    (PAIRS / "ORIENTATION.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    bad = [k for k, v in out.items()
           if isinstance(v, dict) and v.get("verdict") == "INVERTED"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
