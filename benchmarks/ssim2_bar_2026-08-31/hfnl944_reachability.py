#!/usr/bin/env python3
"""Is the near-lossless axis reachable at 944, and is it winnable at all?

Answers the exam's §3.7 open row ("`hf_nearlossless` exists only at 372 width")
with the two facts that settle it, both read from stored bytes:

  R. REACHABILITY. A 944 extraction needs the (reference, distorted) PIXELS.
     The references survive; the 1,200 distorted JXL bitstreams were never
     persisted (`encoded_filename` is blank on every row of the sweep's own
     `pareto.tsv`, and both `refit/distorted/` mirrors are empty), so there is
     nothing to re-extract from. Named-missing-artifact, not a missing command.

  C. CIRCULARITY. `human_score` in that corpus is EXACTLY `ssim2_gpu / 100` on
     all 1,200 rows — it carries no human label at all. So the axis is an ssim2
     SELF-TARGET, which §2.1 of the exam excludes from every "beats" clause by
     name, and the opponent scores 1.0 on it by construction. Re-extracting it
     at 944 could therefore never have satisfied W2's near-lossless clause.

No statistic is computed here (the 1.0 is measured separately by the `panel`
binary, which is the owner); this script only reads bytes and counts.

    hfnl944_reachability.py [--json OUT.json]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

CANON = Path("/mnt/v/zen/zensim-training/canonical-2026-07-15/train")
SWEEP = Path("/mnt/v/output/zensim-jxl-nearlossless/refit")
SOURCES = Path("/mnt/v/input/zensim/sources")
# Every root that serves the corpus to `bake_verdict` under some `--regime`.
ROOTS = [
    "/mnt/v/zen/zensim-training/canonical-2026-07-15/train",
    "/mnt/v/zen/zensim-training/2026-08-30-full-features-372",
    "/mnt/v/zen/zensim-training/2026-08-30-era3-full-features-372",
    "/mnt/v/zen/zensim-training/2026-05-15-full-features",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    out: dict = {}

    # ---- C. the target IS ssim2 -------------------------------------------
    tgt = {}
    for name in ("hf_nearlossless.parquet", "hf_nearlossless_train.parquet",
                 "hf_nearlossless_val.parquet"):
        t = pq.read_table(CANON / name,
                          columns=["ref_basename", "human_score", "ssim2_gpu"]).to_pydict()
        n = len(t["human_score"])
        exact = sum(1 for i in range(n)
                    if t["human_score"][i] * 100.0 == t["ssim2_gpu"][i])
        tgt[name] = {
            "rows": n,
            "refs": len(set(t["ref_basename"])),
            "max_abs_human_x100_minus_ssim2": max(
                abs(t["human_score"][i] * 100.0 - t["ssim2_gpu"][i]) for i in range(n)),
            "rows_exactly_equal": exact,
        }
    out["circularity"] = {
        "verdict": "SELF-TARGET: human_score == ssim2_gpu/100 exactly; the corpus "
                   "carries no human label",
        "per_file": tgt,
        "implication": "peer_ssim2 scores 1.0 on this axis by construction at ANY "
                       "feature width, so the exam's W2 near-lossless clause is "
                       "unwinnable on it — an extraction could not have closed it.",
    }

    # ---- R. the distorted material is gone --------------------------------
    rows = list(csv.DictReader(open(SWEEP / "pareto.tsv"), delimiter="\t"))
    blank = sum(1 for r in rows if not (r["encoded_filename"] or "").strip())
    refs = sorted({Path(r["image_path"]).stem for r in rows})
    have = sum(1 for b in refs if any((SOURCES / f"{b}{e}").exists()
                                      for e in (".png", ".jpg", ".jpeg", ".webp", ".jxl"))
               or (SOURCES / b).exists())
    if have != len(refs):  # tolerate an unknown source extension
        names = {p.stem for p in SOURCES.iterdir() if p.is_file()}
        have = sum(1 for b in refs if b in names)
    dist_dirs = {}
    for d in (SWEEP / "distorted",
              Path("/mnt/tower/output/zensim-jxl-nearlossless/refit/distorted")):
        dist_dirs[str(d)] = {
            "exists": d.exists(),
            "files": len(list(d.iterdir())) if d.exists() else None,
        }
    out["reachability"] = {
        "verdict": "NOT-REACHABLE",
        "missing_artifact": "the 1,200 distorted JXL bitstreams of the "
                            "2026-07-06 post-fix near-lossless sweep "
                            "(zenjxl, q90, butteraugli distance "
                            "{0.005,0.01,0.015,0.02,0.025,0.03})",
        "sweep_rows": len(rows),
        "encoded_filename_blank": blank,
        "reference_image_path_root": sorted({r["image_path"].split("/scratchpad/")[0]
                                             for r in rows}),
        "references_recoverable": {"found": have, "of": len(refs),
                                   "dir": str(SOURCES)},
        "distorted_dirs": dist_dirs,
        "why_regeneration_is_a_substitution":
            "the sweep's encoder is pinned to jxl-encoder@eeb52735 (2026-07-06) and "
            "the stored target is a GPU ssim2 read; re-encoding on today's encoder "
            "and a CPU ssim2 would produce different bitstreams and different "
            "targets, so the row-for-row human_score alignment gate could not hold "
            "and the result would be a different corpus wearing the same name.",
    }

    # ---- the parquet is one file, copied, never re-extracted ---------------
    out["root_copies"] = {}
    for r in ROOTS:
        p = Path(r) / "hf_nearlossless_val.parquet"
        out["root_copies"][r] = {"exists": p.exists(),
                                 "sha256": sha256_file(p) if p.exists() else None,
                                 "symlink_to": str(p.resolve()) if p.is_symlink() else None}
    shas = {v["sha256"] for v in out["root_copies"].values() if v["sha256"]}
    out["root_copies"]["_all_identical"] = len(shas) == 1

    txt = json.dumps(out, indent=2, sort_keys=True)
    if a.json:
        a.json.write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
