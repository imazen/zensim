#!/usr/bin/env python3
"""Phase-2b DVIFM screen — KonFiG segment/admission builder.

Extends the admitted minimal-top segments (reused verbatim, hash-pinned) with
the registered KonFiG originsplit views:

  train  <- konfig_originsplit_train_944.parquet sources {SRC06,SRC28,SRC50}
            (327 rows; role train)
  dev2   <- konfig_originsplit_val_944.parquet sources {SRC01,SRC03,SRC31,SRC45}
            (436 rows; role eval, eval_leg=dev2 — reported separately)

The konfig originsplit TEST view {SRC07,SRC09,SRC17} is not touched. Rows come
from the pinned build pairs table (konfig_pairs.tsv, sha256 recorded in
_MANIFEST_konfig.json); target = human_score * 100 (registered quality-oriented
1 - q_jnd/3.2 rescaled to the KADID/TID 0..100 convention).
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/mnt/v/output/zensim/dvifm-screen2b-2026-09-19")
SEG = ROOT / "segments"
CANON = Path("/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01")
PAIRS = Path("/mnt/v/output/zensim/konfig944/build/konfig_pairs.tsv")
MANIFEST = CANON / "_MANIFEST_konfig.json"

TRAIN_SOURCES = ["SRC06", "SRC28", "SRC50"]
EVAL_SOURCES = ["SRC01", "SRC03", "SRC31", "SRC45"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def main():
    SEG.mkdir(parents=True, exist_ok=True)
    pairs = list(csv.DictReader(open(PAIRS), delimiter="\t"))
    assert len(pairs) == 1090
    manifest_sha = sha(MANIFEST)
    view_train = CANON / "konfig_originsplit_train_944.parquet"
    view_eval = CANON / "konfig_originsplit_val_944.parquet"

    for role, sources, view in (("train", TRAIN_SOURCES, view_train),
                                ("eval", EVAL_SOURCES, view_eval)):
        rows = []
        refs = {}
        for r in pairs:
            s = r["source"]
            if s not in sources:
                continue
            family = f"konfig:{s}"
            refs[family] = dict(corpus="konfig", origin=family,
                                source_family=family, split=role)
            row = dict(task="human", corpus="konfig", origin=family,
                       source_family=family,
                       family=f"konfig_{r['part']}_{r['distortion']}",
                       target=float(r["human_score"]) * 100,
                       reference=r["ref_path"], distorted=r["dist_path"])
            if role == "eval":
                row["eval_leg"] = "dev2"
            rows.append(row)
        assert len(rows) == 109 * len(sources), (role, len(rows))
        authority = dict(
            path=str(MANIFEST), sha256=manifest_sha,
            canonical_view=str(view), canonical_view_sha256=sha(view),
            rule=("konfig originsplit via zenmetrics origin_split.split_of; "
                  "manifest origin_split.map; pairs_tsv_sha256 "
                  + sha(PAIRS)))
        admission = SEG / f"konfig-{role}-admission.json"
        segment = SEG / f"konfig-{role}-segment.json"
        save(admission, dict(schema="zensim-source-admission-v1",
                             authority=authority,
                             sources=[refs[f"konfig:{s}"] for s in sources]))
        save(segment, dict(schema="zensim-feature-segment-v1",
                           role=role, rows=rows))
        print(role, len(rows), len(refs), sha(segment), sha(admission))


if __name__ == "__main__":
    main()
