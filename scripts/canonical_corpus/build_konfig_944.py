#!/usr/bin/env python3
"""Build the KonFiG-IQA training leg at 944 (campaign APPENDIX L, pre-reg e93eba04).

Registered rule (benchmarks/sota944_campaign_2026-08-03.md §L.6): per source —
PartA distortions alphabetical x levels 0..12 (q_jnd = level*0.25), then PartB
motionblur levels 0..30 (q_jnd = level*0.1); dedup per SOURCE on the sha256 of
the DECODED RGB8 pixels of the distorted file, keep first in enumeration order;
target human_score = 1 - q_jnd/3.2 (all values exact binary fractions).
Reproduction gate: the kept (ref_basename, q_jnd) multiset must EQUAL the
2026-07-02 372-era table's exactly (1,090 rows: 85 PartA + 24 PartB per source).

Gates that precede this build (committed 7ed6ac4b): G-L1/G-L2 overlap audits
CLEAN PASS (zero d<=10 flags, min d=17); G-L5 extractor self-consistency
(8 konjnd pairs, 7552/7552 cells exact-equal at the build rev).

Subcommands:
  pairs    stage per-part reference copies + write konfig_pairs.tsv
           (ref_path/dist_path/human_score + source/part/distortion/level/q_jnd/
           dist_px_sha256), enforcing the multiset reproduction gate
  promote  convert the extracted 946-col CSV to the canonical zstd parquet
           (+ q_jnd column, positional-join validated per row), emit the
           origin_split views via THE canonical splitter, run the disjointness
           assertions, and write _MANIFEST_konfig.json

Extraction between the two steps is the frozen wave-7 P1 invocation:
  ZENSIM_AB_MODE=foldapp2 v2_ab_extract konfig_pairs.tsv konfig_944.csv
"""

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, "/home/lilith/work/zen/zenmetrics/scripts/picker")
from origin_split import split_of  # THE canonical splitter — never re-implement

ROOT = Path("/mnt/v/dataset/konfig-iqa/KonFiG-IQA")
BUILD = Path("/mnt/v/output/zensim/konfig944/build")
JULY = Path("/mnt/v/output/zensim-multicodec-probe/konfig_train_2026-07-02.parquet")
DEST = Path(os.environ.get(
    "KONFIG_DEST", "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01"))
SOURCES = ["SRC01", "SRC03", "SRC06", "SRC07", "SRC09",
           "SRC17", "SRC28", "SRC31", "SRC45", "SRC50"]
DISTS = ["colordiffusion", "highsharpen", "jitter", "jpeg2000",
         "lensblur", "motionblur", "multinoise"]
N_FEATURES = 944


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def px_sha(p: Path) -> str:
    im = Image.open(p).convert("RGB")
    return hashlib.sha256(im.tobytes()).hexdigest()


def enumerate_and_dedup():
    """The frozen §L.6 enumeration + per-source content dedup."""
    kept, seen = [], defaultdict(set)
    for s in SOURCES:
        for d in DISTS:
            for lv in range(13):
                p = ROOT / "IMAGES" / "PartA" / s / d / f"{s}_{d}_{lv}.png"
                if not p.is_file():
                    sys.exit(f"ABORT: missing {p}")
                kept.append((s, "PartA", d, lv, lv * 0.25, p))
        for lv in range(31):
            p = (ROOT / "IMAGES" / "PartB" / s / "motionblur"
                 / f"{s}_motionblur_PartB_{lv}.png")
            if not p.is_file():
                sys.exit(f"ABORT: missing {p}")
            kept.append((s, "PartB", "motionblur", lv, lv * 0.1, p))
    if len(kept) != 1220:
        sys.exit(f"ABORT: enumerated {len(kept)} stimuli, want 1220")
    rows = []
    for (s, part, d, lv, q, p) in kept:
        h = px_sha(p)
        if h in seen[s]:
            continue
        seen[s].add(h)
        rows.append((s, part, d, lv, q, p, h))
    return rows


def multiset_gate(rows):
    t = pq.read_table(JULY, columns=["ref_basename", "q_jnd", "human_score"])
    july = Counter((rb, round(q, 9)) for rb, q in
                   zip(t["ref_basename"].to_pylist(), t["q_jnd"].to_pylist()))
    mine = Counter((f"{r[0]}_{r[1]}", round(r[4], 9)) for r in rows)
    if july != mine:
        sys.exit(f"ABORT: (ref_basename, q_jnd) multiset != 2026-07-02 table "
                 f"(july-only {len(july - mine)}, mine-only {len(mine - july)}) "
                 f"— diagnose per §L.6 before proceeding")
    hs = {(rb, round(q, 9)): h for rb, q, h in
          zip(t["ref_basename"].to_pylist(), t["q_jnd"].to_pylist(),
              t["human_score"].to_pylist())}
    worst = max(abs(hs[(f"{r[0]}_{r[1]}", round(r[4], 9))] - (1 - r[4] / 3.2))
                for r in rows)
    if worst != 0.0:
        sys.exit(f"ABORT: human_score formula deviates from July table by {worst}")
    print(f"multiset gate PASS: {len(rows)} rows == 2026-07-02 multiset; "
          f"formula delta 0.0")


def cmd_pairs():
    rows = enumerate_and_dedup()
    if len(rows) != 1090:
        sys.exit(f"ABORT: dedup kept {len(rows)} rows, want 1090")
    multiset_gate(rows)
    refs_dir = BUILD / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    for s in SOURCES:
        src = ROOT / "IMAGES" / "reference_images" / f"{s}_0.png"
        src_sha = px_sha(src)
        for part in ("PartA", "PartB"):
            dst = refs_dir / f"{s}_{part}.png"
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            if px_sha(dst) != src_sha:
                sys.exit(f"ABORT: staged ref {dst} pixel mismatch")
    out = BUILD / "konfig_pairs.tsv"
    with open(out, "w") as f:
        f.write("ref_path\tdist_path\thuman_score\tsource\tpart\tdistortion"
                "\tlevel\tq_jnd\tdist_px_sha256\n")
        for (s, part, d, lv, q, p, h) in rows:
            f.write(f"{refs_dir}/{s}_{part}.png\t{p}\t{1 - q / 3.2:.9f}\t{s}"
                    f"\t{part}\t{d}\t{lv}\t{q:.9f}\t{h}\n")
    print(f"pairs: {len(rows)} rows -> {out}")


def cmd_promote(zensim_commit: str):
    src = BUILD / "konfig_944.csv"
    pairs = list(csv.DictReader(open(BUILD / "konfig_pairs.tsv"), delimiter="\t"))
    with open(src, newline="") as f:
        r = csv.reader(f)
        names = next(r)
        if len(names) != 2 + N_FEATURES:
            sys.exit(f"ABORT: {len(names)} cols, want {2 + N_FEATURES}")
        data = list(r)
    if len(data) != 1090 or len(pairs) != 1090:
        sys.exit(f"ABORT: rows csv={len(data)} pairs={len(pairs)}, want 1090")
    rb, hs, qj = [], [], []
    feats = [[] for _ in range(N_FEATURES)]
    for i, row in enumerate(data):
        p = pairs[i]
        want_rb = f"{p['source']}_{p['part']}"
        if row[0] != want_rb:
            sys.exit(f"ABORT row {i}: ref_basename {row[0]} != pairs {want_rb} "
                     f"(positional identity broken)")
        h = float(row[1])
        if h != float(p["human_score"]):
            sys.exit(f"ABORT row {i}: human_score {row[1]} != pairs "
                     f"{p['human_score']} (positional identity broken)")
        rb.append(row[0])
        hs.append(h)
        qj.append(float(p["q_jnd"]))
        for j in range(N_FEATURES):
            x = float(row[2 + j])
            if x != x or x in (float("inf"), float("-inf")):
                sys.exit(f"ABORT row {i}: non-finite f{j}")
            feats[j].append(x)
    # eval-leg reference-identity assertion (name level; the pixel-level audit
    # is G-L1/G-L2, committed 7ed6ac4b)
    eval_refs = set()
    for fn in DEST.glob("ext_*.parquet"):
        eval_refs |= set(pq.read_table(fn, columns=["ref_basename"])
                         ["ref_basename"].to_pylist())
    ours = set(rb)
    if ours & eval_refs:
        sys.exit(f"ABORT: {len(ours & eval_refs)} ref_basenames shared with "
                 f"eval legs")
    arrays = ([pa.array(rb, type=pa.utf8()),
               pa.array(hs, type=pa.float64()),
               pa.array(qj, type=pa.float64())]
              + [pa.array(feats[j], type=pa.float64())
                 for j in range(N_FEATURES)])
    tbl = pa.table(arrays, names=(["ref_basename", "human_score", "q_jnd"]
                                  + [f"f{j}" for j in range(N_FEATURES)]))
    entries = {}
    dest = DEST / "konfig_944.parquet"
    pq.write_table(tbl, dest, compression="zstd", compression_level=7)
    entries["konfig_944"] = {"parquet": str(dest), "sha256": sha256_file(dest),
                             "rows": tbl.num_rows, "n_sources": len(SOURCES)}
    # origin-split views via THE canonical splitter (numeric source id — SRC
    # prefix stripped so the id leads, per origin_split's contract)
    split_map = {s: split_of(s[3:]) for s in SOURCES}
    if None in split_map.values():
        sys.exit(f"ABORT: unsplittable source in {split_map}")
    for split in ("train", "val", "test"):
        keep = [i for i, name in enumerate(rb)
                if split_map[name.split("_")[0]] == split]
        view = tbl.take(keep)
        vp = DEST / f"konfig_originsplit_{split}_944.parquet"
        pq.write_table(view, vp, compression="zstd", compression_level=7)
        entries[f"originsplit_{split}"] = {
            "parquet": str(vp), "sha256": sha256_file(vp),
            "rows": view.num_rows,
            "sources": sorted(s for s, sp in split_map.items() if sp == split)}
    manifest = {
        "description": (
            "KonFiG-IQA training leg at 944 (Men/Lin/Jenadeleh/Saupe 2021, "
            "arXiv:2108.00201; Konstanz). 10 sources x {PartA 7 distortions x "
            "13 levels @0.25 JND; PartB motionblur x 31 levels @0.1 JND}; "
            "content-dedup to 1,090 rows (85+24 per source), reproducing the "
            "2026-07-02 372-era multiset exactly. human_score = 1 - q_jnd/3.2 "
            "(QUALITY-oriented, verified vs 75,519 raw EXP_III DCR votes; "
            "registered in check_target_orientation.py). Registered: campaign "
            "APPENDIX L (pre-reg e93eba04); gates G-L1/G-L2 CLEAN PASS "
            "(min dHash d=17, zero flags; commit 7ed6ac4b); G-L5 extractor "
            "self-consistency 7552/7552 exact. The PROBE leg is the FULL "
            "table (registered design decision L.6); the originsplit views "
            "exist for any future within-KonFiG instrument. NOTE: KonFiG is "
            "in SSIMULACRA2's tuning set (never a ssim2-comparison corpus)."),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_commit": zensim_commit,
        "driver": "v2_ab_extract (foldapp2), profile codec_target, default toggles",
        "inputs": {
            "dataset_root": str(ROOT),
            "scores_csv_sha256": sha256_file(ROOT / "scores.csv"),
            "exp3_data3_sha256": sha256_file(ROOT / "DATA/EXP_III/data3.csv"),
            "july_372_table": {"path": str(JULY), "sha256": sha256_file(JULY)},
            "pairs_tsv_sha256": sha256_file(BUILD / "konfig_pairs.tsv"),
        },
        "contamination_screen": {
            "tool": "check_holdout_overlap dHash-64 d<=10 + decoded-pixel exact",
            "verdict": ("CLEAN PASS all sets (KonJND-1008, CID22-49, CSIQ-30, "
                        "LIVE-29, AIC3-10): 0 exact hits, zero d<=10 flags, "
                        "global min d=17 — no montages, no user-review queue"),
            "record": "benchmarks/konfig/audit_2026-08-05.meta.md (7ed6ac4b)",
        },
        "extraction_gate": (
            "8 konjnd_bpg pairs re-extracted at this rev vs stored canonical "
            "konjnd_bpg_train_944: 7552/7552 feature cells exact-equal"),
        "origin_split": {"rule": "zenmetrics origin_split.split_of on numeric "
                                 "source id (SRC prefix stripped)",
                         "map": split_map},
        "entries": entries,
    }
    mpath = DEST / "_MANIFEST_konfig.json"
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"promoted -> {DEST} (+ {mpath.name})")
    for k, e in entries.items():
        print(f"  {k}: rows={e['rows']} sha256={e['sha256'][:16]}…")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("pairs", "promote"):
        sys.exit(__doc__)
    if sys.argv[1] == "pairs":
        BUILD.mkdir(parents=True, exist_ok=True)
        cmd_pairs()
    else:
        commit = os.environ.get("ZENSIM_COMMIT") or sys.exit(
            "ABORT: ZENSIM_COMMIT env required")
        cmd_promote(commit)


if __name__ == "__main__":
    main()
