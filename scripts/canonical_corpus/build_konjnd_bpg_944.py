#!/usr/bin/env python3
"""Build the WAVE-7 reference-disjoint KonJND BPG-half training leg at 944.

Registered rule (benchmarks/sota944_campaign_2026-08-03.md §7.1, commit
62f0bcc3): refs = the BPG 504 (SRC0505-SRC1008, the KonJND-1k distribution's
pre-decoded PNGs); pairs = per ref the 20 rank-evenly-spaced picks
idx=round(i*50/19) over the 51-variant ladder sorted by gpu_ssimulacra2
ascending (the 372-era konjnd-dense rule, recovered + verified 1008/1008);
target human_score = gpu_ssimulacra2/100 (the 944-era sibling ssim2-target
scale); split srcnum%10 in {8,9} -> val (101 refs) else train (403 refs).

Subcommands:
  pairs    write konjnd_bpg_{train,val}_pairs.tsv (ref_path/dist_path/human_score)
  promote  convert the two extracted 946-col CSVs to canonical zstd parquets,
           run the disjointness assertions, and emit the _MANIFEST fragment

Extraction between the two steps is the frozen P1 invocation:
  ZENSIM_AB_MODE=foldapp2 v2_ab_extract <pairs.tsv> <out.csv>
"""

import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCORED_CSV = Path("/mnt/v/datasets/KonJND-1k/konjnd_full_scored.csv")
SCORED_SHA = "5749ed6a1ed63eef5204389d15b1ca3249e2b52c75371b17fd8fac6f2434a72f"
N_FEATURES = 944
K_PAIRS = 20


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bpg_ladders():
    """ref -> list of (dist_path, gpu_ssimulacra2), all 51 BPG variants."""
    ladders = {}
    with open(SCORED_CSV, newline="") as f:
        for row in csv.DictReader(f):
            dp = row["decoded_path"]
            m = re.search(r"/(SRC\d{4})_BPG_\d+\.png$", dp)
            if not m:
                continue
            ladders.setdefault(m.group(1), []).append(
                (dp, float(row["gpu_ssimulacra2"]))
            )
    return ladders


def cmd_pairs(out_dir: Path):
    got = sha256_file(SCORED_CSV)
    if got != SCORED_SHA:
        sys.exit(f"ABORT: {SCORED_CSV} sha256 {got} != registered {SCORED_SHA}")
    ladders = load_bpg_ladders()
    if len(ladders) != 504:
        sys.exit(f"ABORT: {len(ladders)} BPG refs, want 504")
    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = [round(i * 50 / 19) for i in range(K_PAIRS)]
    n = {"train": 0, "val": 0}
    files = {
        split: open(out_dir / f"konjnd_bpg_{split}_pairs.tsv", "w")
        for split in ("train", "val")
    }
    for f in files.values():
        f.write("ref_path\tdist_path\thuman_score\n")
    for ref in sorted(ladders):
        ladder = sorted(ladders[ref], key=lambda t: t[1])
        if len(ladder) != 51:
            sys.exit(f"ABORT: {ref} has {len(ladder)} variants, want 51")
        split = "val" if int(ref[3:]) % 10 in (8, 9) else "train"
        ref_path = f"/mnt/v/datasets/KonJND-1k/KonJND-1k/source_image/{ref}.png"
        if not Path(ref_path).is_file():
            sys.exit(f"ABORT: missing source {ref_path}")
        for i in idxs:
            dp, s2 = ladder[i]
            if not Path(dp).is_file():
                sys.exit(f"ABORT: missing distorted {dp}")
            files[split].write(f"{ref_path}\t{dp}\t{s2 / 100.0:.9f}\n")
            n[split] += 1
    for f in files.values():
        f.close()
    if n != {"train": 403 * K_PAIRS, "val": 101 * K_PAIRS}:
        sys.exit(f"ABORT: row counts {n} != registered 8060/2020")
    print(f"pairs: train={n['train']} val={n['val']} -> {out_dir}")


def csv_to_parquet(src: Path, dst: Path, want_rows: int) -> dict:
    with open(src, newline="") as f:
        r = csv.reader(f)
        names = next(r)
        if len(names) != 2 + N_FEATURES:
            sys.exit(f"ABORT {src.name}: {len(names)} cols, want {2 + N_FEATURES}")
        cols = {name: [] for name in names}
        for row in r:
            cols[names[0]].append(row[0])
            for name, v in zip(names[1:], row[1:]):
                x = float(v)
                if x != x or x in (float("inf"), float("-inf")):
                    sys.exit(f"ABORT {src.name}: non-finite in {name}")
                cols[name].append(x)
    nrows = len(cols[names[0]])
    if nrows != want_rows:
        sys.exit(f"ABORT {src.name}: {nrows} rows, want {want_rows}")
    arrays = [pa.array(cols[names[0]], type=pa.utf8())] + [
        pa.array(cols[n], type=pa.float64()) for n in names[1:]
    ]
    tbl = pa.table(arrays, names=names)
    pq.write_table(tbl, dst, compression="zstd", compression_level=7)
    return {
        "parquet": str(dst),
        "sha256": sha256_file(dst),
        "rows": nrows,
        "refs": sorted(set(cols[names[0]])),
    }


def cmd_promote(run_dir: Path, dest_root: Path, zensim_commit: str):
    eval_leg = pq.read_table(
        "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/"
        "ext_konjnd_jpeg_val.parquet",
        columns=["ref_basename"],
    )
    eval_refs = set(eval_leg.column("ref_basename").to_pylist())
    entries = {}
    for split, want in (("train", 8060), ("val", 2020)):
        e = csv_to_parquet(
            run_dir / f"konjnd_bpg_{split}.csv",
            dest_root / f"konjnd_bpg_{split}_944.parquet",
            want,
        )
        refs = set(e.pop("refs"))
        if refs & eval_refs:
            sys.exit(f"ABORT: {split} shares {len(refs & eval_refs)} refs with eval leg")
        e["n_refs"] = len(refs)
        entries[split] = (e, refs)
    if entries["train"][1] & entries["val"][1]:
        sys.exit("ABORT: train/val ref overlap")
    print("assertions PASS: train ∩ val = 0; both ∩ eval-leg = 0")
    manifest = {
        "description": (
            "WAVE-7 reference-disjoint KonJND training leg at 944 — the BPG "
            "half (SRC0505-SRC1008) of KonJND-1k, built from the "
            "distribution's own pre-decoded PNGs (NO BPG decode step exists "
            "or is needed). Registered: sota944 campaign amendment 7 "
            "(commit 62f0bcc3). Rule: 20 rank-evenly-spaced picks/ref over "
            "the ssim2-sorted 51-QP ladder (the recovered 372-era "
            "konjnd-dense rule, verified 1008/1008); human_score = "
            "gpu_ssimulacra2/100; split srcnum%10 in {8,9} -> val. "
            "Reference-disjoint from ext_konjnd_jpeg_val (the bar's KonJND "
            "instrument) by construction, asserted at build."
        ),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_commit": zensim_commit,
        "driver": "v2_ab_extract (foldapp2), profile codec_target, default toggles",
        "input_csv": {"path": str(SCORED_CSV), "sha256": SCORED_SHA},
        "contamination_screen": {
            "tool": "check_holdout_overlap dHash-64 d<=10 vs CID22-49",
            "flags": [
                {
                    "pair": "SRC0611.png vs 3653963.png",
                    "d": 10,
                    "status": (
                        "retained pending user sign-off — montage shows "
                        "categorical content non-match (skyscraper vs "
                        "waterfall); /mnt/v/output/zensim/wave7/"
                        "dhash_d10_SRC0611_vs_3653963_montage.png"
                    ),
                }
            ],
        },
        "extraction_gate": (
            "8 eval-leg pairs re-extracted at this rev vs stored canonical "
            "ext_konjnd_jpeg_val: 7552/7552 feature cells exact-equal"
        ),
        "entries": {k: v[0] for k, v in entries.items()},
    }
    mpath = dest_root / "_MANIFEST_konjnd_bpg.json"
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"promoted -> {dest_root} (+ {mpath.name})")
    for k, (e, _) in entries.items():
        print(f"  {k}: rows={e['rows']} refs={e['n_refs']} sha256={e['sha256'][:16]}…")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("pairs", "promote"):
        sys.exit(__doc__)
    if sys.argv[1] == "pairs":
        out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
            "/mnt/v/output/zensim/wave7"
        )
        cmd_pairs(out)
    else:
        run_dir = Path(os.environ.get("W7_RUN", "/mnt/v/output/zensim/wave7"))
        dest = Path(
            os.environ.get(
                "W7_DEST", "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01"
            )
        )
        commit = os.environ.get("ZENSIM_COMMIT") or sys.exit(
            "ABORT: ZENSIM_COMMIT env required"
        )
        cmd_promote(run_dir, dest, commit)


if __name__ == "__main__":
    main()
