#!/usr/bin/env python3
"""THE single registered CID22-B label unseal for the verdict lane.

Emits pairs/cid22b_unsealed.tsv under the verdict output dir: real
MCOS/100 labels for the 24 CID22-B refs from the frozen source
cid22val_pairs_ab.tsv, in the EXACT row order of pairs/cid22b.tsv
(asserted pairwise; the sealed file is a byte-copy of the 2d sealed
file). Run once, only after every X2/X1 configuration is frozen and
recorded. Writes a manifest recording the unseal event.
"""
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/mnt/v/output/zensim/dvifm-verdict-2026-09-20")
PAIRS = OUT / "pairs"
SEALED = PAIRS / "cid22b.tsv"
UNSEALED = PAIRS / "cid22b_unsealed.tsv"
SOURCE = Path(
    "/mnt/v/dataset/cid22/CID22_validation_set/cid22val_pairs_ab.tsv")
SEALED_2D = Path(
    "/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/pairs/cid22b.tsv")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> int:
    if sha(SEALED) != sha(SEALED_2D):
        raise SystemExit("verdict cid22b.tsv diverges from the 2d sealed "
                         "file — refusing")
    sealed = load(SEALED)
    src = load(SOURCE)
    src_map = {(r["ref_path"], r["dist_path"]): r["human_score"]
               for r in src}
    rows = []
    for r in sealed:
        key = (r["ref_path"], r["dist_path"])
        if key not in src_map:
            raise SystemExit(f"sealed row not in source: {key}")
        rows.append({"ref_path": r["ref_path"],
                     "dist_path": r["dist_path"],
                     "human_score": src_map[key]})
    if UNSEALED.exists():
        raise SystemExit(
            f"{UNSEALED} already exists — the B read is single-use; "
            "refusing to rewrite")
    with open(UNSEALED, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        for r in rows:
            w.writerow([r["ref_path"], r["dist_path"], r["human_score"]])
    manifest = {
        "file": str(UNSEALED),
        "sha256": sha(UNSEALED),
        "rows": len(rows),
        "unsealed_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "source_sha256": sha(SOURCE),
        "sealed_sha256": sha(SEALED),
        "sealed_matches_2d": sha(SEALED) == sha(SEALED_2D),
        "note": ("CID22-B single registered read (verdict lane, X1): real "
                 "MCOS/100 labels unsealed after all X2 arms and the X1 "
                 "configuration were frozen and recorded. Row order "
                 "identical to sealed cid22b.tsv (asserted)."),
    }
    (PAIRS / "cid22b_unsealed.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    vals = [float(r["human_score"]) for r in rows]
    print(f"UNSEALED cid22b: {len(rows)} rows, "
          f"target range {min(vals):.4f}..{max(vals):.4f}, "
          f"mean {sum(vals)/len(vals):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
