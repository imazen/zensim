#!/usr/bin/env python3
"""THE single registered CID22-B label unseal (prereg §5.4 + Amendment 1).

Emits pairs/cid22b_unsealed.tsv: the real MCOS/100 labels for the 24
CID22-B refs, taken from the frozen source cid22val_pairs_ab.tsv, in the
EXACT row order of the sealed pairs/cid22b.tsv (asserted pairwise).

Run once, only after every fit artefact under fits/ is frozen. Writes a
manifest recording the unseal event (timestamp, source sha256, row count).
"""
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/mnt/v/output/zensim/dvifm-screen2d-2026-09-19")
PAIRS = OUT / "pairs"
SEALED = PAIRS / "cid22b.tsv"
UNSEALED = PAIRS / "cid22b_unsealed.tsv"
SOURCE = Path(
    "/mnt/v/dataset/cid22/CID22_validation_set/cid22val_pairs_ab.tsv")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> int:
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
        "note": ("CID22-B single registered read (prereg §5.4): real "
                 "MCOS/100 labels unsealed after all Part-D fits frozen. "
                 "Row order identical to sealed cid22b.tsv (asserted)."),
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
