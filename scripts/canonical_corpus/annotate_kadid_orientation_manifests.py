#!/usr/bin/env python3
"""Record the KADID orientation correction in each ext root's `_MANIFEST.json`.

Companion to `fix_ext_kadid_orientation.py` (which rewrites the table). This writes
the provenance so "which bytes am I looking at, and which bytes was that bake trained
on?" is a grep and not a forensic audit — the failure mode Appendix G was created by.

For each root it updates the `ext_kadid` entry to the CORRECTED sha256 and adds a
`target_orientation` block naming: the transform, the registration, the gate verdict,
and the PRESERVED inverted file plus its sha (which is the sha every pre-2026-08-05
bake's embedded `zentrain.repro` carries for this input).

Idempotent. Usage: annotate_kadid_orientation_manifests.py [--root DIR]... [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOTS = [
    "/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22",
    "/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27",
    "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01",
]
NAME = "ext_kadid.parquet"
PRESERVED = "ext_kadid_INVERTED_2026-08-04.parquet"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def annotate(root: Path, dry: bool) -> dict:
    man = root / "_MANIFEST.json"
    cur, old = root / NAME, root / PRESERVED
    rec = {"root": str(root)}
    if not (man.exists() and cur.exists() and old.exists()):
        rec["status"] = "SKIP (manifest or table missing)"
        return rec
    d = json.loads(man.read_text())
    cur_sha, old_sha = sha256_file(cur), sha256_file(old)
    block = {
        "corrected_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "benchmarks/sota944_campaign_2026-08-03.md REGISTERED APPENDIX H, H.1",
        "determination": "benchmarks/sota944_campaign_2026-08-03.md REGISTERED APPENDIX F",
        "defect": "human_score stored (5 - dmos)/4; KADID's `dmos` is a MOS in disguise "
                  "(raw DCR falls with severity), so the correct transform is (dmos - 1)/4",
        "transform_applied": "human_score := 1 - human_score  (features untouched)",
        "gate": "scripts/canonical_corpus/check_target_orientation.py: "
                "INVERTED -0.582360 -> OK +0.582360 vs 349,800 raw DCR ratings",
        "corrected_sha256": cur_sha,
        "preserved_inverted_file": PRESERVED,
        "preserved_inverted_sha256": old_sha,
        "repro_hazard": (
            "Any bake whose embedded zentrain.repro lists sha256 " + old_sha +
            " for this input was trained on the INVERTED target. Re-running that bake's "
            "argv verbatim will now train on the CORRECTED table and will NOT reproduce "
            "it; substitute " + PRESERVED + " to reproduce. Existing bakes are NOT "
            "retrained or re-verdicted — they are annotated (benchmarks/eval_annotations.json)."
        ),
    }
    changed = False
    for e in d.get("entries", []):
        if e.get("corpus") == "ext_kadid":
            if e.get("sha256") != cur_sha or e.get("target_orientation") != block:
                e["sha256"] = cur_sha
                e["target_orientation"] = block
                changed = True
            rec["entry_updated"] = True
    rec.update(status="UPDATED" if changed else "ALREADY-ANNOTATED",
               corrected_sha256=cur_sha, preserved_sha256=old_sha)
    if changed and not dry:
        man.write_text(json.dumps(d, indent=2) + "\n")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = [annotate(Path(r), a.dry_run) for r in (a.root or DEFAULT_ROOTS)]
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
