#!/usr/bin/env python3
"""Bank-level report: walk /var/tmp/rev4-featbank/bank, validate every set's
manifest + file hashes + key/sidecar row agreement, emit bank/_MANIFEST.json.

Read-only w.r.t. set files; writes only bank/_MANIFEST.json.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

BANK = Path("/var/tmp/rev4-featbank/bank")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    sets = {}
    problems = []
    for d in sorted(BANK.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        mpath = d / "_MANIFEST.json"
        if not mpath.exists():
            problems.append(f"{d.name}: missing _MANIFEST.json")
            continue
        m = json.loads(mpath.read_text())
        entry = {
            "role": m["role"],
            "row_count": m["row_count"],
            "unique_pair_keys": m["unique_pair_keys"],
            "collapsed": m["pixel_identical_stimuli_collapsed"],
            "labels": m["label_handling"],
            "files": {},
        }
        for name, meta in m["files"].items():
            p = d / name if not name.startswith("_sealed/") else BANK / "_sealed" / d.name / Path(name).name
            if not p.exists():
                problems.append(f"{d.name}: missing {name}")
                continue
            h = sha256_file(p)
            if h != meta["sha256"]:
                problems.append(f"{d.name}: {name} sha256 drift")
            entry["files"][name] = meta
        # Check every sidecar, including later feature-family extensions.
        try:
            nk = pq.read_metadata(d / "keys.parquet").num_rows
            sc = sorted(d.glob("features__*.parquet"))
            if not sc:
                problems.append(f"{d.name}: no feature sidecar")
            if nk != m["unique_pair_keys"]:
                problems.append(f"{d.name}: keys {nk} != manifest {m['unique_pair_keys']}")
            entry["keys_rows"] = nk
            entry["sidecar_cols_by_file"] = {}
            for sidecar in sc:
                ns = pq.read_metadata(sidecar).num_rows
                if nk != ns:
                    problems.append(f"{d.name}: keys {nk} != {sidecar.name} {ns}")
                cols = set(pq.read_schema(sidecar).names)
                leaked = [c for c in cols if c.startswith("f") and c[1:].isdigit()
                          and int(c[1:]) in set(m["structural_zero_feature_ids"])]
                if leaked:
                    problems.append(f"{d.name}: structural-zero cols in {sidecar.name}: {leaked}")
                entry["sidecar_cols_by_file"][sidecar.name] = len(cols) - 1
            original = next((p for p in sc if p.name != "features__rev4c1c4.parquet"), None)
            entry["sidecar_cols"] = (entry["sidecar_cols_by_file"][original.name]
                                     if original else 0)
        except Exception as e:
            problems.append(f"{d.name}: validation error {e}")
        sets[d.name] = entry

    report = {
        "schema": "rev4-featbank-bank-report-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bank_root": str(BANK),
        "feature_set_id": (json.loads((BANK / sorted(sets)[0] / "_MANIFEST.json").read_text())["feature_set_id"] if sets else None),
        "sets": sets,
        "problems": problems,
        "n_sets": len(sets),
        "total_unique_keys": sum(e["unique_pair_keys"] for e in sets.values()),
        "total_stimuli": sum(e["row_count"] for e in sets.values()),
    }
    (BANK / "_MANIFEST.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"n_sets": len(sets), "problems": problems,
                      "total_unique_keys": report["total_unique_keys"],
                      "total_stimuli": report["total_stimuli"]}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
