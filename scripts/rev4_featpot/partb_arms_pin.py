"""Pin the Part B candidate arms (C1-C4, csfw, C7, P1, P3 and the restore combinations).

POTENTIAL - ceiling, not a model score. Label-free: reads the promoted bank's
_MANIFEST.json, the three Part B sidecars' parquet SCHEMAS and hashes, and the
admitted tables' pixels_identical flag. Opens no label column or file.
"""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

BANK = Path("/var/tmp/rev4-featbank/bank")
ADMITTED = Path("/var/tmp/rev4-featpot/admitted")
OUT = Path("benchmarks/rev4_featpot_partb_arms_2026-09-25.json")
RESTORE = json.loads(Path("benchmarks/rev4_featpot_restore_arms_2026-09-25.json").read_text())
PIN_SETS = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train", "cid22_a25",
            "kadid_select", "konfig_val", "aic3", "konjnd_bpg_val")
FILES = {"csfw_dvifm": ("features__csfw_dvifm.parquet", 944, 986),
         "c1c4": ("features__rev4c1c4.parquet", 986, 1322),
         "gmsbank": ("features__gmsbank.parquet", 1322, 1502)}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    pins = {}
    for name in PIN_SETS:
        directory = BANK / name
        manifest = json.loads((directory / "_MANIFEST.json").read_text())
        entry = {"manifest_sha256": sha(directory / "_MANIFEST.json"),
                 "keys_sha256": sha(directory / "keys.parquet"),
                 "row_count": manifest["row_count"], "unique_pair_keys": manifest["unique_pair_keys"],
                 "families": {}}
        for family, (filename, lo, hi) in FILES.items():
            path = directory / filename
            names = pq.ParquetFile(path).schema_arrow.names
            assert names == ["pair_key"] + [f"f{i}" for i in range(lo, hi)], (name, family)
            entry["families"][family] = {"file": str(path), "sha256": sha(path),
                                         "manifest_sha256": manifest["files"][filename]["sha256"]}
            assert entry["families"][family]["sha256"] == entry["families"][family]["manifest_sha256"]
        table = pq.read_table(ADMITTED / f"POT_{name}_rev3_944.parquet",
                              columns=["pair_key", "pixels_identical"]).to_pandas()
        flags = table.drop_duplicates("pair_key").pixels_identical.astype(bool)
        entry["identical_keys"] = int(flags.sum())
        entry["identical_stimulus_rows"] = int(table.pixels_identical.astype(bool).sum())
        pins[name] = entry
    c7 = list(range(956, 986))
    c7_f1 = [956 + 6 * level for level in range(5)]
    c8 = list(range(1322, 1502))
    arms = {
        "c1": {"add": list(range(986, 1082))}, "c2": {"add": list(range(1082, 1154))},
        "c3": {"add": list(range(1154, 1298))}, "c4": {"add": list(range(1298, 1322))},
        "all": {"add": list(range(986, 1322)), "note": "C1-C4 together"},
        "csfw": {"add": list(range(944, 956))}, "c7": {"add": c7},
        "p1": {"add": c8, "note": "gmsbank (C8, revised) f1322-1501"},
        "p3": {"add": c8, "peer": ["gmsd", "gmsm"], "note": "P1 plus the reviewed P2 peer columns"},
        "b1": RESTORE["arms"]["B1"], "b1s": RESTORE["arms"]["B1s"],
        "c8n": RESTORE["arms"]["C8n"], "rall": RESTORE["arms"]["ALL"],
    }
    arms["b1"] = {"add": arms["b1"]["add"]}
    arms["b1s"] = {"add": arms["b1s"]["add"], "replaced_f1": c7_f1}
    arms["c8n"] = {"add": arms["c8n"]["add"]}
    arms["rall"] = {"add": arms["rall"]["add"], "note": "restore ALL = A1 + B2 + B1 + C8n additions"}
    out = {"schema": "rev4-featpot-partb-arms-v1", "label": "POTENTIAL — ceiling, not a model score",
           "amendment": "benchmarks/rev4_featpot_partb_arms_amendment_2026-09-25.md",
           "reads": "manifests, parquet schemas, file hashes and the pixels_identical flag; no label",
           "join_key": "pair_key (missing or duplicate key fails the arm)",
           "restore_arms_json_sha256": sha(Path("benchmarks/rev4_featpot_restore_arms_2026-09-25.json")),
           "family_files": {k: {"file": v[0], "first_id": v[1], "width": v[2] - v[1]}
                            for k, v in FILES.items()},
           "peer": {"dir": "/var/tmp/gmsbank/peer_gmsd",
                    "manifest_sha256": "8d2799c8a9ebaa3f25fa7518b3d7591ee7742b696d7f5592b9ace9974d0d27fc",
                    "note": "reviewed P2 peer columns gmsd, gmsm; pinned by p2_data.py constants"},
           "known_properties": {
               "c3_p99_top_bin_saturation_pct_of_cells": {
                   "limit": 1.0, "source": "REVIEW_PARTB.md / PARTB_C1C4_DONE.md item 6",
                   "exceeds_on": {"kadid_train": {"ssim": 0.53, "art": 2.87, "det": 2.88, "mse": 0.00},
                                  "kadid_select": {"ssim": 0.50, "art": 3.26, "det": 3.23, "mse": 0.00},
                                  "tid2013": {"ssim": 0.17, "art": 2.01, "det": 2.01, "mse": 0.00}},
                   "within_limit": "other sets <= 0.08 (ssim) and <= 0.02 (art, det), mse 0.00"},
               "dead_columns": {"ids": list(range(1066, 1074)),
                                "note": "gridblk (Y, scale 3) registered stated-zero; all-zero in every set"}},
           "arms": arms, "sidecar_pins": pins}
    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha(OUT), "bytes": OUT.stat().st_size,
                      "pinned_sets": len(pins), "arms": list(arms)}))


if __name__ == "__main__":
    main()
