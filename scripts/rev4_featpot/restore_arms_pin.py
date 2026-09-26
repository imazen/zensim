"""Pin the restore-cuts arms: column sets, sidecar hashes and label-free A1w drop lists.

POTENTIAL - ceiling, not a model score. Reads FEATURE columns only (pair_key,
ref_basename and f0..f943 of the admitted Rev3 tables; the restore sidecars'
manifests). It opens no label column and no label file. The drop rule is the one
registered in benchmarks/rev4_featpot_restore_cuts_amendment_2026-09-25.md:
constant columns first (ascending id), then the higher id of the highest-|r| pair
among remaining columns, until 60 are dropped.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/var/tmp/rev4-featpot")
BANK = Path("/var/tmp/restore-cuts/bank")
OUT = Path("benchmarks/rev4_featpot_restore_arms_2026-09-25.json")
DROPS = Path("benchmarks/rev4_featpot_restore_arms_droplists_2026-09-25.json")
N_DROP = 60
FEATURES = [f"f{i}" for i in range(944)]
D1_SETS = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
           "cid22_a25", "kadid_select", "konfig_val", "aic3")
D2_SOURCES = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
              "cid22_a25", "aic3", "kadid_select")
PIN_SETS = D1_SETS + ("konjnd_bpg_val",)
FAMILIES = {"mapdev": (1502, 60), "z1max": (1562, 228), "gmsnative": (1790, 30),
            "dvifmgate": (1820, 5)}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arms() -> dict:
    mapdev = list(range(1502, 1562))
    z1max = list(range(1562, 1790))
    a1m = [1502 + 5 * c + s for c in range(12) for s in (0, 1, 2)]
    b2m = [1562 + 19 * c + l for c in range(12)
           for l in (3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18)]
    gate = list(range(1820, 1825))
    c7_f1 = [956 + 6 * level for level in range(5)]
    assert (len(a1m), len(b2m)) == (36, 156)
    return {
        "A1": {"add": mapdev}, "A1m": {"add": a1m},
        "A1w": {"add": mapdev, "drop_rule": "label-free redundancy, see drop_lists"},
        "B2": {"add": z1max}, "B2m": {"add": b2m},
        "B1": {"add": list(range(956, 986)) + gate, "needs": "C7 sidecar (Part B)"},
        "B1s": {"add": [i for i in range(956, 986) if i not in c7_f1] + gate,
                "replaced_f1": c7_f1, "needs": "C7 sidecar (Part B)"},
        "C8n": {"add": list(range(1322, 1502)) + list(range(1790, 1820)),
                "needs": "C8 sidecar (Part B)"},
        "ALL": {"add": mapdev + z1max + list(range(956, 986)) + gate
                + list(range(1322, 1502)) + list(range(1790, 1820)),
                "needs": "C7 and C8 sidecars (Part B)"},
    }


def read_features(name: str):
    path = ROOT / "admitted" / f"POT_{name}_rev3_944.parquet"
    table = pq.read_table(path, columns=["ref_basename"] + FEATURES)
    refs = np.asarray(table.column("ref_basename").to_pylist(), dtype=object)
    x = np.column_stack([table.column(c).to_numpy(zero_copy_only=False) for c in FEATURES])
    return refs, x.astype(np.float64), path


def drop_list(x: np.ndarray) -> list[int]:
    if not np.isfinite(x).all():
        raise ValueError("non-finite feature values in a fit population")
    std = x.std(axis=0)
    constant = [int(i) for i in np.flatnonzero(std == 0)]
    dropped = constant[:N_DROP]
    alive = np.ones(x.shape[1], dtype=bool)
    alive[dropped] = False
    if len(dropped) < N_DROP:
        corr = np.abs(np.corrcoef(x[:, alive], rowvar=False))
        ids = np.flatnonzero(alive)
        corr[np.tril_indices_from(corr)] = -1.0
        live = np.ones(len(ids), dtype=bool)
        while len(dropped) < N_DROP:
            masked = np.where(live[:, None] & live[None, :], corr, -1.0)
            i, j = np.unravel_index(np.argmax(masked), masked.shape)
            dropped.append(int(ids[j]))
            live[j] = False
    return sorted(dropped)


def main() -> None:
    folds = json.loads((ROOT / "folds.json").read_text())["sets"]
    tables = {name: read_features(name) for name in set(D1_SETS) | set(D2_SOURCES)}
    lists = {}
    for name in D1_SETS:
        refs, x, _ = tables[name]
        full = folds[name]["full_refs"]
        lists[f"D1/{name}/full"] = drop_list(x[np.isin(refs, full)])
        for o, outer in enumerate(folds[name]["outer"]):
            fit = sorted(set(full) - set(outer["test_refs"]))
            lists[f"D1/{name}/o{o}"] = drop_list(x[np.isin(refs, fit)])
    for held in D2_SOURCES:
        x = np.vstack([tables[s][1] for s in D2_SOURCES if s != held])
        lists[f"D2/without_{held}"] = drop_list(x)
    pins = {}
    for name in PIN_SETS:
        manifest = BANK / name / "_MANIFEST_restore.json"
        m = json.loads(manifest.read_text())
        pins[name] = {"manifest_sha256": sha(manifest), "row_count": m["row_count"],
                      "feature_set_id": m["feature_set_id"], "build_commit": m["build_commit"],
                      "binary_sha256": m["binary_sha256"],
                      "families": {f: {"sha256": m["families"][f]["sha256"],
                                       "file": str(BANK / name / m["families"][f]["file"])}
                                   for f in FAMILIES}}
    out = {"schema": "rev4-featpot-restore-arms-v1",
           "label": "POTENTIAL — ceiling, not a model score",
           "amendment": "benchmarks/rev4_featpot_restore_cuts_amendment_2026-09-25.md",
           "reads": "feature columns only; no label column or file was opened",
           "join_key": "pair_key (missing or duplicate key fails the arm)",
           "families": {f: {"first_id": a, "width": w} for f, (a, w) in FAMILIES.items()},
           "arms": arms(),
           "inputs": {"folds_json_sha256": sha(ROOT / "folds.json"),
                      "admitted_tables": {n: sha(tables[n][2]) for n in sorted(tables)},
                      "script_sha256": sha(Path(__file__))},
           "sidecar_pins": pins}
    drops = {"schema": "rev4-featpot-restore-a1w-droplists-v1",
             "rule": "constant columns first (ascending id), then the higher id of the "
                     "highest-|r| remaining pair, until 60 dropped; features only",
             "arms_file": str(OUT), "n_drop": N_DROP, "drop_lists": lists}
    DROPS.write_text(json.dumps(drops, separators=(",", ":")) + "\n")
    out["drop_lists_file"] = {"path": str(DROPS), "sha256": sha(DROPS)}
    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps({"arms": str(OUT), "arms_sha256": sha(OUT), "arms_bytes": OUT.stat().st_size,
                      "drops": str(DROPS), "drops_sha256": sha(DROPS),
                      "drops_bytes": DROPS.stat().st_size,
                      "drop_lists": len(lists), "pinned_sets": len(pins)}))


if __name__ == "__main__":
    main()
