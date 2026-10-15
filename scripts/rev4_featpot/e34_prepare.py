"""Label-free E34 preparation. No fitting, prediction, or assessment entrypoint.

P1 binds the existing E33 fit contract before delegating column measurements to
canonical_corpus.check_table_integrity. Only the registered difference columns
are decoded; target and development payloads are not read.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canonical_corpus.check_table_integrity import column_stats  # noqa: E402
from v2_common import COVERAGE_FAMILIES, fx1_declaration, sha  # noqa: E402

CONTRACT_SHA = "3bb4705b8a07307b3dec3956f43d99ae649ea333b6c95817ac389167713ec4d7"
from v2_teacher import POOL_SHA_REV5, POOL_KEYS_SHA, row_keys_sha, selection_sha  # noqa: E402

FIT_TABLES = {
    "safesyn": "wide/main/real/safesyn_fit.parquet",
    "cid22": "wide/main/real/cid22_fit.parquet",
    "coverage": "e15/coverage_pool.parquet",
}


def census(root: Path, contract: Path) -> dict:
    if sha(contract) != CONTRACT_SHA:
        raise ValueError("not the frozen E33 fit contract")
    packet = json.loads(contract.read_text())
    columns = fx1_declaration()["direct"]
    admitted = {}
    coverage_mask = None
    # Match all positive-training-weight tables of A's LODO and full-data routes.
    # C has the same tables; its products are not part of P1's 410-column census.
    for variant in packet["variants"]:
        if variant["name"] not in ("a", "full-a"):
            continue
        for route, records in variant["routes"].items():
            inputs = variant["input_contracts"][route]
            for record in records:
                name = record["name"]
                if inputs[name]["train_w"] <= 0:
                    continue
                if name == "human":
                    stem = "all" if route == "production" else f"without_{route}"
                    rel = f"wide/main/real/human_{stem}_fit.parquet"
                else:
                    rel = FIT_TABLES[name]
                path = root / rel
                sidecar = Path(f"{path}.manifest.json")
                if name == "coverage":
                    # coverage_leg creates cf98 at fit time and removes it later.
                    # Reconstruct its exact row selection from label-free keys,
                    # never call coverage_leg (it reads human_score).
                    if sha(path) != POOL_SHA_REV5:
                        raise ValueError("coverage: parent pool bytes changed")
                    keys_path = path.with_suffix(".keys.parquet")
                    if sha(keys_path) != POOL_KEYS_SHA:
                        raise ValueError("coverage: parent keys changed")
                    keys = pq.read_table(keys_path)
                    families = [f for i, f in enumerate(COVERAGE_FAMILIES) if 0x98 >> i & 1]
                    coverage_mask = np.isin(keys["family"].to_numpy(), families)
                    if selection_sha(np.flatnonzero(coverage_mask)) != record["row_selection_sha256"]:
                        raise ValueError("coverage: selected row indices differ")
                    if row_keys_sha(keys.filter(pa.array(coverage_mask))) != record["row_keys_sha256"]:
                        raise ValueError("coverage: selected row identities differ")
                    admitted[rel] = dict(table_sha256=POOL_SHA_REV5,
                                         manifest_sha256=sha(sidecar),
                                         registered_derived_table_sha256=record["table_sha256"],
                                         selected_keys_sha256=record["row_keys_sha256"],
                                         row_selection_sha256=record["row_selection_sha256"],
                                         rows=inputs[name]["rows"])
                    continue
                if sha(sidecar) != record["manifest_sha256"]:
                    raise ValueError(f"{rel}: manifest differs from frozen contract")
                metadata = json.loads(sidecar.read_text())
                if metadata["formula_revision"] != 5:
                    raise ValueError(f"{rel}: expected Rev5")
                if metadata["table_sha256"] != record["table_sha256"]:
                    raise ValueError(f"{rel}: table declaration differs from contract")
                if rel in admitted:
                    if admitted[rel]["table_sha256"] != record["table_sha256"]:
                        raise ValueError(f"{rel}: inconsistent route binding")
                    continue
                # Hash compressed bytes without decoding labels. Do this only after
                # metadata admission; no directory discovery of other populations.
                if sha(path) != record["table_sha256"]:
                    raise ValueError(f"{rel}: table bytes differ from frozen contract")
                admitted[rel] = dict(table_sha256=record["table_sha256"],
                                     manifest_sha256=record["manifest_sha256"],
                                     rows=inputs[name]["rows"])
    if len(admitted) != 8:
        raise ValueError(f"expected eight distinct fit tables, got {len(admitted)}")
    tables = []
    for rel, binding in sorted(admitted.items()):
        measured = column_stats(str(root / rel), [(i, f"f{i}") for i in columns],
                                row_mask=coverage_mask if rel == FIT_TABLES["coverage"] else None)
        for value in measured.values():
            if value["n"] != binding["rows"] or value["n_nan"] or value["n_inf"]:
                raise ValueError(f"{rel}: unexpected row count or nonfinite difference")
        rows = [{"feature": i, "min": measured[i]["min"], "max": measured[i]["max"]}
                for i in columns]
        tables.append(dict(path=rel, **binding, columns=rows,
                           negative_columns=[row["feature"] for row in rows if row["min"] < 0]))
        print(f"{rel}: {binding['rows']} rows; negative columns {tables[-1]['negative_columns']}", flush=True)
    combined = [{"feature": i, "min": min(t["columns"][j]["min"] for t in tables),
                 "max": max(t["columns"][j]["max"] for t in tables)} for j, i in enumerate(columns)]
    return dict(schema="e34-feature-census-v1", labels_read=False,
                decoded_columns=[f"f{i}" for i in columns], contract_sha256=CONTRACT_SHA,
                root=str(root.resolve()), tables=tables, columns=combined,
                negative_columns=[r["feature"] for r in combined if r["min"] < 0],
                rows_include_repeated_lodo_subsets=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a fresh receipt")
    result = census(args.root, args.contract)
    result["command"] = sys.argv
    result["commit"] = subprocess.check_output(
        ["jj", "log", "-r", "@", "--no-graph", "-T", "commit_id"], text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
