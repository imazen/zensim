"""Prepare pinned C1-C4 D2 MLP source and target-free eval tables."""

import argparse
import json
from pathlib import Path

import numpy as np

from lodo_bvls import EVAL, SOURCES
from candidate_data import ARMS, ROOT, columns, load, sha
import candidate_mlp as cm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=[*ARMS, *(f"{name}_perm" for name in ARMS)], required=True)
    args = parser.parse_args()
    cm.FEATURES, cm.CANONICAL_IDS = columns(args.arm)
    root = ROOT / "candidates/d2_mlp/tables" / args.arm
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    names = sorted(set(SOURCES) | set(EVAL.values()))
    for name in names:
        data, meta = load(name, args.arm)
        indices = np.arange(len(data))
        record = {"rows": len(data), "references": data.ref_basename.nunique(),
                  "label_source": meta, "tables": {}}
        if name in SOURCES:
            bounds = cm.fit_bounds(data, indices)
            fit = root / f"{name}.fit.parquet"
            cm.table(data, indices, bounds, fit)
            record["bounds"] = bounds
            record["tables"]["fit"] = {"path": str(fit), "sha256": sha(fit),
                                        "manifest_sha256": sha(Path(f"{fit}.manifest.json"))}
        eval_data = data.copy()
        eval_data["target"] = 0.0
        eval_path = root / f"{name}.eval.parquet"
        cm.table(eval_data, indices, (0.0, 1.0), eval_path)
        record["tables"]["eval"] = {"path": str(eval_path), "sha256": sha(eval_path),
                                      "manifest_sha256": sha(Path(f"{eval_path}.manifest.json"))}
        records[name] = record
        print(json.dumps({"set": name, "rows": len(data), "refs": record["references"],
                          "fit_sha256": record["tables"].get("fit", {}).get("sha256"),
                          "eval_sha256": record["tables"]["eval"]["sha256"]}), flush=True)
    output = root / "receipt.json"
    output.write_text(json.dumps({"schema": "rev4-featpot-c1c4-d2-mlp-tables-v1", "arm": args.arm,
                                  "canonical_candidate_ids": cm.CANONICAL_IDS,
                                  "source_sets": list(SOURCES), "eval_substitution": EVAL,
                                  "tables": records}, indent=2) + "\n")
    print(json.dumps({"receipt": str(output), "sha256": sha(output)}), flush=True)


if __name__ == "__main__":
    main()
