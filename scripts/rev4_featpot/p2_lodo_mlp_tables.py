"""Prepare pinned, quarantined D2 MLP source and target-free eval tables."""

import argparse
import json
from pathlib import Path

import numpy as np

from data import load as load_bank
from lodo_bvls import EVAL, SOURCES
from mlp_probe import table as bank_table, fit_bounds as bank_bounds
from p2_data import ROOT, load as load_peer, sha
from p2_mlp import table as peer_table, fit_bounds as peer_bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["r0", "p2", "p2_perm"], required=True)
    args = parser.parse_args()
    root = ROOT / "p2/d2_mlp/tables" / args.arm
    root.mkdir(parents=True, exist_ok=True)
    prepare = bank_table if args.arm == "r0" else peer_table
    bounds_fn = bank_bounds if args.arm == "r0" else peer_bounds
    records = {}
    names = sorted(set(SOURCES) | set(EVAL.values()))
    for name in names:
        if args.arm == "r0":
            data, meta = load_bank(name)
        else:
            data, meta = load_peer(name, args.arm)
            data = data.rename(columns={"gmsd": "f944", "gmsm": "f945"})
        indices = np.arange(len(data))
        record = {"rows": len(data), "references": data.ref_basename.nunique(),
                  "label_source": meta, "tables": {}}
        if name in SOURCES:
            bounds = bounds_fn(data, indices)
            fit = root / f"{name}.fit.parquet"
            prepare(data, indices, bounds, fit)
            record["bounds"] = bounds
            record["tables"]["fit"] = {"path": str(fit), "sha256": sha(fit),
                                        "manifest_sha256": sha(Path(f"{fit}.manifest.json"))}
        eval_data = data.copy()
        eval_data["target"] = 0.0
        eval_path = root / f"{name}.eval.parquet"
        prepare(eval_data, indices, (0.0, 1.0), eval_path)
        record["tables"]["eval"] = {"path": str(eval_path), "sha256": sha(eval_path),
                                      "manifest_sha256": sha(Path(f"{eval_path}.manifest.json"))}
        records[name] = record
        print(json.dumps({"set": name, "rows": len(data), "refs": record["references"],
                          "fit_sha256": record["tables"].get("fit", {}).get("sha256"),
                          "eval_sha256": record["tables"]["eval"]["sha256"]}), flush=True)
    output = root / "receipt.json"
    output.write_text(json.dumps({"schema": "rev4-featpot-d2-mlp-tables-v1", "arm": args.arm,
                                  "source_sets": list(SOURCES), "eval_substitution": EVAL,
                                  "tables": records}, indent=2) + "\n")
    print(json.dumps({"receipt": str(output), "sha256": sha(output)}), flush=True)


if __name__ == "__main__":
    main()
