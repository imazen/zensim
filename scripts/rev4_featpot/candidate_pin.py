"""Create a label-free C1-C4 input receipt after the Part B report exists.

The receipt is only an inventory. Commit its exact hashes and the dated
candidate sign-mask amendment before any candidate fitting or label read.
"""

import json

import numpy as np
import pyarrow.parquet as pq

from admit_bank import PINNED_MANIFEST_SHA, PROMOTED_BANK, load_pinned_set
from candidate_data import PIN, REPORT, SIDECAR, sha


def main() -> None:
    if not REPORT.is_file():
        raise FileNotFoundError(REPORT)
    receipt = {"schema": "rev4-featpot-c1c4-input-v1", "report_sha256": sha(REPORT),
               "sets": {}, "label": "POTENTIAL — ceiling, not a model score"}
    for name in PINNED_MANIFEST_SHA:
        old = load_pinned_set(name)  # component SHA checks, no target decode
        directory = PROMOTED_BANK / name
        current = json.loads((directory / "_MANIFEST.json").read_text())
        for filename, component in old["files"].items():
            if current["files"].get(filename, {}).get("sha256") != component["sha256"]:
                raise ValueError(f"{name}: promoted base component changed: {filename}")
        sidecar = directory / SIDECAR
        side_sha = sha(sidecar)
        if current["files"].get(SIDECAR, {}).get("sha256") != side_sha:
            raise ValueError(f"{name}: sidecar manifest hash mismatch")
        expected = ["pair_key"] + [f"f{i}" for i in range(986, 1322)]
        schema = pq.ParquetFile(sidecar).schema_arrow
        if schema.names != expected or any(str(field.type) != "float" for field in schema[1:]):
            raise ValueError(f"{name}: unexpected sidecar schema")
        keys = pq.read_table(directory / "keys.parquet", columns=["pair_key"]).column(0).to_pylist()
        tab = pq.read_table(sidecar)
        observed = tab.column("pair_key").to_pylist()
        if (len(observed) != old["unique_pair_keys"] or observed != keys or
                len(set(observed)) != len(observed)):
            raise ValueError(f"{name}: candidate pair_key order/uniqueness mismatch")
        values = tab.select(expected[1:]).to_pandas().to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: nonfinite sidecar value")
        receipt["sets"][name] = {
            "rows": old["row_count"], "unique_pair_keys": old["unique_pair_keys"],
            "manifest_sha256": sha(directory / "_MANIFEST.json"),
            "keys_sha256": sha(directory / "keys.parquet"),
            "sidecar_sha256": side_sha,
        }
        print(json.dumps({"set": name, **receipt["sets"][name]}), flush=True)
    PIN.parent.mkdir(parents=True, exist_ok=True)
    PIN.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pin": str(PIN), "sha256": sha(PIN), "sets": len(receipt["sets"])}))


if __name__ == "__main__":
    main()
