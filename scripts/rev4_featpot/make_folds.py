"""Deterministic reference-grouped Rev4 POTENTIAL folds (no targets read).

The manifest is an experimental input, not a model score. Each reference is
assigned once, independently of rows and labels; all variants stay together.
"""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("/var/tmp/rev4-featpot")
SETS = (
    "kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
    "cid22_a25", "aic3", "kadid_select", "konfig_val",
)
SEED = 20260923


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def partition(refs: list[str], dataset: str, tag: str, parts: int) -> list[list[str]]:
    ordered = sorted(refs, key=lambda ref: (digest(f"{SEED}:{dataset}:{tag}:{ref}"), ref))
    return [ordered[i::parts] for i in range(parts)]


def main() -> None:
    result = {"schema": "rev4-featpot-reference-folds-v1", "seed": SEED, "sets": {}}
    for name in SETS:
        path = ROOT / "admitted" / f"POT_{name}_rev3_944.parquet"
        refs = pq.read_table(path, columns=["ref_basename"]).column(0).to_pylist()
        unique = sorted(set(refs))
        if len(unique) < 5:
            raise ValueError(f"{name}: {len(unique)} references cannot support five folds")
        outer = partition(unique, name, "outer", 5)
        rounds = []
        for o, test in enumerate(outer):
            fit_refs = sorted(set(unique) - set(test))
            inner = partition(fit_refs, name, f"inner-{o}", 4)
            if any(not fold for fold in inner):
                raise ValueError(f"{name}: empty inner fold in outer {o}")
            rounds.append({"test_refs": test, "inner_val_refs": inner})
        result["sets"][name] = {
            "input": str(path), "rows": len(refs), "references": len(unique),
            "full_refs": unique,
            "full_inner_val_refs": partition(unique, name, "full-inner", 4),
            "outer": rounds,
        }
        print(json.dumps({"set": name, "rows": len(refs), "references": len(unique),
                          "outer_ref_counts": [len(x) for x in outer],
                          "min_inner_ref_count": min(len(x) for r in rounds for x in r["inner_val_refs"])}))
    target = ROOT / "folds.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"folds_path": str(target), "folds_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
