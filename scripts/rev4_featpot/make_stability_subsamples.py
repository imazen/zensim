"""Fix the 200 reference half-samples for baseline lasso stability.

This reads the target-free fold manifest only. The output is an experimental
input under /var/tmp, not a result, and is shared by all feature arms.
"""

import hashlib
import json
import random
from pathlib import Path


ROOT = Path("/var/tmp/rev4-featpot")
SEED = 20260923
B = 200


def main() -> None:
    folds = json.loads((ROOT / "folds.json").read_text())
    if folds["seed"] != SEED:
        raise ValueError("fold seed differs from preregistration")
    result = {"schema": "rev4-featpot-stability-refs-v1", "seed": SEED,
              "subsamples_per_set": B, "unit": "reference", "sets": {}}
    for name, fold in folds["sets"].items():
        refs = fold["full_refs"]
        count = len(refs) // 2
        if count < 2 or len(set(refs)) != len(refs):
            raise ValueError(f"{name}: invalid reference list")
        dataset_seed = int.from_bytes(
            hashlib.sha256(f"{SEED}:{name}:stability".encode()).digest()[:8], "big")
        rng = random.Random(dataset_seed)
        samples = [sorted(rng.sample(refs, count)) for _ in range(B)]
        if any(len(sample) != count or len(set(sample)) != count for sample in samples):
            raise ValueError(f"{name}: invalid half-reference sample")
        result["sets"][name] = {"references": len(refs), "sampled_references": count,
                                "dataset_seed": dataset_seed, "samples": samples}
        print(json.dumps({"set": name, "references": len(refs), "sampled": count,
                          "draws": B, "unique_draws": len({tuple(s) for s in samples})}))
    output = ROOT / "stability_subsamples.json"
    output.write_text(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output),
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
