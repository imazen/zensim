#!/usr/bin/env python3
"""Probe unchanged public serving with pinned bakes and one identity fixture."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


PINS = [
    "f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400",
    "1bf8f3afbf0d4c760e3fcfb1e36fa3ca8c2c19f838b0f6719b8c681fa682fdec",
    "54118c54ce7fe90bf1e9dcec52f7f822caf54e93e5b1d1a95a2bbd37f741b688",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("models", nargs=3, type=Path)
    args = parser.parse_args()
    # Finish input admission before any scoring. No label or corpus loader.
    assert [digest(p) for p in args.models] == PINS, "production model pins"
    assert args.program.is_file() and args.fixture.is_file()
    assert args.output_dir.is_dir()
    result = {
        "scope": "identity fixture only; serving revision contract, no quality gate",
        "program_sha256": digest(args.program),
        "fixture_sha256": digest(args.fixture),
        "fixture": str(args.fixture),
        "model_sha256": PINS,
        "cases": [],
    }
    for revision in [None, "5"]:
        for seed, model in enumerate(args.models):
            env = dict(os.environ)
            env.pop("ZENSIM_FORMULA_REV", None)
            if revision is not None:
                env["ZENSIM_FORMULA_REV"] = revision
            command = [str(args.program), str(model), str(args.fixture), str(args.fixture)]
            proc = subprocess.run(command, env=env, text=True, capture_output=True)
            name = f"seed{seed}-revision-{revision or 'unset'}.log"
            with (args.output_dir / name).open("x") as output:
                output.write(proc.stdout + proc.stderr)
            if revision is None:
                observed = (
                    proc.returncode == 1
                    and "cannot be mixed" in proc.stdout
                    and "IDENTITY REFUSED" in proc.stdout
                )
            else:
                observed = (
                    proc.returncode == 0
                    and "SERVED  score=100.000000" in proc.stdout
                    and "IDENTITY (ref vs ref) score=100.000000" in proc.stdout
                )
            result["cases"].append({
                "seed": seed,
                "process_revision": revision,
                "returncode": proc.returncode,
                "expected_contract_observed": observed,
                "command": command,
                "log": name,
            })
            print(f"seed {seed}, revision {revision}: contract observed={observed}", flush=True)
    result["all_expected_contracts_observed"] = all(
        case["expected_contract_observed"] for case in result["cases"]
    )
    with (args.output_dir / "revision-pin-results.json").open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    assert result["all_expected_contracts_observed"], "unexpected serving behavior"


if __name__ == "__main__":
    main()
