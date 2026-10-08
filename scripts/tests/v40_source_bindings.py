"""Bind retained V40 executables to a reachable, exact trainer/admission tree."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--producer", required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["jj", "log", "-r", args.producer, "--no-graph", "-T", "commit_id"],
        cwd=args.source, text=True,
    ).strip()
    if commit != args.producer:
        raise ValueError("an exact producer commit is required")
    paths = subprocess.check_output(
        ["jj", "file", "list", "-r", commit, "Cargo.lock",
         "Cargo.toml", "zensim/src", "zensim/Cargo.toml", "zensim-train-core",
         "zensim-validate/src", "zensim-validate/examples", "zensim-validate/Cargo.toml"],
        cwd=args.source, text=True,
    ).splitlines()
    checks = []
    for rel in paths:
        if not (rel.endswith(".rs") or Path(rel).name in ("Cargo.lock", "Cargo.toml")):
            continue
        recorded = subprocess.check_output(["jj", "file", "show", "-r", commit, rel], cwd=args.source)
        current = (args.source / rel).read_bytes()
        if recorded != current:
            raise ValueError(f"uncommitted trainer/admission drift: {rel}")
        checks.append(dict(path=rel, producer_sha256=digest(recorded),
                           current_sha256=digest(current), byte_equal=True))
    binaries = ("zensim_mlp_train", "bake_dial_refit", "panel",
                "predict_features_with_bake", "inspect_qualified_checkpoint")
    record = dict(schema="v40-trainer-admission-source-bindings-v2",
                  binary_producer_commit=commit, current_source_commit=commit,
                  source_checks=checks,
                  binaries={name: dict(producer_commit=commit,
                                      sha256=digest((args.bundle / "bin" / name).read_bytes()))
                            for name in binaries})
    with (args.bundle / "SOURCE_BINDINGS.json").open("x") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(producer=commit, source_files=len(checks), binaries=len(binaries))))


if __name__ == "__main__":
    main()
