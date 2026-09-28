#!/usr/bin/env python3
"""Development stage of run_full_eval.sh; orchestration only, no model/stat math.

Build the Rust owners once before timing. New recipes require physically separate,
explicitly admitted train/eval segments. Test/terminal segments and legacy mixed
caches are forbidden. Historical v1 recipes remain records, not executable inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recipe", type=Path)
    ap.add_argument("out", type=Path, help="fresh output directory")
    ap.add_argument("--cache", type=Path, help="retired mixed-cache option; rejected under strict split policy")
    ap.add_argument("--ceiling-stage", choices=("all", "prepare", "fit", "audit", "report"), default="all")
    ap.add_argument("--ceiling-panel", type=Path, help="raw-error-capable Rust panel for ceiling audit")
    args = ap.parse_args()
    recipe = json.loads(args.recipe.read_text())
    from feature_screen_ceiling import execute
    return execute(args, recipe)


if __name__ == "__main__":
    main()
