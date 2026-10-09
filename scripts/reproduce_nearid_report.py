#!/usr/bin/env python3
"""Reproduce the frozen NEARID report without opening pixels or labels."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    expected = json.loads((args.frozen / "SUMMARY.json").read_text())
    assert expected["rows"] == 1944
    args.output.mkdir()  # Refuse an existing evidence directory.
    inputs = {}
    for model in ("seed0", "B", "A"):
        name = f"{model}.jsonl"
        inputs[name] = digest(args.frozen / name)
        shutil.copyfile(args.frozen / name, args.output / name)
        assert digest(args.output / name) == inputs[name]
    # Exercise the merged dispatcher, rather than calling the helper directly.
    subprocess.run([sys.executable, str(Path(__file__).with_name("prodqual_label_free.py")),
                    "--nearid-summary", str(args.output)], check=True)
    outputs = {}
    for name in ("SUMMARY.json", "scores.tsv", "SUMMARY.md"):
        actual = digest(args.output / name)
        assert actual == digest(args.frozen / name), f"report changed: {name}"
        outputs[name] = actual
    result = {"pass": True, "rows": 1944, "input_sha256": inputs,
              "report_sha256": outputs,
              "scope": "exact numerical and text report reproduction from frozen scored rows; no rescore",
              "figures": "PNG and SVG regenerated; SVG timestamps are not compared"}
    with (args.output / "REPRODUCED.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("NEARID report reproduction PASS: SUMMARY.json, scores.tsv and SUMMARY.md byte-identical")


if __name__ == "__main__":
    main()
