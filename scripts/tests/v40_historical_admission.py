"""Historical population admission regression; all payloads are synthetic.

Every negative must refuse before *any* table open, including admitted groups.
The same zero-open oracle fails on the retained V40R3 trainer.
Permitted ordinary historical recipes still complete training.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=False)
    owner = Path(__file__).with_name("v40_review_admission.py")
    sys.argv = [
        str(owner),
        "--binary",
        str(args.binary),
        "--dest",
        str(args.dest / "fixtures"),
    ]
    fixture_owner = runpy.run_path(str(owner), run_name="inventory_fixture_owner")
    f, _, _, groups = fixture_owner["d1_fixture"](dense=True)
    reports = []

    def run(name, flags, extra=(), positive=False):
        dest = args.dest / name
        dest.mkdir()
        argv = fixture_owner["d1_argv"](
            groups,
            fixture_owner["upiq"].columns("by_v2fy"),
            1853,
            [*flags, "--historical-replay", "synthetic compatibility probe"],
        )
        argv = [str(dest / "model.bin") if str(v) == "@OUT" else str(v) for v in argv]
        trace = dest / "open.trace"
        proc = subprocess.run(
            [
                "strace",
                "-qq",
                "-f",
                "-e",
                "trace=open,openat,openat2",
                "-o",
                str(trace),
                *argv,
            ],
            capture_output=True,
            text=True,
        )
        (dest / "argv.json").write_text(json.dumps(argv, indent=2) + "\n")
        (dest / "stdout.log").write_text(proc.stdout)
        (dest / "stderr.log").write_text(proc.stderr)
        payloads = [*(g[1] for g in groups), *extra]
        names = {str(p) for p in payloads} | {str(p.resolve()) for p in payloads}
        opens = [
            line
            for line in trace.read_text().splitlines()
            if (
                any('"' + p + '"' in line for p in names)
                or ('.parquet"' in line and '.keys.parquet"' not in line)
            )
            and "O_PATH" not in line
            and "O_DIRECTORY" not in line
            and "= -1" not in line
        ]
        written = (dest / "model.bin").exists()
        passed = (
            (proc.returncode == 0 and written and bool(opens))
            if positive
            else (proc.returncode == 2 and not written and not opens)
        )
        result = dict(
            case=name,
            positive=positive,
            status="PASS" if passed else "FAIL",
            exit_code=proc.returncode,
            payload_opens=len(opens),
            model_written=written,
        )
        reports.append(result)
        (dest / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        declarations = {
            "VAL": dict(role="val", split="development"),
            "validation": dict(role="validation"),
            "T0": dict(tier="T0"),
            "terminal": dict(split="terminal"),
            "test": dict(role="test"),
            "AIC": dict(human_sources=["aic3"], data_role="holdout-only"),
            "holdout": dict(data_role="holdout-only"),
            "forbidden-key": dict(role="train", keys_role="val"),
        }
        routes = (
            "manifest",
            "anchor-parquet",
            "cross-codec-eq-parquet",
            "pjnd-passthrough-parquet",
            "konjnd-aggregation-parquet",
        )
        for role_index, (role, declaration) in enumerate(declarations.items()):
            for route in routes:
                for alias in (False, True):
                    dest = args.dest / f"input-{role_index}-{route}-{alias}"
                    dest.mkdir()
                    sentinel = dest / "ordinary.parquet"
                    sentinel.write_bytes(b"SYNTHETIC PROTECTED ROLE PAYLOAD")
                    Path(str(sentinel) + ".manifest.json").write_text(
                        json.dumps(declaration)
                    )
                    path = sentinel
                    if alias:
                        path = dest / "alias.parquet"
                        path.symlink_to(sentinel)
                    if route == "manifest":
                        manifest = dest / "inventory.toml"
                        manifest.write_text(
                            f'[inputs.admitted_first]\npath = {json.dumps(str(groups[0][1]))}\nsha256 = "{sha(groups[0][1])}"\n'
                            f'[inputs.extra]\npath = {json.dumps(str(path))}\nsha256 = "{sha(path)}"\n'
                        )
                        flags = ["--manifest", manifest]
                    else:
                        flags = ["--" + route, path]
                    run(f"refuse-{role}-{route}-{alias}", flags, [path, sentinel])
        # Ordinary TRAIN declarations cannot hide forbidden label-free row keys.
        import pyarrow as pa
        import pyarrow.parquet as pq

        for role in ("val", "T0", "aic3"):
            for route in routes:
                dest = args.dest / f"key-{role}-{route}"
                dest.mkdir()
                sentinel = dest / "ordinary.parquet"
                sentinel.write_bytes(b"SYNTHETIC KEY-DECLARED PROTECTED PAYLOAD")
                Path(str(sentinel) + ".manifest.json").write_text('{"role":"train"}')
                pq.write_table(
                    pa.table({"role": [role], "row_id": [0]}),
                    sentinel.with_suffix(".keys.parquet"),
                    compression="zstd",
                )
                if route == "manifest":
                    manifest = dest / "inventory.toml"
                    manifest.write_text(
                        f'[inputs.extra]\npath = {json.dumps(str(sentinel))}\nsha256 = "{sha(sentinel)}"\n'
                    )
                    flags = ["--manifest", manifest]
                else:
                    flags = ["--" + route, sentinel]
                run(f"refuse-key-{role}-{route}", flags, [sentinel])
        # Group declarations themselves also precede every group payload.
        sidecar = Path(str(groups[-1][1]) + ".manifest.json")
        original = sidecar.read_bytes()
        for role, declaration in declarations.items():
            sidecar.write_text(json.dumps(declaration))
            run("refuse-group-" + role, [])
        sidecar.write_bytes(original)
        run("positive-permitted-groups", [], positive=True)
        sentinel = args.dest / "ordinary-train.bin"
        sentinel.write_bytes(b"SYNTHETIC ORDINARY TRAIN INPUT")
        Path(str(sentinel) + ".manifest.json").write_text('{"role":"train"}')
        manifest = args.dest / "positive.toml"
        manifest.write_text(
            f'[inputs.extra]\npath = {json.dumps(str(sentinel))}\nsha256 = "{sha(sentinel)}"\n'
        )
        run(
            "positive-permitted-extra",
            ["--manifest", manifest],
            [sentinel],
            positive=True,
        )
    finally:
        f.doCleanups()
    (args.dest / "RESULT.json").write_text(
        json.dumps(
            dict(
                binary=str(args.binary), trainer_sha256=sha(args.binary), cases=reports
            ),
            indent=2,
        )
        + "\n"
    )
    failures = [r for r in reports if r["status"] != "PASS"]
    if failures:
        raise AssertionError(failures)
    print(f"PASS: {len(reports)} inventory cases; all negative payload opens zero")


if __name__ == "__main__":
    main()
