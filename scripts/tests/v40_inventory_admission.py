"""Actual-trainer inventory regression; all payloads are synthetic.

Every negative must refuse before *any* table open, including admitted groups.
Manifest provenance policy positives repeat an admitted table/key/declaration.
The exact same oracle fails on the retained V40R2 trainer.
"""
import argparse
import hashlib
import json
import os
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
    sys.argv = [str(owner), "--binary", str(args.binary), "--dest", str(args.dest / "fixtures")]
    fixture_owner = runpy.run_path(str(owner), run_name="inventory_fixture_owner")
    f, _, _, groups = fixture_owner["d1_fixture"](dense=True)
    reports = []

    def run(name, flags, extra=(), positive=False):
        dest = args.dest / name
        dest.mkdir()
        argv = fixture_owner["d1_argv"](groups, fixture_owner["upiq"].columns("by_v2fy"), 1853, flags)
        argv = [str(dest / "model.bin") if str(v) == "@OUT" else str(v) for v in argv]
        trace = dest / "open.trace"
        proc = subprocess.run(["strace", "-qq", "-f", "-e", "trace=open,openat,openat2", "-o", str(trace), *argv], capture_output=True, text=True)
        (dest / "argv.json").write_text(json.dumps(argv, indent=2) + "\n")
        (dest / "stdout.log").write_text(proc.stdout)
        (dest / "stderr.log").write_text(proc.stderr)
        payloads = [*(g[1] for g in groups), *extra]
        names = {str(p) for p in payloads} | {str(p.resolve()) for p in payloads}
        opens = [line for line in trace.read_text().splitlines()
                 if (any('"' + p + '"' in line for p in names)
                     or ('.parquet"' in line and '.keys.parquet"' not in line))
                 and "O_PATH" not in line and "O_DIRECTORY" not in line and "= -1" not in line]
        written = (dest / "model.bin").exists()
        passed = (proc.returncode == 0 and written and bool(opens)) if positive else (proc.returncode == 2 and not written and not opens)
        result = dict(case=name, positive=positive, status="PASS" if passed else "FAIL", exit_code=proc.returncode, payload_opens=len(opens), model_written=written)
        reports.append(result)
        (dest / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        for role, declaration in {
            "VAL": dict(role="val", split="development"),
            "T0": dict(tier="T0"),
            "AIC": dict(human_sources=["aic3"], data_role="holdout-only"),
            "forbidden-key": dict(role="train", keys_role="val"),
            "unregistered-TRAIN": dict(role="train"),
        }.items():
            dest = args.dest / ("inputs-" + role)
            dest.mkdir()
            sentinel = dest / "unregistered.parquet"
            sentinel.write_bytes(b"SYNTHETIC UNADMITTED PAYLOAD")
            Path(str(sentinel) + ".manifest.json").write_text(json.dumps(declaration))
            for late in (False, True):
                entries = [("unregistered", sentinel)]
                if late:
                    entries.insert(0, ("admitted_first", groups[0][1]))
                manifest = dest / f"inventory-{late}.toml"
                manifest.write_text("".join(f'[inputs.{key}]\npath = {json.dumps(str(path))}\nsha256 = "{sha(path)}"\n' for key, path in entries))
                run(f"manifest-{role}-late-{late}", ["--manifest", manifest], [sentinel])
        arbitrary = args.dest / "arbitrary-provenance.json"
        arbitrary.write_text('{"role":"T0"}')
        manifest = args.dest / "arbitrary.toml"
        manifest.write_text(f'[inputs.provenance]\npath = {json.dumps(str(arbitrary))}\nsha256 = "{sha(arbitrary)}"\n')
        run("manifest-unbound-nontable", ["--manifest", manifest], [arbitrary])
        for flag in ("anchor-parquet", "cross-codec-eq-parquet", "pjnd-passthrough-parquet", "konjnd-aggregation-parquet"):
            for ancestry in ("_sealed", "holdout", "ordinary"):
                for symlink in (False, True):
                    dest = args.dest / f"sentinel-{flag}-{ancestry}-{symlink}"
                    actual = dest / ancestry / "sentinel.parquet"
                    actual.parent.mkdir(parents=True)
                    actual.write_bytes(b"SYNTHETIC AUXILIARY PAYLOAD")
                    Path(str(actual) + ".manifest.json").write_text('{"role":"val","tier":"T0"}')
                    path = actual
                    if symlink:
                        path = dest / "alias.parquet"
                        path.symlink_to(actual)
                    run(f"aux-{flag}-{ancestry}-{symlink}", ["--" + flag, path], [path, actual])
        # Native manifest application must also refuse a late auxiliary path.
        dest = args.dest / "manifest-aux"
        dest.mkdir()
        auxiliary = dest / "ordinary.parquet"
        auxiliary.write_bytes(b"SYNTHETIC AUXILIARY PAYLOAD")
        manifest = dest / "aux.toml"
        manifest.write_text(f'[training]\nanchor_parquet = {json.dumps(str(auxiliary))}\n')
        run("manifest-anchor", ["--manifest", manifest], [auxiliary])
        for label, path in (("table", groups[0][1]), ("keys", groups[0][1].with_suffix(".keys.parquet")), ("declaration", Path(str(groups[0][1]) + ".manifest.json"))):
            manifest = args.dest / f"positive-{label}.toml"
            manifest.write_text(f'[inputs.admitted]\npath = {json.dumps(str(path))}\nsha256 = "{sha(path)}"\n')
            run("positive-" + label, ["--manifest", manifest], positive=True)
    finally:
        f.doCleanups()
    (args.dest / "RESULT.json").write_text(json.dumps(dict(binary=str(args.binary), trainer_sha256=sha(args.binary), cases=reports), indent=2) + "\n")
    failures = [r for r in reports if r["status"] != "PASS"]
    if failures:
        raise AssertionError(failures)
    print(f"PASS: {len(reports)} inventory cases; all negative payload opens zero")


if __name__ == "__main__":
    main()
