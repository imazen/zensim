"""Trace the actual extraction binary: invalid admission opens zero EXR/label files.

This local ingestion probe is not a fleet/image smoke. strace is a required
caller dependency; no test is silently skipped if it is unavailable.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--admission", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(Path(args.admission).read_text())
    cases = []
    for key, value in [("role", "val"), ("role", "development"), ("tier", "T0"),
            ("formula_revision", 4), ("authority", "other")]:
        a = copy.deepcopy(baseline)
        a[key] = value
        cases.append((f"{key}-{value}", a, False))
    for key, value in [("dataset", "live"), ("distorted_rel", "../tid2013/image.png"),
            ("condition_id", "l-i20-l-03-4")]:
        a = copy.deepcopy(baseline)
        a["rows"][-1][key] = value
        cases.append((f"last-member-{key}", a, False))
    a = copy.deepcopy(baseline)
    a["rows"].append(a["rows"][0])
    cases.append(("extra-member", a, False))
    a = copy.deepcopy(baseline)
    a["rows"][-1] = a["rows"][0]
    cases.append(("duplicate-member", a, False))
    cases.append(("wrong-allowlist-pin", baseline, True))
    results = []
    for name, a, wrong_pin in cases:
        path = dest / f"{name}.json"
        path.write_text(json.dumps(a))
        pin = "0" * 64 if wrong_pin else hashlib.sha256(path.read_bytes()).hexdigest()
        trace = dest / f"{name}.openat.log"
        run = subprocess.run(["strace", "-f", "-e", "trace=openat", "-o", str(trace), args.binary,
            "--training-allowlist", str(path), "--allowlist-sha256", pin, "--out", str(dest / f"{name}.tsv")],
            env={**os.environ, "ZENSIM_FORMULA_REV": "5"}, capture_output=True, text=True)
        (dest / f"{name}.stderr.log").write_text(run.stderr)
        opens = [line for line in trace.read_text().splitlines()
                 if '"/mnt/v/datasets/' in line or '"/mnt/v/output/zenmetrics/upiq-pu/' in line]
        assert run.returncode != 0 and "UPIQ-380 allowlisted extraction" in run.stderr, name
        assert not opens, (name, opens)
        assert not (dest / f"{name}.tsv").exists(), name
        results.append(dict(case=name, returncode=run.returncode, dataset_payload_opens=len(opens)))
    (dest / "BINARY_REFUSALS_PASS.json").write_text(json.dumps(dict(status="PASS", cases=results,
        actual_binary=True, dataset_payload_opens=0, scope="11 invalid metadata/pin cases; openat trace of actual local binary"), indent=2) + "\n")
    print(json.dumps(dict(status="PASS", cases=len(results), dataset_payload_opens=0)))


if __name__ == "__main__":
    main()
