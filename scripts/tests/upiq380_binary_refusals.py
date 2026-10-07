"""Trace actual UPIQ binaries: admission/CLI refusals precede payload access.

strace is required. The optional prior binary reproduces the three reviewed
admission defects with synthetic paths; it never uses the real legacy labels.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess


def trace_binary(binary, argv, dest, name):
    trace = dest / f"{name}.syscalls.log"
    run = subprocess.run(
        ["strace", "-f", "-e", "trace=openat,newfstatat,statx", "-o", str(trace), str(binary), *argv],
        env={**os.environ, "ZENSIM_FORMULA_REV": "5"}, capture_output=True, text=True,
    )
    (dest / f"{name}.stderr.log").write_text(run.stderr)
    lines = trace.read_text().splitlines()
    payloads = [line for line in lines if "openat(" in line and (
        '"/mnt/v/datasets/' in line or '"/mnt/v/output/zenmetrics/upiq-pu/' in line)]
    assert not payloads, (name, payloads)
    assert run.returncode != 0, name
    return run, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--admission", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--prior-binary", default="")
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
    image_root = dest / "nonexistent-images"
    # Let the prior binary reach the exact reviewed content-directory boundary.
    (image_root / "narwaria").mkdir(parents=True)
    wrong_ids = copy.deepcopy(baseline)
    wrong_ids["image_root"] = str(image_root)
    wrong_ids["requested_ids"][0] = 0
    assert len(wrong_ids["requested_ids"]) == 420
    assert wrong_ids["requested_ids"] == sorted(set(wrong_ids["requested_ids"]))
    cases.append(("same-width-different-ids", wrong_ids, False))
    results = []
    wrong_id_argv = None
    for name, a, wrong_pin in cases:
        path = dest / f"{name}.json"
        path.write_text(json.dumps(a))
        pin = "0" * 64 if wrong_pin else hashlib.sha256(path.read_bytes()).hexdigest()
        argv = ["--training-allowlist", str(path), "--allowlist-sha256", pin,
                "--out", str(dest / f"{name}.tsv")]
        run, lines = trace_binary(args.binary, argv, dest, name)
        assert "UPIQ-380 allowlisted extraction" in run.stderr, name
        if name == "same-width-different-ids":
            assert "not the registered TRAIN-only" in run.stderr, name
            assert not any(str(image_root) in line for line in lines), name
            wrong_id_argv = argv
        assert not (dest / f"{name}.tsv").exists(), name
        results.append(dict(case=name, returncode=run.returncode, dataset_payload_opens=0))

    tripwire = dest / "synthetic-subjective.csv"
    tripwire.write_text("synthetic_header_tripwire\n")
    out = dest / "must-not-exist.csv"
    common = ["--subjective", str(tripwire), "--images", str(image_root), "--out", str(out)]
    cli_cases = [
        ("missing-training-allowlist-value", ["--training-allowlist"]),
        ("pin-without-training-allowlist", ["--allowlist-sha256", "0" * 64]),
        ("missing-pin-value", ["--training-allowlist", args.admission, "--allowlist-sha256"]),
        ("option-as-allowlist-value", ["--training-allowlist", "--allowlist-sha256", "0" * 64]),
        ("duplicate-allowlist", ["--training-allowlist", args.admission, "--training-allowlist=other", "--allowlist-sha256", "0" * 64]),
        ("duplicate-pin", ["--training-allowlist", args.admission, "--allowlist-sha256", "0" * 64, "--allowlist-sha256=other"]),
        ("empty-allowlist-assignment", ["--training-allowlist="]),
        ("empty-pin-assignment", ["--allowlist-sha256="]),
    ]
    for name, suffix in cli_cases:
        run, lines = trace_binary(args.binary, common + suffix, dest, name)
        assert "UPIQ-380 allowlisted extraction" in run.stderr, name
        assert not any(str(tripwire) in line or str(image_root) in line for line in lines), name
        assert not out.exists(), name
        results.append(dict(case=name, returncode=run.returncode, dataset_payload_opens=0,
                            synthetic_label_opens=0, image_path_accesses=0))

    if args.prior_binary:
        before = []
        for name, suffix in cli_cases[:2]:
            run, lines = trace_binary(args.prior_binary, common + suffix, dest, f"before-{name}")
            opened = any("openat(" in line and str(tripwire) in line for line in lines)
            assert opened and "condition_id" in run.stderr, name
            before.append(dict(case=name, synthetic_label_opened=True, real_dataset_payload_opens=0))
        run, lines = trace_binary(args.prior_binary, wrong_id_argv, dest, "before-same-width-different-ids")
        assert any(str(image_root / "narwaria/01") in line for line in lines)
        assert "not the registered TRAIN-only" not in run.stderr
        before.append(dict(case="same-width-different-ids", reached_image_path_checks=True,
                           real_dataset_payload_opens=0))
        (dest / "PRIOR_DEFECTS_REPRODUCED.json").write_text(json.dumps(before, indent=2) + "\n")

    report = dict(status="PASS", cases=results, actual_binary=True, dataset_payload_opens=0,
                  binary_sha256=hashlib.sha256(Path(args.binary).read_bytes()).hexdigest(),
                  scope="11 original refusals, exact-ID refusal before image metadata, 8 malformed CLI refusals before synthetic labels")
    (dest / "BINARY_REFUSALS_PASS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", cases=len(results), dataset_payload_opens=0)))


if __name__ == "__main__":
    main()
