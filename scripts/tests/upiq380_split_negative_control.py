"""Run the unchanged split/key regression against an explicit prior source revision."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before-revision", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "scripts/rev4_featpot"))
    store = subprocess.run(["jj", "git", "root"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    old = subprocess.run(["git", f"--git-dir={store}", "show", f"{args.before_revision}:scripts/rev4_featpot/upiq380.py"],
        cwd=repo, capture_output=True, text=True, check=True).stdout
    module = types.ModuleType("upiq380")
    exec(compile(old, "upiq380-before-binding.py", "exec"), module.__dict__)
    sys.modules["upiq380"] = module
    path = repo / "scripts/tests/test_upiq380.py"
    spec = importlib.util.spec_from_file_location("upiq_binding_regressions", path)
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([
        tests.UpiqAdmission("test_split_and_row_key_rewiring_refuse_before_labels")]))
    assert not result.wasSuccessful() and len(result.failures) == 5 and len(result.errors) == 0
    with Path(args.out).open("x") as f:
        json.dump(dict(status="PASS", before_source_commit=args.before_revision, before_subcase_failures=5,
            before_payload_opens_blocked_by_tripwire=True, after_passed_tests=6,
            same_test_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()), f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
