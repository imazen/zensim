#!/usr/bin/env python3
"""Generate a frozen zenbench pair confirmation plan for unresolved audit cells."""
import argparse
import hashlib
import json
from pathlib import Path

from path_audit import METRIC_PATTERN, MODES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--control", required=True, action="append", help="NAME=PATH; repeat for each control")
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--general-bake", required=True, type=Path)
    ap.add_argument("--codec-bake", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--inner-rounds", type=int, default=32)
    ap.add_argument("--timer-error-ns", required=True, type=float)
    ap.add_argument("--scope", required=True)
    args = ap.parse_args()
    rows = json.loads(args.summary.read_text())
    controls = {name: Path(path) for name, path in (item.split("=", 1) for item in args.control)}
    cells = sorted({(r["control"], r["cell"]) for r in rows if r["control"] in controls and not r["within_margin"]})
    if not cells:
        ap.error("no unresolved cells for this control")
    here = Path(__file__).resolve().parent
    inputs = {"launcher_source": str(here / "path_audit_launch.py"),
              "candidate_binary": str(args.candidate.resolve()),
              "general_bake": str(args.general_bake.resolve()),
              "codec_bake": str(args.codec_bake.resolve()),
              "checker_source": str(here / "path_audit_check.py"),
              "packet_source": str(here / "path_audit.py")}
    cases = []
    for control, cell in cells:
        profile, size, mode = cell.split("-", 2)
        width, height = size.split("x")
        route, threads, tier = MODES[mode]
        cases.append({"id": control + "-" + cell, "inputs": {**inputs, "control_binary": str(controls[control].resolve())}, "fields": {
            "profile": profile, "route": route, "width": width, "height": height,
            "threads": str(threads), "tier": tier, "cpus": "2" if threads == 1 else "0-7"}})
    def arm(name, key, revision):
        return {"id": name, "program": "/usr/bin/python3", "args": [str(here / "path_audit_launch.py"), "/usr/bin/taskset", "-c", "${cpus}",
                "${" + key + "}", "bench", "${profile}", "${route}", "${width}", "${height}",
                "${threads}", "${tier}", str(args.inner_rounds)], "timeout_seconds": 120,
                "revision": revision}
    plan = {"schema": 1, "baseline": arm("control", "control_binary", json.dumps({name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in controls.items()}, sort_keys=True)),
            "candidates": [arm("candidate", "candidate_binary", "sha256:" + hashlib.sha256(args.candidate.read_bytes()).hexdigest())], "cases": cases,
            "metric": {"extract": {"kind": "regex", "pattern": METRIC_PATTERN, "capture": 1},
                       "unit": "ns", "expected": 1, "resolution": args.timer_error_ns},
            "check": {"kind": "command", "program": "/usr/bin/python3", "args": [
                str(here / "path_audit_check.py"), "${baseline}/stdout.txt", "${candidate}/stdout.txt"],
                "timeout_seconds": 30},
            "warmup": "Example excludes fixture/reference creation and its 100ms warmup; metric is the median of fixed zenbench rounds. Correctness hashing is outside timing.",
            "blocks": args.blocks, "aa": True, "deadline_seconds": 10800, "seed": 73829,
            "env": {"PATH_AUDIT_GENERAL_BAKE": "${general_bake}",
                    "PATH_AUDIT_CODEC_BAKE": "${codec_bake}", "ZENBENCH_MODE": "iterate",
                    "PYTHONDONTWRITEBYTECODE": "1"},
            "record_env": ["RAYON_NUM_THREADS", "OMP_NUM_THREADS", "ZENBENCH_CPU_POLICY"],
            "confirmation": {"margin_pct": 2.0, "control_margin_pct": 2.0,
                             "family_confidence": 0.95, "scope": args.scope, "accept": "non_inferiority"}}
    with args.output.open("x") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")
    print(f"{len(cases)} cases, {len(cases) * args.blocks * 3} fresh invocations")


if __name__ == "__main__":
    main()
