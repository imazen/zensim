#!/usr/bin/env python3
"""Compare frozen path_audit executables with interleaved fresh-process runs.

Run under the machine's resource wrapper, exclusive heavy-work lock, and an enforced
uniform CPU cpuset. Timing is inside zenbench, never subprocess elapsed time. Keep the
output directory: raw packets, RSS files, executable hashes and progress are evidence.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

MODES = {
    "scalar": ("compute", 1, "scalar"),
    "serial": ("compute", 1, "auto"),
    "parallel": ("compute", 8, "auto"),
    "cached": ("cached", 1, "auto"),
    "scratch": ("scratch", 1, "auto"),
    "map_serial": ("map", 1, "auto"),
    "map_parallel": ("map", 8, "auto"),
    "planar_serial": ("planar", 1, "auto"),
    "planar_parallel": ("planar", 8, "auto"),
}


def packet(text):
    return json.loads(next(line for line in reversed(text.splitlines()) if line.startswith('{"fingerprint"')))


def summary(rows, arms, output, margin):
    groups = {}
    for row in rows:
        if row["action"] == "bench":
            groups.setdefault(row["cell"], {}).setdefault(row["arm"], []).append(row)
    report = []
    rng = random.Random(73829)
    for cell, data in groups.items():
        for control in arms[1:]:
            # First arm is candidate, subsequent arms are explicit controls.
            a = {r["round"]: r for r in data.get(arms[0], [])}
            b = {r["round"]: r for r in data.get(control, [])}
            pairs = [(a[k], b[k]) for k in sorted(a.keys() & b.keys())]
            if not pairs:
                continue
            values = [math.log(x["median_ns"] / y["median_ns"]) for x, y in pairs]
            boots = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(5000))
            lo, hi = (math.exp(boots[i]) for i in (125, 4874))
            report.append({"cell": cell, "candidate": arms[0], "control": control,
                "process_pairs": len(pairs), "median_ratio": math.exp(statistics.median(values)),
                "ci95_low": lo, "ci95_high": hi,
                "within_margin": len(pairs) >= 6 and hi <= 1 + margin})
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    memory = {}
    for row in rows:
        if row["action"] == "memory":
            memory.setdefault(row["cell"], {}).setdefault(row["arm"], []).append(row["max_rss_kib"])
    (output / "memory.json").write_text(json.dumps(memory, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", required=True, help="NAME=executable; candidate first")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--profiles", default="E,ECodec,A,PreviewV0_1,PreviewV0_2")
    ap.add_argument("--sizes", default="64x64,257x193,1024x768,2048x1536")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--inner-rounds", type=int, default=12)
    ap.add_argument("--memory-rounds", type=int, default=3)
    ap.add_argument("--serial-cpu", default="2")
    ap.add_argument("--parallel-cpus", default="0-7")
    ap.add_argument("--max-slowdown", type=float, default=0.02,
                    help="one-sided equivalence bound; report inconclusive cells, never hide them")
    args = ap.parse_args()
    arms = dict(x.split("=", 1) for x in args.arm)
    assert len(arms) == len(args.arm) and len(arms) >= 2
    assert not args.output.exists(), "use a fresh output directory; old evidence is protected"
    args.output.mkdir(parents=True)
    metadata = {"argv": vars(args) | {"output": str(args.output)}, "pid": os.getpid(),
                "started": time.time(), "binaries": {n: {"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for n,p in arms.items()}}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cells = [(p, size, mode) for p in args.profiles.split(',') for size in args.sizes.split(',') for mode in args.modes.split(',')]
    rows = []
    fingerprints = {}
    completed = 0
    required = len(cells) * len(arms) * (args.rounds + args.memory_rounds)
    with (args.output / "rows.jsonl").open('w') as ledger:
        for profile, size, mode in cells:
            width, height = size.split('x')
            route, threads, tier = MODES[mode]
            cell = f"{profile}-{size}-{mode}"
            for round_no in range(args.rounds + args.memory_rounds):
                action = "bench" if round_no < args.rounds else "memory"
                names = list(arms)
                # Rotate order so no arm systematically follows a cold/hot predecessor.
                names = names[round_no % len(names):] + names[:round_no % len(names)]
                if (round_no // len(names)) % 2:
                    names.reverse()
                for name in names:
                    stem = f"{cell}-{round_no}-{name}"
                    cmd = ["taskset", "-c", args.serial_cpu if threads == 1 else args.parallel_cpus,
                           str(Path(arms[name]).resolve()), action, profile, route, width, height,
                           str(threads), tier, str(args.inner_rounds)]
                    rss_path = (args.output / (stem + ".rss")).resolve()
                    if action == "memory":
                        cmd = ["/usr/bin/time", "-v", "-o", str(rss_path), *cmd]
                    status = {"completed": completed, "required": required, "cell": cell, "arm": name,
                              "round": round_no, "action": action, "command": cmd, "updated": time.time()}
                    (args.output / "progress.json").write_text(json.dumps(status, indent=2) + "\n")
                    run = subprocess.run(cmd, cwd=args.output, capture_output=True, text=True)
                    (args.output / (stem + ".stdout")).write_text(run.stdout)
                    (args.output / (stem + ".stderr")).write_text(run.stderr)
                    if run.returncode:
                        raise RuntimeError(f"{stem} exited {run.returncode}; see retained stderr")
                    result = packet(run.stdout)
                    expected = fingerprints.setdefault(cell, result["fingerprint"])
                    if result["fingerprint"] != expected:
                        raise RuntimeError(f"output mismatch {stem}: {result['fingerprint']} != {expected}")
                    row = {"cell": cell, "arm": name, "round": round_no, "action": action,
                           "fingerprint": result["fingerprint"]}
                    if action == "bench":
                        timing = result["timing"]
                        assert not timing["incomplete"] and not timing["unreliable"], stem
                        assert timing["cpu_placement"]["status"] == "enforced_uniform", stem
                        comp = timing["comparisons"][0]
                        row["median_ns"] = comp["benchmarks"][0]["summary"]["median"]
                    else:
                        rss = rss_path.read_text()
                        row["max_rss_kib"] = int(next(l.split(':')[-1] for l in rss.splitlines() if "Maximum resident set size" in l))
                    rows.append(row)
                    ledger.write(json.dumps(row) + "\n")
                    ledger.flush()
                    completed += 1
            summary(rows, list(arms), args.output, args.max_slowdown)
            print(f"{completed}/{required} {cell}", flush=True)
    result = summary(rows, list(arms), args.output, args.max_slowdown)
    (args.output / "progress.json").write_text(json.dumps({"completed": completed, "required": required, "finished": time.time(), "inconclusive_or_slower": sum(not r["within_margin"] for r in result)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
