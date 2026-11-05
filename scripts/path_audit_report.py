#!/usr/bin/env python3
"""Combine retained path audits without dropping slow or inconclusive samples.

Each input must use the same frozen binaries and candidate/control ordering.
Writes a compact TSV plus complete combined summaries into a fresh directory.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

from path_audit import summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-slowdown", type=float, default=0.02)
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    hashes = {}
    rows = []
    fingerprints = {}
    arms = None
    for run_id, path in enumerate(args.runs):
        progress = json.loads((path / "progress.json").read_text())
        if "finished" not in progress or progress["completed"] != progress["required"]:
            raise RuntimeError(f"run incomplete: {path}")
        manifest = json.loads((path / "manifest.json").read_text())
        current = {k: v["sha256"] for k, v in manifest["binaries"].items()}
        if arms is None:
            arms = list(current)
            hashes = current
        if list(current) != arms or current != hashes:
            raise RuntimeError("binary identity/order changed")
        for line in (path / "rows.jsonl").read_text().splitlines():
            row = json.loads(line)
            expected = fingerprints.setdefault(row["cell"], row["fingerprint"])
            if expected != row["fingerprint"]:
                raise RuntimeError("output fingerprint changed")
            row["round"] = f"{run_id}:{row['round']}"
            rows.append(row)
    report = summary(rows, arms, args.output, args.max_slowdown)
    memory = json.loads((args.output / "memory.json").read_text())
    with (args.output / "comparisons.tsv").open("w") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["cell", "control", "pairs", "time_ratio", "ci95_low", "ci95_high",
                         "within_margin", "candidate_rss_kib", "control_rss_kib",
                         "candidate_rss_range", "control_rss_range"])
        for item in report:
            cell, control = item["cell"], item["control"]
            a = memory.get(cell, {}).get(arms[0], [])
            b = memory.get(cell, {}).get(control, [])
            writer.writerow([cell, control, item["process_pairs"],
                *(f"{item[k]:.6f}" for k in ("median_ratio", "ci95_low", "ci95_high")),
                item["within_margin"], statistics.median(a) if a else "",
                statistics.median(b) if b else "",
                f"{min(a)}-{max(a)}" if a else "", f"{min(b)}-{max(b)}" if b else ""])
    with (args.output / "memory.tsv").open("w") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["cell", "arm", "processes", "median_rss_kib", "min_rss_kib", "max_rss_kib"])
        for cell, values in memory.items():
            for arm, rss in values.items():
                writer.writerow([cell, arm, len(rss), statistics.median(rss), min(rss), max(rss)])
    (args.output / "binaries.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"{len(fingerprints)} exact cells; {len(report)} comparisons; "
          f"{sum(not r['within_margin'] for r in report)} outside timing bound")


if __name__ == "__main__":
    main()
