#!/usr/bin/env python3
"""Assemble the committed E5A JSON record: results + lane provenance.

Usage:
  e5a_record.py --results /var/tmp/e5a-render/results.json \
    --out benchmarks/rev4_e5a_render_2026-09-23.json
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lane", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    lane = Path(args.lane)
    results = json.loads(Path(args.results).read_text())

    head = subprocess.run(
        ["jj", "log", "-r", "@-", "--no-graph", "--no-pager", "-T", "commit_id"],
        cwd=lane,
        capture_output=True,
        text=True,
    ).stdout.strip()

    inputs = {}
    for name, p in [
        ("e5a_scores", "/var/tmp/e5a-render/e5a_scores.tsv"),
        ("peer", "/var/tmp/e5a-render/peer.tsv"),
        ("dssim", "/var/tmp/e5a-render/dssim.tsv"),
        ("tuner", "/var/tmp/e5a-render/tuner.parquet"),
        ("pairs", "/var/tmp/e5a-render/pairs.tsv"),
    ]:
        fp = Path(p)
        if fp.exists():
            inputs[name] = {"path": p, "sha256": sha_file(fp)}

    record = {
        "schema": "rev4-e5a-render-record-v1",
        "lane": "quarantine/devin/e5a-render",
        "lane_commit": head,
        "prereg": "benchmarks/e5a-render_prereg_2026-09-23.md",
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "inputs": inputs,
        "results": results,
    }
    out = Path(args.out)
    out.write_text(json.dumps(record, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
