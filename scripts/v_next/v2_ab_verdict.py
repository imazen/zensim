#!/usr/bin/env python3
"""v2-vs-v1 trainability A/B verdict (docs/V2_TRAINABILITY_AB_2026-07-19.md).

Forwards each arm's best-checkpoint bake over its held-out feature CSVs and
computes the stats via the CANONICAL owners — `predict_features_with_bake`
(forward pass) and `panel` (zenstats Mohammadi panel). This script is packing
glue only: no stat math, no bake parsing (per the no-duplication rule).

Usage: python3 scripts/v_next/v2_ab_verdict.py /mnt/v/output/zensim/v2-ab-2026-07-19
"""
import csv
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PREDICT = REPO / "target/release/predict_features_with_bake"
PANEL = REPO / "target/release/panel"
CORPORA = ["cid22val", "csiq", "live"]  # held-out only
ARMS = ["v1", "v2"]


def load_csv(path: Path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        fcols = [i for i, h in enumerate(header) if h.startswith("f") and h[1:].isdigit()]
        hcol = header.index("human_score")
        targets, rows = [], []
        for line in r:
            targets.append(float(line[hcol]))
            rows.append([float(line[i]) for i in fcols])
    return targets, rows


def forward(bake: Path, rows) -> list[float]:
    n_feat, n_rows = len(rows[0]), len(rows)
    with tempfile.NamedTemporaryFile(suffix=".blob", delete=False) as tf:
        tf.write(struct.pack("<II", n_feat, n_rows))
        for row in rows:
            tf.write(struct.pack(f"<{n_feat}f", *row))
        blob = tf.name
    out = subprocess.run(
        [str(PREDICT), "--bake", str(bake), "--bake-post", "raw", "--features-file", blob],
        capture_output=True, text=True, check=True,
    )
    Path(blob).unlink()
    scores = [float(x) for x in out.stdout.split()]
    assert len(scores) == n_rows, f"{len(scores)} scores != {n_rows} rows"
    return scores


def run_panel(preds, targets) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as tf:
        tf.write("predicted\ttarget\n")
        for p, t in zip(preds, targets):
            tf.write(f"{p}\t{t}\n")
        tsv = tf.name
    out = subprocess.run(
        [str(PANEL), "--input", tsv, "--json"], capture_output=True, text=True, check=True
    )
    Path(tsv).unlink()
    return json.loads(out.stdout)["groups"][0]


def main():
    ab = Path(sys.argv[1])
    results = {}
    for arm in ARMS:
        bake = ab / f"{arm}_arm.bin"
        results[arm] = {}
        for corpus in CORPORA:
            targets, rows = load_csv(ab / f"{arm}_{corpus}.csv")
            preds = forward(bake, rows)
            results[arm][corpus] = run_panel(preds, targets)
    # Verdict per the pre-registered bands (docs/V2_TRAINABILITY_AB_2026-07-19.md)
    def s(arm, c):
        return abs(results[arm][c]["srocc"])  # quality-oriented; |.| guards sign convention

    deltas = {c: s("v2", c) - s("v1", c) for c in CORPORA}
    mean_delta = sum(deltas.values()) / len(deltas)
    win = mean_delta >= -0.010 and all(d >= -0.020 for d in deltas.values())
    kill = any(d <= -0.030 for d in deltas.values())
    verdict = "KILL" if kill else ("WIN" if win else "BETWEEN")

    report = {
        "per_corpus": {
            c: {"v1_srocc": s("v1", c), "v2_srocc": s("v2", c), "delta": deltas[c]}
            for c in CORPORA
        },
        "mean_delta": mean_delta,
        "verdict": verdict,
        "full_panels": results,
    }
    out_path = ab / "verdict.json"
    out_path.write_text(json.dumps(report, indent=2))
    for c in CORPORA:
        r = report["per_corpus"][c]
        print(f"{c:10s} v1={r['v1_srocc']:.4f} v2={r['v2_srocc']:.4f} delta={r['delta']:+.4f}")
    print(f"mean_delta={mean_delta:+.4f}  VERDICT={verdict}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
