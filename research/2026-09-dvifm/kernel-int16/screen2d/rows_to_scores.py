#!/usr/bin/env python3
"""Join a per-row comparator score source to the (unsealed) pairs TSV and
emit the standard `ref_path,dist_path,target,score,E` score CSV that
metrics.py consumes.

Sources:
  --audit-jsonl PATH   extractor audit channel (canonical-feature-audit
                       schema): keys reference/distorted plus the ssim2
                       score at peer_ssim2.score.
  --bake-tsv PATH      ensemble_score_rows output (idx,human,score);
                       idx is the parquet row index = pairs row index.

usage:
  rows_to_scores.py --pairs unsealed.tsv --audit-jsonl a.jsonl \
      --out scores/x.csv
  rows_to_scores.py --pairs unsealed.tsv --bake-tsv b.tsv --out y.csv
"""
import argparse
import csv
import json
import sys


def load_pairs(p):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def emit(rows, scores, out):
    assert len(rows) == len(scores), (len(rows), len(scores))
    assert all(s is not None for s in scores)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ref_path", "dist_path", "target", "score", "E"])
        for r, s in zip(rows, scores):
            w.writerow([r["ref_path"], r["dist_path"],
                        float(r["human_score"]) * 100.0, s, -s])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--audit-jsonl")
    ap.add_argument("--bake-tsv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load_pairs(a.pairs)
    index = {(r["ref_path"], r["dist_path"]): i
             for i, r in enumerate(rows)}
    scores = [None] * len(rows)
    n = 0

    if a.audit_jsonl:
        with open(a.audit_jsonl) as f:
            for line in f:
                e = json.loads(line)
                if "peer_ssim2" not in e:
                    continue
                key = (e["reference"], e["distorted"])
                i = index[key]
                scores[i] = float(e["peer_ssim2"]["score"])
                n += 1
    elif a.bake_tsv:
        with open(a.bake_tsv) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                scores[int(r["idx"])] = float(r["score"])
                n += 1
    else:
        ap.error("need --audit-jsonl or --bake-tsv")
    assert n == len(rows), (n, len(rows))
    emit(rows, scores, a.out)
    print(f"{a.out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
