#!/usr/bin/env python3
"""Characterise metrics on zensim's canonical corruption packet (2026-09-08).

Two readings, both from existing zensim protocols:

* gate (`scripts/v_next/corruption_gate_eval.py`'s rule): a corruption PASSES
  at q20 when the metric rates it worse than the same origin's native q20 JPEG
  anchor; pass@q10 likewise against the q10 anchor. Non-inert corruptions
  only (inert attempts changed no pixel and are labelled noncorrupt).
* detection at a matched false-positive rate (the corruption-head records'
  reading): a threshold on the metric's damage is set so that a fraction
  alpha of the HONEST pairs of the same split (the q10/q20 anchors and the
  native JPEG XL / AVIF supplement) exceed it; detection is the fraction of
  non-inert corruptions above it.

Damage is oriented "higher = worse": dvifmish E; butteraugli max-norm;
minus the quality score for fast-ssim2 and zensim peers.

Rows are deduplicated as the owner's serving report does
(`scripts/v_next/corruption_gate_eval.py`, summarize): one row per (origin,
reference pixels, distorted pixels), keyed by the decoded-pixel SHA-256s the
extractor audit recorded for every pair. Positives are the non-inert
corruptions; negatives everything else (honest anchors, native JXL/AVIF, and
the inert attempts, which collapse to one identity pair per origin). Identical
pixel pairs must carry identical labels and, for every model, identical
damage; anything else stops the script. The matched-FP threshold is set on
the unique negatives. Per-family rates deduplicate within each family.

Usage: corruption_eval.py <split> <out.json> name=path[:kind] ...
  kind: dvifmish (batch TSV, column `distortion`), quality (peer CSV/TSV with a
  `score` column, higher = better), butteraugli (zenmetrics TSV, `butteraugli_max`).
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAIRS = Path("/mnt/v/output/zensim/dvifmish-eval-2026-09-22/pairs")
AUDIT = Path("/var/tmp/dvifmish/peers")  # the Rev3 extractor's audit: decoded-pixel SHA-256 per pair
ALPHAS = (0.01, 0.05)


def read_rows(path):
    delim = "," if str(path).endswith(".csv") else "\t"
    return list(csv.DictReader(open(path), delimiter=delim))


def damage(path, kind, keys):
    rows = read_rows(path)
    assert len(rows) == len(keys), (path, len(rows), len(keys))
    # every input here carries its pair's paths: check them, not just the count
    assert [(r["ref_path"], r["dist_path"]) for r in rows] == keys, (path, "row keys")
    if kind == "dvifmish":
        return np.array([float(r["distortion"]) for r in rows])
    if kind == "butteraugli":
        return np.array([float(r["butteraugli_max"]) for r in rows])
    return -np.array([float(r["score"]) for r in rows])


def pixel_keys(split, pairs):
    """(origin, ref pixels, dist pixels) per pair, from the extractor audit (pair order)."""
    path = AUDIT / f"audit_corruption_{split}.jsonl"
    aud = [json.loads(line) for line in open(path)]
    aud = [a for a in aud if "reference" in a]
    assert len(aud) == len(pairs), (path, len(aud), len(pairs))
    out = []
    for a, r in zip(aud, pairs):
        assert (a["reference"], a["distorted"]) == (r["ref_path"], r["dist_path"]), (path, "row keys")
        out.append((r["source"], a["reference_pixels_sha256"], a["distorted_pixels_sha256"]))
    return out


def unique_rows(idx, pk):
    """First row of each distinct pixel pair among `idx`, in pair order."""
    seen, out = set(), []
    for i in idx:
        if pk[i] not in seen:
            seen.add(pk[i])
            out.append(i)
    return np.asarray(out, dtype=int)


def main():
    split, out = sys.argv[1], Path(sys.argv[2])
    pairs = read_rows(PAIRS / f"corruption_{split}.tsv")
    n = len(pairs)
    keys = [(r["ref_path"], r["dist_path"]) for r in pairs]
    kind = [r["kind"] for r in pairs]
    src = [r["source"] for r in pairs]
    fam = [r["codec"] for r in pairs]
    pk = pixel_keys(split, pairs)
    corr = np.array([k == "corruption" and r["inert"] == "0" for k, r in zip(kind, pairs)])
    groups = defaultdict(list)
    for i, key in enumerate(pk):
        groups[key].append(i)
    for rows in groups.values():
        assert len({bool(corr[i]) for i in rows}) == 1, ("duplicate label conflict", rows[:4])
    pos = unique_rows(np.nonzero(corr)[0], pk)
    neg = unique_rows(np.nonzero(~corr)[0], pk)
    anchor = {q: {s: i for i, (k, s) in enumerate(zip(kind, src)) if k == f"anchor_q{q}"}
              for q in (10, 20)}
    res = {"split": split, "pairs": n, "unique_pixel_pairs": len(groups),
           "removed_duplicates": n - len(groups), "positives": len(pos), "negatives": len(neg),
           "raw_corruptions": int(corr.sum()), "origins": len(set(src)),
           "dedup": "one row per (origin, reference pixels, distorted pixels), as corruption_gate_eval.py",
           "models": {}}
    for spec in sys.argv[3:]:
        name, rest = spec.split("=", 1)
        path, _, k = rest.partition(":")
        d = damage(path, k or "dvifmish", keys)
        for rows in groups.values():
            assert len({float(d[i]) for i in rows}) == 1, (name, "identical pixels, different damage", rows[:4])
        m = {}
        for q in (10, 20):
            ok = d[pos] > np.array([d[anchor[q][src[i]]] for i in pos])
            m[f"pass_q{q}"] = float(np.mean(ok))
            m[f"pass_q{q}_count"] = int(ok.sum())
        for a in ALPHAS:
            tau = float(np.quantile(d[neg], 1.0 - a))
            m[f"detect_at_fp{int(a * 100)}"] = float(np.mean(d[pos] > tau))
        per = defaultdict(list)
        for i in np.nonzero(corr)[0]:
            per[fam[i]].append(i)
        m["pass_q20_by_family"] = {}
        for f, idx in sorted(per.items()):
            u = unique_rows(idx, pk)
            m["pass_q20_by_family"][f] = float(np.mean([d[i] > d[anchor[20][src[i]]] for i in u]))
        res["models"][name] = m
        print(f"{split:8s} {name:28s} pass@q20 {m['pass_q20']:.4f} ({m['pass_q20_count']}/{len(pos)})  "
              f"pass@q10 {m['pass_q10']:.4f}  detect@FP1% {m['detect_at_fp1']:.3f}  "
              f"detect@FP5% {m['detect_at_fp5']:.3f}")
    out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
