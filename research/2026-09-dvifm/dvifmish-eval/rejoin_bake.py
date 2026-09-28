#!/usr/bin/env python3
"""Join `ensemble_score_rows` output back to a pair list, keyed.

The Rev3 extractor (pairs-tsv corpus) writes its feature rows in the pair
list's order STABLY SORTED BY REFERENCE BASENAME, and `ensemble_score_rows`
numbers its output rows (`idx`) in that feature-row order. So bake row j
belongs to pair perm[j], perm = the stable argsort of the pair list by
basename(ref_path). The positional join in the phase-2d `rows_to_scores.py
--bake-tsv` assumed perm = identity and mis-paired every list that was not
already reference-sorted (reported by the paper-holdout lane, 2026-09-23;
here: cid22_49, nncd, corruption_validate, corruption_train).

The extractor's audit channel is in pair-list order and is joined by path,
so the fast-ssim2 scores were never affected.

For every row this checks that the feature table's ref_basename, human_score
and any extra-target column the pair list also has (source, inert) agree
with the pair it is mapped to, and that the bake's `human` agrees too; it
refuses to write anything otherwise. Output: the rows_to_scores.py layout
(ref_path, dist_path, target = 100 * human_score, score, E = -score) in
pair-list order, with the same float formatting.

Usage: rejoin_bake.py PAIRS.tsv FEATURES.csv BAKE.tsv OUT.csv [--skip COL,...]

--skip names extra-target columns not to compare: aic4_ptc's feature table
carries the codec label from before the AIC-4 codec fix (1f0aad23), when the
pair builder took it from the image number.
"""
import csv
import os
import sys


def same(a, b):
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    return abs(x - y) <= 1e-6 * max(1.0, abs(y))


def main():
    pairs_p, feat_p, bake_p, out_p = sys.argv[1:5]
    skip = set(sys.argv[6].split(",")) if len(sys.argv) > 6 and sys.argv[5] == "--skip" else set()
    pairs = list(csv.DictReader(open(pairs_p), delimiter="\t"))
    n = len(pairs)
    perm = sorted(range(n), key=lambda i: os.path.basename(pairs[i]["ref_path"]))  # stable
    with open(feat_p) as f:
        r = csv.reader(f)
        hdr = next(r)
        keys = [(k, h) for k, h in enumerate(hdr) if not (h[:1] == "f" and h[1:].isdigit())]
        feat = [[row[k] for k, _ in keys] for row in r]
    names = [h for _, h in keys]
    shared = [h for h in names if h in pairs[0] and h not in ("ref_basename", "human_score") and h not in skip]
    bake = list(csv.DictReader(open(bake_p), delimiter="\t"))
    assert len(feat) == n and len(bake) == n, (n, len(feat), len(bake))
    out = [None] * n
    for j in range(n):
        p = pairs[perm[j]]
        fr = dict(zip(names, feat[j]))
        assert int(bake[j]["idx"]) == j, (j, bake[j]["idx"])
        assert fr["ref_basename"] == os.path.basename(p["ref_path"]), (j, fr["ref_basename"], p["ref_path"])
        assert same(fr["human_score"], p["human_score"]), (j, fr["human_score"], p["human_score"])
        assert same(bake[j]["human"], p["human_score"]), (j, bake[j]["human"], p["human_score"])
        for h in shared:
            assert same(fr[h], p[h]), (j, h, fr[h], p[h])
        s = float(bake[j]["score"])
        out[perm[j]] = [p["ref_path"], p["dist_path"], float(p["human_score"]) * 100.0, s, -s]
    with open(out_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ref_path", "dist_path", "target", "score", "E"])
        w.writerows(out)
    moved = sum(1 for j, i in enumerate(perm) if i != j)
    extra = f", {'/'.join(shared)}" if shared else ""
    print(f"{out_p}: {n} rows, {moved} moved; ref_basename, human{extra} agree on every row")


if __name__ == "__main__":
    main()
