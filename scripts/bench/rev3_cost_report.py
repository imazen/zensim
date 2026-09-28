#!/usr/bin/env python3
"""Summarise a `rev3_cost_ab.sh` run: paired revision-1 vs revision-3 arms,
with the revision-independent `fast_ssim2` arm as the cross-block drift anchor.

zenbench writes one machine-readable line per (group, benchmark) into the file
its header names; this parses those lines out of each block log's companion
file, pools blocks per revision, and reports mean/median/min plus the anchor's
own movement. If the anchor moved more than the arms did, the box moved and the
comparison is not attributable — that is reported, not smoothed over.
"""
import json
import pathlib
import re
import statistics
import sys

# zenbench prints each duration in whichever unit reads best, so a
# `ms`-only pattern silently DROPS rows (a sub-millisecond `mad=950.63µs` cost
# this script four arms on the first run, and a dropped arm in one revision but
# not the other is how a paired comparison stops being paired).
_N = r"([\d.]+)(ms|µs|us|s)"
FIELD = re.compile(
    r"group=(\S+)\s+benchmark=(\S+).*?"
    r"min=" + _N + r" mean=" + _N + r" median=" + _N + r" mad=" + _N +
    r"\s+\|\s+n=(\d+) cv=([\d.]+)%"
)
_SCALE = {"ms": 1.0, "µs": 1e-3, "us": 1e-3, "s": 1e3}


def parse_block(path: pathlib.Path):
    """zenbench's own results file, located from the block log's header line."""
    for line in path.read_text(errors="replace").splitlines():
        if "[zenbench] results" in line:
            results = pathlib.Path(line.split("→")[-1].strip())
            if results.exists():
                out = []
                for r in results.read_text(errors="replace").splitlines():
                    m = FIELD.search(r)
                    if m:
                        g = m.groups()
                        out.append(
                            {
                                "group": g[0],
                                "arm": g[1],
                                "min": float(g[2]) * _SCALE[g[3]],
                                "mean": float(g[4]) * _SCALE[g[5]],
                                "median": float(g[6]) * _SCALE[g[7]],
                                "mad": float(g[8]) * _SCALE[g[9]],
                                "n": int(g[10]),
                                "cv": float(g[11]),
                            }
                        )
                return out
    return []


def main(out_dir: str) -> int:
    root = pathlib.Path(out_dir)
    # Arms are revision labels ("1", "3") in one-binary mode or binary labels
    # ("A", "B") in two-binary mode; the pairing logic is the same.
    per_rev: dict[str, list] = {}
    for log in sorted(root.glob("block*_rev*.txt")):
        rev = log.stem.split("_rev")[1]
        per_rev.setdefault(rev, [])
        # Only COMPLETE blocks. A block still running has a partial results
        # file, and pooling it silently drops arms from one revision and not
        # the other — which is how a paired comparison stops being paired.
        if "total:" not in log.read_text(errors="replace"):
            print(f"SKIP {log.name}: still running", file=sys.stderr)
            continue
        rows = parse_block(log)
        if not rows:
            print(f"WARNING: no parsable results in {log.name}", file=sys.stderr)
        per_rev[rev].extend(rows)

    def pool(rev, group, arm, field="median"):
        vals = [r[field] for r in per_rev[rev] if r["group"] == group and r["arm"] == arm]
        return statistics.median(vals) if vals else None

    groups = sorted({r["group"] for rows in per_rev.values() for r in rows})
    arms = sorted({r["arm"] for rows in per_rev.values() for r in rows})
    labels = sorted(per_rev)
    if len(labels) != 2:
        print(f"expected exactly two arm labels, found {labels}", file=sys.stderr)
        return 1
    la, lb = labels
    def key(label):  # "rev1_median_ms" stays as it was; binaries become "binA_median_ms"
        return (f"rev{label}" if label.isdigit() else f"bin{label}") + "_median_ms"
    report = {"arms": [la, lb], "groups": {}, "blocks_per_revision": {
        k: len({r["group"] for r in v}) and len(v) // max(1, len(groups) * max(1, len(arms)))
        for k, v in per_rev.items()}}

    for g in groups:
        anchor1, anchor3 = pool(la, g, "fast_ssim2"), pool(lb, g, "fast_ssim2")
        drift = None
        if anchor1 and anchor3:
            drift = 100.0 * (anchor3 - anchor1) / anchor1
        rows = {}
        for a in arms:
            v1, v3 = pool(la, g, a), pool(lb, g, a)
            if v1 and v3:
                rows[a] = {
                    key(la): round(v1, 3),
                    key(lb): round(v3, 3),
                    "delta_pct": round(100.0 * (v3 - v1) / v1, 2),
                }
        report["groups"][g] = {"anchor_drift_pct": None if drift is None else round(drift, 2),
                               "arms": rows}

    print(json.dumps(report, indent=1))
    for g, gd in report["groups"].items():
        print(f"\n== {g} ==  anchor(fast_ssim2) drift {gd['anchor_drift_pct']}%")
        ka, kb = key(la), key(lb)
        print(f"{'arm':<18}{ka[:-10] + ' ms':>10}{kb[:-10] + ' ms':>10}{'delta':>10}")
        for a, r in sorted(gd["arms"].items(), key=lambda kv: kv[1]["delta_pct"]):
            print(f"{a:<18}{r[ka]:>10.2f}{r[kb]:>10.2f}{r['delta_pct']:>9.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
