#!/usr/bin/env python3
"""wlin7_bars.py — the round-7 REPORTING VIEW over `bake_verdict --full-json` outputs.

This computes NO statistic. It reads `rank.<corpus>.srocc_signed` (never
`bands[].srocc`, never `srocc` — the unsigned field) out of fulleval JSONs the
owner already wrote, compares each to the frozen round-6 bar, and prints the
table plus the registered maximin-margin tie-break. KonJND is additionally shown
as |SROCC| per its convention (its target is a PJND parameter), and the bar is
applied to the magnitude.

Bars and selection rule: benchmarks/wlin_round7_rawframe_2026-08-30.md §4.
"""
import argparse
import json
import sys
from pathlib import Path

# Frozen round-6 bars, verbatim.
BARS = {"konjnd": 0.40, "hfnlproxy": 0.40, "cid22": 0.845,
        "nonphoto": 0.865, "imazen26": 0.875}
ORDER = ["cid22", "konjnd", "nonphoto", "imazen26", "hfnlproxy"]
LABEL = {"cid22": "cid22", "konjnd": "|kon|", "nonphoto": "nonphoto",
         "imazen26": "imazen26", "hfnlproxy": "hfnl"}
# B's reference row is READ from a fulleval JSON (--b-fulleval), never hardcoded:
# the same-pair-restricted cut and the full cut differ by up to 0.07 on the family
# axes (the v1-width row restriction was size-correlated), so a constant here would
# silently mix two rulers. The fallback below is R1b §8.4's RESTRICTED read and is
# labelled as such wherever it is used.
B_ROW_RESTRICTED = {"cid22": 0.8763, "konjnd": 0.5183, "nonphoto": 0.9093,
                    "imazen26": 0.9142, "hfnlproxy": 0.3553}


def read(p: Path):
    j = json.loads(p.read_text())
    r = j.get("rank", {})
    out = {}
    for c in ORDER:
        v = r.get(c)
        if v is None:
            out[c] = None
            continue
        s = v.get("srocc_signed")
        out[c] = abs(s) if c == "konjnd" else s
    return out, j


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fullevals", nargs="+")
    ap.add_argument("--extra", nargs="*", default=["kadid", "tid"],
                    help="extra rank corpora to print (context, not bars)")
    ap.add_argument("--tsv", help="also write the table as TSV")
    ap.add_argument("--b-fulleval", help="fulleval JSON for the reference model (B). "
                                         "Its rank block is READ, never recomputed.")
    ap.add_argument("--b-label", default="B (shipped 372)")
    a = ap.parse_args()

    if a.b_fulleval:
        b_row, _bj = read(Path(a.b_fulleval))
        b_label = a.b_label
        if any(v is None for v in b_row.values()):
            print("B fulleval is missing an axis; aborting rather than mixing rulers",
                  file=sys.stderr)
            return 2
    else:
        b_row, b_label = B_ROW_RESTRICTED, "B (R1b §8.4 RESTRICTED cut)"

    rows = []
    for f in a.fullevals:
        p = Path(f)
        vals, j = read(p)
        margins = [(vals[c] - BARS[c]) / BARS[c] for c in ORDER if vals[c] is not None]
        n_pass = sum(1 for c in ORDER if vals[c] is not None and vals[c] >= BARS[c])
        n_beats_b = sum(1 for c in ORDER if vals[c] is not None and vals[c] >= b_row[c])
        rows.append({"name": j.get("name") or p.stem, "vals": vals,
                     "pass": n_pass, "maximin": min(margins) if margins else float("nan"),
                     "beats_b": n_beats_b, "size": j.get("model", {}).get("bytes"),
                     "extra": {c: (j.get("rank", {}).get(c) or {}).get("srocc_signed")
                               for c in a.extra},
                     "path": str(p)})

    hdr = ["arm"] + [LABEL[c] for c in ORDER] + ["bars/5", "maximin", ">=B/5"] + a.extra
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    for r in sorted(rows, key=lambda r: (-r["pass"], -r["maximin"])):
        cells = [r["name"]]
        for c in ORDER:
            v = r["vals"][c]
            cells.append("—" if v is None else
                         ("**%.4f**" % v if v >= BARS[c] else "%.4f" % v))
        cells += ["%d" % r["pass"], "%+.3f" % r["maximin"], "%d" % r["beats_b"]]
        cells += ["—" if r["extra"][c] is None else "%.4f" % r["extra"][c] for c in a.extra]
        print("| " + " | ".join(cells) + " |")
    print("| *%s* | " % b_label +
          " | ".join("*%.4f*" % b_row[c] for c in ORDER) +
          " | *%d* | *%+.3f* | *—* | %s |" % (
              sum(1 for c in ORDER if b_row[c] >= BARS[c]),
              min((b_row[c] - BARS[c]) / BARS[c] for c in ORDER),
              " | ".join("—" for _ in a.extra)))
    print("\nbars: " + "  ".join("%s>=%.3f" % (LABEL[c], BARS[c]) for c in ORDER))
    if a.tsv:
        with open(a.tsv, "w") as fh:
            fh.write("arm\t" + "\t".join(ORDER) + "\tbars\tmaximin\tbeats_b\tpath\n")
            for r in rows:
                fh.write(r["name"] + "\t" + "\t".join(
                    "" if r["vals"][c] is None else "%.6f" % r["vals"][c] for c in ORDER)
                    + "\t%d\t%.6f\t%d\t%s\n" % (r["pass"], r["maximin"], r["beats_b"], r["path"]))
        print("wrote " + a.tsv, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
