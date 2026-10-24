#!/usr/bin/env python3
"""Compare two perf_bits.sh output dirs: every dump must be byte-identical for a bit-identical change; for a
value-changing candidate, report per dump how many values moved and by how much (relative to max(|a|, 1e-30)).

  perf_bits_compare.py BASE_DIR CAND_DIR [--allow-diff]

Exit status 1 when any dump differs (or is missing) unless --allow-diff. Record: benchmarks/byconstruction_2026-10-22.md.
"""
import argparse
import struct
import sys
from pathlib import Path


def values(path):
    """The dump's values (score first, then each emitted feature), from their hex bit patterns."""
    return [struct.unpack("<d", int(line, 16).to_bytes(8, "little"))[0] for line in path.read_text().split()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--allow-diff", action="store_true")
    a = ap.parse_args()
    base, cand = Path(a.base), Path(a.cand)
    names = sorted(p.name for p in base.glob("*.hex"))
    if not names:
        sys.exit(f"no dumps in {base}")
    bad, worst = 0, 0.0
    for n in names:
        c = cand / n
        if not c.exists():
            print(f"MISSING {n}")
            bad += 1
            continue
        if c.read_bytes() == (base / n).read_bytes():
            continue
        va, vb = values(base / n), values(c)
        if len(va) != len(vb):
            print(f"LENGTH {n}: {len(va)} vs {len(vb)}")
            bad += 1
            continue
        moved = [(i, x, y) for i, (x, y) in enumerate(zip(va, vb)) if struct.pack("<d", x) != struct.pack("<d", y)]
        rel = max(abs(y - x) / max(abs(x), 1e-30) for _, x, y in moved)
        worst = max(worst, rel)
        score = " (score moved)" if moved[0][0] == 0 else ""
        print(f"DIFF {n}: {len(moved)} of {len(va)} values, max rel {rel:.3e}{score}, first index {moved[0][0]}")
        bad += 1
    print(f"{len(names) - bad} of {len(names)} dumps identical; worst relative change {worst:.3e}")
    if bad and not a.allow_diff:
        sys.exit(1)


if __name__ == "__main__":
    main()
