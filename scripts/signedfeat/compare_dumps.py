#!/usr/bin/env python3
"""Compare two `signedfeat_dump_vector` output dirs cell by cell over the common prefix.
Usage: compare_dumps.py <baseline dir> <candidate dir>   (prints cells compared and differing cells; exit 1 on any difference)."""
import sys
from pathlib import Path
import numpy as np

a, b = Path(sys.argv[1]), Path(sys.argv[2])
names = sorted(p.name for p in a.glob("*.f64"))
assert names == sorted(p.name for p in b.glob("*.f64")), "different pair sets"
cells = diff = 0
for n in names:
    x, y = np.fromfile(a / n, dtype="<u8"), np.fromfile(b / n, dtype="<u8")
    k = min(x.size, y.size)
    d = int(np.count_nonzero(x[:k] != y[:k]))
    cells += k
    diff += d
    if d:
        print(f"{n}: {d} differing cells of {k} (widths {x.size} vs {y.size})")
print(f"pairs={len(names)} cells_compared={cells} differing_cells={diff} widths={x.size},{y.size}")
sys.exit(1 if diff else 0)
