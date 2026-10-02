"""Design log E9 (b), kept as a report by E9′/E9″: leave-one-block-out over the seven legacy blocks of the 944 bank.

  python3 e9_lobo.py grid  --program-sha P --data-sha D --out SPEC.json
  python3 e9_lobo.py score --root ROOT          # -> <root>/compare/e9_lobo.json

Cells: `r0-<block>@h32:H128` for basic, peaks, masked, iw, v2, append, append2 and the base `r0@h32:H128`; seeds 0-9,
both heads, five design folds (recipe C of E8). Statistic per block, head and source: the seed-paired Δ(R0 − R0∖block)
of the held-out signed SROCC, i.e. what the block adds on top of every other legacy column. Descriptive only; a block
whose Δ is <= 0 here and whose lean-base gain is <= 0 is reported as "adds nothing in this regime". No sealed label is read.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from v2_common import HEADS, LEGACY_BLOCKS, SOURCE_ORDER, V2

SEEDS = range(10)
RECIPE = ":H128"
BASE = f"r0@h32{RECIPE}"


def spec_of(block: str) -> str:
    return f"r0-{block}@h32{RECIPE}"


def cmd_grid(args) -> int:
    specs = [BASE, *(spec_of(b) for b in LEGACY_BLOCKS)]
    cells = [{"name": f"{sp}__{h}/without_{s}_s{i}",
              "argv": ["v2_lodo_mlp.py", "--spec", sp, "--head", h, "--heldout", s, "--seed-index", str(i), "--root", str(V2)]}
             for sp in specs for h in HEADS for s in SOURCE_ORDER for i in SEEDS]
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo), "out": args.out}))
    return 0


def signed(spec: str, head: str, source: str, seed: int) -> float | None:
    path = V2 / "cells" / f"{spec}__{head}" / f"without_{source}_s{seed}" / "result.json"
    if not path.is_file():
        return None
    y = pq.read_table(V2 / "wide" / "main" / "real" / f"{source}.keys.parquet", columns=["target"]).column(0).to_numpy()
    return float(spearmanr(json.loads(path.read_text())["prediction"], y)[0])


def cmd_score(args) -> int:
    rows, missing = {}, 0
    for block in LEGACY_BLOCKS:
        for head in HEADS:
            per = {}
            for s in SOURCE_ORDER:
                vals = []
                for i in SEEDS:
                    a, b = signed(BASE, head, s, i), signed(spec_of(block), head, s, i)
                    if a is None or b is None:
                        missing += 1
                        continue
                    vals.append(a - b)  # positive: removing the block loses held-out rank
                per[s] = {"delta": float(np.mean(vals)) if vals else None,
                          "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None, "n": len(vals)}
            v = [p["delta"] for p in per.values() if p["delta"] is not None]
            rows[f"{block}/{head}"] = {"columns": len(LEGACY_BLOCKS[block]), "per_source": per,
                                       "mean": float(np.mean(v)) if len(v) == len(SOURCE_ORDER) else None,
                                       "positive_sources": sum(x > 0 for x in v)}
    out = {"schema": "rev4-featpot-e9-lobo-v1", "base": BASE, "seeds": len(SEEDS), "rows": rows, "missing_cells": missing,
           "status": "INCOMPLETE" if missing else "complete"}
    dest = V2 / "compare" / "e9_lobo.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for key, r in rows.items():
        m = r["mean"]
        print(f"  {key:10s} cols {r['columns']:4d}  Δ(R0 − R0∖block) {m if m is not None else float('nan'):+.4f}  positive {r['positive_sources']}/5")
    print(json.dumps({"status": out["status"], "missing_cells": missing, "out": str(dest)}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["grid", "score"])
    ap.add_argument("--root", help="instrument root; read by v2_common from argv")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    args = ap.parse_args()
    return cmd_grid(args) if args.cmd == "grid" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
