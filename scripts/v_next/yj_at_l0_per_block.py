#!/usr/bin/env python3
"""Per-block L0 importance for a single ZNPR v3 bake.

Reads the bake's L0 layer weights and computes
`importance[i] = scaler_scale[i] * sum_h(|L0[h, i]|)` per input
feature `i`, then aggregates per the canonical 372-feature block
schema (basic / peak / masked / iw_pool).

Used to compare per-block mass between the current v11 ship and the
YJ-autotransforms retrain (task #214 Phase 2).

Output: prints a per-block mass table + emits a TSV.

Usage:
  python3 scripts/v_next/yj_at_l0_per_block.py --bake <path> [--out <tsv>]

Optionally pass --baseline-bake <path> for an A/B compare table.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed", file=sys.stderr)
    sys.exit(2)


def block_for(idx: int) -> str:
    """Canonical 372-feature block schema from zensim/CLAUDE.md."""
    if idx < 156:
        return "basic"
    if idx < 228:
        return "peak"
    if idx < 300:
        return "masked"
    return "iw_pool"


def l0_importance(bake_path: Path) -> tuple[np.ndarray, int]:
    """Use zenpredict via a subprocess to extract L0 importance.

    Returns (importance, n_inputs). importance[i] is the L0 mass for
    feature i: `scaler_scale[i] * sum_h(|L0[h, i]|)`.
    """
    # Build via the dump_l0_importance binary which already does the
    # math. Run it pointed at a temp dir containing only this bake,
    # then parse the emitted CSV.
    import shutil
    import subprocess
    import tempfile

    bake_path = bake_path.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # The binary walks `zensim/weights/`. Create that subdir under tmp + symlink only this bake.
        # Easiest: build our own Rust importance script via cargo example.
        # Alternative: use Python to read the ZNPR v3 bake directly.
        # Take the alternative path.
        pass

    # Read ZNPR v3 manually. Format spec lives in zenanalyze/zenpredict
    # but the layout is approximately:
    #   header: magic("ZNPR"), version (1 byte), flags (...), n_inputs
    #   layers: each with (in_dim, out_dim, dtype, weights, biases)
    #   metadata: bytes
    # This is too brittle for ad-hoc parsing. Instead, use the trained
    # bake's own JSON inspector if available.
    raise NotImplementedError("Use bake_verdict instead; or copy the bake to weights/ dir + run dump_l0_importance + grep the CSV row")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", type=Path, required=True)
    ap.add_argument("--baseline-bake", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # Run dump_l0_importance with both bakes in zensim/weights/. The
    # binary already emits a CSV — we grep + post-process.
    import csv
    import shutil
    import subprocess

    bakes_dir = Path("/home/lilith/work/zen/zensim/zensim/weights")
    placed: list[Path] = []
    try:
        # Place the bake(s) in weights/ (copy with unique name so the
        # binary picks them up).
        bake_target = bakes_dir / args.bake.name
        if not bake_target.exists():
            shutil.copy(args.bake, bake_target)
            placed.append(bake_target)
        if args.baseline_bake:
            base_target = bakes_dir / args.baseline_bake.name
            if not base_target.exists():
                shutil.copy(args.baseline_bake, base_target)
                placed.append(base_target)

        # Run the binary
        binary = Path("/home/lilith/work/zen/zensim/target/debug/examples/dump_l0_importance")
        subprocess.run([str(binary)], check=True, stdout=subprocess.DEVNULL)

        # Read /tmp/zensim_l0_importance.csv
        csv_path = Path("/tmp/zensim_l0_importance.csv")
        if not csv_path.exists():
            raise RuntimeError("dump_l0_importance.csv not emitted")

        rows = []
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                rows.append(row)
        # Filter to bakes we care about
        bake_names = {args.bake.name}
        if args.baseline_bake:
            bake_names.add(args.baseline_bake.name)
        rows = [r for r in rows if r["bake_name"] in bake_names]
        if not rows:
            raise RuntimeError(f"no rows found for {bake_names}; check the binary's output")

        # Aggregate per-block per-bake
        results: dict[str, dict[str, float]] = {b: {} for b in bake_names}
        for r in rows:
            b = r["bake_name"]
            idx = int(r["feature_index"])
            imp = float(r["importance"])
            blk = block_for(idx)
            results[b][blk] = results[b].get(blk, 0.0) + imp

        # Emit
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w") as f:
                w = csv.writer(f, delimiter="\t")
                cols = ["block", *sorted(bake_names)]
                if args.baseline_bake and len(bake_names) == 2:
                    cols.append("delta_pct")
                w.writerow(cols)
                for blk in ("basic", "peak", "masked", "iw_pool"):
                    row = [blk]
                    vals = {}
                    for b in sorted(bake_names):
                        v = results[b].get(blk, 0.0)
                        row.append(f"{v:.4f}")
                        vals[b] = v
                    if args.baseline_bake and len(bake_names) == 2:
                        bn = args.baseline_bake.name
                        on = args.bake.name
                        if bn in vals and on in vals and vals[bn] > 0:
                            delta = (vals[on] - vals[bn]) / vals[bn] * 100.0
                            row.append(f"{delta:+.2f}%")
                        else:
                            row.append("n/a")
                    w.writerow(row)

        # Stdout summary
        print("Per-block L0 mass (sum of importance per block):")
        for b in sorted(bake_names):
            print(f"\n  {b}:")
            total = sum(results[b].get(blk, 0.0) for blk in ("basic", "peak", "masked", "iw_pool"))
            for blk in ("basic", "peak", "masked", "iw_pool"):
                v = results[b].get(blk, 0.0)
                pct = v / total * 100.0 if total > 0 else 0.0
                print(f"    {blk:<10} {v:>10.4f}  ({pct:>5.1f}%)")

        if args.baseline_bake and len(bake_names) == 2:
            bn = args.baseline_bake.name
            on = args.bake.name
            print(f"\nDelta ({on} − {bn}):")
            print(f"  {'block':<10} | {'baseline':>10} | {'candidate':>10} | {'Δ (abs)':>10} | {'Δ %':>9}")
            for blk in ("basic", "peak", "masked", "iw_pool"):
                bv = results[bn].get(blk, 0.0)
                ov = results[on].get(blk, 0.0)
                delta = ov - bv
                pct = (delta / bv * 100.0) if bv > 0 else 0.0
                print(f"  {blk:<10} | {bv:>10.4f} | {ov:>10.4f} | {delta:>+10.4f} | {pct:>+8.2f}%")
            # Also baseline/candidate share-of-mass (normalized to 1.0)
            print("\nNormalized share of mass:")
            bt = sum(results[bn].get(blk, 0.0) for blk in ("basic", "peak", "masked", "iw_pool"))
            ot = sum(results[on].get(blk, 0.0) for blk in ("basic", "peak", "masked", "iw_pool"))
            print(f"  {'block':<10} | {'baseline%':>10} | {'candidate%':>10} | {'Δ pp':>8}")
            for blk in ("basic", "peak", "masked", "iw_pool"):
                bp = results[bn].get(blk, 0.0) / bt * 100.0 if bt > 0 else 0.0
                op = results[on].get(blk, 0.0) / ot * 100.0 if ot > 0 else 0.0
                print(f"  {blk:<10} | {bp:>10.2f} | {op:>10.2f} | {op - bp:>+7.2f}")

        return 0
    finally:
        # Clean up copies
        for p in placed:
            try:
                p.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
