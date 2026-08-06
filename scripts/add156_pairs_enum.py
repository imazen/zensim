#!/usr/bin/env python3
"""Enumerate the REGISTERED APPENDIX U cell grid — mechanically, from the
committed layout decoder.

Four rules, frozen in appendix U §U.4. Singletons (S) are the control that
makes "pair" a testable claim: a pair is only interesting if it beats BOTH of
its members alone.

    S  singleton — every candidate slot in the arm's pool
    W  within-cell: all pairs of locals inside one (block, scale, channel)
    C  cross-channel: same (block, scale, local), the 3 channel pairs
    Z  cross-scale-adjacent: same (block, channel, local), scales (0,1)(1,2)(2,3)
    G  family / gain-loss partners inside a cell (arm B; arm B's W is
       HF-priority-only, so G carries the families that W does not reach)

Plus the G-U1/G-U2 structural gate cells, which must come out ZERO.

The layout comes from `scripts/featsub/k128_stage_map.py` — the committed
decoder, not a second copy of the arithmetic.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "featsub"))
from k128_stage_map import (  # noqa: E402
    APP_LOCALS, A2_LOCALS, V2_LOCALS, decode, decode_v1_372,
)

# ---- arm A: v1-372 peak72 / masked72 / iw72 (base = off + scale*18 + ch*6) ---
A_BLOCKS = {"peak72": 156, "masked72": 228, "iw72": 300}
A_NLOC = 6


def a_idx(block: str, scale: int, ch: int, local: int) -> int:
    return A_BLOCKS[block] + scale * 18 + ch * 6 + local


# ---- arm B: folded-944 ------------------------------------------------------
def v2_idx(scale: int, ch: int, local: int) -> int:
    return 372 + scale * 87 + ch * 29 + local


def app_idx(scale: int, ch: int, local: int) -> int:
    return 720 + scale * 51 + ch * 17 + local


def a2_idx(scale: int, local: int) -> int:
    return 924 + scale * 5 + local


# HF-priority locals for arm B's dense within-cell enumeration (appendix U §U.3
# ranks 1, 2, 4, 5). Named, not index-literal, so a layout change breaks loudly.
HF_LOCAL_NAMES = [
    "HF_GAIN", "HF_LOSS", "HF_MAG_LOSS",                        # rank 4
    "SSIM_SOFT_PEAK", "ART_SOFT_PEAK", "DET_SOFT_PEAK",         # rank 5
    "PJND_TRANSDUCER", "PJND_FRAGILITY",                        # rank 1
    "PJND_TRANSDUCER_LOW_K", "PJND_TRANSDUCER_HIGH_K",          # rank 1
    "BLOCKINESS", "RINGING", "BANDING", "EDGE_WIDTH_CHANGE",    # rank 2
]
# G-rule families (arm B): each is a set of local NAMES inside one cell.
V2_FAMILIES = {
    "hf": ["HF_GAIN", "HF_LOSS", "HF_MAG_LOSS"],
    "softpeak": ["SSIM_SOFT_PEAK", "ART_SOFT_PEAK", "DET_SOFT_PEAK"],
    "pjnd": ["PJND_TRANSDUCER", "PJND_FRAGILITY",
             "PJND_TRANSDUCER_LOW_K", "PJND_TRANSDUCER_HIGH_K"],
    "artifact": ["BLOCKINESS", "RINGING", "BANDING", "EDGE_WIDTH_CHANGE"],
}
APP_FAMILIES = {
    "contrast": ["CONTRAST_GAIN", "CONTRAST_LOSS"],
    "lum": ["LUM_DARK_ERR", "LUM_MID_ERR", "LUM_BRIGHT_ERR"],
}
A2_FAMILIES = {"bandvis": ["BANDVIS_GAIN", "BANDVIS_LOSS"]}


def li(names: list[str], want: str) -> int:
    return names.index(want)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    cells: dict[tuple[str, tuple[int, ...]], tuple[str, str]] = {}

    def add(arm: str, kind: str, blk: str, idxs) -> None:
        key = (arm, tuple(sorted(set(idxs))))
        # A pair that collapses to one index is a singleton; drop the dup.
        if len(key[1]) == 0:
            return
        cells.setdefault(key, (kind, blk))

    # ============================== ARM A ====================================
    for blk in A_BLOCKS:
        for s in range(4):
            for c in range(3):
                for l in range(A_NLOC):                       # S
                    add("A", "S", blk, [a_idx(blk, s, c, l)])
                for l1, l2 in combinations(range(A_NLOC), 2):  # W
                    add("A", "W", blk, [a_idx(blk, s, c, l1), a_idx(blk, s, c, l2)])
        for s in range(4):                                     # C
            for l in range(A_NLOC):
                for c1, c2 in combinations(range(3), 2):
                    add("A", "C", blk, [a_idx(blk, s, c1, l), a_idx(blk, s, c2, l)])
        for c in range(3):                                     # Z
            for l in range(A_NLOC):
                for s in range(3):
                    add("A", "Z", blk, [a_idx(blk, s, c, l), a_idx(blk, s + 1, c, l)])

    # ============================== ARM B ====================================
    # ---- v2-348
    for s in range(4):
        for c in range(3):
            for l in range(29):                                # S
                add("B", "S", "v2_348", [v2_idx(s, c, l)])
            if s <= 1:                                         # W (HF-priority)
                hf = [li(V2_LOCALS, n) for n in HF_LOCAL_NAMES]
                for l1, l2 in combinations(hf, 2):
                    add("B", "W", "v2_348", [v2_idx(s, c, l1), v2_idx(s, c, l2)])
            for fam in V2_FAMILIES.values():                   # G
                fl = [li(V2_LOCALS, n) for n in fam]
                for l1, l2 in combinations(fl, 2):
                    add("B", "G", "v2_348", [v2_idx(s, c, l1), v2_idx(s, c, l2)])
    for s in range(4):                                         # C
        for l in range(29):
            for c1, c2 in combinations(range(3), 2):
                add("B", "C", "v2_348", [v2_idx(s, c1, l), v2_idx(s, c2, l)])
    for c in range(3):                                         # Z
        for l in range(29):
            for s in range(3):
                add("B", "Z", "v2_348", [v2_idx(s, c, l), v2_idx(s + 1, c, l)])

    # ---- append-204
    for s in range(4):
        for c in range(3):
            for l in range(17):                                # S
                add("B", "S", "append204", [app_idx(s, c, l)])
            for fam in APP_FAMILIES.values():                  # G
                fl = [li(APP_LOCALS, n) for n in fam]
                for l1, l2 in combinations(fl, 2):
                    add("B", "G", "append204", [app_idx(s, c, l1), app_idx(s, c, l2)])
    for s in range(4):                                         # C
        for l in range(17):
            for c1, c2 in combinations(range(3), 2):
                add("B", "C", "append204", [app_idx(s, c1, l), app_idx(s, c2, l)])
    for c in range(3):                                         # Z
        for l in range(17):
            for s in range(3):
                add("B", "Z", "append204", [app_idx(s, c, l), app_idx(s + 1, c, l)])

    # ---- append2-20 (Y only: no cross-channel rule)
    for s in range(4):
        for l in range(5):                                     # S
            add("B", "S", "append2_20", [a2_idx(s, l)])
        for fam in A2_FAMILIES.values():                       # G
            fl = [li(A2_LOCALS, n) for n in fam]
            for l1, l2 in combinations(fl, 2):
                add("B", "G", "append2_20", [a2_idx(s, l1), a2_idx(s, l2)])
    for l in range(5):                                         # Z
        for s in range(3):
            add("B", "Z", "append2_20", [a2_idx(s, l), a2_idx(s + 1, l)])

    # ===================== structural gate cells (must be ZERO) ==============
    # G-U1: candidates INSIDE f0..155 are a provable no-op at ADD156's lam.
    #       12 unused basic slots, spread over scales/channels/locals.
    base28 = {6, 8, 11, 14, 17, 19, 22, 24, 26, 34, 37, 89, 91, 93, 94, 116,
              120, 121, 122, 124, 128, 136, 137, 138, 140, 146, 150, 155}
    unused = [i for i in range(156) if i not in base28]
    for i in unused[::11][:12]:
        add("A", "GU1", "v1basic156", [i])
    # G-U2: f156..371 at root B are folded structural zeros (replicates T.R5).
    for i in range(156, 372, 18):
        add("B", "GU2", "zeros156_371", [i])

    rows = []
    for (arm, idxs), (kind, blk) in cells.items():
        dec = decode_v1_372 if arm == "A" else decode
        names = []
        for i in idxs:
            b, s, c, n, _p = dec(i)
            names.append(f"{n}@s{s}{c or ''}")
        cid = "-".join(str(i) for i in idxs)
        rows.append((cid, arm, kind, blk, ",".join(str(i) for i in idxs),
                     len(idxs), "+".join(names)))
    rows.sort(key=lambda r: (r[1], r[2], [int(x) for x in r[4].split(",")]))

    with open(a.out, "w") as f:
        f.write("cell_id\tarm\tkind\tblock\tindices\tn_cand\tnames\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"{len(rows)} unique cells", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
