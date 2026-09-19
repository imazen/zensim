#!/usr/bin/env python3
"""joint-core-v1 DVIFM constants fit — f16-capped caches + row subsets.

Wraps the phase-2d standalone fitter (fit_standalone.py, imported as a
module — identical numerics) with:

  PlaneCacheF16 -- reads the capped/f16 caches written by
                   `extract_features_372col --dvifm-cap --dvifm-quant f16`.
                   Index 'offset' stays in BYTES; elements are 2B.
  RowView       -- deterministic row subset of a cache (stride or list);
                   the fit domain shrinks ROWS first per the work order.

Usage: fit_core.py <spec.json> <out.json> <surfaces_dir>
                   <plane>=<bin> [<plane>=<bin> ...]
                   --pairs-tsv <tsv> --legs a,b,c --row-stride K
"""
import json, math, sys, time
from pathlib import Path

sys.path.insert(0, '/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/tools')
import numpy as np
import fit_standalone as fs


class PlaneCacheF16(fs.PlaneCache):
    """Same layout as PlaneCache but the record element is u16-f16."""

    def __init__(self, bin_path):
        self.bin_path = str(bin_path)
        self.index = fs.load_index(bin_path)
        self.data = np.memmap(bin_path, dtype=np.dtype("<f2"), mode="r")
        self.n_rows = len(self.index)
        self.nb = np.zeros((self.n_rows, fs.LEVELS), np.int64)
        self.row_elem = np.zeros((self.n_rows, fs.LEVELS), np.int64)
        for i, e in enumerate(self.index):
            base = e["offset"] // 2          # f16 elements
            cnt = np.asarray(e["level_records"], np.int64)
            self.nb[i] = cnt
            self.row_elem[i] = base + np.cumsum(
                np.concatenate([[0], cnt]))[:5] * fs.REC
        self.nz_rows = [np.nonzero(self.nb[:, l])[0] for l in range(fs.LEVELS)]
        self._sub = [None] * fs.LEVELS
        self._full = [None] * fs.LEVELS
        from collections import OrderedDict
        self._recs_sub = OrderedDict()


class RowView(PlaneCacheF16):
    """Row subset of an f16 cache: keeps the file, narrows the row space."""

    def __init__(self, bin_path, rows):
        super().__init__(bin_path)
        rows = np.asarray(rows, np.int64)
        self.nb = self.nb[rows]
        self.row_elem = self.row_elem[rows]
        self.n_rows = len(rows)
        self.nz_rows = [np.nonzero(self.nb[:, l])[0]
                        for l in range(fs.LEVELS)]


def main():
    args = sys.argv[1:]
    spec_path, out_path, surfaces_dir = args[0], args[1], args[2]
    planes = {}
    i = 3
    pairs_tsv = None
    keep_legs = None
    row_stride = 1
    while i < len(args):
        a = args[i]
        if a == "--pairs-tsv":
            pairs_tsv = args[i + 1]
            i += 2
        elif a == "--legs":
            keep_legs = set(args[i + 1].split(","))
            i += 2
        elif a == "--row-stride":
            row_stride = int(args[i + 1])
            i += 2
        else:
            p, _, path = a.partition("=")
            planes[p] = path
            i += 1

    import csv
    rows_meta = [r for r in csv.DictReader(open(pairs_tsv), delimiter="\t")]
    # NB: the pairs file is cell-blocked (a ref's ~32 encode cells sit
    # contiguous, score-ascending), so a bare `i % stride` ALIASES onto one
    # block position (observed: stride-16 -> mean(y) 19.1 vs 51.9 full).
    # Deterministic uniform subset instead: seeded permutation, take first
    # n//stride indices, sorted back to file order for cache views.
    leg_ok = [i for i, r in enumerate(rows_meta)
              if keep_legs is None or r["leg"] in keep_legs]
    if row_stride > 1:
        rng = np.random.default_rng(20260920)
        k = max(1, len(leg_ok) // row_stride)
        sel = sorted(rng.choice(leg_ok, size=k, replace=False).tolist())
    else:
        sel = leg_ok
    y = np.asarray([float(rows_meta[i]["human_score"]) for i in sel])
    caches = {p: RowView(b, sel) for p, b in planes.items()}
    n = next(iter(caches.values())).n_rows
    assert len(y) == n, (len(y), n)

    spec = json.loads(Path(spec_path).read_text())
    Path(surfaces_dir).mkdir(parents=True, exist_ok=True)
    prov = {
        "caches": {p: b for p, b in planes.items()},
        "pairs_tsv": pairs_tsv,
        "legs": sorted(keep_legs) if keep_legs else "all",
        "row_stride": row_stride,
        "row_sample": "seeded uniform subset (rng 20260920, n=len//stride) "
                      "— bare stride aliases the cell-blocked file order",
        "fit_rows": int(n),
        "record_format": "f16 capped (cap + strides in .index.jsonl)",
        "init_spec": {"path": spec_path, "sha256": fs.sha(spec_path)},
        "estimator": fs.Params and
        f"Adam on deterministic <={fs.ADAM_BLOCK_CAP}-block per-row stride "
        "subsample over the capped records; reported losses on kept records",
        "objective": "Amendment-1 refit-map MSE: yhat=A*exp(-lam*E)+B, "
                     "A>0; (A,B) constrained LS per lam, lam golden-"
                     "sectioned; SROCC/KROCC of -E recorded per cell",
        "grid": {"c0": fs.GRID_C0.tolist(), "beta": fs.GRID_BETA.tolist(),
                 "spacing": "log", "edge_extend_points": fs.GRID_EXTEND},
        "param_bounds": {"g": [fs.G_LO, fs.G_HI], "beta": [fs.B_LO, fs.B_HI],
                         "c0": [fs.C0_LO, fs.C0_HI],
                         "lam": [fs.LAM_LO, fs.LAM_HI]},
        "adam": {"level_steps": fs.ADAM_STEPS_LEVEL,
                 "head_steps": fs.ADAM_STEPS_HEAD, "lr": fs.ADAM_LR,
                 "starts": fs.N_STARTS, "max_sweeps": fs.MAX_SWEEPS,
                 "rel_stop": fs.REL_STOP},
    }
    t0 = time.time()
    fs.fit_variant("core_native3", tuple(planes.keys()), caches, y,
                   spec["levels"], out_path, surfaces_dir, prov)
    print(f"FIT WALL {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
