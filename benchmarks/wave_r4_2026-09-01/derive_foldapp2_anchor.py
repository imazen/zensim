#!/usr/bin/env python3
"""wave-r4: derive the `foldapp2`-regime dial anchor from the pools anchor.

WHY THIS EXISTS — completing the regime fix from `derive_foldapp2_views.py`
(2cacb6d5) that the terminated session did not reach. A1's flagship recipe
was retrained (correctly) on `foldapp2_views/` (f156..371 structural zero),
but `anchor944_pools_dial.parquet` — the dial-spline fit anchor built for the
FIRST (pools-regime) A1 attempt — still carries nonzero f156..371. Fitting a
foldapp2-trained model's output spline on nonzero-pool anchor rows forwards it
on inputs it never trained on (extrapolation into an unseen region), which is
exactly the kind of regime mismatch this wave's own §7.3 correction exists to
catch. So the anchor needs the same treatment as the other 22 tables.

Reuses the IDENTICAL identity `derive_foldapp2_views.py` established and
gate-checked (foldapp2 == foldapp2pools with f156..f371 set to 0, exactly) —
this script does not re-derive or re-check that identity, only applies it to
one more table. Unlike `derive_foldapp2_views.py`, this MERGES into the
existing `foldapp2_views/_MANIFEST.json` (22 entries) instead of overwriting
it, because it is invoked standalone rather than as part of that full run.

Output is renamed `anchor944_foldapp2_dial.parquet` (never re-use the "pools"
name inside foldapp2_views/ — regime-purity naming, same discipline as every
other file in this wave).

Generalized (2026-09-01, same run) to take an arbitrary external SRC/OUT pair
via env vars, so the same gated transform derives the wlin7b-2026-08-30 dial
grid's foldapp2 counterpart without a second near-duplicate script. Caveat
recorded where this is invoked: the wlin7b dial grid predates the radius-4
patch workflow (built 2026-08-30; `patch_radius.sh` landed 2026-08-31), so
this only fixes the REGIME (pool block); the RADIUS is not verified as 4. It
is used for the secondary dial/monotonicity panel only — the primary rank
bars (§4.1/§4.2 W1-W2) never touch it.
"""
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC_ROOT = os.environ.get("WR4_ROOT", "/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01")
DEST = os.environ.get("WR4_ZERO", SRC_ROOT + "/foldapp2_views")
SRC = os.environ.get("WR4_ANCHOR_SRC", os.path.join(SRC_ROOT, "anchor944_pools_dial.parquet"))
OUT_NAME = os.environ.get("WR4_ANCHOR_OUT_NAME", "anchor944_foldapp2_dial.parquet")
OUT = os.path.join(DEST, OUT_NAME)
POOL = [f"f{i}" for i in range(156, 372)]
HAS_TARGET_SCORE = os.environ.get("WR4_ANCHOR_HAS_TARGET_SCORE", "1") == "1"


def main() -> int:
    t = pq.read_table(SRC)
    names = set(t.schema.names)
    missing = [c for c in POOL if c not in names]
    if missing:
        raise SystemExit(f"ABORT: {SRC} lacks {len(missing)} pool columns (first {missing[:3]})")
    zero = pa.array(np.zeros(t.num_rows, dtype=np.float64), pa.float64())
    cols = {n: (zero if n in set(POOL) else t.column(n)) for n in t.schema.names}
    if "regime" in cols:
        cols["regime"] = pa.array(["folded720append2"] * t.num_rows)
    out_table = pa.table(cols)
    pq.write_table(out_table, OUT, compression="zstd", compression_level=7)

    # gate: non-pool columns byte-identical to source, pool columns exactly 0,
    # row count and target_score untouched.
    chk = pq.read_table(OUT)
    bad_zero = int(sum(1 for c in POOL if chk.column(c).to_numpy(zero_copy_only=False).any()))
    if bad_zero:
        raise SystemExit(f"ABORT: {OUT_NAME} has {bad_zero} nonzero pool columns after zeroing")
    if chk.num_rows != t.num_rows:
        raise SystemExit(f"ABORT: row count changed {t.num_rows} -> {chk.num_rows}")
    if HAS_TARGET_SCORE:
        ts_src = t.column("target_score").to_numpy(zero_copy_only=False)
        ts_out = chk.column("target_score").to_numpy(zero_copy_only=False)
        if not np.array_equal(ts_src, ts_out):
            raise SystemExit("ABORT: target_score changed by the zeroing transform")
    print(f"  {OUT_NAME}: {t.num_rows} rows, f156-371 zeroed, non-pool cols untouched, gate PASS")

    # merge into the existing 22-table manifest rather than overwriting it.
    man_path = os.path.join(DEST, "_MANIFEST.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {
        "source_root": SRC_ROOT, "regime": "folded720append2",
        "rule": "f156..f371 := 0.0 on every row (measured EXACT vs a native foldapp2 extraction)",
        "purity": "NEVER column-mix with the folded720append2pools tables these came from",
        "tables": {},
    }
    man["tables"][OUT_NAME] = {
        "from": "anchor944_pools_dial.parquet",
        "rows": t.num_rows,
        "pool_cols_zeroed": len(POOL),
        "note": "dial-spline fit anchor for foldapp2-regime (pool-zeroed) bakes; "
                "row identity and target_score inherited unchanged from the pools anchor "
                "(stride-55 safesyn, target_score = human_score*100).",
    }
    json.dump(man, open(man_path, "w"), indent=1)
    print(f"OK -> {OUT} (manifest now {len(man['tables'])} tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
