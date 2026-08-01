#!/usr/bin/env python3
"""G-BF1/G-BF2 gate for the 944 backfill (PLAN_SOTA944 P1, task #9).

For a (new-944, old-924) parquet pair, verify:
  G-BF1: f0..f923 BITWISE identical row-for-row (compared at the OLD file's
         stored dtype — the kadis pair stores f32, everything else f64; the
         new file must store the SAME dtype for its first 924 f-columns).
  G-BF2: row counts identical; every non-feature column of OLD exists in NEW
         and is exactly equal (floats at bit-pattern level, other types by
         value), i.e. keys + target columns carried.
  Plus:  f156..f371 structurally zero in both; f924..f943 present, finite,
         bounded [0,1]; HL bins (f924+s*5+{3,4}) exactly 0 when
         --expect-hl-zero (the SDR routes — structural gating).

Any mismatch = STOP per the plan (byte-stability is proven at both tips; a
mismatch means wrong inputs/recipe, not drift). Exit 0 pass / 1 fail.

Usage:
  gate_backfill944.py --new NEW.parquet --old OLD.parquet \
      [--report out.json] [--no-expect-hl-zero] [--col-batch 48]
"""

import argparse
import json
import sys

import numpy as np
import pyarrow.parquet as pq

HL_LOCALS = (3, 4)  # idx_append2 HL_BIN1/2 within the per-scale block of 5
N_SCALES = 4
APPEND2_PER_SCALE = 5


def bits(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.float64:
        return a.view(np.uint64)
    if a.dtype == np.float32:
        return a.view(np.uint32)
    return a


def col_np(tbl, name, dtype=None):
    a = tbl[name].combine_chunks().to_numpy(zero_copy_only=False)
    if dtype is not None:
        a = np.asarray(a, dtype)
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--report")
    ap.add_argument("--no-expect-hl-zero", action="store_true")
    ap.add_argument("--col-batch", type=int, default=48)
    args = ap.parse_args()

    pf_new, pf_old = pq.ParquetFile(args.new), pq.ParquetFile(args.old)
    s_new, s_old = pf_new.schema_arrow, pf_old.schema_arrow
    rep = {
        "new": args.new,
        "old": args.old,
        "rows_new": pf_new.metadata.num_rows,
        "rows_old": pf_old.metadata.num_rows,
        "checks": {},
    }
    fails = []

    def check(name, ok, detail=""):
        rep["checks"][name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            fails.append(f"{name}: {detail}")
        print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f" ({detail})" if detail else ""))

    def is_f(n):
        return n.startswith("f") and n[1:].isdigit()

    # --- G-BF2: rows + schema shape ---
    check("rows_equal", rep["rows_new"] == rep["rows_old"],
          f"new={rep['rows_new']} old={rep['rows_old']}")
    old_f = [n for n in s_old.names if is_f(n)]
    new_f = [n for n in s_new.names if is_f(n)]
    old_nonf = [n for n in s_old.names if not is_f(n)]
    check("old_has_f0..f923", len(old_f) == 924 and set(old_f) == {f"f{i}" for i in range(924)},
          f"{len(old_f)} f-cols")
    check("new_has_f0..f943", len(new_f) == 944 and set(new_f) == {f"f{i}" for i in range(944)},
          f"{len(new_f)} f-cols")
    missing_nonf = [n for n in old_nonf if n not in s_new.names]
    check("nonfeat_cols_carried", not missing_nonf, f"missing={missing_nonf}")
    if fails:
        print("GATE: FAIL (structural)")
        _write(rep, args.report, False)
        return 1

    fdtype_old = s_old.field("f0").type
    fdtype_new = s_new.field("f0").type
    check("f_dtype_match", str(fdtype_old) == str(fdtype_new),
          f"old={fdtype_old} new={fdtype_new}")

    # --- G-BF2: non-feature columns exact ---
    for n in old_nonf:
        t_o = pq.read_table(args.old, columns=[n])
        t_n = pq.read_table(args.new, columns=[n])
        f_o = t_o.schema.field(n).type
        if str(f_o) in ("float", "double", "halffloat"):
            a_o, a_n = col_np(t_o, n), col_np(t_n, n)
            same = a_o.dtype == a_n.dtype and np.array_equal(bits(a_o), bits(a_n))
        else:
            same = t_o[n].to_pylist() == t_n[n].to_pylist()
        check(f"col_{n}_exact", same)

    # --- G-BF1: f0..f923 bitwise, batched; + structural-zero + append2 checks ---
    n_mismatch_cols = 0
    first_bad = None
    zero_ok = True
    a2_stats = {"min": np.inf, "max": -np.inf, "nonfinite": 0}
    hl_nonzero = 0
    hl_idx = {924 + s * APPEND2_PER_SCALE + l for s in range(N_SCALES) for l in HL_LOCALS}
    for lo in range(0, 944, args.col_batch):
        cols = [f"f{i}" for i in range(lo, min(lo + args.col_batch, 944))]
        old_cols = [c for c in cols if int(c[1:]) < 924]
        t_n = pq.read_table(args.new, columns=cols)
        t_o = pq.read_table(args.old, columns=old_cols) if old_cols else None
        for c in cols:
            i = int(c[1:])
            a_n = col_np(t_n, c)
            if i < 924:
                a_o = col_np(t_o, c)
                if not (a_o.dtype == a_n.dtype and np.array_equal(bits(a_o), bits(a_n))):
                    n_mismatch_cols += 1
                    if first_bad is None:
                        d = ~(bits(a_o) == bits(a_n))
                        r = int(np.argmax(d))
                        first_bad = (c, r, float(a_o[r]), float(a_n[r]))
                if 156 <= i < 372 and not (np.all(a_o == 0.0) and np.all(a_n == 0.0)):
                    zero_ok = False
            else:
                a64 = np.asarray(a_n, np.float64)
                fin = np.isfinite(a64)
                a2_stats["nonfinite"] += int((~fin).sum())
                if fin.any():
                    a2_stats["min"] = min(a2_stats["min"], float(a64[fin].min()))
                    a2_stats["max"] = max(a2_stats["max"], float(a64[fin].max()))
                if i in hl_idx:
                    hl_nonzero += int(np.count_nonzero(a64))
    check("GBF1_f0..f923_bitwise", n_mismatch_cols == 0,
          f"mismatch_cols={n_mismatch_cols} first_bad={first_bad}")
    check("f156..f371_structural_zero", zero_ok)
    check("append2_finite", a2_stats["nonfinite"] == 0, f"nonfinite={a2_stats['nonfinite']}")
    check("append2_bounded_0_1",
          a2_stats["min"] >= 0.0 and a2_stats["max"] <= 1.0,
          f"range=[{a2_stats['min']:.6g},{a2_stats['max']:.6g}]")
    if not args.no_expect_hl_zero:
        check("hl_bins_zero_sdr", hl_nonzero == 0, f"nonzero={hl_nonzero}")

    ok = not fails
    print(f"GATE: {'PASS' if ok else 'FAIL'} ({args.new})")
    rep["append2_range"] = [a2_stats["min"], a2_stats["max"]]
    _write(rep, args.report, ok)
    return 0 if ok else 1


def _write(rep, path, ok):
    rep["verdict"] = "PASS" if ok else "FAIL"
    if path:
        with open(path, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
