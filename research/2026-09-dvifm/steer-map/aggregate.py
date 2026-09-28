#!/usr/bin/env python3
"""Aggregate dvifm_steer_study pairs.jsonl into the benchmark record tables.

Usage: aggregate.py <out_dir>  — reads pairs.jsonl + rects.bin + summary.json,
writes aggregate.json next to them and prints the markdown tables.

rects.bin record layout per rect (15 f32 LE):
  size_idx, x0, y0, x1, y1, dvifm_comb, dvifm_l0..l4, attr, refgain, sse, delta_s
"""
import json
import math
import statistics as st
import struct
import sys
from pathlib import Path

RECT_F32 = 15
SIZES = [16, 32, 64, 128, 256]


def q(vals):
    vals = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p25": None,
                "p75": None, "p90": None}
    n = len(vals)
    return {
        "n": n,
        "mean": st.fmean(vals),
        "median": st.median(vals),
        "p25": vals[n // 4] if n >= 4 else vals[0],
        "p75": vals[(3 * n) // 4] if n >= 4 else vals[-1],
        "p90": vals[min(n - 1, int(0.9 * n))] if n >= 10 else vals[-1],
    }


def linfit(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = st.fmean(xs)
    my = st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-30:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def read_rects(row, blob):
    off, ln = row["rects_off"], row["rects_len"]
    n = ln // (RECT_F32 * 4)
    out = []
    for k in range(n):
        v = struct.unpack_from("<15f", blob, off + k * RECT_F32 * 4)
        out.append({
            "size": SIZES[int(v[0])],
            "rect": v[1:5],
            "dvifm": v[5],
            "lvl": v[6:11],
            "attr": v[11],
            "refgain": v[12],
            "sse": v[13],
            "ds": v[14],
        })
    return out


def rel_errors(rects, key):
    """Calibrated relative error for predictor `key` (or int level index)
    within this (pair,size) rect set: per-set least-squares calibrate, then
    |pred_hat - ds| / (|ds| + floor), floor = 0.1*median|ds|."""
    if isinstance(key, int):
        xs = [r["lvl"][key] for r in rects]
    else:
        xs = [r[key] for r in rects]
    ys = [r["ds"] for r in rects]
    fit = linfit(xs, ys)
    if fit is None:
        return []
    slope, icpt = fit
    med = st.median(abs(y) for y in ys)
    floor = 0.1 * med + 1e-9
    return [abs(slope * x + icpt - y) / (abs(y) + floor) for x, y in zip(xs, ys)]


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    rows = [json.loads(l) for l in open(out / "pairs.jsonl") if l.strip()]
    summary = json.load(open(out / "summary.json"))
    blob = (out / "rects.bin").read_bytes()

    s2_keys = ["srocc_dvifm", "srocc_attr", "srocc_refgain", "srocc_sse",
               "pearson_dvifm", "pearson_attr", "precision_q_dvifm",
               "precision_q_attr", "kendall_dvifm", "kendall_attr"]
    s3_keys = ["srocc_dvifm_ssim2", "srocc_attr_ssim2", "srocc_dvifm_butter",
               "srocc_attr_butter", "srocc_judges", "srocc_dvifm_agree",
               "srocc_attr_agree", "precision_agree_dvifm",
               "precision_agree_attr", "kendall_dvifm_ssim2",
               "kendall_attr_ssim2", "agree_frac"]

    def agg(rr, section, keys):
        return {k: q([r[section][k] for r in rr]) for k in keys}

    result = {
        "summary": summary,
        "n_rows": len(rows),
        "overall": {"s2": agg(rows, "s2", s2_keys),
                    "s3": agg(rows, "s3", s3_keys)},
        "s2_by_size": {},
        "relerr_by_size": {},
        "relerr_by_level": {},
        "additivity": {},
        "leak": {},
        "by_leg": {}, "by_codec": {}, "by_band": {}, "by_group": {},
        "dvifm_levels_s2": [], "dvifm_levels_s3_ssim2": [],
        "dvifm_levels_s3_butter": [],
    }

    # ---- per-size S2 metrics + calibrated relative error ----
    size_metrics = {s: {k: [] for k in s2_keys} for s in SIZES}
    size_relerr = {s: {k: [] for k in ("dvifm", "attr", "refgain", "sse")}
                   for s in SIZES}
    lvl_relerr = {l: [] for l in range(5)}
    add_resid, add_pairs, add_joint_ratio = [], [], []
    leak_lv = {l: [] for l in range(5)}
    leak_attr = []

    for row in rows:
        rects = read_rects(row, blob)
        by_size = {}
        for r in rects:
            by_size.setdefault(r["size"], []).append(r)
        for s, rs in by_size.items():
            bs = row["s2"]["by_size"].get(str(s))
            if bs:
                for k in s2_keys:
                    size_metrics[s][k].append(bs[k])
            for key in ("dvifm", "attr", "refgain", "sse"):
                size_relerr[s][key] += rel_errors(rs, key)
            for l in range(5):
                lvl_relerr[l] += rel_errors(rs, l)
        for t in row["s2"].get("add_tests", []):
            s_ab = t["ds_a"] + t["ds_b"]
            add_resid.append(t["residual"])
            add_pairs.append((abs(t["residual"]), abs(s_ab)))
            if abs(s_ab) > 1e-9:
                add_joint_ratio.append(t["ds_joint"] / s_ab)
        for l, v in enumerate(row["s2"].get("leak_dvifm_levels") or []):
            if v is not None:
                leak_lv[l].append(v)
        if row["s2"].get("leak_attr") is not None:
            leak_attr.append(row["s2"]["leak_attr"])

    for s in SIZES:
        result["s2_by_size"][s] = {k: q(v) for k, v in size_metrics[s].items()}
        result["relerr_by_size"][s] = {k: q(v)
                                       for k, v in size_relerr[s].items()}
    result["relerr_by_level"] = {str(l): q(v) for l, v in lvl_relerr.items()}
    floor = (0.1 * st.median(s for _, s in add_pairs) if add_pairs else 0.0) + 1e-9
    result["additivity"] = {
        "n": len(add_resid),
        "residual": q(add_resid),
        "abs_rel_residual": q([r / (s + floor) for r, s in add_pairs]),
        "joint_over_sum": q(add_joint_ratio),
        "rel_floor": floor,
    }
    result["leak"] = {
        "dvifm_levels": {str(l): q(v) for l, v in leak_lv.items()},
        "attr": q(leak_attr),
    }

    for l in range(5):
        result["dvifm_levels_s2"].append(
            q([r["s2"]["srocc_dvifm_levels"][l] for r in rows]))
        result["dvifm_levels_s3_ssim2"].append(
            q([r["s3"]["srocc_dvifm_levels_ssim2"][l] for r in rows]))
        result["dvifm_levels_s3_butter"].append(
            q([r["s3"]["srocc_dvifm_levels_butter"][l] for r in rows]))

    for field, dest in [("leg", "by_leg"), ("codec", "by_codec"),
                        ("band", "by_band"), ("group", "by_group")]:
        groups = {}
        for r in rows:
            groups.setdefault(r[field], []).append(r)
        for name, rs in sorted(groups.items()):
            result[dest][name] = {
                "n": len(rs),
                "s2": agg(rs, "s2", s2_keys),
                "s3": agg(rs, "s3", s3_keys),
            }

    json.dump(result, open(out / "aggregate.json", "w"), indent=2)

    def f(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    o = result["overall"]
    print(f"n={len(rows)} pairs, parity_max={summary['ssim2_parity_max']:.2e}, "
          f"skipped_size={summary['n_skipped_size']}")
    print("\n## S2 pooled — rank agreement with finite-edit ΔS")
    for k in s2_keys:
        a = o["s2"][k]
        print(f"  {k:24s} mean={f(a['mean'])} median={f(a['median'])} "
              f"[{f(a['p25'])},{f(a['p75'])}]")
    print("\n## S2 by rect size — srocc dvifm/attr/refgain/sse "
          "(mean over pairs)")
    for s in SIZES:
        g = result["s2_by_size"][s]
        if g["srocc_dvifm"]["n"]:
            print(f"  {s:3d}px n_pairs={g['srocc_dvifm']['n']:3d}  "
                  f"{f(g['srocc_dvifm']['mean'])} / {f(g['srocc_attr']['mean'])}"
                  f" / {f(g['srocc_refgain']['mean'])} / "
                  f"{f(g['srocc_sse']['mean'])}")
    print("\n## S2 calibrated relative error — median/p90 by size "
          "(dvifm/attr/refgain/sse)")
    for s in SIZES:
        g = result["relerr_by_size"][s]
        if g["dvifm"]["n"]:
            print(f"  {s:3d}px n_rects={g['dvifm']['n']:5d}  "
                  f"{f(g['dvifm']['median'])}/{f(g['dvifm']['p90'])}  "
                  f"{f(g['attr']['median'])}/{f(g['attr']['p90'])}  "
                  f"{f(g['refgain']['median'])}/{f(g['refgain']['p90'])}  "
                  f"{f(g['sse']['median'])}/{f(g['sse']['p90'])}")
    print("\n## S2 calibrated relative error — median/p90 by DVIFM level")
    for l in range(5):
        g = result["relerr_by_level"][str(l)]
        print(f"  level {l}: n={g['n']:5d}  {f(g['median'])}/{f(g['p90'])}")
    print("\n## additivity (64px disjoint pairs)")
    a = result["additivity"]
    print(f"  n={a['n']}  residual mean={f(a['residual']['mean'])} "
          f"median={f(a['residual']['median'])} "
          f"|rel| median={f(a['abs_rel_residual']['median'])} "
          f"p90={f(a['abs_rel_residual']['p90'])} "
          f"joint/sum median={f(a['joint_over_sum']['median'])}")
    print("\n## pyramid leak (frac |Δfield| outside 64px edit)")
    for l in range(5):
        g = result["leak"]["dvifm_levels"][str(l)]
        print(f"  dvifm level {l}: n={g['n']} mean={f(g['mean'])} "
              f"median={f(g['median'])} p90={f(g['p90'])}")
    g = result["leak"]["attr"]
    print(f"  attr density  : n={g['n']} mean={f(g['mean'])} "
          f"median={f(g['median'])} p90={f(g['p90'])}")
    print("\n## S3 — agreement vs judges")
    for k in s3_keys:
        a = o["s3"][k]
        print(f"  {k:24s} mean={f(a['mean'])} median={f(a['median'])} "
              f"[{f(a['p25'])},{f(a['p75'])}]")
    print("\n## S3 by content group (d_ssim2/a_ssim2/d_butter/a_butter/judges)")
    for name, g in result["by_group"].items():
        s3 = g["s3"]
        print(f"  {name:10s} n={g['n']:3d}  {f(s3['srocc_dvifm_ssim2']['mean'])}"
              f" / {f(s3['srocc_attr_ssim2']['mean'])} / "
              f"{f(s3['srocc_dvifm_butter']['mean'])} / "
              f"{f(s3['srocc_attr_butter']['mean'])} / "
              f"{f(s3['srocc_judges']['mean'])}")


if __name__ == "__main__":
    main()
