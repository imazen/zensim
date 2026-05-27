#!/usr/bin/env python3
"""Characterize zensim's quality-monotonicity on the 2026-05-27 dense
zenjpeg sweep.

READ-ONLY analysis. For every (source, size, knob_tuple) curve, order by q
ascending and compute the q-step reversal rate for both metric_zensim_gpu and
metric_ssim2_gpu: the fraction of adjacent-q steps where the score *decreases*
as q *increases*. A monotone quality metric -> ~0.

Breakdowns: content class (from image_basename prefix), q-band, size class,
and cell (subsampling / progressive / sharp_yuv / effort). Severity = the
magnitude (score-point drop) of each reversal, not just the rate.

Output: a TSV of per-curve stats + console summary tables (which the
companion .md doc transcribes). No retrain, no weight change.

Usage:
    python3 scripts/analyze_compression_monotonicity.py \
        --parquet /mnt/v/zen/picker-dense-full-2026-05-27/parquet/picker_dense_full_zenjpeg.parquet \
        --out-dir /mnt/v/zen/picker-dense-full-2026-05-27/monotonicity_analysis
"""
import argparse
import json
import os
import collections

import numpy as np
import pyarrow.parquet as pq


def content_class(src):
    """Class from the source-name prefix. gen-<class>__... ; hex-named = photo."""
    for p in ("chart", "doc", "line", "mixed", "screen"):
        if src.startswith("gen-" + p):
            return p
    return "photo"  # hex-named real-photo tiles


def q_band(q):
    if q < 30:
        return "low (q<30)"
    if q < 70:
        return "mid (30<=q<70)"
    return "high (q>=70)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cols = ["image_basename", "q", "knob_tuple_json", "size_class",
            "metric_zensim_gpu", "metric_ssim2_gpu"]
    t = pq.read_table(args.parquet, columns=cols)
    n = t.num_rows
    basename = t.column("image_basename").to_pylist()
    q = np.asarray(t.column("q").to_pylist(), dtype=np.int64)
    knob = t.column("knob_tuple_json").to_pylist()
    size_class = t.column("size_class").to_pylist()
    zz = np.asarray(t.column("metric_zensim_gpu").to_pylist(), dtype=np.float64)
    ss = np.asarray(t.column("metric_ssim2_gpu").to_pylist(), dtype=np.float64)
    print(f"loaded {n} rows")

    # group key: (source, size, knob)  -> a quality curve over q
    # image_basename is "<source>@<sizeclass>"; split source off.
    src = [b.split("@", 1)[0] for b in basename]
    curves = collections.defaultdict(list)  # key -> list of row idx
    for i in range(n):
        curves[(src[i], size_class[i], knob[i])].append(i)

    # per-curve reversal accounting
    # A "step" is an adjacent pair on the q-sorted curve. A "reversal" = score
    # goes DOWN as q goes UP. We tally per step, attributing each step's stats
    # to the content class, q-band (of the lower-q endpoint), size, and cell.
    per_class = collections.defaultdict(lambda: dict(steps=0, rev=0, mag=[]))
    per_qband = collections.defaultdict(lambda: dict(steps=0, rev=0, mag=[]))
    per_size = collections.defaultdict(lambda: dict(steps=0, rev=0, mag=[]))
    per_cell = collections.defaultdict(lambda: dict(steps=0, rev=0, mag=[]))
    per_class_ss = collections.defaultdict(lambda: dict(steps=0, rev=0, mag=[]))

    agg = dict(steps=0, rev=0, mag=[])
    agg_ss = dict(steps=0, rev=0, mag=[])

    worst = []  # (drop_magnitude, key, q_lo, q_hi, z_lo, z_hi, ss_lo, ss_hi)
    # per-curve: also record the full-range collapse q_low->q_high (e.g. q30 vs q90)
    curve_rows = []  # for TSV

    for key, idxs in curves.items():
        s, sz, kb = key
        cls = content_class(s)
        order = sorted(idxs, key=lambda i: q[i])
        qs = q[order]
        zs = zz[order]
        sss = ss[order]
        c_steps = c_rev = 0
        c_steps_ss = c_rev_ss = 0
        c_maxdrop = 0.0
        for a, b in zip(range(len(order) - 1), range(1, len(order))):
            dz = zs[b] - zs[a]  # change as q increases
            dss = sss[b] - sss[a]
            qb = q_band(qs[a])
            # zensim
            agg["steps"] += 1
            per_class[cls]["steps"] += 1
            per_qband[qb]["steps"] += 1
            per_size[sz]["steps"] += 1
            per_cell[kb]["steps"] += 1
            c_steps += 1
            if dz < 0:
                drop = -dz
                agg["rev"] += 1
                agg["mag"].append(drop)
                per_class[cls]["rev"] += 1
                per_class[cls]["mag"].append(drop)
                per_qband[qb]["rev"] += 1
                per_qband[qb]["mag"].append(drop)
                per_size[sz]["rev"] += 1
                per_size[sz]["mag"].append(drop)
                per_cell[kb]["rev"] += 1
                per_cell[kb]["mag"].append(drop)
                c_rev += 1
                c_maxdrop = max(c_maxdrop, drop)
            # ssim2 reference
            agg_ss["steps"] += 1
            per_class_ss[cls]["steps"] += 1
            c_steps_ss += 1
            if dss < 0:
                agg_ss["rev"] += 1
                agg_ss["mag"].append(-dss)
                per_class_ss[cls]["rev"] += 1
                per_class_ss[cls]["mag"].append(-dss)
                c_rev_ss += 1
        # full-range collapse: q_low end vs q_high end (does q-high score < q-low?)
        z_collapse = zs[0] - zs[-1]   # >0 means high-q scored LOWER than low-q
        ss_collapse = sss[0] - sss[-1]
        curve_rows.append((s, cls, sz, kb, len(order),
                           c_rev, c_steps, c_maxdrop,
                           float(zs[0]), float(zs[-1]), z_collapse,
                           c_rev_ss, c_steps_ss,
                           float(sss[0]), float(sss[-1]), ss_collapse))
        # worst exemplars by the biggest single reversal drop
        worst.append((c_maxdrop, s, cls, sz, kb, float(zs.min()), float(zs.max()),
                      z_collapse, c_rev, c_steps))

    def pct(d):
        return 100.0 * d["rev"] / d["steps"] if d["steps"] else 0.0

    def magstats(d):
        m = np.asarray(d["mag"]) if d["mag"] else np.asarray([0.0])
        return (float(m.mean()), float(np.median(m)), float(m.max()))

    # ---- write TSV ----
    tsv = os.path.join(args.out_dir, "per_curve_monotonicity.tsv")
    with open(tsv, "w") as f:
        f.write("source\tclass\tsize\tcell\tn_q\t"
                "zen_rev\tzen_steps\tzen_max_drop\tzen_q_lo_score\tzen_q_hi_score\tzen_lo_minus_hi\t"
                "ss_rev\tss_steps\tss_q_lo_score\tss_q_hi_score\tss_lo_minus_hi\n")
        for r in curve_rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"wrote {tsv} ({len(curve_rows)} curves)")

    # ---- summary ----
    out = []
    out.append("=== AGGREGATE q-step reversal rate ===")
    zm = magstats(agg)
    sm = magstats(agg_ss)
    out.append(f"zensim: {pct(agg):.3f}%  ({agg['rev']}/{agg['steps']} steps)  "
               f"drop mean={zm[0]:.3f} median={zm[1]:.3f} max={zm[2]:.3f}")
    out.append(f"ssim2 : {pct(agg_ss):.3f}%  ({agg_ss['rev']}/{agg_ss['steps']} steps)  "
               f"drop mean={sm[0]:.3f} median={sm[1]:.3f} max={sm[2]:.3f}")

    out.append("\n=== zensim reversal rate by CONTENT CLASS (vs ssim2) ===")
    out.append(f"{'class':8s} {'z_rate%':>8s} {'z_meanDrop':>11s} {'z_maxDrop':>10s} "
               f"{'ss_rate%':>9s} {'ss_meanDrop':>12s} {'n_steps':>9s}")
    for cls in sorted(per_class, key=lambda c: -pct(per_class[c])):
        zm = magstats(per_class[cls])
        sm = magstats(per_class_ss[cls])
        out.append(f"{cls:8s} {pct(per_class[cls]):8.3f} {zm[0]:11.3f} {zm[2]:10.2f} "
                   f"{pct(per_class_ss[cls]):9.3f} {sm[0]:12.4f} {per_class[cls]['steps']:9d}")

    out.append("\n=== zensim reversal rate by Q-BAND (lower-q endpoint) ===")
    out.append(f"{'qband':16s} {'z_rate%':>8s} {'z_meanDrop':>11s} {'z_maxDrop':>10s} {'n_steps':>9s}")
    for qb in ["low (q<30)", "mid (30<=q<70)", "high (q>=70)"]:
        d = per_qband[qb]
        zm = magstats(d)
        out.append(f"{qb:16s} {pct(d):8.3f} {zm[0]:11.3f} {zm[2]:10.2f} {d['steps']:9d}")

    out.append("\n=== zensim reversal rate by SIZE CLASS ===")
    out.append(f"{'size':10s} {'z_rate%':>8s} {'z_meanDrop':>11s} {'z_maxDrop':>10s} {'n_steps':>9s}")
    szorder = sorted(per_size, key=lambda s: int(s.replace("sz", "")))
    for sz in szorder:
        d = per_size[sz]
        zm = magstats(d)
        out.append(f"{sz:10s} {pct(d):8.3f} {zm[0]:11.3f} {zm[2]:10.2f} {d['steps']:9d}")

    out.append("\n=== zensim reversal rate by CELL (knob tuple), sorted worst-first ===")
    out.append(f"{'rate%':>7s} {'meanDrop':>9s} {'maxDrop':>8s}  cell")
    for kb in sorted(per_cell, key=lambda c: -pct(per_cell[c])):
        d = per_cell[kb]
        zm = magstats(d)
        out.append(f"{pct(d):7.3f} {zm[0]:9.3f} {zm[2]:8.2f}  {kb}")

    # cell axis marginals
    def axis_rate(field, valfn):
        groups = collections.defaultdict(lambda: dict(steps=0, rev=0, mag=[]))
        for kb, d in per_cell.items():
            kv = json.loads(kb)
            g = groups[str(valfn(kv))]
            g["steps"] += d["steps"]
            g["rev"] += d["rev"]
            g["mag"].extend(d["mag"])
        return groups
    out.append("\n=== cell-axis marginals (zensim reversal rate) ===")
    for name, fn in [("subsampling", lambda k: k["subsampling"]),
                     ("progressive", lambda k: k["progressive"]),
                     ("sharp_yuv", lambda k: k["sharp_yuv"]),
                     ("effort", lambda k: k["effort"])]:
        groups = axis_rate(name, fn)
        out.append(f"-- {name} --")
        for val in sorted(groups):
            d = groups[val]
            zm = magstats(d)
            out.append(f"   {name}={val:8s} rate={pct(d):7.3f}%  meanDrop={zm[0]:.3f} maxDrop={zm[2]:.2f}")

    # full-range collapse: fraction of curves where q_HIGH end < q_LOW end
    n_curves = len(curve_rows)
    z_collapsed = sum(1 for r in curve_rows if r[10] > 0)   # zen_lo_minus_hi > 0
    ss_collapsed = sum(1 for r in curve_rows if r[15] > 0)
    out.append("\n=== full-curve collapse (q-HIGH end scored LOWER than q-LOW end) ===")
    out.append(f"zensim: {z_collapsed}/{n_curves} curves ({100.0*z_collapsed/n_curves:.2f}%)")
    out.append(f"ssim2 : {ss_collapsed}/{n_curves} curves ({100.0*ss_collapsed/n_curves:.2f}%)")
    # collapse by class
    out.append("collapse by class (zensim):")
    cls_coll = collections.Counter()
    cls_tot = collections.Counter()
    for r in curve_rows:
        cls_tot[r[1]] += 1
        if r[10] > 0:
            cls_coll[r[1]] += 1
    for cls in sorted(cls_tot, key=lambda c: -(cls_coll[c] / cls_tot[c])):
        out.append(f"   {cls:8s} {cls_coll[cls]}/{cls_tot[cls]} ({100.0*cls_coll[cls]/cls_tot[cls]:.1f}%)")

    # worst exemplars by max single-step drop
    out.append("\n=== WORST exemplars (largest single-step zensim drop) ===")
    worst.sort(key=lambda w: -w[0])
    out.append(f"{'maxDrop':>8s} {'class':6s} {'size':7s} {'zmin':>8s} {'zmax':>8s} "
               f"{'lo-hi':>8s} {'revs':>5s}  source / cell")
    for w in worst[:25]:
        md, s, cls, sz, kb, zmn, zmx, zcol, crev, cst = w
        out.append(f"{md:8.2f} {cls:6s} {sz:7s} {zmn:8.2f} {zmx:8.2f} {zcol:8.2f} "
                   f"{crev:3d}/{cst:<2d}  {s} | {kb}")

    summary = "\n".join(out)
    print(summary)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\nwrote {os.path.join(args.out_dir, 'summary.txt')}")


if __name__ == "__main__":
    main()
