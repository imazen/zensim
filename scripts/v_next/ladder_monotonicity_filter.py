#!/usr/bin/env python3
"""All-metrics-agree compression-ladder monotonicity filter.

For synthetic (non-human) compression training data, an RD ladder — the set
of encodes of one (image, codec) at increasing byte cost — should buy MORE
perceptual quality as it spends more bytes. A ladder where more bytes yields
LESS quality is a noisy training label (bad encoder params, a metric dead
zone, quantizer weirdness) and is a candidate to drop.

The refinement (user, 2026-07-14): **only drop a ladder when ALL metrics
agree its quality is non-monotonic in bytes** — if only one metric (e.g.
ssim2) sees a reversal while the others see a clean ramp, that's that
metric's noise, not a broken encode, so keep it. Filtering on a single
metric over-drops; requiring unanimous agreement filters only the genuinely
broken ladders.

This tool measures both thresholds so the over-filtering the single-metric
rule would cause is quantified, and writes a per-ladder keep/drop table.

  usage: ladder_monotonicity_filter.py SIDECAR.parquet [--eps 0.5]
         [--out-keep keep.parquet] [--sort bytes|q]

SIDECAR must carry: image_path, codec, encoded_bytes, and one or more of the
score_* metric columns below. Ladders are grouped by (image_path, codec) and
sorted by encoded_bytes ascending (more bytes = should be higher quality).
"""
import argparse
import os
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

# metric column -> polarity: +1 = higher is better, -1 = lower is better.
METRIC_POLARITY = {
    "score_zensim_gpu": +1,
    "score_ssim2": +1,
    "score_ssim2_gpu": +1,
    "score_iwssim_gpu": +1,
    "score_cvvdp_imazen_v0_0_1": +1,
    "score_cvvdp_cpu_imazen_v0_1_0": +1,
    "score_butteraugli_max_gpu": -1,
    "score_butteraugli_pnorm3_gpu": -1,
    "score_dssim": -1,
}


def ladder_nonmono(quality_dir_values, eps):
    """A quality-direction sequence (already sign-flipped so higher=better),
    ordered by increasing bytes, is non-monotonic if any adjacent step drops
    by more than `eps` (a material RD reversal). Returns (is_nonmono,
    worst_drop)."""
    d = np.diff(quality_dir_values)
    worst = float(-d.min()) if d.size and d.min() < 0 else 0.0
    return bool(d.size and (d < -eps).any()), worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sidecar")
    ap.add_argument("--eps", type=float, default=0.5,
                    help="material RD-reversal tolerance in each metric's own units "
                         "(default 0.5; scaled per-metric by its IQR)")
    ap.add_argument("--sort", choices=["bytes", "q"], default="bytes")
    ap.add_argument("--out-keep", help="write a parquet of kept rows")
    ap.add_argument("--out-table", help="write the per-ladder keep/drop TSV")
    a = ap.parse_args()

    t = pq.read_table(a.sidecar)
    cols = set(t.column_names)
    metrics = [m for m in METRIC_POLARITY if m in cols]
    if not metrics:
        raise SystemExit(f"no known score_* metric columns in {a.sidecar}")
    print(f"[filter] {t.num_rows} rows; metrics: {metrics}")

    img = t["image_path"].to_pylist()
    codec = t["codec"].to_pylist()
    ebytes = np.asarray(t["encoded_bytes"].to_pylist(), dtype=float)
    sortkey = ebytes
    if a.sort == "q" and "q" in cols:
        try:
            sortkey = np.asarray([float(x) for x in t["q"].to_pylist()], dtype=float)
        except (TypeError, ValueError):
            sortkey = ebytes
    # quality-direction matrix: each metric flipped so higher = better.
    qmat = {}
    scale = {}
    for m in metrics:
        v = np.asarray(t[m].to_pylist(), dtype=float) * METRIC_POLARITY[m]
        qmat[m] = v
        # per-metric eps scaled by robust spread so `--eps` is comparable
        # across metrics on wildly different unit scales.
        iqr = np.subtract(*np.percentile(v[np.isfinite(v)], [75, 25])) if np.isfinite(v).any() else 1.0
        scale[m] = max(iqr, 1e-9)

    # group row indices by (image, codec)
    groups = {}
    for i in range(len(img)):
        groups.setdefault((img[i], codec[i]), []).append(i)

    n_ladders = 0
    per_metric_nonmono = {m: 0 for m in metrics}
    size_nonmono = 0
    ssim2_alone_drop = 0            # ssim2 (or first ↑ metric) says non-mono
    all_agree_drop = 0             # ALL metrics say non-mono → the filter
    any_drop = 0                   # ANY metric says non-mono (the over-filter)
    keep_idx = []
    table_rows = []
    ssim_key = "score_ssim2" if "score_ssim2" in metrics else ("score_ssim2_gpu" if "score_ssim2_gpu" in metrics else metrics[0])

    for (im, cd), idxs in groups.items():
        if len(idxs) < 3:
            keep_idx.extend(idxs)      # too short to judge — keep
            continue
        order = sorted(idxs, key=lambda k: sortkey[k])
        n_ladders += 1
        # size monotonic? (bytes should strictly increase along the sort; if
        # sorting by bytes this is trivially true, so only meaningful for q-sort)
        b = ebytes[order]
        size_bad = bool(np.diff(b).min() < 0) if b.size > 1 else False
        size_nonmono += size_bad
        flags = {}
        worst = {}
        for m in metrics:
            nm, w = ladder_nonmono(qmat[m][order], a.eps * scale[m])
            flags[m] = nm
            worst[m] = w
            per_metric_nonmono[m] += nm
        ssim2_bad = flags.get(ssim_key, False)
        all_bad = all(flags[m] for m in metrics)
        any_bad = any(flags[m] for m in metrics)
        ssim2_alone_drop += ssim2_bad
        all_agree_drop += all_bad
        any_drop += any_bad
        # THE filter: drop only when all metrics agree it's non-monotonic.
        if all_bad:
            table_rows.append((im, cd, len(idxs), "DROP",
                               ";".join(f"{m}:{worst[m]:.2f}" for m in metrics)))
        else:
            keep_idx.extend(idxs)
            table_rows.append((im, cd, len(idxs), "keep",
                               ";".join(f"{m}:{'x' if flags[m] else '-'}" for m in metrics)))

    print(f"\n[filter] ladders (≥3 rungs): {n_ladders}")
    print("  per-metric non-monotone ladders:")
    for m in metrics:
        print(f"    {m:32s} {per_metric_nonmono[m]:6d}  ({100*per_metric_nonmono[m]/max(1,n_ladders):.1f}%)")
    print(f"\n  {ssim_key} ALONE would drop:  {ssim2_alone_drop:6d}  ({100*ssim2_alone_drop/max(1,n_ladders):.1f}%)")
    print(f"  ANY-metric would drop:          {any_drop:6d}  ({100*any_drop/max(1,n_ladders):.1f}%)   <- the over-filter")
    print(f"  ALL-metrics-AGREE drops:        {all_agree_drop:6d}  ({100*all_agree_drop/max(1,n_ladders):.1f}%)   <- THE filter")
    over = ssim2_alone_drop - all_agree_drop
    print(f"\n  single-metric over-filtering avoided: {over} ladders "
          f"({100*over/max(1,ssim2_alone_drop):.1f}% of the single-metric drops were metric noise, not broken encodes)")

    if a.out_keep:
        kept = t.take(pa.array(sorted(keep_idx)))
        pq.write_table(kept, a.out_keep, compression="zstd")
        print(f"\n  kept {kept.num_rows}/{t.num_rows} rows -> {a.out_keep}")
    if a.out_table:
        with open(a.out_table, "w") as f:
            f.write("image_path\tcodec\tn_rungs\tverdict\tdetail\n")
            for r in table_rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        print(f"  per-ladder table -> {a.out_table}")


if __name__ == "__main__":
    main()
