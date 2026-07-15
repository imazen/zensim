#!/usr/bin/env python3
"""Independent-reference knob consistency AT SCALE — the open gap, closed with data
we already have (no new GPU). Joins the fill4 4-metric sidecar (cvvdp/butteraugli/
dssim/iwssim on the picker corpus, keyed by encoded_filename, 4.2M encodes) onto the
re-forwarded CURRENT-B/A ab_rescored q-ladders (~950k rows, 4 codecs).

For each knob M in {B, A, ssim2} and each independent reference REF, eta^2(REF|M-decile)
= fraction of REF variance pinned down by M's level (higher = M is a better knob for
that reference). butteraugli + dssim are references B was NEVER trained on (cleanest
independence; B's cid head saw an ssim2+cvvdp mix, so cvvdp is semi-circular for B)."""
import sys
from pathlib import Path

import numpy as np, pandas as pd, pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canonical_corpus"))
from join_safety import safe_metric_join  # noqa: E402

RF = "/mnt/v/output/zensim-multicodec-probe/knob_reforward"
AB = "/mnt/v/output/zensim/ab_rescored_2026-07-05"
FILL4 = "/mnt/v/datasets/fill4-6codec-2026-07-01/fill4metrics_sidecar_2026-07-02.parquet"
CODECS = ["zenjpeg_lossy", "zenavif_lossy", "zenjxl_lossy", "zenwebp_lossy"]
REFS = ["score_cvvdp", "score_butteraugli", "score_dssim", "score_iwssim"]
KNOBS = [("b", "B"), ("a", "A"), ("ssim2", "ssim2")]

print("loading fill4 sidecar (4.2M rows)...")
f4 = pq.read_table(FILL4, columns=["encoded_filename"] + REFS).to_pandas()

def eta2(metric, ref, nbins=10):
    m = ~(np.isnan(metric) | np.isnan(ref))
    metric, ref = metric[m], ref[m]
    if len(ref) < nbins * 5 or np.var(ref) == 0:
        return np.nan
    order = np.argsort(metric, kind="mergesort")
    grand, n, between = ref.mean(), len(ref), 0.0
    for b in np.array_split(order, nbins):
        between += len(b) * (ref[b].mean() - grand) ** 2
    return (between / n) / np.var(ref)

rows, pooled = [], {}
for codec in CODECS:
    rf = pq.read_table(f"{RF}/{codec}.parquet").to_pandas()
    ab = pq.read_table(f"{AB}/{codec}.b.parquet", columns=["encoded_filename", "score_ssim2"]).to_pandas()
    assert np.allclose(rf["ssim2"].to_numpy(), ab["score_ssim2"].to_numpy(), atol=1e-6), codec
    rf["encoded_filename"] = ab["encoded_filename"].to_numpy()
    # Routed through the join_safety owner (the CI grep-gate's whole point).
    # `encoded_filename` IS the full per-pair key: it names one encode, so an
    # inner merge cannot ref-broadcast — verified 2026-07-15, the sidecar is
    # 4,214,382 rows / 4,214,382 unique keys. safe_metric_join re-checks that
    # uniqueness on every run rather than trusting this comment, which is the
    # difference between a guard and a promise. One call per ref because it
    # attaches one metric column at a time.
    df = rf
    for ref in REFS:
        df = safe_metric_join(df, f4, ["encoded_filename"], ref, how="inner")
    cov = len(df) / len(rf)
    for ref in REFS:
        rv = df[ref].to_numpy()
        rec = {"codec": codec.replace("zen", "").replace("_lossy", ""), "ref": ref.replace("score_", ""),
               "n": len(df), "cov": cov}
        for col, lab in KNOBS:
            e = eta2(df[col].to_numpy(), rv)
            rec[f"eta2_{lab}"] = e
            pooled.setdefault((ref, lab), []).append((df[col].to_numpy(), rv))
        rows.append(rec)

R = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:.4f}")
print("\n=== eta^2(REF | knob-decile) per codec — higher = knob pins that reference better ===")
print(R.to_string(index=False))

print("\n=== POOLED across all 4 codecs (the universal-knob test) ===")
print(f"{'ref':14} {'eta2_B':>8} {'eta2_A':>8} {'eta2_ssim2':>11} {'n':>9}  winner")
for ref in REFS:
    r = ref.replace("score_", "")
    es = {}
    for lab in ["B", "A", "ssim2"]:
        cat_m = np.concatenate([m for m, _ in pooled[(ref, lab)]])
        cat_r = np.concatenate([rr for _, rr in pooled[(ref, lab)]])
        es[lab] = eta2(cat_m, cat_r)
    n = sum(len(m) for m, _ in pooled[(ref, "B")])
    win = max(es, key=es.get)
    print(f"{r:14} {es['B']:8.4f} {es['A']:8.4f} {es['ssim2']:11.4f} {n:9d}  {win}"
          + ("  (indep of B)" if r in ("butteraugli", "dssim") else "  (semi-circular for B)"))
