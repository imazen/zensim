#!/usr/bin/env python3
"""Cross-metric CONSENSUS outlier check — adjudicate a bake against several
INDEPENDENT reference metrics at once, so a single noisy metric can't produce a
false alarm and a real blind spot is caught only when the independents AGREE.

This is the honest form of "sanity-check against other metrics": rather than
trust one reference (which may share the bake's biases or have floor artifacts —
see bake_outlier_gate.py's ssim2-floor caveat), we require a QUORUM of unrelated
metrics to agree that the bake is out of line before flagging a row.

Per row, on a corpus that carries the 372 features AND >=2 independent metric
columns (e.g. KADIS-700k-gpu: butteraugli, ssim2, cvvdp, dssim, iwssim):
  1. rank-normalize the bake's dial and each reference metric to [0,1]
     (polarity-aligned: distance metrics like butteraugli are inverted so
      higher = better everywhere);
  2. consensus_r = median of the reference ranks; spread = their IQR;
  3. flag a row when |bake_rank - consensus_r| > T>  AND the references agree
     among themselves (spread < S) — i.e. the bake is the odd one out, not the
     metrics disagreeing with each other.
Reports overall rank agreement of the bake with each metric, and the worst
high-confidence blind spots (bake far from a tight independent consensus).

  usage: xmetric_consensus.py --bake B.bin --corpus kadis.parquet \
             --metrics score_ssim2_gpu:+ score_butteraugli_max_gpu:- \
                       score_cvvdp_cpu_imazen_v0_1_0:+
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.abspath("."))
# reuse the faithful forward from the outlier gate (same transform code as the bakes)
import importlib.util  # noqa: E402
_g = importlib.util.spec_from_file_location(
    "gate", Path(__file__).parent / "bake_outlier_gate.py")
gate = importlib.util.module_from_spec(_g)
_g.loader.exec_module(gate)


def rank01(v):
    return (rankdata(v) - 1) / max(1, len(v) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--metrics", nargs="+", required=True,
                    help="COL:+ (quality, higher=better) or COL:- (distance, higher=worse)")
    ap.add_argument("--flag-thresh", type=float, default=0.30,
                    help="|bake_rank - consensus_rank| to flag (rank units, 0..1)")
    ap.add_argument("--agree-iqr", type=float, default=0.20,
                    help="max IQR among reference ranks for the consensus to count as tight")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    bk = gate.load_bake(a.bake)
    t = pq.read_table(a.corpus)
    cols = sorted([c for c in t.schema.names
                   if (c.startswith("f") and c[1:].isdigit()) or c.startswith("feat_")],
                  key=lambda c: int("".join(ch for ch in c if ch.isdigit())))[:372]
    F = np.column_stack([np.asarray(t[c], dtype=float) for c in cols])
    names = (t["source_filename"].to_pylist() if "source_filename" in t.schema.names
             else t["ref_basename"].to_pylist() if "ref_basename" in t.schema.names
             else [str(i) for i in range(len(F))])

    # odd-dim corruption sanity (the w11 GPU-extractor signature): a masked/IW
    # feature bit-constant across a whole source's rows is extraction garbage.
    # Cheap global proxy: fraction of rows whose masked block (f228..f299) has
    # any value > 5 (corpora top ~2). Report it; don't silently trust.
    masked = F[:, 228:300]
    susp = float((masked > 5).any(axis=1).mean())

    dial = gate.dial(bk, gate.raw_forward(bk, F))
    braw = rank01(dial)

    refs = []
    print(f"=== xmetric consensus: {os.path.basename(a.bake)} on {os.path.basename(a.corpus)} "
          f"(n={len(F):,}) ===")
    print(f"feature odd-dim-corruption proxy (masked>5): {100*susp:.3f}% of rows"
          f"{'  <-- INVESTIGATE (GPU odd-dim?)' if susp > 0.01 else '  (clean)'}")
    for spec in a.metrics:
        col, pol = spec.rsplit(":", 1)
        v = np.asarray(t[col], dtype=float)
        ok = np.isfinite(v)
        r = rank01(v[ok] if not ok.all() else v)
        if pol == "-":
            r = 1.0 - r
        full = np.full(len(F), np.nan)
        full[ok] = r if not ok.all() else r
        refs.append((col, full))
        sr = spearmanr(dial[ok], v[ok]).statistic
        print(f"  B vs {col:<34} SROCC {abs(sr):.4f} (n={int(ok.sum()):,})")

    R = np.vstack([r for _, r in refs])           # (n_metrics, n_rows), polarity-aligned ranks
    valid = np.isfinite(R).all(axis=0)
    cons = np.nanmedian(R, axis=0)
    q75, q25 = np.nanpercentile(R, 75, axis=0), np.nanpercentile(R, 25, axis=0)
    iqr = q75 - q25
    dev = np.abs(braw - cons)
    tight = iqr < a.agree_iqr
    flag = valid & tight & (dev > a.flag_thresh)
    nflag = int(flag.sum())
    print(f"\nhigh-confidence B outliers (|B - tight independent consensus| > {a.flag_thresh}): "
          f"{nflag} ({100*nflag/max(1,valid.sum()):.3f}% of {int(valid.sum()):,} multi-metric rows)")
    order = np.argsort(-(dev * flag))[:a.top]
    hdr = "name | B_rank | consensus | IQR | " + " ".join(c.replace("score_", "")[:10] for c, _ in refs)
    print("  worst (B rank vs tight consensus — B is the odd one out):")
    print("   ", hdr)
    for i in order:
        if not flag[i]:
            continue
        mv = " ".join(f"{R[j, i]:.2f}" for j in range(len(refs)))
        print(f"    {str(names[i])[:30]:<30} {braw[i]:.2f}   {cons[i]:.2f}   {iqr[i]:.2f}   {mv}")

    # distortion-type breakdown: is the divergence a coherent content/distortion
    # CLASS (design difference) or scattered (a bug)? Concentration => former.
    dcol = next((c for c in ("dist_name", "dist_type", "codec") if c in t.schema.names), None)
    if dcol is not None:
        dv = np.asarray(t[dcol].to_pylist(), dtype=object)
        allc = {}
        flc = {}
        for d in dv:
            allc[d] = allc.get(d, 0) + 1
        for d in dv[flag]:
            flc[d] = flc.get(d, 0) + 1
        print(f"\n  flagged-row {dcol} concentration (flagged% within type; base rate {100*nflag/max(1,valid.sum()):.1f}%):")
        rows = sorted(((flc.get(d, 0) / allc[d], d, flc.get(d, 0), allc[d]) for d in allc), reverse=True)
        for rate, d, f_, tot in rows[:14]:
            bar = "#" * int(rate * 40)
            print(f"    {str(d)[:26]:<26} {100*rate:>5.1f}% ({f_:>5}/{tot:<6}) {bar}")
    # B agreement with each metric among the flagged rows — which metric does B side WITH?
    print("\n  on flagged rows, mean |B - each ref| (LOW = B sides with that metric):")
    for j, (col, _) in enumerate(refs):
        md = float(np.nanmean(np.abs(braw[flag] - R[j, flag])))
        print(f"    {col.replace('score_',''):<30} {md:.3f}")


if __name__ == "__main__":
    main()
