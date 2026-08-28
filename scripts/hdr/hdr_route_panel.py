#!/usr/bin/env python3
"""HDR-route gate panel (amended form, registered in
benchmarks/hdr944_retrain_wave_2026-08-28.md). Computes, for each bake, on
the mc944 t1 VAL leg (census-clean, never trained):
  - per-codec swing FIDELITY = model p50 swing / target p50 swing over the
    pooled q-ladder (bar 0.65..1.5)
  - HG-mono = per-(rendition,codec) fraction of non-decreasing adjacent
    steps, on codecs whose target swing >= 25 (bar >= 0.93)
Forward = predict_features_with_bake (owner); no stats re-derived.

usage: hdr_route_panel.py <bake.bin> [<bake2.bin> ...] [--parquet P]
"""
import argparse, os, struct, subprocess, sys, tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("bakes", nargs="+")
ap.add_argument("--parquet", default="/mnt/v/zen/zensim-training/hdrgrid-mc944-t1-2026-08-27/hdrgrid_mc944_t2_val.parquet")
a = ap.parse_args()

t = pq.read_table(a.parquet)
fcols = [f"feat_{i}" for i in range(944)]
X = np.column_stack([np.asarray(t[c].to_pylist(), np.float32) for c in fcols])
target = np.asarray(t["human_score"].to_pylist(), float) * 100.0
codec = t["codec"].to_pylist(); rend = t["image_path"].to_pylist(); qv = t["q"].to_pylist()
tool = os.environ.get("ZL_PREDICT", str(REPO / "target/release/predict_features_with_bake"))

def swing_and_mono(vals):
    """vals: array aligned to rows. Returns per-codec (swing, mono)."""
    pooled = defaultdict(lambda: defaultdict(list))   # codec -> q -> [v]
    ladders = defaultdict(lambda: defaultdict(list))  # codec -> rendition -> [(q, v)]
    for c, r, q, v in zip(codec, rend, qv, vals):
        pooled[c][q].append(v); ladders[c][r].append((q, v))
    out = {}
    for c, qmap in pooled.items():
        qs = sorted(qmap)
        p50 = {q: float(np.median(qmap[q])) for q in qs}
        swing = p50[qs[-1]] - p50[qs[0]]
        monos = []
        for r, pts in ladders[c].items():
            pts = sorted(pts)
            d = np.diff([v for _, v in pts])
            if len(d): monos.append(float((d >= -0.05).mean()))
        out[c] = (swing, float(np.mean(monos)))
    return out

tgt = swing_and_mono(target)
print("target swings:", {c: round(s, 2) for c, (s, _) in tgt.items()})
for bake in a.bakes:
    with tempfile.NamedTemporaryFile(suffix=".wire", delete=False) as f:
        f.write(struct.pack("<II", X.shape[1], X.shape[0]))
        f.write(X.astype("<f4").tobytes())
        wire = f.name
    try:
        r = subprocess.run([tool, "--bake", bake, "--features-file", wire],
                           capture_output=True, text=True, check=True)
    finally:
        os.unlink(wire)
    pred = np.array([float(v) for v in r.stdout.split()])
    assert len(pred) == len(target), (len(pred), len(target))
    mdl = swing_and_mono(pred)
    cells, ok = [], True
    for c in sorted(tgt):
        ts, _ = tgt[c]; ms, mono = mdl[c]
        fid = ms / ts if ts else float("nan")
        fid_ok = 0.65 <= fid <= 1.5
        mono_ok = mono >= 0.93 if ts >= 25 else None
        ok &= fid_ok and (mono_ok is not False)
        cells.append(f"{c}: fid={fid:.2f}{'✓' if fid_ok else '✗'}"
                     + (f" mono={mono:.3f}{'✓' if mono_ok else '✗'}" if mono_ok is not None else f" mono={mono:.3f}(ungated)"))
    print(f"{os.path.basename(bake):<36} {'PASS' if ok else 'FAIL'}  " + " | ".join(cells))
