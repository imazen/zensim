#!/usr/bin/env python3
"""H-HID TERMINAL READ — run ONCE at balance-campaign end (registered:
benchmarks/balance_campaign_2026-08-28.md). Forwards finalists over the
sealed hidden KADIS panel (never in any train slice; sha-manifested) and
reports srocc_signed vs each of the 7 independent metric targets + the
H-MAXIMIN comparison. Forward = predict_features_with_bake (owner);
stats = zen_stats.panel_batch (owner).

usage: hidden_terminal_read.py <bake.bin|name=path> ...
"""
import os, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib import zen_stats  # noqa: E402

PANEL = "/mnt/v/zen/zensim-training/balance-hidden-2026-08-28/kadis_hidden_20k.parquet"
TARGETS = ["score_ssim2_gpu", "score_butteraugli_max_gpu", "score_cvvdp_cpu_imazen_v0_1_0",
           "score_iwssim_gpu", "score_dssim_gpu", "score_zensim_gpu"]
NEG = {"score_butteraugli_max_gpu", "score_dssim_gpu"}  # distances: negate for quality orientation

t = pq.read_table(PANEL)
X = np.column_stack([np.asarray(t[f"f{i}"].to_pylist(), np.float32) for i in range(944)])
ys = {k: np.asarray(t[k].to_pylist(), float) * (-1 if k in NEG else 1) for k in TARGETS}
tool = os.environ.get("ZL_PREDICT", str(REPO / "target/release/predict_features_with_bake"))
print(f"hidden panel: {t.num_rows} rows (touch-once read at {__import__('datetime').datetime.utcnow().isoformat()}Z)")
rows = []
for spec in sys.argv[1:]:
    name, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
    with tempfile.NamedTemporaryFile(suffix=".wire", delete=False) as f:
        f.write(struct.pack("<II", X.shape[1], X.shape[0])); f.write(X.astype("<f4").tobytes())
        wire = f.name
    try:
        r = subprocess.run([tool, "--bake", path, "--features-file", wire], capture_output=True, text=True, check=True)
    finally:
        os.unlink(wire)
    pred = [float(v) for v in r.stdout.split()]
    assert len(pred) == t.num_rows
    stats = zen_stats.panel_batch([(k, pred, ys[k].tolist()) for k in TARGETS], stats="srocc")
    got = {}
    for label, row in (stats.items() if isinstance(stats, dict) else [(x["label"], x) for x in stats]):
        got[label] = row["srocc"] if isinstance(row, dict) else row
    rows.append((name, got))
hdr = f"{'candidate':<26}" + "".join(f"{k.replace('score_','').replace('_gpu',''):>16}" for k in TARGETS) + f"{'MIN':>8}"
print(hdr)
for name, got in rows:
    vals = [got[k] for k in TARGETS]
    print(f"{name:<26}" + "".join(f"{v:>16.4f}" for v in vals) + f"{min(vals):>8.4f}")
