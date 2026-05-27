#!/usr/bin/env python3
"""Re-calibrate v47-strict's output dial: replace the degenerate auto-spline
with a monotone PCHIP fit on the TRUE pre-spline (tanh-pin) output over the
multiband anchor (per-row target_score in [0,100]).

Workflow (the established V10 spline-retrofit path):
  1. strip the old spline -> nospline bake (preserves per-sample-α + tanh-pin)
  2. predict --bake-post raw on the nospline bake = tanh-pin output (pre-spline)
  3. fit monotone PCHIP: tanh-pin quantiles -> target_score quantiles
  4. reconstruct JSON (inspect --weights), drop old spline, inject new one (hex)
  5. re-bake via `zenpredict bake`

Rank-invariant by construction (monotone spline) => SROCC unchanged.
The empirical bake_verdict (pooled G1 + panel + blur ladder) is the verdict.

Usage: python3 scripts/v_next/recal_v47_dial.py <orig_bake> <out_bake>
"""
import sys, subprocess, struct, os, json, tempfile
import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr

ZP = "/home/lilith/work/zen/zenanalyze/target/release/zenpredict"
PRED = "./target/release/predict_features_with_bake"
ANCHOR = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet"
CID22 = "/mnt/v/zen/zensim-training/2026-05-15-full-features/cid22_features_372col_2026-05-15.parquet"


def feats_to_tmp(parquet):
    t = pq.read_table(parquet)
    fcols = sorted([c for c in t.column_names if c.startswith('f') and c[1:].isdigit()],
                   key=lambda c: int(c[1:]))
    n = t.num_rows
    feats = np.zeros((n, len(fcols)), dtype=np.float32)
    for i, c in enumerate(fcols):
        feats[:, i] = t.column(c).to_numpy().astype(np.float32)
    f = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    f.write(struct.pack('<II', len(fcols), n)); f.write(feats.tobytes()); f.close()
    return f.name, t


def raw_preds(bake, parquet):
    tmp, t = feats_to_tmp(parquet)
    r = subprocess.run([PRED, '--bake', bake, '--bake-post', 'raw', '--features-file', tmp],
                       capture_output=True, text=True, timeout=300)
    os.unlink(tmp)
    if r.returncode != 0:
        sys.exit(f"predict failed: {r.stderr[:400]}")
    return np.array([float(x) for x in r.stdout.split() if x and not x.startswith('#')]), t


def main():
    orig, out = sys.argv[1], sys.argv[2]
    nospline = out + ".nospline.bin"

    # 1. strip the old spline
    subprocess.run(["python3", "scripts/v_next/strip_spline_metadata.py",
                    "--bake", orig, "--out", nospline, "--zenpredict-bin", ZP], check=True)

    # 2. tanh-pin (pre-spline) preds on the anchor
    tp, at = raw_preds(nospline, ANCHOR)
    tgt = at.column('target_score').to_numpy().astype(float)
    print(f"\ntanh-pin (pre-spline) on anchor: range [{tp.min():.5f},{tp.max():.5f}] "
          f"spread={tp.max()-tp.min():.5f}, corr w/ target={np.corrcoef(tp,tgt)[0,1]:.4f}")

    # 3. monotone PCHIP: tanh-pin quantiles -> target_score
    n_knots = 18
    edges = np.percentile(tp, np.linspace(1, 99, n_knots))
    kx, ky = [], []
    lo = tp < edges[0]
    if lo.sum() >= 2:
        kx.append(float(np.median(tp[lo]))); ky.append(float(np.median(tgt[lo])))
    for i in range(len(edges) - 1):
        m = (tp >= edges[i]) & (tp < edges[i + 1])
        if m.sum() >= 2:
            kx.append(float(np.median(tp[m]))); ky.append(float(np.median(tgt[m])))
    hi = tp >= edges[-1]
    if hi.sum() >= 2:
        kx.append(float(np.median(tp[hi]))); ky.append(float(np.median(tgt[hi])))
    cx, cy = [kx[0]], [ky[0]]
    for i in range(1, len(kx)):
        if kx[i] > cx[-1] + 1e-7 and ky[i] >= cy[-1]:
            cx.append(kx[i]); cy.append(ky[i])
    # NEG-TAIL: if the bottom has multiple dial=0 knots, PCHIP extrapolates
    # FLAT (slope 0) below them -> heavy corruption clamps at 0, killing the
    # negative-tail resolution. Drop all but the HIGHEST-x dial=0 knot so the
    # endpoint slope is the (steep, positive) bottom-segment slope -> PCHIP
    # extrapolates NEGATIVE below the anchor's worst. The honest [0,100] range
    # is unchanged; only worse-than-honest-q20 inputs go below 0.
    if os.environ.get("NEG_TAIL") == "1":
        zeros = [i for i, y in enumerate(cy) if y <= 1e-6]
        if len(zeros) > 1:
            keep = zeros[-1]  # highest-x zero knot
            drop = set(zeros[:-1])
            cx = [x for i, x in enumerate(cx) if i not in drop]
            cy = [y for i, y in enumerate(cy) if i not in drop]
    print(f"fit {len(cx)} monotone knots (tanh-pin -> dial):")
    for x, y in zip(cx, cy):
        print(f"  {x:11.6f} -> {y:7.3f}")
    if len(cx) < 3:
        sys.exit("DEGENERATE (<3 knots) — sibling-ship instead.")
    spline = PchipInterpolator(cx, cy, extrapolate=True)

    # verify rank-invariance on CID22 (pre-spline tanh-pin -> calibrate)
    cp, ct = raw_preds(nospline, CID22)
    mcos = ct.column('human_score').to_numpy().astype(float) * 100.0
    cal = spline(cp)
    print(f"\nCID22 rank check: raw SROCC={spearmanr(cp,mcos).statistic:.4f}  "
          f"calibrated SROCC={spearmanr(cal,mcos).statistic:.4f}  "
          f"cal pctl p5={np.percentile(cal,5):.1f} p50={np.percentile(cal,50):.1f} p95={np.percentile(cal,95):.1f}")

    # 4. payload + reconstruct JSON with new spline injected
    payload = struct.pack('<I', len(cx))
    for x, y in zip(cx, cy):
        payload += struct.pack('<ff', float(x), float(y))
    payload_hex = payload.hex()

    insp = json.loads(subprocess.run([ZP, "inspect", orig, "--weights"],
                                     capture_output=True, text=True, check=True).stdout)
    out_layers = [{"in_dim": l["in_dim"], "out_dim": l["out_dim"], "activation": l["activation"],
                   "dtype": l["dtype"], "weights": l["weights"], "biases": l["biases"]}
                  for l in insp["layers"]]
    md = []
    for e in insp.get("metadata", []):
        if e["key"] == "zentrain.output_calibration_spline":
            continue
        item = {"key": e["key"], "type": e["kind"]}
        if "value_hex" in e:
            item["hex"] = e["value_hex"]
        elif "value_text" in e:
            item["text"] = e["value_text"]
        elif "value_f32_array" in e:
            item["f32"] = e["value_f32_array"]
        md.append(item)
    md.append({"key": "zentrain.output_calibration_spline", "type": "bytes", "hex": payload_hex})

    sh = insp.get("schema_hash", 0)
    sh = int(sh, 16) if isinstance(sh, str) and sh.startswith("0x") else int(sh)
    req = {"schema_hash": sh, "flags": 0, "compressed": True,
           "scaler_mean": insp["scaler_mean"], "scaler_scale": insp["scaler_scale"],
           "layers": out_layers, "metadata": md}
    jp = out + ".tmp.json"
    open(jp, "w").write(json.dumps(req))
    subprocess.run([ZP, "bake", jp, out], check=True)
    os.unlink(jp); os.unlink(nospline)
    print(f"\nwrote recalibrated bake -> {out}")


if __name__ == '__main__':
    main()
