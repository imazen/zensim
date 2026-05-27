#!/usr/bin/env python3
"""STANDARD zensim bake packing path: pack (per-layer zerobias + f16 + lz4)
THEN calibrate (refit the output spline on the PACKED network).

Why this order: zerobias/quantization preserves RANK (signs intact) but
shifts the network's raw outputs, so a spline fit on the f32 network maps
the packed network's identity output to the wrong dial value (identity drops).
Refitting the spline on the packed network re-anchors identity — at the small
packed size. Per-layer zerobias additionally protects identity-critical layers
(e.g. the per-sample-alpha n_hidden×n_hidden passthrough, which is tiny in
bytes but precision-sensitive): hammer the bulk encoder, leave the last layer
near-lossless.

Standard recipe (recommended): --dtype f16 --zerobias-bulk 0.005 --protect-last
  -> bulk encoder zerobias'd + f16, last layer untouched, spline refit on the
     packed net -> small AND identity-exact.

Usage:
  python3 scripts/v_next/pack_and_calibrate.py <orig.bin> <out.bin> \
      [--dtype f16|f32] [--zerobias-bulk TAU] [--protect-last] [--neg-tail] \
      [--anchor PARQUET] [--verify-cid22 PARQUET]
"""
import sys, os, json, struct, subprocess, tempfile, argparse
import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr

ZP = "/home/lilith/work/zen/zenanalyze/target/release/zenpredict"
PRED = "./target/release/predict_features_with_bake"
ANCHOR = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet"
CID22 = "/mnt/v/zen/zensim-training/2026-05-15-full-features/cid22_features_372col_2026-05-15.parquet"
SPLINE_KEY = "zentrain.output_calibration_spline"


def raw_preds(bake, parquet):
    t = pq.read_table(parquet)
    fcols = sorted([c for c in t.column_names if c.startswith('f') and c[1:].isdigit()],
                   key=lambda c: int(c[1:]))
    feats = np.zeros((t.num_rows, len(fcols)), dtype=np.float32)
    for i, c in enumerate(fcols):
        feats[:, i] = t.column(c).to_numpy().astype(np.float32)
    f = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    f.write(struct.pack('<II', len(fcols), t.num_rows)); f.write(feats.tobytes()); f.close()
    r = subprocess.run([PRED, '--bake', bake, '--bake-post', 'raw', '--features-file', f.name],
                       capture_output=True, text=True, timeout=300)
    os.unlink(f.name)
    if r.returncode != 0:
        sys.exit(f"predict failed: {r.stderr[:400]}")
    return np.array([float(x) for x in r.stdout.split() if x and not x.startswith('#')]), t


def fit_spline_knots(tp, tgt, neg_tail):
    edges = np.percentile(tp, np.linspace(1, 99, 18))
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
    if neg_tail:
        zeros = [i for i, y in enumerate(cy) if y <= 1e-6]
        if len(zeros) > 1:
            drop = set(zeros[:-1])
            cx = [x for i, x in enumerate(cx) if i not in drop]
            cy = [y for i, y in enumerate(cy) if i not in drop]
    return cx, cy


def build_json(insp, dtype, zb_bulk, protect_last):
    """Reconstruct BakeRequest JSON: drop spline, per-layer zerobias, set dtype."""
    n_layers = len(insp["layers"])
    layers = []
    z_counts = []
    for li, l in enumerate(insp["layers"]):
        w = np.array(l["weights"], dtype=np.float64)
        is_last = (li == n_layers - 1)
        tau = 0.0 if (protect_last and is_last) else zb_bulk
        if tau > 0:
            mask = np.abs(w) < tau
            w[mask] = 0.0
            z_counts.append((li, int(mask.sum()), w.size))
        else:
            z_counts.append((li, 0, w.size))
        # last layer kept f32 when protecting; bulk uses requested dtype
        ldtype = "f32" if (protect_last and is_last) else dtype
        layers.append({"in_dim": l["in_dim"], "out_dim": l["out_dim"],
                       "activation": l["activation"], "dtype": ldtype,
                       "weights": w.tolist(), "biases": l["biases"]})
    return layers, z_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orig"); ap.add_argument("out")
    ap.add_argument("--dtype", default="f16")
    ap.add_argument("--zerobias-bulk", type=float, default=0.005)
    ap.add_argument("--protect-last", action="store_true")
    ap.add_argument("--neg-tail", action="store_true")
    a = ap.parse_args()

    insp = json.loads(subprocess.run([ZP, "inspect", a.orig, "--weights"],
                                     capture_output=True, text=True, check=True).stdout)
    layers, zc = build_json(insp, a.dtype, a.zerobias_bulk, a.protect_last)
    print("per-layer zerobias (zeroed/total):", [f"L{li}:{z}/{t}" for li, z, t in zc])

    md = [{"key": e["key"], "type": e["kind"],
           **({"hex": e["value_hex"]} if "value_hex" in e else
              {"text": e["value_text"]} if "value_text" in e else
              {"f32": e["value_f32_array"]})}
          for e in insp.get("metadata", []) if e["key"] != SPLINE_KEY]
    sh = insp.get("schema_hash", 0)
    sh = int(sh, 16) if isinstance(sh, str) and sh.startswith("0x") else int(sh)

    def bake(layers, md, path):
        req = {"schema_hash": sh, "flags": 0, "compressed": True,
               "scaler_mean": insp["scaler_mean"], "scaler_scale": insp["scaler_scale"],
               "layers": layers, "metadata": md}
        jp = path + ".json"; open(jp, "w").write(json.dumps(req))
        subprocess.run([ZP, "bake", jp, path], check=True, capture_output=True)
        os.unlink(jp)

    # 1. packed network WITHOUT spline -> get its tanh-pin outputs
    nosp = a.out + ".nospline.bin"
    bake(layers, md, nosp)
    tp, at = raw_preds(nosp, ANCHOR)
    tgt = at.column('target_score').to_numpy().astype(float)
    print(f"packed tanh-pin range [{tp.min():.4f},{tp.max():.4f}] corr={np.corrcoef(tp,tgt)[0,1]:.4f}")

    # 2. fit spline ON THE PACKED NETWORK (re-anchors identity)
    cx, cy = fit_spline_knots(tp, tgt, a.neg_tail)
    payload = struct.pack('<I', len(cx)) + b"".join(struct.pack('<ff', x, y) for x, y in zip(cx, cy))

    # 3. inject spline into the packed JSON -> final
    md2 = md + [{"key": SPLINE_KEY, "type": "bytes", "hex": payload.hex()}]
    bake(layers, md2, a.out)
    os.unlink(nosp)

    # verify
    spline = PchipInterpolator(cx, cy, extrapolate=True)
    cp, ct = raw_preds(a.out + ".nospline.bin" if False else a.out, CID22)  # final has spline -> raw=post-spline
    mcos = ct.column('human_score').to_numpy().astype(float) * 100.0
    print(f"\nFINAL {a.out}  size={os.path.getsize(a.out)} bytes")
    print(f"  CID22 SROCC (post-spline)={spearmanr(cp,mcos).statistic:.4f}  "
          f"cal pctl p5={np.percentile(cp,5):.1f} p95={np.percentile(cp,95):.1f}")


if __name__ == '__main__':
    main()
