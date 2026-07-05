#!/usr/bin/env python3
"""w11: fix the Profile-B linear pick's webp feature-OOD blindness (2026-07-05).

MEASURED BUG (commits 2e655672 + 64e432fb): 3/24 webp dial ladders score
-80..-12 under `lp_ens-Pline-cid80-anchored-f16` while ssim2 says 61-91.
DIAGNOSIS (this script's `diagnose`): the cid component head
(`hdrmix-lasso0.002-raw` @ tau0, fit on 7,410 HDR-JXL rows only) carries
w[f235] = -0.0034; f235 (masked-block scale-0 Y flatness-weighted feature)
sits at 46..275 on the trio's webp encodes — z = +927..+5,497 under the
HDR-mix scaler (three orders of magnitude beyond BOTH hdr_v3mix max 0.27
and safesyn max 0.64) — contributing -3.1..-18.5 to the raw pred. The raw
pred falls below the dial spline's bottom knot and the deliberately-uncapped
downward extrapolation produces the -80s. The tau0 head explodes on ~0.95%
of bigcodec_val rows too (min -938), so the instability is not webp-only.

FIX (this script): refit the cid head with the missing SDR content class
in-corpus — a small deterministic stratified slice of canonical safesyn
(the trio's source universe; ssim2-derived human_score target, minmax01'd
per group by the gram machinery = the same target convention every group
gets) — then re-ensemble with the unchanged kon head
(`canonhdr15-bvls-raw` @ tau0.005) via the same convex-blend machinery,
selection on train-legal axes ONLY. bigcodec mass stays OUT (falsified
poison per linear_projections_2026-07-03.md).

Slice selection rule (deterministic, seed-free, evidence-picked):
  1. per-feature tails: for each of the 372 features, the top-8 and
     bottom-8 safesyn rows by value (OOD tails are the measured failure
     mode; this covers every feature direction where hdr_v3mix is
     near-degenerate but SDR content is not, f235 included);
  2. stride backbone: every 16th row of safesyn (row order groups
     ref x codec x q, so the stride covers the bulk distribution).
  Union, deduped by row index.

Subcommands: diagnose / slice / fit / ensemble / bake / mitigate.
Verification (trio + full panel + dial) runs via bake_verdict +
upiq_panel.py outside this script; see
benchmarks/linear_projections_2026-07-03.md "w11 refit".
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SPEC = importlib.util.spec_from_file_location(
    "lp", Path(__file__).parent / "linear_projections_2026-07-03.py")
lp = importlib.util.module_from_spec(SPEC)
sys.modules["lp"] = lp
SPEC.loader.exec_module(lp)

SCRATCH = lp.SCRATCH                      # linear-probe dir
W11 = Path("/mnt/v/output/zensim-multicodec-probe/w11-webp-ood")
SAFESYN = lp.CANON / "safesyn.parquet"
GRID = Path("/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29.parquet")
TRIO = ["a06b91d3d8419aad_513x769", "a9143f4b78fe5a13_513x769",
        "c37e9ae52fbab790_1022x818"]
FCOLS = lp.FCOLS
N_FEAT = lp.N_FEAT
CID_KEY, CID_TAU = "hdrmix-lasso0.002-raw", 0.0        # shipped cid head
KON_KEY, KON_TAU = "canonhdr15-bvls-raw", 0.005        # shipped kon head

SLICE_PQ = W11 / "safesyn_w11_slice_2026-07-05.parquet"
SLICE_GRAM = SCRATCH / "grams" / "safesyn_w11.npz"

# Pre-registered sweep (before any panel): slice weight x lasso lambda.
# 0.02/0.05 were the pre-registered EXTENSION round (rigor policy: push to
# the claimed benefit before falsifying) after the first round failed gates.
SLICE_WEIGHTS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
LAMS = [0.001, 0.002]
TAUS = [0.0, 0.005]
ALPHAS = [0.80]          # shipped P-line composition kept fixed (cid:kon)

# ---------------------------------------------------------------------------
# MEASURED VERDICT (2026-07-05) — read benchmarks/linear_projections_2026-07-03.md
# "w11" section before reusing any of this:
#
# 1. THE PREMISE WAS WRONG. The trio's -80s were NOT model blindness: the
#    dial grid's stored features for 9 (image,codec) ladders are extraction
#    garbage (masked/IW blocks 34..489, bit-constant across the whole q
#    ladder; zensim-gpu odd-dim pathology). Fresh CPU extraction on fresh
#    re-encodes gives 0.003..0.025 there, and the SHIPPED bake scores the
#    fresh trio sane + correctly ordered (92.1 / 81.5 / 79.0).
# 2. THE CORPUS REFIT IS FALSIFIED: every slice weight in {0.02..1.0} costs
#    held-out CID22 (-0.017..-0.032) and usually KonJND; the ssim2-anchored
#    cid22tr selection axis moves UP while real CID22 moves DOWN (the
#    ssim2-target trap, now measured on a selection axis).
# 3. What survives: `fit`'s OOD-health columns (the tau0 cid head really is
#    fragile on real extreme inputs: 0.95% of bigcodec_val below raw -2,
#    min -938 via f155/f52/f216 heavy tails; every lasso0.001+slice fit
#    zeroes f235+f155 and holds min >= -1.6) and `mitigate` (bottom-floor
#    bounds garbage-input damage at 0 with ZERO corruption-gate cost on
#    this bake, but does NOT restore knob usability - garbage in stays
#    garbage out).
# ---------------------------------------------------------------------------


def _load_grid():
    t = pq.read_table(GRID, columns=["image_id", "codec", "q"] + FCOLS)
    img = np.array(t["image_id"].to_pylist())
    cod = np.array(t["codec"].to_pylist())
    q = np.asarray(t["q"], dtype=float)
    F = np.column_stack([np.asarray(t[c], dtype=np.float64) for c in FCOLS])
    return img, cod, q, F


def _head(key, tau):
    z = np.load(SCRATCH / "fits" / f"{key}.npz")
    w = z["w"].astype(np.float64).copy()
    if tau > 0:
        w[np.abs(w) < tau] = 0.0
    return w, float(z["bias"]), z["mu"].astype(np.float64), z["sd"].astype(np.float64)


def cmd_diagnose(args) -> int:
    img, cod, q, F = _load_grid()
    hi = (cod == "webp") & (q >= 90)
    istrio = np.isin(img, TRIO)
    for name, key, tau in [("cid", CID_KEY, CID_TAU), ("kon", KON_KEY, KON_TAU)]:
        w, b, mu, sd = _head(key, tau)
        Z = (F - mu) / sd
        p = Z @ w + b
        print(f"\n{name} ({key}@{tau}): healthy webp q>=90 median "
              f"{np.median(p[hi & ~istrio]):+.3f}")
        for im in TRIO:
            m = hi & (img == im)
            r = np.nonzero(m)[0][np.argmax(q[m])]
            act = np.nonzero(np.abs(w) > 0)[0]
            c = Z[r, act] * w[act]
            j = act[np.argmin(c)]
            print(f"  {im}: pred [{p[m].min():+.3f},{p[m].max():+.3f}]  "
                  f"worst feature f{j} w={w[j]:+.4f} z={Z[r, j]:+.1f} "
                  f"contrib={Z[r, j] * w[j]:+.3f}")
    return 0


def cmd_slice(args) -> int:
    """Build the deterministic stratified safesyn slice + its raw-space gram."""
    W11.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    t = pq.read_table(SAFESYN, columns=["ref_basename", "human_score"] + FCOLS)
    n = len(t)
    hs = np.asarray(t["human_score"], dtype=np.float64)
    F = np.column_stack([np.asarray(t[c], dtype=np.float64) for c in FCOLS])
    keep = np.isfinite(hs) & np.all(np.isfinite(F), axis=1)
    print(f"safesyn rows {n}, finite {keep.sum()} ({time.time()-t0:.0f}s load)")

    idx = set()
    fin = np.nonzero(keep)[0]
    Ff = F[fin]
    for f in range(N_FEAT):
        o = np.argsort(Ff[:, f], kind="stable")
        idx.update(fin[o[:8]].tolist())       # bottom-8
        idx.update(fin[o[-8:]].tolist())      # top-8
    n_tails = len(idx)
    idx.update(fin[::16].tolist())            # stride backbone
    idx = np.array(sorted(idx))
    print(f"slice: {n_tails} tail rows + stride -> {len(idx)} total "
          f"({100.0 * len(idx) / n:.1f}% of safesyn)")

    cols = {"ref_basename": pa.array([t["ref_basename"][int(i)].as_py() for i in idx]),
            "human_score": pa.array(hs[idx])}
    for k, c in enumerate(FCOLS):
        cols[c] = pa.array(F[idx, k])
    ot = pa.table(cols)
    meta = {b"w11_selection_rule":
            b"per-feature top-8+bottom-8 tails (372 feats) UNION stride-16 backbone; "
            b"deterministic, seed-free; source canonical-2026-05-21/train/safesyn.parquet",
            b"w11_rows": str(len(idx)).encode()}
    pq.write_table(ot.replace_schema_metadata(meta), SLICE_PQ, compression="zstd")
    sha = hashlib.sha256(SLICE_PQ.read_bytes()).hexdigest()
    print(f"wrote {SLICE_PQ} rows={len(idx)} sha256={sha}")

    # raw-space gram in the exact format MixGram expects
    lo, span = lp.minmax01_bounds(hs[keep])
    y = np.clip((hs[idx] - lo) / span, 0.0, 1.0)
    X = F[idx]
    save = {"raw__S": X.T @ X, "raw__s": X.sum(axis=0), "raw__n": float(len(idx)),
            "raw__dropped": 0,
            "raw__q_human_score": X.T @ y,
            "raw__Y1_human_score": float(y.sum()),
            "raw__Y2_human_score": float((y * y).sum())}
    np.savez_compressed(SLICE_GRAM, **save)
    print(f"gram -> {SLICE_GRAM} (target minmax01 bounds from FULL safesyn: "
          f"lo={lo:.4f} span={span:.4f})")
    manifest = {"slice_parquet": str(SLICE_PQ), "rows": int(len(idx)),
                "sha256": sha, "n_tail_rows": int(n_tails),
                "selection_rule": "per-feature top8+bottom8 tails UNION stride-16",
                "source": str(SAFESYN),
                "target_minmax01": [lo, span]}
    (W11 / "_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return 0


def _register_mixes():
    for wgt in SLICE_WEIGHTS:
        lp.MIXES_HDR[f"hdrmixsafe{wgt:g}"] = [
            ("hdr_v3mix", 1.0, "human_score"),
            ("safesyn_w11", wgt, "human_score"),
        ]
    lp.GROUPS["safesyn_w11"] = (SLICE_PQ, ["human_score"])


def _ood_health(w, b, mu, sd):
    X, _ = lp._val_xy("bigcodec_val", "raw")
    p = (X - mu) / sd @ w + b
    return float((p < -2).mean()), float(p.min())


def cmd_fit(args) -> int:
    """Refit sweep: hdr_v3mix 1.0 + safesyn_w11 {weights} x lasso {lams}."""
    _register_mixes()
    img, cod, q, F = _load_grid()
    hi = (cod == "webp") & (q >= 90)
    rows = []
    for wgt in SLICE_WEIGHTS:
        mg = lp.MixGram("raw", lp.MIXES_HDR[f"hdrmixsafe{wgt:g}"])
        for lam in LAMS:
            w, b = mg.lasso(lam)
            key = f"hdrmixsafe{wgt:g}-lasso{lam:g}-raw"
            np.savez_compressed(SCRATCH / "fits" / f"{key}.npz",
                                w=w, bias=b, mu=mg.mu, sd=mg.sd,
                                space="raw", desc="|".join(mg.desc))
            vm = lp.val_metrics(w, b, mg.mu, mg.sd, "raw")
            frac, pmin = _ood_health(w, b, mg.mu, mg.sd)
            Z = (F - mg.mu) / mg.sd
            p = Z @ w + b
            trio_lo = min(p[hi & (img == im)].min() for im in TRIO)
            nact = int((np.abs(w) > 1e-7).sum())
            f235 = float(w[235])
            rows.append(dict(key=key, n_active=nact, w_f235=f235,
                             ood_frac_lt_m2=frac, bigval_min=pmin,
                             trio_min=float(trio_lo), **vm))
            print(f"{key:34s} act={nact:3d} w235={f235:+.5f} "
                  f"cid22tr={vm['cid22tr_sel']:+.4f} bigval={vm['bigcodec_val']:+.4f} "
                  f"hdrmixval={vm['hdr_valmix']:+.4f} guard={abs(vm['konjnd_guard']):.4f} "
                  f"ood<-2={frac*100:.3f}% min={pmin:.1f} trio_min={trio_lo:+.3f}",
                  flush=True)
    with open(W11 / "fit_table.json", "w") as f:
        json.dump(rows, f, indent=1)
    return 0


def cmd_ensemble(args) -> int:
    """Blend cid'(refit) + kon (unchanged) at the shipped alpha, anchor-znorm,
    -> single raw-space linear layer npz (same collapse math as cmd_ensemble
    in the campaign script)."""
    key = args.cid_key
    tau = float(args.cid_tau)
    az = np.load(SCRATCH / "val" / "anchor.npz")
    A = az["raw"].astype(np.float64)
    heads = []
    for k, t in [(key, tau), (KON_KEY, KON_TAU)]:
        z = np.load(SCRATCH / "fits" / f"{k}.npz")
        w = z["w"].astype(np.float64).copy()
        if t > 0:
            w[np.abs(w) < t] = 0.0
        mu = z["mu"].astype(np.float64)
        sd = z["sd"].astype(np.float64)
        v = w / sd
        c = float(z["bias"]) - float(mu @ v)
        p = A @ v + c
        heads.append((v, c, float(p.mean()), float(p.std())))
    for a in [float(x) for x in args.alphas.split(",")]:
        alpha = np.array([a, 1.0 - a])
        V = np.column_stack([v / s for (v, c, m, s) in heads])
        C = np.array([(c - m) / s for (v, c, m, s) in heads])
        w_ens = V @ alpha
        b_ens = float(C @ alpha)
        out = f"ens-w11-{args.tag}-cid{int(a*100)}"
        np.savez_compressed(SCRATCH / "fits" / f"{out}.npz",
                            w=w_ens, bias=b_ens, mu=np.zeros(N_FEAT),
                            sd=np.ones(N_FEAT), space="raw",
                            desc=f"w11: {key}@{tau}:{a:.2f}+{KON_KEY}@{KON_TAU}:{1-a:.2f}")
        vm = lp.val_metrics(w_ens, b_ens, np.zeros(N_FEAT), np.ones(N_FEAT), "raw")
        print(f"{out}: cid22tr={vm['cid22tr_sel']:+.4f} bigval={vm['bigcodec_val']:+.4f} "
              f"hdrval={vm['hdr_val']:+.4f} guard={abs(vm['konjnd_guard']):.4f}", flush=True)
    return 0


def cmd_bake(args) -> int:
    """tau0 bake via the campaign's bake_candidate (spline-on-packed over the
    dial100 anchor = the shared-anchor convention for SDR)."""
    (SCRATCH / "bakes").mkdir(exist_ok=True)
    for key in args.keys.split(","):
        out = SCRATCH / "bakes" / f"lp_{key}-tau0-f16.bin"
        info = lp.bake_candidate(key, 0.0, out)
        print(json.dumps(info), flush=True)
    return 0


# --- mitigation study (eval-only, numpy; replicates the Rust evaluators) ---
def pchip_derivs(xs, ys):
    n = len(xs)
    if n == 2:
        s = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return np.array([s, s])
    h = np.diff(xs)
    s = np.diff(ys) / h
    d = np.zeros(n)
    for k in range(1, n - 1):
        if s[k - 1] * s[k] <= 0:
            d[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k])

    def endpoint(h0, h1, s0, s1):
        e = ((2 * h0 + h1) * s0 - h0 * s1) / (h0 + h1)
        if e * s0 <= 0:
            return 0.0
        if s0 * s1 <= 0 and abs(e) > 3 * abs(s0):
            return 3 * s0
        return e
    d[0] = endpoint(h[0], h[1], s[0], s[1])
    d[-1] = endpoint(h[-2], h[-3], s[-2], s[-3])
    return d


def spline_apply(x, xs, ys, d, bottom_floor=False):
    """zensim-validate semantics: below-bottom linear (uncapped) unless
    bottom_floor; above-top linear capped at 100."""
    n = len(xs)
    if x <= xs[0]:
        return ys[0] if bottom_floor else ys[0] + d[0] * (x - xs[0])
    if x >= xs[-1]:
        return min(ys[-1] + d[-1] * (x - xs[-1]), 100.0)
    hi = np.searchsorted(xs, x)
    lo = hi - 1
    h = xs[hi] - xs[lo]
    t = (x - xs[lo]) / h
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t * t * (3 - 2 * t)
    h11 = t * t * (t - 1)
    return h00 * ys[lo] + h10 * h * d[lo] + h01 * ys[hi] + h11 * h * d[hi]


def bake_spline(bake_path: Path):
    """Extract the output-calibration spline knots from a ZNPR bake via
    `zenpredict inspect` (JSON-ish dump with a `value_hex` field)."""
    r = subprocess.run([str(lp.BAKER), "inspect", str(bake_path)],
                       capture_output=True, text=True)
    lines = r.stdout.splitlines()
    for i, line in enumerate(lines):
        if "output_calibration_spline" not in line:
            continue
        for j in range(i, min(i + 4, len(lines))):
            if "value_hex" in lines[j]:
                hx = lines[j].split(":", 1)[1].strip().strip('",')
                b = bytes.fromhex(hx)
                n = struct.unpack("<I", b[:4])[0]
                kx, ky = [], []
                for k in range(n):
                    x, y = struct.unpack("<ff", b[4 + 8 * k:12 + 8 * k])
                    kx.append(x)
                    ky.append(y)
                return np.array(kx, dtype=float), np.array(ky, dtype=float)
    raise RuntimeError(f"no spline metadata found in {bake_path}")


def cmd_mitigate(args) -> int:
    """OLD anchored bake + bottom-floor clamp (eval-only): does flooring alone
    restore trio knob-usability? Uses the bake's own f16 weights + stored
    spline; floor = bottom-knot y."""
    bake = Path(args.bake)
    kx, ky = bake_spline(bake)
    d = pchip_derivs(kx, ky)
    print(f"spline: {len(kx)} knots x[{kx[0]:.3f},{kx[-1]:.3f}] "
          f"y[{ky.min():.1f},{ky.max():.1f}]")
    z = np.load(SCRATCH / "fits" / f"{args.fit_key}.npz")
    w = z["w"].astype(np.float16).astype(np.float64)   # as the f16 bake stores
    b = float(z["bias"])
    img, cod, q, F = _load_grid()
    p = F @ w + b if bool(z["mu"].any()) is False else ((F - z["mu"]) / z["sd"]) @ w + b
    for floor in (False, True):
        s = np.array([spline_apply(x, kx, ky, d, bottom_floor=floor) for x in p])
        tag = "floor@bottom-knot" if floor else "current (uncapped)"
        print(f"\n--- {tag} ---")
        for im in TRIO:
            m = (cod == "webp") & (img == im)
            o = np.argsort(q[m])
            qs = q[m][o]
            ss = s[m][o]
            top = ss[qs >= 90]
            print(f"  {im}: ladder min={ss.min():.1f} max={ss.max():.1f} "
                  f"q>=90 [{top.min():.1f},{top.max():.1f}] "
                  f"n_distinct_scores={len(np.unique(np.round(ss, 1)))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("diagnose")
    sub.add_parser("slice")
    sub.add_parser("fit")
    e = sub.add_parser("ensemble")
    e.add_argument("--cid-key", required=True)
    e.add_argument("--cid-tau", default="0")
    e.add_argument("--tag", required=True)
    e.add_argument("--alphas", default="0.80")
    bk = sub.add_parser("bake")
    bk.add_argument("--keys", required=True)
    m = sub.add_parser("mitigate")
    m.add_argument("--bake", required=True)
    m.add_argument("--fit-key", default="ens-Pline-cid80")
    args = ap.parse_args()
    return {"diagnose": cmd_diagnose, "slice": cmd_slice, "fit": cmd_fit,
            "ensemble": cmd_ensemble, "bake": cmd_bake,
            "mitigate": cmd_mitigate}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
