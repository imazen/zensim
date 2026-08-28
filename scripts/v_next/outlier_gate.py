#!/usr/bin/env python3
"""G-OUT v2 — the ACCEPTED final form (user, 2026-08-27; variant study in
sdr_pure_retrain_wave_2026-08-28.md "G-OUT VARIANT STUDY").

Clauses, per candidate axis (peers CALIBRATE, never gated):
  R rate      axis OR <= best-peer OR + 0.005            (panel-owned stat)
  S severity  axis p99|chart-z| <= min(best-peer p99, 12.0)
  B backstop  axis max|chart-z| <= 35                    (single-pair catcher)
  D bounded   emissions in [bottom_knot_dial - span/3, 100 + 5]
              span = top_knot_dial - bottom_knot_dial; the /3 allowance is the
              neg-tail design's sanctioned below-knot extrapolation zone.
              D applies on EVERY axis (unboundedness is a model property);
              R/S/B gate all axes for SDR candidates, on-route axes for HDR.

Chart-z = OLS(pred~target) residual / (MAD * 1.4826), computed on RAW preds —
never on 4PL-mapped values (MEASURED: the mapping saturates unbounded
emissions; t2 emits < -50 on 8.78% of kadid pairs yet its mapped-space
or/z_rmse beat the bounded incumbent's).

Usage:
  outlier_gate.py --peer peer_a.json --peer peer_b.json \
      [--range name=lo:hi] [--onroute name=ax1,ax2] cand1.json cand2.json ...
Ranges default from the bake spline decode recorded in the wave md; pass
--range to override. Emits a per-axis clause table + PASS/FAIL per candidate.
"""
import argparse, json, sys
import numpy as np

S_CEIL, S_TOL_R, B_CEIL, D_TOP = 12.0, 0.005, 35.0, 105.0

# Declared dial ranges: spline (bottom_knot_dial, top_knot_dial) decoded from
# zentrain.output_calibration_spline via zenpredict inspect (2026-08-27).
DECLARED = {
    "W10L9P_s4005_packed": (5.11, 87.09),
    "W10L9P_s4004_packed": (5.11, 87.09),   # sibling seeds share the anchor fit shape
    "W10L9P_s4003_packed": (5.11, 87.09),
    "W10L9_s4003_packed": (5.42, 86.96),
    "HDR944_L1T1_s4005_hfpack": (0.00, 96.14),
    "HDR944R_t2_s4003_hfpack": (-146.92, 93.43),
}

def chart_z(pred, mos):
    x = np.asarray(mos, float); y = np.asarray(pred, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 100: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    mad = np.median(np.abs(res - np.median(res))) * 1.4826 or 1e-9
    return res / mad, y

def axis_stats(o, ax):
    blk = (o.get("per_pair") or {}).get(ax)
    if not isinstance(blk, dict) or "pred" not in blk: return None
    tcol = next((k for k in ("mos", "jnd", "pjnd", "target") if k in blk), None)
    if tcol is None: return None
    z, y = chart_z(blk["pred"], blk[tcol])
    if z is None: return None
    rb = (o.get("rank") or {}).get(ax) or {}
    return {"or": rb.get("or"), "p99": float(np.percentile(np.abs(z), 99)),
            "max": float(np.max(np.abs(z))),
            "pred_min": float(np.min(y)), "pred_max": float(np.max(y)), "n": len(z)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer", action="append", default=[], help="peer fulleval json (calibration)")
    ap.add_argument("--range", action="append", default=[], help="name=lo:hi declared dial range override")
    ap.add_argument("--onroute", action="append", default=[], help="name=ax1,ax2 R/S/B axes for HDR candidates")
    ap.add_argument("--axes", default="cid22,imazen26,nonphoto,hfnlproxy,kadid,live",
                    help="R/S/B gated axes (the accepted study scope); other axes are reported, not gated")
    ap.add_argument("candidates", nargs="+")
    args = ap.parse_args()
    ranges = dict(DECLARED)
    for spec in args.range:
        name, lohi = spec.split("="); lo, hi = lohi.split(":")
        ranges[name] = (float(lo), float(hi))
    onroute = {}
    for spec in args.onroute:
        name, axs = spec.split("="); onroute[name] = set(axs.split(","))
    peers = [json.load(open(p)) for p in args.peer]
    cands = [json.load(open(p)) for p in args.candidates]
    axes = sorted({ax for o in cands for ax in (o.get("per_pair") or {})})
    peer_best = {}
    for ax in axes:
        stats = [s for s in (axis_stats(p, ax) for p in peers) if s]
        if stats:
            peer_best[ax] = {"or": min((s["or"] for s in stats if s["or"] is not None), default=None),
                             "p99": min(s["p99"] for s in stats)}
    print(f"{'candidate':<26}{'axis':<11}{'OR':>7}{'barR':>7}{'p99':>7}{'barS':>7}{'max':>7}{'rng':>16}{'floorD':>8}  clauses")
    overall = {}
    for o, path in zip(cands, args.candidates):
        name = o.get("name") or path
        decl = ranges.get(name)
        floor = decl[0] - (decl[1] - decl[0]) / 3.0 if decl else None
        gate_axes = onroute.get(name)
        fails = []
        for ax in axes:
            s = axis_stats(o, ax)
            if s is None: continue
            pb = peer_best.get(ax, {})
            gated_scope = set(args.axes.split(","))
            rsb = (gate_axes is None or ax in gate_axes) and ax in gated_scope
            cl = []
            if rsb and pb.get("or") is not None and s["or"] is not None:
                cl.append(("R", s["or"] <= pb["or"] + S_TOL_R))
            if rsb and pb.get("p99") is not None:
                cl.append(("S", s["p99"] <= min(pb["p99"], S_CEIL)))
            if rsb:
                cl.append(("B", s["max"] <= B_CEIL))
            if floor is not None:
                cl.append(("D", floor <= s["pred_min"] and s["pred_max"] <= D_TOP))
            verdict = " ".join(f"{c}{'+' if ok else '-'}" for c, ok in cl)
            fails += [f"{ax}:{c}" for c, ok in cl if not ok]
            barR = (pb.get("or") + S_TOL_R) if pb.get("or") is not None else float("nan")
            barS = min(pb["p99"], S_CEIL) if pb.get("p99") is not None else float("nan")
            print(f"{name[:25]:<26}{ax:<11}{s['or'] if s['or'] is not None else float('nan'):>7.3f}{barR:>7.3f}"
                  f"{s['p99']:>7.2f}{barS:>7.2f}{s['max']:>7.1f}"
                  f"{'['+format(s['pred_min'],'.0f')+','+format(s['pred_max'],'.0f')+']':>16}"
                  f"{floor if floor is not None else float('nan'):>8.1f}  {verdict}")
        overall[name] = fails
        print(f"{'':26}=> {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}\n")
    return 0 if all(not f for f in overall.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
