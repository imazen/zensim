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
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

def axis_stats(o, ax):
    """Read native complete-population tails, or explicitly assess legacy raw rows."""
    assessment = next(iter((o.get("scatter_assessment", {}).get(ax) or {}).values()), None)
    if assessment is None:
        blk = (o.get("per_pair") or {}).get(ax)
        if not isinstance(blk, dict) or "pred" not in blk:
            return None
        tcol = next((k for k in ("mos", "jnd", "pjnd", "target") if k in blk), None)
        if tcol is None:
            return None
        # Legacy summaries are not silently extrapolated from capped plot samples.
        if len(blk["pred"]) != (o.get("rank", {}).get(ax) or {}).get("n"):
            return None
        from lib.zen_stats import scatter
        assessment = scatter(blk["pred"], blk[tcol])
    if assessment.get("status") != "MEASURED":
        return None
    raw = assessment["raw"]
    # A zero robust scale is unmeasured, never an epsilon-derived gate pass.
    if raw["p99"] is None or raw["max"] is None:
        return None
    return dict(raw, n=assessment["n"], **{"or": (o.get("rank", {}).get(ax) or {}).get("or")})


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
    axes = sorted(set(args.axes.split(",")) | {ax for o in cands for ax in (o.get("rank") or {})})
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
        missing = []
        for ax in axes:
            s = axis_stats(o, ax)
            if s is None:
                if ax in (o.get("rank") or {}) or (ax in set(args.axes.split(",")) and (gate_axes is None or ax in gate_axes)):
                    missing.append(ax + ":statistics")
                continue
            pb = peer_best.get(ax, {})
            gated_scope = set(args.axes.split(","))
            rsb = (gate_axes is None or ax in gate_axes) and ax in gated_scope
            cl = []
            if rsb and (pb.get("or") is None or pb.get("p99") is None):
                missing.append(ax + ":peer-bars")
            if rsb and s["or"] is None:
                missing.append(ax + ":candidate-outlier-ratio")
            if floor is None:
                missing.append(ax + ":declared-range")
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
        overall[name] = dict(fails=fails, missing=missing)
        print(f"{'':26}=> {'FAIL ' + ', '.join(fails) if fails else 'INCOMPLETE ' + ', '.join(missing) if missing else 'PASS'}\n")
    return 1 if any(v["fails"] for v in overall.values()) else 2 if any(v["missing"] for v in overall.values()) else 0

if __name__ == "__main__":
    sys.exit(main())
