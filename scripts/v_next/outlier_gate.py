#!/usr/bin/env python3
"""G-OUT: worst-outlier gates per chart (registered in
sdr_pure_retrain_wave_2026-08-28.md). Reads fulleval per_pair blocks."""
import json, sys
import numpy as np

BARS = {"extreme_z": 6.0, "severe_z": 4.0, "severe_frac": 0.002, "rank_disp": 0.6}

def axis_outliers(pred, mos):
    x = np.asarray(mos, float); y = np.asarray(pred, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 100:
        return None
    A = np.vstack([x, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    mad = np.median(np.abs(res - np.median(res))) * 1.4826 or 1e-9
    z = res / mad
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    disp = np.abs(rx - ry) / max(n - 1, 1)
    worst = np.argsort(-np.abs(z))[:5]
    return {"n": n, "max_abs_z": float(np.max(np.abs(z))),
            "n_extreme": int((np.abs(z) > BARS["extreme_z"]).sum()),
            "frac_severe": float((np.abs(z) > BARS["severe_z"]).mean()),
            "max_rank_disp": float(disp.max()),
            "worst5": [{"z": round(float(z[i]), 2), "pred": round(float(y[i]), 2),
                        "target": round(float(x[i]), 2)} for i in worst]}

def main():
    for path in sys.argv[1:]:
        o = json.load(open(path))
        name = o.get("name", path)
        pp = o.get("per_pair") or {}
        fails = []
        print(f"== {name}")
        for ax, blk in sorted(pp.items()):
            if not isinstance(blk, dict) or "pred" not in blk:
                continue
            tcol = next((k for k in ("mos", "jnd", "pjnd", "target") if k in blk), None)
            if tcol is None:
                continue
            r = axis_outliers(blk["pred"], blk[tcol])
            if r is None:
                continue
            bad = []
            if r["n"] >= 500 and r["n_extreme"] > 0: bad.append(f"EXTREME x{r['n_extreme']}")
            if r["n"] >= 500 and r["frac_severe"] > BARS["severe_frac"]: bad.append(f"severe {r['frac_severe']:.4f}")
            if r["n"] >= 1000 and r["max_rank_disp"] > BARS["rank_disp"]: bad.append(f"disp {r['max_rank_disp']:.2f}")
            flag = " ⚠ " + ",".join(bad) if bad else ""
            print(f"   {ax:<12} n={r['n']:<6} max|z|={r['max_abs_z']:>5.2f} sev={r['frac_severe']:.4f} maxdisp={r['max_rank_disp']:.2f}{flag}")
            if bad: fails.append((ax, bad, r["worst5"]))
        if fails:
            print(f"   G-OUT: FAIL on {len(fails)} axes")
            for ax, bad, w5 in fails:
                print(f"     {ax}: worst5 {w5}")
        else:
            print("   G-OUT: PASS")

if __name__ == "__main__":
    main()
