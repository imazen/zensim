#!/usr/bin/env python3
"""fastclass2 arm table. READS bake_verdict --full-json fullevals and
bake_verdict --gaddr-json gradings; forms only k-seed means and spreads.
No statistic that an owner owns is recomputed here."""
import json, glob, os, re, sys, statistics as st

W = "/mnt/v/output/zensim/fastclass2-2026-09-05"
AXES = [("composite", None), ("cid22", "rank"), ("konjnd", "rank"), ("aic3", "rank"),
        ("csiq", "rank"), ("live", "rank"), ("imazen26", "rank"), ("nonphoto", "rank"),
        ("hfnlproxy", "rank"), ("aic4", "rank"), ("sdr25", "rank")]

def axis(d, name, where):
    if where is None:
        return d.get(name)
    r = (d.get("rank") or {}).get(name)
    if not r:
        return None
    v = r.get("srocc_signed", r.get("srocc"))
    return abs(v) if name == "konjnd" and v is not None else v

def gaddr(name):
    p = f"{W}/gaddr/gaddr_{name}_packed_ladder.json"
    if not os.path.exists(p):
        p = f"{W}/gaddr/gaddr_{name}_ladder.json"
    if not os.path.exists(p):
        return None, None
    a = json.load(open(p))
    a = a.get("addressability", a)
    ck = {c["id"]: c for c in a["checks"]}
    a7 = ck.get("A7r", {}).get("measured")
    return a7, a.get("contract")

def load():
    g = {}
    for f in sorted(glob.glob(f"{W}/*.fulleval.json")):
        d = json.load(open(f))
        n = d["name"]
        g.setdefault(re.sub(r"_s\d+$", "", n), []).append(d)
    return g

def cell(vs, fmt="%.4f"):
    vs = [v for v in vs if v is not None]
    return fmt % st.mean(vs) if vs else "-"

def main():
    g = load()
    keys = ["arm", "k", "composite", "cid22", "konjnd", "aic3", "csiq", "live",
            "imazen26", "nonphoto", "hfnlproxy", "mono", "A7r", "contract", "bytes"]
    rows = []
    for k in sorted(g):
        ds = g[k]
        a7s, cons = [], []
        for d in ds:
            a, c = gaddr(d["name"])
            if a is not None: a7s.append(a)
            if c: cons.append(c)
        r = [k, str(len(ds))]
        for a, w in AXES[:1] + AXES[1:9]:
            r.append(cell([axis(d, a, w) for d in ds]))
        r.append(cell([(d.get("dial") or {}).get("mono_pct") for d in ds]))
        r.append(("%.1f" % st.mean(a7s)) if a7s else "-")
        r.append("/".join(sorted(set(cons))) if cons else "-")
        r.append(cell([(d.get("model") or {}).get("file_bytes") for d in ds], "%.0f"))
        rows.append(r)
    w = [max(len(str(x[i])) for x in [keys] + rows) for i in range(len(keys))]
    print(" ".join(h.ljust(w[i]) for i, h in enumerate(keys)))
    for r in rows:
        print(" ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    print()
    print("per-seed spread (min..max):")
    for a in ("cid22", "konjnd", "composite"):
        print(f"  --- {a}")
        for k in sorted(g):
            vs = sorted(x for x in (axis(d, a, None if a == "composite" else "rank")
                                    for d in g[k]) if x is not None)
            if vs:
                print(f"    {k:30s} k={len(vs)} mean={st.mean(vs):.4f} "
                      f"[{vs[0]:.4f}..{vs[-1]:.4f}] spread={vs[-1]-vs[0]:.4f}")

if __name__ == "__main__":
    main()
