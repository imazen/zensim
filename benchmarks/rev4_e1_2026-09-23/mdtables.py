#!/usr/bin/env python3
"""Rev4 E1 — extra markdown tables (peer points per band; band SROCC; exploratory split).

Formatting only; every number is read from analyze.py / explore.py output.
"""
import json
import sys

E = json.load(open("/var/tmp/rev4-e1/e1_full.json"))
X = json.load(open("/var/tmp/rev4-e1/explore.json"))
PEERS = ["ssim2", "butter_p3", "butter_max", "iwssim", "pub_iwssim", "pub_msssim", "dssim", "pub_psnry",
         "cvvdp_fhd", "pub_cvvdp", "cvvdp_4k"]
Z = ["B", "C", "D", "R915_fast", "R915_rich", "V0_2"]
CN = {"cid22a": "CID22-A", "csiq": "CSIQ", "konjnd404": "KonJND-404", "aic3": "AIC-3", "aic4crop": "AIC-4 crop",
      "aic4full": "AIC-4 full", "sdr25": "SDR25"}


def f(x):
    return "—" if x is None else f"{x:.3f}"


which = sys.argv[1]
if which == "peers":
    print("| corpus | band | stat | " + " | ".join(PEERS) + " |")
    print("|---|---|---|" + "---|" * len(PEERS))
    for c, cr in E["corpora"].items():
        st = "srocc" if c == "konjnd404" else "pairwise"
        for b, e in cr["bands"].items():
            if c == "konjnd404" and b == "ALL":
                continue
            M = e[st]["metrics"]
            print(f"| {CN[c]} | {b} | {'SROCC' if st == 'srocc' else 'acc'} | " +
                  " | ".join(f(M[p]["point"]) if p in M else "—" for p in PEERS) + " |")
elif which == "srocc":
    print("| corpus | band | n | best peer | SSIMULACRA2 | " + " | ".join(Z) + " |")
    print("|---|---|---|---|---|" + "---|" * len(Z))
    for c, cr in E["corpora"].items():
        for b, e in cr["bands"].items():
            if c == "konjnd404" and b == "ALL":
                continue
            blk = e["srocc"]
            M, bp = blk["metrics"], blk["best_peer"]
            row = [CN[c], b, str(e["n_stim"]), f"{bp} {M[bp]['point']:.3f}", f"{M['ssim2']['point']:.3f}"]
            for z in Z:
                if z not in M:
                    row.append("—")
                    continue
                d = blk["deltas"][f"{z}-{bp}"]
                mk = "**" if d["ci"][1] < 0 else ("*" if d["ci"][0] > 0 else "")
                row.append(f"{M[z]['point']:.3f} {mk}{d['point']:+.3f}{mk}")
            print("| " + " | ".join(row) + " |")
elif which == "explore":
    print("| corpus | band | pair type | n pairs | best peer | B − ssim2 [CI] | C − ssim2 [CI] | B − best | C − best |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c, cc in X["corpora"].items():
        for b, ent in cc.items():
            for w, v in ent.items():
                bp = v["best_peer"]
                d = v["d"]

                def g(k):
                    if k not in d:
                        return "—"
                    p, lo, hi = d[k]
                    mk = "**" if hi < 0 else ("*" if lo > 0 else "")
                    return f"{mk}{p:+.4f} [{lo:+.4f}, {hi:+.4f}]{mk}"
                print(f"| {CN[c]} | {b} | {w} | {v['n_pairs']} | {bp} | {g('B-ssim2')} | {g('C-ssim2')} | "
                      f"{g('B-' + bp) if bp != 'ssim2' else '='} | {g('C-' + bp) if bp != 'ssim2' else '='} |")
