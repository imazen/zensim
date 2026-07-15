#!/usr/bin/env python3
"""Optimal-input-blend search (user 2026-07-15). Sweeps corpus-blend configs — now that
kadid/tid ssim2_gpu is FIXED (§3.18) they're usable positives again — trains each over N seeds,
scores the FULL held-out Mohammadi panel per corpus (blend_lib), and ranks by a goal-aware
composite (CID22 primary + non-photo + KonJND, guards the rest). Saves the top-K payloads (npz)
+ their averaged panels (json) for the bandwise dashboard, and a results TSV to benchmarks/.

  usage: blend_search.py [--seeds 1,7] [--topk 4] [--out-dir /mnt/v/output/zensim/reports/blend]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import blend_lib as B

# round 2 — break the CID22↔nonphoto trade: suppress bigcodec's HQ-saturated rows (where it
# disagrees with CID22), sweep div weight, add depth. base blend = safesyn+cid22+kadid+tid+bigcodec+kadis.
_KT = {"safesyn": 1, "cid22_train": 1, "kadid": 1.0, "tid": 1.0, "bigcodec": 1.0, "kadis": 0.3}
ROUND2 = [
    ("r2-base",          dict(_KT), {}),
    ("r2-hq0.1",         dict(_KT), {"hq_weight": 0.1}),
    ("r2-hq0(dropHQ)",   dict(_KT), {"hq_weight": 0.0}),
    ("r2-hqband80",      dict(_KT), {"hq_band": 80.0, "hq_weight": 0.2}),
    ("r2-div0.3",        {**_KT, "bigcodec": 0.3}, {}),
    ("r2-div0.7",        {**_KT, "bigcodec": 0.7}, {}),
    ("r2-div0.5hq0.1",   {**_KT, "bigcodec": 0.5}, {"hq_weight": 0.1}),
    ("r2-winsor0.5",     dict(_KT), {"winsor_pct": 0.5}),
    ("r2-2layer",        dict(_KT), {"layers": 2}),
    ("r2-2layer-div0.5", {**_KT, "bigcodec": 0.5}, {"layers": 2}),
    ("r2-2layer-hq0.1",  dict(_KT), {"layers": 2, "hq_weight": 0.1}),
    ("r2-ssim2only",     {"safesyn": 1, "cid22_train": 1}, {}),
]
# round 3 — 2-layer BROKE the CID22↔nonphoto trade (r2, +0.03 CID22 & +0.02 nonphoto over 1-layer).
# Confirm at 5 seeds + explore depth/width to squeeze more. div0.5 gave the best CID22 in r2.
ROUND3 = [
    ("r3-2L",            dict(_KT), {"layers": 2}),
    ("r3-2L-div0.5",     {**_KT, "bigcodec": 0.5}, {"layers": 2}),
    ("r3-2L-div0.7",     {**_KT, "bigcodec": 0.7}, {"layers": 2}),
    ("r3-2L-H96",        dict(_KT), {"layers": 2, "hidden": 96}),
    ("r3-2L-H128",       dict(_KT), {"layers": 2, "hidden": 128}),
    ("r3-2L-H96-div0.5", {**_KT, "bigcodec": 0.5}, {"layers": 2, "hidden": 96}),
    ("r3-2L-ep700",      dict(_KT), {"layers": 2, "epochs": 700}),
    ("r3-1L-ref",        dict(_KT), {"layers": 1}),
    ("r3-ssim2only-2L",  {"safesyn": 1, "cid22_train": 1}, {"layers": 2}),
]
# (label, blend spec {corpus: weight}, hp-overrides). The systematic sweep of "does adding the
# now-clean kadid/tid help, at what weight, traded against div/kadis".
ROUND1 = [
    ("base(§8.35)",       {"safesyn": 1, "cid22_train": 1, "bigcodec": 1.0, "kadis": 0.3}, {}),
    ("+kadid",            {"safesyn": 1, "cid22_train": 1, "kadid": 1.0, "bigcodec": 1.0, "kadis": 0.3}, {}),
    ("+tid",              {"safesyn": 1, "cid22_train": 1, "tid": 1.0, "bigcodec": 1.0, "kadis": 0.3}, {}),
    ("+kadid+tid",        {"safesyn": 1, "cid22_train": 1, "kadid": 1.0, "tid": 1.0, "bigcodec": 1.0, "kadis": 0.3}, {}),
    ("+kt@0.5",           {"safesyn": 1, "cid22_train": 1, "kadid": 0.5, "tid": 0.5, "bigcodec": 1.0, "kadis": 0.3}, {}),
    ("+kt@2",             {"safesyn": 1, "cid22_train": 1, "kadid": 2.0, "tid": 2.0, "bigcodec": 1.0, "kadis": 0.3}, {}),
    ("+kt,div0.5",        {"safesyn": 1, "cid22_train": 1, "kadid": 1.0, "tid": 1.0, "bigcodec": 0.5, "kadis": 0.3}, {}),
    ("+kt,div1.5",        {"safesyn": 1, "cid22_train": 1, "kadid": 1.0, "tid": 1.0, "bigcodec": 1.5, "kadis": 0.3}, {}),
    ("+kt,kadis0.6",      {"safesyn": 1, "cid22_train": 1, "kadid": 1.0, "tid": 1.0, "bigcodec": 1.0, "kadis": 0.6}, {}),
    ("+kt,H96",           {"safesyn": 1, "cid22_train": 1, "kadid": 1.0, "tid": 1.0, "bigcodec": 1.0, "kadis": 0.3}, {"hidden": 96}),
    ("+kt,H128",          {"safesyn": 1, "cid22_train": 1, "kadid": 1.0, "tid": 1.0, "bigcodec": 1.0, "kadis": 0.3}, {"hidden": 128}),
    ("+kt,ss60k",         {"safesyn": 1, "cid22_train": 1, "kadid": 1.0, "tid": 1.0, "bigcodec": 1.0, "kadis": 0.3}, {"safesyn_cap": 60000}),
]


def avg_panels(panels):
    """Average a list of score_all() dicts (per corpus, aggregate + per-band SROCC)."""
    out = {}
    for corp in B.VAL_CORPORA:
        agg = {}
        for k in ["srocc", "srocc_abs", "plcc", "krocc", "zrmse", "n"]:
            vals = [p[corp][k] for p in panels if np.isfinite(p[corp].get(k, np.nan))]
            agg[k] = float(np.mean(vals)) if vals else float("nan")
        agg["sign"] = panels[0][corp]["sign"]
        if "bands" in panels[0][corp]:
            bands = {}
            for bn in panels[0][corp]["bands"]:
                sv = [p[corp]["bands"][bn]["srocc"] for p in panels
                      if np.isfinite(p[corp]["bands"][bn].get("srocc", np.nan))]
                bands[bn] = {"srocc": float(np.mean(sv)) if sv else float("nan"),
                             "n": panels[0][corp]["bands"][bn]["n"]}
            agg["bands"] = bands
        out[corp] = agg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,7")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--round", default="1", choices=["1", "2", "3"])
    ap.add_argument("--out-dir", default="/mnt/v/output/zensim/reports/blend")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    CONFIGS = {"1": ROUND1, "2": ROUND2, "3": ROUND3}[a.round]
    rtag = f"r{a.round}"

    cols = ["cid22", "nonphoto", "konjnd", "aic3", "aic4", "kadid", "tid"]
    print(f"{'config':16} " + " ".join(f"{c[:8]:>8}" for c in cols) + "  compos  rej")
    results = []
    for label, spec, hpov in CONFIGS:
        hp = {"hidden": 64, "epochs": 400, "winsor_pct": 0.1, "div_cap": 120000,
              "hq_band": 85.0, "hq_weight": 0.3, **hpov}
        t0 = time.time()
        payloads, panels = [], []
        for s in seeds:
            p = B.train_blend(spec, hp, s)
            payloads.append(p); panels.append(B.score_all(p))
        avg = avg_panels(panels)
        comp, reject = B.composite(avg)

        def show(c):
            v = avg[c]["srocc_abs"] if B.VAL_CORPORA[c][2] < 0 else avg[c]["srocc"]
            return v if np.isfinite(v) else float("nan")
        print(f"{label:16} " + " ".join(f"{show(c):+8.4f}" for c in cols)
              + f"  {comp:6.3f}  {'REJ' if reject else ' ok'}   ({time.time()-t0:.0f}s)")
        results.append({"label": label, "spec": spec, "hp": hp, "composite": comp,
                        "reject": bool(reject), "panel": avg,
                        # keep the median-seed payload for the dashboard/bake
                        "payload_idx": len(seeds) // 2, "_payloads": payloads})

    # rank non-rejected by composite; save top-K payloads + panels
    ranked = sorted([r for r in results if not r["reject"]], key=lambda r: -r["composite"])
    print(f"\nTOP {a.topk} (non-rejected, by composite):")
    saved = []
    for i, r in enumerate(ranked[:a.topk]):
        pth = od / f"blend_{rtag}_{i}_{r['label'].replace('(', '').replace(')', '').replace('§', 's').replace(',', '_').replace('+', 'p').replace('@', 'a').replace('.', '')}.npz"
        # strip non-array dict fields (spec/hp live in the json) so np.load needs no allow_pickle
        arr_only = {k: v for k, v in r["_payloads"][r["payload_idx"]].items() if k not in ("spec", "hp")}
        np.savez(str(pth), **arr_only)
        saved.append({"label": r["label"], "npz": str(pth), "spec": r["spec"], "hp": r["hp"],
                      "composite": r["composite"], "panel": r["panel"]})
        print(f"  #{i} {r['label']:16} composite {r['composite']:.3f}  CID22 "
              f"{r['panel']['cid22']['srocc']:.4f}  nonphoto {r['panel']['nonphoto']['srocc']:.4f}  -> {pth.name}")

    (od / f"blend_results_{rtag}_2026-07-15.json").write_text(json.dumps(
        {"seeds": seeds, "saved": saved,
         "all": [{k: v for k, v in r.items() if k != "_payloads"} for r in results]}, indent=2))
    # committed TSV summary
    bench = Path.home() / f"work/zen/zensim/benchmarks/blend_search_{rtag}_2026-07-15.tsv"
    with open(bench, "w") as f:
        f.write("config\t" + "\t".join(cols) + "\tcomposite\treject\n")
        for r in results:
            def show(c):
                v = r["panel"][c]["srocc_abs"] if B.VAL_CORPORA[c][2] < 0 else r["panel"][c]["srocc"]
                return f"{v:.4f}" if np.isfinite(v) else "nan"
            f.write(f"{r['label']}\t" + "\t".join(show(c) for c in cols)
                    + f"\t{r['composite']:.4f}\t{int(r['reject'])}\n")
    print(f"\nwrote {od/'blend_results_2026-07-15.json'} + {bench}")


if __name__ == "__main__":
    main()
