#!/usr/bin/env python3
"""Peer dial curves for the gauntlet dial panel (user: peers on ALL charts).

Builds `dial.curves` blocks ({codec: [[q, p25, p50, p75], ...]}) for the four
peer rows from the scored dial-grid ladder (persisted decoded pixels of the
2026-07-27 dial re-encode; scores = zenmetrics batch, cvvdp CPU + GPU trio on
the wsl 5070). mono_pct = per-(image,codec)-ladder fraction of non-decreasing
adjacent steps in q order (presentation-grade replication of the dial panel's
G3 reading; provenance stated in the row). Butteraugli is negated for the
quality orientation so curves rise with q like every other row.
"""
import csv, json, os
import numpy as np

R = "/mnt/v/output/zensim/reports/refmetrics"
OUT = "/mnt/v/output/zensim/reports/fulleval"
SRC = {
    "ssim2": ("dialgrid_ssim2_gpu.tsv", "ssim2_gpu", +1),
    "butteraugli": ("dialgrid_butteraugli_gpu.tsv", "butteraugli_max_gpu", -1),
    "iwssim": ("dialgrid_iwssim_gpu.tsv", "iwssim_gpu", +1),
    "cvvdp": ("dialgrid_cvvdp.tsv", "cvvdp_cpu_imazen_v0_1_0", +1),
}

def main():
    for peer, (tsv, col, sign) in SRC.items():
        p = os.path.join(R, tsv)
        if not os.path.exists(p):
            print(f"{peer}: {tsv} absent — skipped"); continue
        rows = list(csv.DictReader(open(p), delimiter="\t"))
        if not rows or col not in rows[0]:
            cols = [c for c in rows[0] if c not in ("ref_path","dist_path","codec","q","knob_tuple_json")]
            col2 = cols[-1] if cols else None
            if col2 is None: print(f"{peer}: no metric col — skipped"); continue
            colr = col2
        else:
            colr = col
        by = {}
        lad = {}
        for r in rows:
            try:
                q = float(r["q"]); v = sign * float(r[colr])
            except (KeyError, TypeError, ValueError):
                continue
            cd = {"zenjpeg": "jpeg", "zenwebp": "webp", "zenjxl": "jxl", "zenavif": "avif"}.get(r["codec"], r["codec"])
            by.setdefault(cd, {}).setdefault(q, []).append(v)
            img = os.path.basename(r["ref_path"])
            lad.setdefault((img, cd), []).append((q, v))
        curves = {}
        for cd, qs in by.items():
            curves[cd] = [[q, float(np.percentile(vs, 25)), float(np.percentile(vs, 50)),
                           float(np.percentile(vs, 75))] for q, vs in sorted(qs.items())]
        steps = ok = 0
        for (img, cd), pts in lad.items():
            pts.sort()
            for (qa, va), (qb, vb) in zip(pts[:-1], pts[1:]):
                steps += 1; ok += vb >= va - 1e-9
        jp = os.path.join(OUT, f"peer_{peer}.fulleval.json")
        doc = json.load(open(jp))
        doc["dial"] = {"curves": curves, "mono_pct": (ok / steps if steps else None),
                       "tied_pct": None,
                       "provenance": ("scored dial-grid ladder (persisted 2026-07-27 pixels); "
                                      "mono = per-(image,codec) non-decreasing step fraction, "
                                      "peer-computed (presentation-grade); "
                                      + ("negated (distance metric)" if sign < 0 else "as-is"))}
        json.dump(doc, open(jp, "w"), indent=1)
        print(f"{peer}: {sum(len(v) for v in curves.values())} curve points, "
              f"codecs {sorted(curves)}, mono {ok}/{steps} = {ok/steps:.3f}")

if __name__ == "__main__":
    main()
