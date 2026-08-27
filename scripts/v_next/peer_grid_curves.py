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

CSRC = {
    "ssim2": ("corruption_ssim2_gpu.tsv", +1),
    "butteraugli": ("corruption_butteraugli_gpu.tsv", -1),
    "iwssim": ("corruption_iwssim_gpu.tsv", +1),
    "cvvdp": ("corruption_cvvdp.tsv", +1),
}

def corruption_blocks():
    """Peer `corruption` blocks matching the board shape {n_triples, pass_q10,
    pass_q20, per_family}: an entry PASSES when the metric scores the
    corruption below its q10/q20 anchor of the same base name (the
    corruption-gate semantics; entry = gb82_dog__<family>__<params>__{corruption,q10,q20})."""
    for peer, (tsv, sign) in CSRC.items():
        p = os.path.join(R, tsv)
        if not os.path.exists(p):
            print(f"corr {peer}: absent — skipped"); continue
        val = {}
        col = None
        for r in csv.DictReader(open(p), delimiter="\t"):
            if col is None:
                col = [c for c in r if c not in ("ref_path", "dist_path", "entry")][-1]
            try:
                val[r["entry"]] = sign * float(r[col])
            except (TypeError, ValueError):
                pass
        bases = {}
        for e, v in val.items():
            for suf in ("__corruption", "__q10", "__q20"):
                if e.endswith(suf):
                    bases.setdefault(e[: -len(suf)], {})[suf[2:]] = v
        n = p10 = p20 = 0
        fam = {}
        for b, d in bases.items():
            if "corruption" not in d or "q10" not in d or "q20" not in d:
                continue
            n += 1
            f = b.split("__")[1] if "__" in b else "?"
            fam.setdefault(f, [0, 0, 0])
            fam[f][0] += 1
            if d["corruption"] < d["q10"]: p10 += 1; fam[f][1] += 1
            if d["corruption"] < d["q20"]: p20 += 1; fam[f][2] += 1
        jp = os.path.join(OUT, f"peer_{peer}.fulleval.json")
        doc = json.load(open(jp))
        doc["corruption"] = {"n_triples": n, "pass_q10": p10 / n if n else None,
                             "pass_q20": p20 / n if n else None,
                             "per_family": {f: {"n": c[0], "pass_q10": c[1] / c[0], "pass_q20": c[2] / c[0]}
                                            for f, c in sorted(fam.items())},
                             "provenance": "peer-scored persisted corruption_gate PNGs; gate semantics (corruption below its own q-anchors)"}
        json.dump(doc, open(jp, "w"), indent=1)
        print(f"corr {peer}: {n} triples, pass_q10 {p10/n:.3f}, pass_q20 {p20/n:.3f}")

if __name__ == "__main__":
    main()
    corruption_blocks()
