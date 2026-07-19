#!/usr/bin/env python3
"""diffmap-RD probe analysis (2026-07-18) — turns the probe matrix into the two
pre-registered verdicts of docs/RD_TARGET_EVAL_DESIGN_2026-07-18.md:

HELPFUL  — bytes saved vs the same codec's no-diffmap baseline at EQUAL
           ACHIEVED JUDGE SCORE (log-bytes interpolation on the baseline's
           per-image score→bytes frontier; never at nominal distance/target).
           Judges: ssim2 + butteraugli + zensim (uniform zenmetrics build) —
           per-judge tables so home-turf cells are visible (#38 convention).
EFFICIENT — passes / encode_ms / |achieved−T| residuals per driver.

Reads:  $RD/jxl/manifest_*.tsv, $RD/zenjpeg/probe.tsv, $RD/judge_{ssim2,butteraugli,zensim}.tsv
Writes: $RD/analysis_summary.md (+ prints it)
"""
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

RD = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim/rd-target-eval-2026-07"
CLASS = {"codec_wiki": "screen", "gmessages": "screen"}  # rest = photo


def load_judges():
    """dist_path -> {judge: score}. Butteraugli negated (higher=better uniformly)."""
    out = defaultdict(dict)
    specs = [
        ("ssim2", f"{RD}/judge_ssim2.tsv", ("ssim2", "ssim2_gpu"), +1),
        ("butter", f"{RD}/judge_butteraugli.tsv", ("butteraugli_pnorm3", "butteraugli_max"), -1),
        ("zensim", f"{RD}/judge_zensim.tsv", ("zensim",), +1),
    ]
    for name, path, cols, sign in specs:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                v = next((r[c] for c in cols if c in r and r[c] not in ("", "nan")), None)
                if v is not None:
                    out[r["dist_path"]][name] = sign * float(v)
    return out


def interp_bytes(frontier, score):
    """log-bytes at `score` on a per-image (score, bytes) frontier; None if outside."""
    pts = sorted(frontier)
    if not pts or score < pts[0][0] or score > pts[-1][0]:
        return None
    for (s0, b0), (s1, b1) in zip(pts, pts[1:]):
        if s0 <= score <= s1:
            if s1 == s0:
                return b0
            t = (score - s0) / (s1 - s0)
            return math.exp(math.log(b0) + t * (math.log(b1) - math.log(b0)))
    return pts[-1][1]


def helpful(cells, baseline_label, judges, codec):
    """cells: list of dicts {image, label, bytes, dist_path}. Returns
    {(judge, driver, class): [bytes_saved_frac,...]}."""
    frontiers = defaultdict(list)  # (judge, image) -> [(score, bytes)]
    for c in cells:
        if c["label"] != baseline_label:
            continue
        for j, s in judges.get(c["dist_path"], {}).items():
            frontiers[(j, c["image"])].append((s, c["bytes"]))
    saved = defaultdict(list)
    for c in cells:
        if c["label"] == baseline_label:
            continue
        for j, s in judges.get(c["dist_path"], {}).items():
            base = interp_bytes(frontiers.get((j, c["image"]), []), s)
            if base and base > 0:
                cls = CLASS.get(c["image"], "photo")
                saved[(j, c["label"], cls)].append(1.0 - c["bytes"] / base)
    return saved


def fmt_saved(saved, drivers, judges=("ssim2", "butter", "zensim")):
    lines = [
        "| driver | class | " + " | ".join(f"{j} med% (n)" for j in judges) + " |",
        "|---|---|" + "---|" * len(judges),
    ]
    for d in drivers:
        for cls in ("photo", "screen"):
            cells = []
            for j in judges:
                v = saved.get((j, d, cls), [])
                cells.append(f"{100*st.median(v):+.1f} ({len(v)})" if v else "—")
            lines.append(f"| {d} | {cls} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    judges = load_judges()
    out = ["# diffmap-RD probe analysis — auto-generated\n"]

    # ---- jxl ----
    jxl = []
    import glob
    for mf in glob.glob(f"{RD}/jxl/manifest_*.tsv"):
        with open(mf) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                jxl.append({
                    "image": r["image"], "label": r["label"],
                    "bytes": int(r["bytes"]), "dist_path": r["dist_path"],
                    "ms": float(r["encode_ms"]), "op": r["distance"],
                })
    if jxl:
        drivers = sorted({c["label"] for c in jxl} - {"none"})
        out.append("## jxl-encoder — bytes saved vs `none` baseline at equal judged score\n")
        out.append(fmt_saved(helpful(jxl, "none", judges, "jxl"), drivers))
        out.append("\n### efficiency (median encode_ms per image class)\n")
        eff = defaultdict(list)
        for c in jxl:
            eff[(c["label"], CLASS.get(c["image"], "photo"))].append(c["ms"])
        out.append("| driver | photo ms | screen ms |")
        out.append("|---|---|---|")
        for d in ["none", *drivers]:
            p = eff.get((d, "photo"), []); s = eff.get((d, "screen"), [])
            out.append(f"| {d} | {st.median(p):.0f} | {st.median(s):.0f} |" if p and s else f"| {d} | — | — |")

    # ---- zenjpeg ----
    zp = f"{RD}/zenjpeg/probe.tsv"
    if os.path.exists(zp):
        zj = []
        with open(zp) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                png = f"{RD}/zenjpeg/{r['driver']}__{r['image']}__t{int(float(r['target']))}.png"
                zj.append({
                    "image": r["image"], "label": r["driver"], "target": float(r["target"]),
                    "bytes": int(r["bytes"]), "achieved": float(r["achieved_score"]),
                    "passes": int(r["passes"]), "ms": float(r["encode_ms"]), "dist_path": png,
                })
        drivers = sorted({c["label"] for c in zj} - {"global"})
        out.append("\n## zenjpeg — bytes saved vs `global` baseline at equal judged score\n")
        out.append(fmt_saved(helpful(zj, "global", judges, "zenjpeg"), drivers))
        out.append("\n### efficiency + targeting residual (B-scale drivers)\n")
        out.append("| driver | med passes | med ms | med abs(achieved−T) | n |")
        out.append("|---|---|---|---|---|")
        for d in ["global", "aq", "picker"]:
            v = [c for c in zj if c["label"] == d]
            if not v:
                continue
            res = [abs(c["achieved"] - c["target"]) for c in v]
            out.append(
                f"| {d} | {st.median([c['passes'] for c in v]):.0f} "
                f"| {st.median([c['ms'] for c in v]):.0f} "
                f"| {st.median(res):.2f} | {len(v)} |"
            )
        v = [c for c in zj if c["label"] == "aq_add156_abs"]
        if v:
            res = [abs(c["achieved"] - c["target"]) for c in v]
            out.append(
                f"| aq_add156_abs (own dial) | {st.median([c['passes'] for c in v]):.0f} "
                f"| {st.median([c['ms'] for c in v]):.0f} | {st.median(res):.2f} | {len(v)} |"
            )

    text = "\n".join(out) + "\n"
    with open(f"{RD}/analysis_summary.md", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
