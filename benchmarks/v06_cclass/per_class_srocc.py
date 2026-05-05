#!/usr/bin/env python3
"""Compute per-content-class SROCC on CID22 from a perpair eval CSV.

The compare_v06.py script bands by ssim2 score, but the rebalance
hypothesis specifically asks whether content-class conditioning helps
on non-photo content. We cluster CID22 reference images into content
classes using zenanalyze features (edge_density, dct_compressibility,
flat_color_block_ratio) and report SROCC per cluster.

Usage:
  per_class_srocc.py <perpair.csv> <zenanalyze_union_cclass.tsv> [out.md]
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load_perpair(path: str) -> dict:
    """Returns dict of {dataset: {ref_stem: {h:[...], v04:[...]}}}"""
    out = defaultdict(lambda: defaultdict(lambda: {"h": [], "v04": []}))
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                h = float(r["human_score"])
                v = float(r["v04_distance"])
            except (ValueError, KeyError):
                continue
            if not (np.isfinite(h) and np.isfinite(v)):
                continue
            stem = r.get("reference_stem") or ""
            if not stem:
                ref = r.get("reference_path") or r.get("ref_path") or r.get("reference") or ""
                if not ref:
                    continue
                stem = Path(ref).stem
            ds = r["dataset"]
            out[ds][stem]["h"].append(h)
            out[ds][stem]["v04"].append(v)
    return out


def load_cclass_tsv(path: str) -> dict:
    """Returns {stem: cclass_name} from cclass_photo/screen/lineart/synthetic/document."""
    out = {}
    cclass_cols = ["cclass_photo", "cclass_screen", "cclass_lineart",
                   "cclass_synthetic", "cclass_document"]
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            stem = parts[0]
            for c in cclass_cols:
                if c in idx:
                    try:
                        v = float(parts[idx[c]])
                        if v > 0.5:
                            out[stem] = c.replace("cclass_", "")
                            break
                    except ValueError:
                        pass
    return out


def srocc(a, b):
    if len(a) < 4:
        return float("nan")
    return abs(stats.spearmanr(a, b).correlation)


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: per_class_srocc.py <perpair.csv> <cclass.tsv> [out.md]", file=sys.stderr)
        return 2
    perpair = sys.argv[1]
    cclass = sys.argv[2]
    out_md = sys.argv[3] if len(sys.argv) > 3 else None

    by_ds = load_perpair(perpair)
    cclass_map = load_cclass_tsv(cclass)

    lines = []
    lines.append(f"# Per-content-class SROCC on {Path(perpair).name}\n")
    lines.append(f"\nCID22 reference stems are bucketed by `cclass_*` columns from\n")
    lines.append(f"`{Path(cclass).name}` (synthetic-v2 stem-prefix heuristic).\n\n")
    for ds in sorted(by_ds.keys()):
        if ds != "CID22":
            continue
        per_class = defaultdict(lambda: {"h": [], "v04": []})
        unclassified_h = []
        unclassified_v = []
        unclass_stems = set()
        for stem, vs in by_ds[ds].items():
            cc = cclass_map.get(stem)
            if cc is None:
                unclassified_h.extend(vs["h"])
                unclassified_v.extend(vs["v04"])
                unclass_stems.add(stem)
                continue
            per_class[cc]["h"].extend(vs["h"])
            per_class[cc]["v04"].extend(vs["v04"])
        all_h = []
        all_v = []
        for stem, vs in by_ds[ds].items():
            all_h.extend(vs["h"])
            all_v.extend(vs["v04"])
        global_srocc = srocc(all_v, all_h)

        lines.append(f"## {ds} (global SROCC={global_srocc:.4f}, n_pairs={len(all_h)}, n_refs={len(by_ds[ds])})\n\n")
        lines.append("| cclass | n_refs | n_pairs | SROCC |\n")
        lines.append("|---|--:|--:|--:|\n")
        for cc in ["photo", "screen", "lineart", "synthetic", "document"]:
            d = per_class.get(cc)
            if not d:
                lines.append(f"| {cc} | 0 | 0 | — |\n")
                continue
            n_refs = sum(1 for stem, vs in by_ds[ds].items() if cclass_map.get(stem) == cc)
            r = srocc(d["v04"], d["h"])
            lines.append(f"| {cc} | {n_refs} | {len(d['h'])} | {r:.4f} |\n")
        if unclass_stems:
            r = srocc(unclassified_v, unclassified_h) if unclassified_h else float("nan")
            r_str = f"{r:.4f}" if np.isfinite(r) else "—"
            lines.append(f"| (unclassified) | {len(unclass_stems)} | {len(unclassified_h)} | {r_str} |\n")
        lines.append("\n")
        if unclass_stems:
            lines.append(f"<details><summary>{len(unclass_stems)} CID22 refs not in cclass TSV</summary>\n\n")
            for s in sorted(unclass_stems):
                lines.append(f"- `{s}`\n")
            lines.append("\n</details>\n\n")

    text = "".join(lines)
    if out_md:
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(out_md).write_text(text)
        print(f"wrote {out_md}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
