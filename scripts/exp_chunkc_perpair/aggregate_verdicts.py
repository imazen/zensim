#!/usr/bin/env python3
"""Parse bake_verdict markdowns and emit a comparison table.

Reads:
  - chunkc_s{1..5}_verdict.md
  - ship_balanced_verdict.md
  - ship_compression_verdict.md

Emits per-corpus aggregate SROCC, PLCC, KROCC, OR, PWRC, Z-RMSE.
"""
import argparse
import re
import sys
from pathlib import Path


def parse_aggregate(text, corpus):
    """Parse the per-corpus aggregate row.

    Looks for:
      ## <corpus> (n=N)
      ### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)
      | Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
      |---|---:|---:|---:|---:|---:|---:|
      | V_X bake | X | X | X | X | X | X |
    """
    # Find the section start
    corpus_label_map = {
        "cid22": "CID22",
        "kadid": "KADIK10k",
        "tid": "TID2013",
        "konjnd": "KonJND-1k (full)",
        "aic3": "AIC-3 CTC",
    }
    label = corpus_label_map[corpus]
    # Find the section
    sec_pat = re.compile(rf"^## {re.escape(label)} \(n=(\d+)\)", re.M)
    m = sec_pat.search(text)
    if not m:
        return None
    start = m.start()
    # Find next "## " section
    nxt = re.search(r"^## ", text[start + 1:], re.M)
    end = (start + 1 + nxt.start()) if nxt else len(text)
    sec = text[start:end]
    # Find the aggregate table
    agg_pat = re.compile(r"\| V_X bake \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|")
    am = agg_pat.search(sec)
    if not am:
        return None
    return {
        "n": int(m.group(1)),
        "srocc": float(am.group(1)),
        "plcc": float(am.group(2)),
        "krocc": float(am.group(3)),
        "or": float(am.group(4)),
        "pwrc": float(am.group(5)),
        "z_rmse": float(am.group(6)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verdicts-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    vdir = Path(a.verdicts_dir)
    corpora = ["cid22", "kadid", "tid", "konjnd", "aic3"]
    bakes = {
        "chunkc_s1": vdir / "chunkc_s1_verdict.md",
        "chunkc_s2": vdir / "chunkc_s2_verdict.md",
        "chunkc_s3": vdir / "chunkc_s3_verdict.md",
        "chunkc_s4": vdir / "chunkc_s4_verdict.md",
        "chunkc_s5": vdir / "chunkc_s5_verdict.md",
        "ship_balanced (V_22-mix-LARGE+iwssim)": vdir / "ship_balanced_verdict.md",
        "ship_compression (V_24-per-sample-α s4)": vdir / "ship_compression_verdict.md",
    }
    rows = {}
    for tag, path in bakes.items():
        if not path.exists():
            print(f"WARN: missing {path}", file=sys.stderr)
            continue
        text = path.read_text()
        rows[tag] = {}
        for c in corpora:
            r = parse_aggregate(text, c)
            rows[tag][c] = r

    # Emit markdown table per corpus
    lines = []
    lines.append("# EXP-CHUNKC-PERPAIR — Verdict Comparison")
    lines.append("")
    for c in corpora:
        lines.append(f"## {c}")
        lines.append("")
        lines.append("| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |")
        lines.append("|---|--:|---:|---:|---:|---:|---:|---:|")
        for tag, per_corpus in rows.items():
            r = per_corpus.get(c)
            if r is None:
                lines.append(f"| {tag} | — | — | — | — | — | — | — |")
                continue
            lines.append(f"| {tag} | {r['n']} | {r['srocc']:.4f} | {r['plcc']:.4f} | {r['krocc']:.4f} | {r['or']:.4f} | {r['pwrc']:.4f} | {r['z_rmse']:.4f} |")
        lines.append("")

    # Summary: per-seed CI, median row
    lines.append("## Summary: per-seed median + Compression-ship delta")
    lines.append("")
    lines.append("| Corpus | seeds CID22 median | ship_compression | Δ vs ship_compression | ship_balanced | Δ vs ship_balanced |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for c in corpora:
        seed_sroccs = []
        for s in range(1, 6):
            r = rows.get(f"chunkc_s{s}", {}).get(c)
            if r is not None:
                seed_sroccs.append(r["srocc"])
        if not seed_sroccs:
            continue
        seed_sroccs.sort()
        median = seed_sroccs[len(seed_sroccs) // 2] if seed_sroccs else None
        ship_c = rows.get("ship_compression (V_24-per-sample-α s4)", {}).get(c)
        ship_b = rows.get("ship_balanced (V_22-mix-LARGE+iwssim)", {}).get(c)
        d_c = (median - ship_c["srocc"]) if (ship_c and median is not None) else None
        d_b = (median - ship_b["srocc"]) if (ship_b and median is not None) else None
        sc = f"{ship_c['srocc']:.4f}" if ship_c else "NA"
        sb = f"{ship_b['srocc']:.4f}" if ship_b else "NA"
        dc_s = f"{d_c:+.4f}" if d_c is not None else "NA"
        db_s = f"{d_b:+.4f}" if d_b is not None else "NA"
        lines.append(f"| {c} | {median:.4f} | {sc} | {dc_s} | {sb} | {db_s} |")

    Path(a.out).write_text("\n".join(lines))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
