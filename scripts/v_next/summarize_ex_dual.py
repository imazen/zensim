#!/usr/bin/env python3
"""Summarize an EX-DUAL λ-sweep: read each verdict.md, extract aggregate
SROCC/PLCC/PWRC/Z-RMSE per corpus, emit a comparison table."""
import re
import sys
from pathlib import Path


def parse_verdict(path: Path) -> dict:
    """Return {corpus: {SROCC, PLCC, KROCC, OR, PWRC, Z-RMSE, n}}."""
    text = path.read_text()
    out = {}
    # Match "## CORPUS (n=N)" then later "| V_X bake | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |"
    # Verdict format from bake_verdict bin.
    for m in re.finditer(r"## ([A-Za-z0-9_-]+) \(n=(\d+)\).*?\| V_X bake \| ([\d.\-]+) \| ([\d.\-]+) \| ([\d.\-]+) \| ([\d.\-]+) \| ([\d.\-]+) \| ([\d.\-]+) \|", text, re.DOTALL):
        corpus, n, srocc, plcc, krocc, or_, pwrc, zrmse = m.groups()
        out[corpus] = {
            "n": int(n),
            "SROCC": float(srocc),
            "PLCC": float(plcc),
            "KROCC": float(krocc),
            "OR": float(or_),
            "PWRC": float(pwrc),
            "Z-RMSE": float(zrmse),
        }
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("USAGE: summarize_ex_dual.py <out_dir>")
        sys.exit(2)
    out_dir = Path(sys.argv[1])
    verdict_files = sorted(out_dir.glob("exdual_l*_seed*.verdict.md"))
    if not verdict_files:
        print(f"No verdict files in {out_dir}")
        sys.exit(1)
    rows = []
    for vf in verdict_files:
        m = re.search(r"exdual_l([\d.]+)_seed(\d+)\.verdict\.md", vf.name)
        if not m:
            continue
        lam, seed = m.groups()
        scores = parse_verdict(vf)
        rows.append((lam, seed, vf.name, scores))

    corpora = ["CID22", "KADIK10k", "TID2013", "KonJND-1k", "AIC-3"]
    print(f"# EX-DUAL sweep summary ({len(rows)} bakes)\n")
    print("| λ | seed | " + " | ".join(f"{c} SROCC | {c} Z-RMSE" for c in corpora) + " |")
    print("|---|---|" + "---|---|" * len(corpora))
    for lam, seed, name, scores in rows:
        cells = []
        for c in corpora:
            sc = scores.get(c, {})
            srocc = sc.get("SROCC")
            zrmse = sc.get("Z-RMSE")
            cells.append(f"{srocc:.4f}" if srocc is not None else "—")
            cells.append(f"{zrmse:.3f}" if zrmse is not None else "—")
        print(f"| {lam} | {seed} | " + " | ".join(cells) + " |")
    print("\n## Full panel per λ\n")
    for lam, seed, name, scores in rows:
        print(f"### λ={lam} seed={seed} (`{name}`)\n")
        print("| corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for c in corpora:
            sc = scores.get(c)
            if sc is None:
                continue
            print(
                f"| {c} | {sc['n']} | {sc['SROCC']:.4f} | {sc['PLCC']:.4f} | "
                f"{sc['KROCC']:.4f} | {sc['OR']:.4f} | {sc['PWRC']:.4f} | {sc['Z-RMSE']:.3f} |"
            )
        print()


if __name__ == "__main__":
    main()
