#!/usr/bin/env python3
"""WAVE 11 — the pooled k=8 family table + registered per-axis outcome calls.

Registration: `benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX K
(§K.4 aggregation, §K.5 outcome tiers). Every threshold is read from that
registration; nothing is chosen here.

**This script computes NO statistics** (wave10_matrix.py discipline). Every
scalar is READ from an owner's output:
  * per-cell endpoint scalars   <- `freeze_check --tsv` (the profile owner)
  * aic3/aic4/imazen26/best_val <- the fulleval JSON's `rank.*.srocc` / `repro`
  * selection                   <- `freeze_check --select` (captured verbatim)
The only arithmetic is the REGISTERED aggregation (median / min / max over the
pooled 8) and the registered band comparisons of §K.5 — the experiment's
decision rule, not a statistic.

Usage:
    wave11_family.py [--fulleval-dir DIR] [--out-dir benchmarks/wave11]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FREEZE = os.environ.get("ZL_FREEZE_CHECK",
                        str(Path(os.environ.get("CARGO_TARGET_DIR", REPO / "target")) /
                            "release" / "freeze_check"))

W11_SEEDS = [4101, 4103, 4105, 4107, 4109, 4111]      # K.2, frozen
FAMILY = [f"W11_s{s}" for s in W11_SEEDS] + ["W10L9_s4001", "W10L9_s4003"]
L9_PAIR = ["W10L9_s4001", "W10L9_s4003"]
INCUMBENT = ["H_co3abpg_s2501", "H_co3abpg_s2503", "H_co3abpg_s2507"]

# §H.4 bands, unchanged (K.4). higher-is-better on every banded axis.
BANDS = {
    "cid22": 0.010, "konjnd_abs": 0.076, "nonphoto": 0.010, "csiq": 0.096,
    "live": 0.050, "m3a": 0.092, "sdr25": 0.020, "aic3": 0.011, "aic4": 0.010,
    "imazen26": 0.010, "hfnl_perref": 0.247, "mono": 0.024,
}
HEADLINE = ["cid22", "konjnd_abs", "live", "hfnl_perref", "mono"]   # K.4, frozen
NO_BAND = ["kadid_signed", "tid_signed", "tied", "n_pass",
           "bal_composite", "product_composite", "best_val"]


def freeze_tsv(paths: list[Path]) -> dict[str, dict]:
    hdr = subprocess.run([FREEZE, "--tsv-header"], capture_output=True, text=True,
                         check=True).stdout.strip().split("\t")
    out = {}
    for p in paths:
        r = subprocess.run([FREEZE, "--fulleval", str(p),
                            "--profile", "balanced-2026-08-04", "--tsv"],
                           capture_output=True, text=True)
        row = [ln for ln in r.stdout.splitlines() if ln.strip() and not ln.startswith("#")]
        if not row:
            print(f"  WARN no tsv row for {p.name}: {r.stderr.strip()[:200]}", file=sys.stderr)
            continue
        d = dict(zip(hdr, row[-1].split("\t")))
        out[d.get("name") or p.name.replace(".fulleval.json", "")] = d
    return out


def num(v):
    if isinstance(v, str) and "/" in v:
        try:
            return float(v.split("/", 1)[0])
        except ValueError:
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(fe_dir: Path, names: list[str]) -> dict[str, dict]:
    paths = [fe_dir / f"{n}.fulleval.json" for n in names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"  MISSING fullevals ({len(missing)}): {', '.join(missing)}", file=sys.stderr)
    have = [p for p in paths if p.exists()]
    tsv = freeze_tsv(have)
    cells = {}
    for p in have:
        name = p.name.replace(".fulleval.json", "")
        j = json.loads(p.read_text())
        row = {k: num(v) for k, v in tsv.get(name, {}).items()}
        row["name"] = name
        for c in ("aic3", "aic4", "imazen26"):
            row[c] = (j.get("rank", {}).get(c) or {}).get("srocc")
        row["best_val"] = (j.get("repro") or {}).get("best_val")
        row["verdict"] = tsv.get(name, {}).get("verdict")
        row["fails"] = tsv.get(name, {}).get("fails")
        cells[name] = row
    return cells


def classify(axis: str, med: float, pair_lo: float, pair_hi: float,
             inc_mean: float) -> str:
    """The K.5 tier for one axis (all axes higher-better)."""
    band = BANDS[axis]
    if pair_lo <= med <= pair_hi:
        return "HOLDS"
    dist = (pair_lo - med) if med < pair_lo else (med - pair_hi)
    if dist <= band:
        return "HOLDS-WITHIN-NOISE"
    if med < inc_mean - band:
        return "COLLAPSE"
    if med < pair_lo:                       # regressed toward the incumbent
        survives = med > inc_mean + band
        return f"REGRESSION({'survives' if survives else 'not-survived'})"
    return "ABOVE-RANGE"                    # better than the pair by > band


def fmt(v, nd=5):
    return "" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fulleval-dir", default="/mnt/v/output/zensim/reports/fulleval")
    ap.add_argument("--out-dir", default=str(REPO / "benchmarks" / "wave11"))
    args = ap.parse_args()
    fe = Path(args.fulleval_dir)
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)

    fam = load(fe, FAMILY)
    inc = load(fe, INCUMBENT)
    if len(fam) < len(FAMILY):
        print("REFUSING: family incomplete", file=sys.stderr)
        return 1

    axes = list(BANDS) + NO_BAND
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── per-cell table ────────────────────────────────────────────────────
    cells_tsv = outd / f"wave11_cells_{date}.tsv"
    with cells_tsv.open("w") as f:
        f.write("cell\tverdict\tfails\t" + "\t".join(axes) + "\n")
        for n in FAMILY + INCUMBENT:
            row = fam.get(n) or inc.get(n)
            f.write(n + "\t" + str(row.get("verdict")) + "\t" + str(row.get("fails")) +
                    "\t" + "\t".join(fmt(row.get(a)) for a in axes) + "\n")

    # ── the K.4/K.5 summary ──────────────────────────────────────────────
    summary_tsv = outd / f"wave11_family_summary_{date}.tsv"
    calls = {}
    with summary_tsv.open("w") as f:
        f.write("axis\tband\tfam_median\tfam_min\tfam_max\tpair_lo\tpair_hi\t"
                "incumbent_mean\theadline\tcall\n")
        for a in axes:
            vals = [fam[n][a] for n in FAMILY if fam[n].get(a) is not None]
            med = statistics.median(vals) if vals else None
            lo, hi = (min(vals), max(vals)) if vals else (None, None)
            pv = [fam[n][a] for n in L9_PAIR if fam[n].get(a) is not None]
            plo, phi = (min(pv), max(pv)) if pv else (None, None)
            iv = [inc[n][a] for n in INCUMBENT if inc.get(n, {}).get(a) is not None]
            imean = sum(iv) / len(iv) if iv else None
            call = ""
            if a in BANDS and None not in (med, plo, phi, imean):
                call = classify(a, med, plo, phi, imean)
                calls[a] = call
            f.write(f"{a}\t{BANDS.get(a, '')}\t{fmt(med)}\t{fmt(lo)}\t{fmt(hi)}\t"
                    f"{fmt(plo)}\t{fmt(phi)}\t{fmt(imean)}\t"
                    f"{'Y' if a in HEADLINE else ''}\t{call}\n")

    # ── selection (owner output, captured verbatim) ──────────────────────
    sel_txt = outd / f"wave11_select_{date}.txt"
    sel = subprocess.run([FREEZE, "--select"] +
                         [str(fe / f"{n}.fulleval.json") for n in FAMILY],
                         capture_output=True, text=True)
    sel_txt.write_text(sel.stdout + ("\n--- stderr ---\n" + sel.stderr if sel.stderr.strip() else ""))

    # ── meta ─────────────────────────────────────────────────────────────
    def _git(*cmd):
        try:
            return subprocess.run(["git", "-C", str(REPO), *cmd],
                                  capture_output=True, text=True).stdout.strip()
        except Exception:
            return ""
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "registration": "benchmarks/sota944_campaign_2026-08-03.md REGISTERED APPENDIX K",
        "family": FAMILY, "l9_pair": L9_PAIR, "incumbent": INCUMBENT,
        "bands_H4": BANDS, "headline_axes": HEADLINE,
        "rule_K5": "HOLDS iff median in pair range; WITHIN-NOISE iff dist<=band; "
                   "COLLAPSE iff median < incumbent_mean - band; REGRESSION else "
                   "(survives iff median > incumbent_mean + band)",
        "stat_source": "freeze_check --tsv + fulleval rank.*.srocc + repro.best_val; "
                       "median/min/max is the registered K.4 aggregation",
        "headline_calls": calls,
        "commit": _git("rev-parse", "HEAD") or None,
    }
    (outd / f"wave11_family_{date}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {cells_tsv}\n      {summary_tsv}\n      {sel_txt}")
    for a in HEADLINE:
        print(f"  {a:12s} -> {calls.get(a, '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
