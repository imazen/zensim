#!/usr/bin/env python3
"""WAVE 10 — build the leave-one-out MARGINAL-VALUE MATRIX.

Registration: `benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX H
(§H.3 endpoints, §H.4 noise bands, §H.5 decision rules). Every threshold below is
read from that registration; nothing is chosen here.

**This script computes NO statistics.** Every number it prints is READ from an
owner's output:
  * per-cell endpoint scalars  <- `freeze_check --tsv` (the bar/profile owner)
  * aic3/aic4/imazen26/best_val <- the fulleval JSON's own `rank.*.srocc` / `repro`
  * floor counts + `--select` rank <- `freeze_check`
The only arithmetic performed is the registered DIFFERENCE of two read values and
the registered comparison of that difference against a frozen band (§H.5), which
is the experiment's decision rule, not a statistic.

Usage:
    wave10_matrix.py [--fulleval-dir DIR] [--out-dir benchmarks/wave10]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FREEZE = os.environ.get("ZL_FREEZE_CHECK",
                        str(Path(os.environ.get("CARGO_TARGET_DIR", REPO / "target")) /
                            "release" / "freeze_check"))

ARMS = [("L0", None), ("L1", "safesyn"), ("L2", "cid22_train"), ("L3", "kadid"),
        ("L4", "tid"), ("L5", "bigcodec"), ("L6", "kadis"), ("L7", "tsafesyn"),
        ("L8", "ttbig"), ("L9", "tkadis"), ("L10", "konjnd_bpg")]
PAIRED_SEEDS = [4001, 4003]          # shared across every arm -> paired comparison
L0_SEEDS = [4001, 4003, 4007]
INCUMBENT = ["H_co3abpg_s2501", "H_co3abpg_s2503", "H_co3abpg_s2507"]

# §H.4 FROZEN noise bands. higher_is_better drives the marginal-value sign only.
BANDS = {
    "cid22":       (0.010, True),
    "konjnd_abs":  (0.076, True),
    "nonphoto":    (0.010, True),
    "csiq":        (0.096, True),
    "live":        (0.050, True),
    "m3a":         (0.092, True),
    "sdr25":       (0.020, True),
    "aic3":        (0.011, True),
    "aic4":        (0.010, True),
    "imazen26":    (0.010, True),
    "hfnl_perref": (0.247, True),
    "mono":        (0.024, True),      # 2.4 pp, expressed as a fraction
}
# Reported with NO band (§H.3/§H.4): KADID's target changed this wave; best_val is
# not comparable across arms; floor count and composites are read-outs.
NO_BAND = ["kadid_signed", "tid_signed", "tied", "n_pass",
           "bal_composite", "product_composite", "best_val"]


def cell(arm: str, seed: int) -> str:
    return f"W10{arm}_s{seed}"


def freeze_tsv(paths: list[Path]) -> dict[str, dict]:
    """One `freeze_check --tsv` row per fulleval. Read, never recomputed."""
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
    """Parse a freeze_check TSV cell. `n_pass` arrives as "7/8" — keep the
    numerator so the floor count is a comparable number; everything else is a
    plain float or blank."""
    if isinstance(v, str) and "/" in v:
        head = v.split("/", 1)[0]
        try:
            return float(head)
        except ValueError:
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(fe_dir: Path, names: list[str]) -> dict[str, dict]:
    """Endpoint scalars per cell: freeze_check TSV + the four the TSV omits."""
    paths = [fe_dir / f"{n}.fulleval.json" for n in names]
    have = [p for p in paths if p.exists()]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"  MISSING fullevals ({len(missing)}): {', '.join(missing)}", file=sys.stderr)
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


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fulleval-dir", default="/mnt/v/output/zensim/reports/fulleval")
    ap.add_argument("--out-dir", default=str(REPO / "benchmarks" / "wave10"))
    a = ap.parse_args()
    fe = Path(a.fulleval_dir)
    outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)

    names = [cell("L0", s) for s in L0_SEEDS]
    for arm, _ in ARMS[1:]:
        names += [cell(arm, s) for s in PAIRED_SEEDS]
    cells = load(fe, names + INCUMBENT)

    axes = list(BANDS) + NO_BAND
    # ---- per-cell table --------------------------------------------------
    with open(outd / "wave10_cells_2026-08-05.tsv", "w") as f:
        f.write("arm\tdropped\tseed\tcell\tverdict\tfails\t" + "\t".join(axes) + "\n")
        for arm, drop in ARMS:
            for s in (L0_SEEDS if arm == "L0" else PAIRED_SEEDS):
                c = cells.get(cell(arm, s))
                if not c:
                    continue
                f.write(f"{arm}\t{drop or '-'}\t{s}\t{c['name']}\t{c.get('verdict')}\t"
                        f"{c.get('fails')}\t" +
                        "\t".join("" if c.get(x) is None else f"{c[x]:.6g}" for x in axes) + "\n")
        for n in INCUMBENT:
            c = cells.get(n)
            if c:
                f.write(f"INCUMBENT\t-\t-\t{n}\t{c.get('verdict')}\t{c.get('fails')}\t" +
                        "\t".join("" if c.get(x) is None else f"{c[x]:.6g}" for x in axes) + "\n")

    # ---- the marginal-value matrix (§H.5 paired rule) ---------------------
    rows = []
    for arm, drop in ARMS[1:]:
        for ax in axes:
            band, hib = BANDS.get(ax, (None, True))
            ds = []
            for s in PAIRED_SEEDS:
                A, B = cells.get(cell(arm, s)), cells.get(cell("L0", s))
                if A and B and A.get(ax) is not None and B.get(ax) is not None:
                    ds.append(A[ax] - B[ax])
            if not ds:
                rows.append(dict(arm=arm, dropped=drop, axis=ax, n=0, call="NO DATA"))
                continue
            m = mean(ds)
            consistent = len(ds) == len(PAIRED_SEEDS) and (
                all(d > 0 for d in ds) or all(d < 0 for d in ds))
            if band is None:
                call = "NO BAND (reported only)"
            elif abs(m) > band and consistent:
                call = "OUTSIDE NOISE"
            elif abs(m) > band:
                call = "inside noise (seeds disagree)"
            else:
                call = "inside noise"
            rows.append(dict(arm=arm, dropped=drop, axis=ax, n=len(ds),
                             delta_mean=m, deltas=ds, band=band,
                             marginal_value=(-m if hib else m), call=call))
    with open(outd / "wave10_marginal_matrix_2026-08-05.tsv", "w") as f:
        f.write("arm\tdropped_leg\taxis\tn_seeds\tdelta_mean\tdelta_s4001\tdelta_s4003"
                "\tband\tmarginal_value\tcall\n")
        for r in rows:
            d = r.get("deltas") or []
            f.write("\t".join([
                r["arm"], str(r["dropped"]), r["axis"], str(r["n"]),
                "" if r.get("delta_mean") is None else f"{r['delta_mean']:+.6g}",
                f"{d[0]:+.6g}" if len(d) > 0 else "",
                f"{d[1]:+.6g}" if len(d) > 1 else "",
                "" if r.get("band") is None else f"{r['band']:g}",
                "" if r.get("marginal_value") is None else f"{r['marginal_value']:+.6g}",
                r["call"]]) + "\n")

    # ---- L0 vs incumbent (unpaired, §H.5) --------------------------------
    with open(outd / "wave10_l0_vs_incumbent_2026-08-05.tsv", "w") as f:
        f.write("axis\tL0_mean\tL0_min\tL0_max\tH_mean\tH_min\tH_max\tdelta\tband"
                "\tranges_overlap\tcall\n")
        for ax in axes:
            L = [cells[cell("L0", s)][ax] for s in L0_SEEDS
                 if cells.get(cell("L0", s)) and cells[cell("L0", s)].get(ax) is not None]
            H = [cells[n][ax] for n in INCUMBENT
                 if cells.get(n) and cells[n].get(ax) is not None]
            if not L or not H:
                f.write(f"{ax}\t\t\t\t\t\t\t\t\t\tNO DATA\n")
                continue
            band = BANDS.get(ax, (None, True))[0]
            d = mean(L) - mean(H)
            overlap = not (min(L) > max(H) or min(H) > max(L))
            call = ("NO BAND (reported only)" if band is None else
                    "OUTSIDE NOISE" if (abs(d) > band and not overlap) else
                    "inside noise (ranges overlap)" if abs(d) > band else "inside noise")
            f.write(f"{ax}\t{mean(L):.6g}\t{min(L):.6g}\t{max(L):.6g}\t{mean(H):.6g}\t"
                    f"{min(H):.6g}\t{max(H):.6g}\t{d:+.6g}\t"
                    f"{'' if band is None else f'{band:g}'}\t{overlap}\t{call}\n")

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "benchmarks/sota944_campaign_2026-08-03.md REGISTERED APPENDIX H",
        "rule_H5": "OUTSIDE NOISE iff |mean paired delta| > band AND both seeds same sign",
        "bands_H4": {k: v[0] for k, v in BANDS.items()},
        "no_band_axes": NO_BAND,
        "stat_source": "freeze_check --tsv (per-cell scalars); fulleval rank.*.srocc for "
                       "aic3/aic4/imazen26; repro.best_val. NO statistic is computed here.",
        "cells_found": sorted(cells),
        "cells_missing": [n for n in names + INCUMBENT if n not in cells],
    }
    (outd / "wave10_matrix_2026-08-05.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {outd}/wave10_{{cells,marginal_matrix,l0_vs_incumbent}}_2026-08-05.tsv")
    print(f"cells found {len(cells)} / {len(names) + len(INCUMBENT)}; "
          f"missing: {meta['cells_missing']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
