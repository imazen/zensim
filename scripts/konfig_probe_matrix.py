#!/usr/bin/env python3
"""APPENDIX L — the KonFiG weight-probe PAIRED Δ TABLE (pre-reg e93eba04).

Registration: `benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX L
(§L.7 cells, §L.8 bands + decision rule, §L.9 outcomes). Every threshold is read
from that registration; nothing is chosen here.

**This script computes NO statistics** (the wave-10 matrix discipline): every
scalar is READ from an owner's output via `wave10_matrix.load` (freeze_check
--tsv + the fulleval JSON's own fields). The only arithmetic is the registered
§L.8 difference `Δ_{w,s}(a) = value(KFG_w, s, a) − value(W11, s, a)` and its
comparison against the frozen §H.4 band, which is the experiment's decision
rule, not a statistic.

Cell sources:
  * KFG cells       <- the STANDARD fulleval dir (harvested by the probe lane)
  * W11 baselines   <- the konfig session's pair-harvest dir (SAME gate-passed
                       instrument build as the KFG cells; isolation documented
                       in benchmarks/konfig/audit_2026-08-05.meta.md)
  * wave-11 k=8 dispersion reference <- the standard dir's sota944_W11_* /
                       W11_* fullevals when present (reported as context only;
                       absent cells are listed, never invented)

Usage:
    konfig_probe_matrix.py [--out-dir benchmarks/konfig]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import wave10_matrix as w10  # the owner of the read helpers + frozen bands

SEEDS = [4101, 4103]
DOSES = [("KFG25", 0.25), ("KFG75", 0.75)]
# Cared-about axes for the §L.9 outcome call = the registered H.3 endpoint set
# that carries a band (§L.8). KonJND + HF-NL are the thesis axes (L.1) and are
# flagged in the rendering; the outcome rule reads ALL banded axes.
AXES = list(w10.BANDS)


def pick(cells: dict, name: str) -> dict | None:
    return cells.get(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fulleval-dir", default="/mnt/v/output/zensim/reports/fulleval")
    ap.add_argument("--pairharvest-dir",
                    default="/mnt/v/output/zensim/konfig944/pairharvest/fulleval")
    ap.add_argument("--out-dir", default=str(REPO / "benchmarks" / "konfig"))
    a = ap.parse_args()
    outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)

    kfg_names = [f"{tag}_s{s}" for tag, _ in DOSES for s in SEEDS]
    kfg = w10.load(Path(a.fulleval_dir), kfg_names)
    base = w10.load(Path(a.pairharvest_dir), [f"W11_s{s}" for s in SEEDS])

    # wave-11 k=8 dispersion reference (context): standard-dir stems, both the
    # sota944_-prefixed and plain forms, whichever exist.
    w11_family = []
    for s in (4101, 4103, 4105, 4107, 4109, 4111):
        for stem in (f"sota944_W11_s{s}", f"W11_s{s}"):
            if (Path(a.fulleval_dir) / f"{stem}.fulleval.json").exists():
                w11_family.append(stem)
                break
    fam = w10.load(Path(a.fulleval_dir), w11_family) if w11_family else {}

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cells_tsv = outd / f"konfig_probe_cells_{ts}.tsv"
    with open(cells_tsv, "w") as f:
        f.write("cell\trole\t" + "\t".join(AXES + w10.NO_BAND) + "\n")
        for name in [f"W11_s{s}" for s in SEEDS]:
            c = pick(base, name)
            if c:
                f.write(name + "\tbaseline\t" +
                        "\t".join(str(c.get(x)) for x in AXES + w10.NO_BAND) + "\n")
        for name in kfg_names:
            c = pick(kfg, name)
            if c:
                f.write(name + "\tprobe\t" +
                        "\t".join(str(c.get(x)) for x in AXES + w10.NO_BAND) + "\n")

    # ---- the registered paired Δ table -----------------------------------
    matrix_tsv = outd / f"konfig_probe_matrix_{ts}.tsv"
    outside = {tag: [] for tag, _ in DOSES}
    with open(matrix_tsv, "w") as f:
        f.write("dose\taxis\tdelta_s4101\tdelta_s4103\tmean_delta\tband\t"
                "signs_agree\tcall\n")
        for tag, w in DOSES:
            for ax in AXES:
                band, _hib = w10.BANDS[ax]
                ds = {}
                for s in SEEDS:
                    kc = pick(kfg, f"{tag}_s{s}")
                    bc = pick(base, f"W11_s{s}")
                    kv = kc.get(ax) if kc else None
                    bv = bc.get(ax) if bc else None
                    ds[s] = (kv - bv) if (kv is not None and bv is not None) else None
                have = [d for d in ds.values() if d is not None]
                if len(have) < 2:
                    call = "NOT-MEASURABLE"
                    mean_d = have[0] if have else None
                    agree = ""
                else:
                    mean_d = sum(have) / 2
                    agree = (ds[SEEDS[0]] > 0) == (ds[SEEDS[1]] > 0)
                    call = ("OUTSIDE" if abs(mean_d) > band and agree else "inside")
                    if call == "OUTSIDE":
                        outside[tag].append((ax, mean_d))
                f.write(f"{tag}\t{ax}\t{ds[SEEDS[0]]}\t{ds[SEEDS[1]]}\t{mean_d}\t"
                        f"{band}\t{agree}\t{call}\n")

    # ---- k=8 dispersion reference (context) ------------------------------
    fam_tsv = outd / f"konfig_probe_w11family_{ts}.tsv"
    with open(fam_tsv, "w") as f:
        f.write("axis\tn_cells\tmin\tmax\n")
        for ax in AXES:
            vals = [c.get(ax) for c in fam.values() if c.get(ax) is not None]
            f.write(f"{ax}\t{len(vals)}\t{min(vals) if vals else ''}\t"
                    f"{max(vals) if vals else ''}\n")

    print(f"wrote {cells_tsv.name}, {matrix_tsv.name}, {fam_tsv.name}")
    for tag, _ in DOSES:
        if outside[tag]:
            print(f"{tag}: OUTSIDE-NOISE axes: "
                  + ", ".join(f"{ax} ({d:+.4f})" for ax, d in outside[tag]))
        else:
            print(f"{tag}: all axes inside noise")
    return 0


if __name__ == "__main__":
    main()
