#!/usr/bin/env python3
"""CID22-B on the 23 clean references — a correction of the SAME spent read.

The verdict lane's single registered CID22-B read (2026-09-21, 24 refs,
2,100 rows) included `3316926_opo25u.png`, which is the same picture as the
CID22-A reference `844297.png` (dHash 0, luma NCC 0.9999; audit in
`benchmarks/dvifmish_cid22_audit_2026-09-22.md`). This script re-scores the
already-exposed per-row scores of that read with that reference's rows
removed. No label is read that was not already read; no model is refit.

It reuses the verdict lane's own statistics (`build_verdict.metrics`,
`build_verdict.ref_boot_delta`: SROCC/KROCC/5-parameter-logistic PLCC and a
2,000-draw reference-cluster bootstrap, seed 20260920).

Usage: cid22b_clean23.py <verdict_scores_dir> <scratch_dir> <out.json>
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verdict-x1x2"))
import build_verdict as bv  # noqa: E402

DUP = "3316926_opo25u.png"
METHODS = ["dvifm_gate", "fastssim2", "bake_prof_b", "bake_prof_d",
           "bake_r915_basic228", "bake_r915_y60"]


def filtered_copy(src, dst):
    rows = list(csv.DictReader(open(src)))
    keep = [r for r in rows if Path(r["ref_path"]).name != DUP]
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(keep)
    return len(rows), len(keep)


def main():
    sdir, scratch, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    scratch.mkdir(parents=True, exist_ok=True)
    res = {"duplicate_removed": DUP, "same_as": "844297.png (CID22-A)",
           "kind": "correction of the single spent CID22-B read (no new exposure)",
           "all24": {}, "clean23": {}, "bootstrap24": {}, "bootstrap23": {}}
    paths24, paths23 = {}, {}
    for m in METHODS:
        p = sdir / f"{m}__on__cid22b.csv"
        q = scratch / f"{m}__on__cid22b23.csv"
        n_all, n_keep = filtered_copy(p, q)
        paths24[m], paths23[m] = p, q
        for key, path in (("all24", p), ("clean23", q)):
            ref, tgt, sc, _ = bv.read_scores(path)
            d = bv.metrics(ref, tgt, sc)
            d["n"] = int(len(tgt))
            d["n_refs"] = int(len(set(ref)))
            res[key][m] = d
    for m in METHODS[1:]:
        res["bootstrap24"][f"dvifm_gate-vs-{m}"] = bv.ref_boot_delta(paths24[m], paths24["dvifm_gate"])
        res["bootstrap23"][f"dvifm_gate-vs-{m}"] = bv.ref_boot_delta(paths23[m], paths23["dvifm_gate"])
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"{'method':22} {'SROCC24':>8} {'SROCC23':>8} {'KROCC24':>8} {'KROCC23':>8} {'PLCC24':>8} {'PLCC23':>8}  n24/n23")
    for m in METHODS:
        a, b = res["all24"][m], res["clean23"][m]
        print(f"{m:22} {a['srocc']:8.4f} {b['srocc']:8.4f} {a['krocc']:8.4f} {b['krocc']:8.4f} "
              f"{a['plcc']:8.4f} {b['plcc']:8.4f}  {a['n']}/{b['n']}")
    for k in res["bootstrap23"]:
        a, b = res["bootstrap24"][k], res["bootstrap23"][k]
        print(f"{k:36} 24: {a['delta_mean']:+.4f} [{a['ci95'][0]:+.4f},{a['ci95'][1]:+.4f}] "
              f"P<=0 {a['p_delta_le_0']:.3f} | 23: {b['delta_mean']:+.4f} "
              f"[{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] P<=0 {b['p_delta_le_0']:.3f}")


if __name__ == "__main__":
    main()
