#!/usr/bin/env python3
"""Failure-profile artifacts: what failure evidence each board cell carries, and
the ladder-inversion results, as flat tables.

READS ONLY. Every column is copied out of a `*.fulleval.json` (which
`bake_verdict` wrote); nothing is re-derived. Where a statistic does not exist
for a cell the column is the literal `NOT_MEASURED`, never blank and never 0.

Emits into --out-dir:
  failure_inventory_2026-08-31.tsv   one row per board cell x corpus
  ladder_inversions_2026-08-31.tsv   one row per board cell x split x zone
  worst_ladders_2026-08-31.tsv       the named worst ladders per cell
  _MANIFEST.json                     provenance + sha256 of each table
and prints the discussion-set summary the campaign doc quotes.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path

NM = "NOT_MEASURED"
ZONES = ["q<50", "q50-85", "q>=85"]
# corpora whose label is distortion-oriented: a negative SROCC there is the
# CONVENTION, not an inversion (EXPECTED_ORIENTATION registry).
JND = {"konjnd", "aic4", "sdr25"}


def v(x):
    return NM if x is None else x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fulleval-dir", default="/mnt/v/output/zensim/reports/fulleval", type=Path)
    ap.add_argument("--out-dir", default="/mnt/v/output/zensim/failure-profiles-2026-08-31",
                    type=Path)
    ap.add_argument("--sets", default=None, help="comma-separated board names to summarise")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    docs = [json.loads(f.read_text()) for f in sorted(a.fulleval_dir.glob("*.fulleval.json"))]

    inv, lad, worst = [], [], []
    inv.append("\t".join(["bake", "n_inputs", "corpus", "n", "srocc_signed", "per_ref_mean",
                          "frac_refs_backwards", "per_ref_n", "or", "z_rmse", "pwrc",
                          "train_eq_val", "n_bands", "worst_band", "worst_band_srocc",
                          "worst_band_n", "worst_band_span"]))
    lad.append("\t".join(["bake", "grid", "split", "key", "zone", "n_pairs", "inv_material",
                          "inv_rate", "flat", "codec_sat", "n_ladders", "ladders_with_inv",
                          "frac_ladders_with_inv", "ladders_ends_backwards",
                          "frac_ladders_ends_backwards", "inv_mag_med", "inv_mag_max"]))
    worst.append("\t".join(["bake", "image_id", "codec", "content_class", "zone",
                            "end_delta", "n_rungs", "worst_step"]))
    for o in docs:
        name = o.get("name")
        for c, r in sorted((o.get("rank") or {}).items()):
            if not isinstance(r, dict):
                continue
            bands = r.get("bands") or []
            usable = [b for b in bands if b.get("srocc_signed") is not None]
            wb = min(usable, key=lambda b: b["srocc_signed"]) if usable else None
            inv.append("\t".join(str(x) for x in [
                name, v(o.get("n_inputs")), c, v(r.get("n")), v(r.get("srocc_signed")),
                v(r.get("per_ref_mean")), v(r.get("frac_negative")), v(r.get("per_ref_n")),
                v(r.get("or")), v(r.get("z_rmse")), v(r.get("pwrc")), v(r.get("train_eq_val")),
                len(bands) if bands else NM,
                wb["band"] if wb else NM, wb["srocc_signed"] if wb else NM,
                wb["n"] if wb else NM, wb.get("span") if wb else NM]))
        z = (o.get("dial") or {}).get("zones")
        if not isinstance(z, dict):
            continue
        grid = str(z.get("grid") or "").rsplit("/", 1)[-1]
        for c in z.get("cells") or []:
            lad.append("\t".join(str(x) for x in [
                name, grid, c.get("split"), c.get("key"), c.get("zone"), c.get("n_pairs"),
                c.get("inv_material"), v(c.get("inv_rate")), c.get("flat"), c.get("codec_sat"),
                c.get("n_ladders"), c.get("ladders_with_inv"),
                v(c.get("frac_ladders_with_inv")), c.get("ladders_ends_backwards"),
                v(c.get("frac_ladders_ends_backwards")), v(c.get("inv_mag_med")),
                v(c.get("inv_mag_max"))]))
        for w in z.get("worst_ladders") or []:
            worst.append("\t".join(str(x) for x in [
                name, w.get("image_id"), w.get("codec"), w.get("class"), w.get("zone"),
                round(float(w.get("end_delta", 0.0)), 4), w.get("n_rungs"),
                round(float(w.get("worst_step", 0.0)), 4)]))

    files = {}
    for fn, rows in (("failure_inventory_2026-08-31.tsv", inv),
                     ("ladder_inversions_2026-08-31.tsv", lad),
                     ("worst_ladders_2026-08-31.tsv", worst)):
        p = a.out_dir / fn
        p.write_text("\n".join(rows) + "\n")
        files[fn] = {"rows": len(rows) - 1, "bytes": p.stat().st_size,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        print(f"wrote {p}  ({len(rows) - 1} rows)")
    (a.out_dir / "_MANIFEST.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what": "Failure-profile evidence tables: per-corpus failure inventory, the "
                "ladder-inversion split (bake_verdict dial.zones, scheme "
                "ladder-inversion-2026-08-31) and the named worst ladders.",
        "source": "READ from /mnt/v/output/zensim/reports/fulleval/*.fulleval.json — no "
                  "statistic is recomputed here; NOT_MEASURED means the verdict carries "
                  "no such value.",
        "producer": "zensim scripts/v_next/failure_profile_report.py",
        "n_board_cells": len(docs), "files": files}, indent=1))

    # ---- summary for the campaign doc ---------------------------------------
    names = a.sets.split(",") if a.sets else []
    by = {o.get("name"): o for o in docs}
    if names:
        print("\n=== ladder inversions, discussion set ===")
        hdr = ["bake", "grid", "zone", "pairs", "inv", "inv%", "ladders", ">=1 inv",
               "ends bwd", "worst pt"]
        print("\t".join(hdr))
        for n in names:
            o = by.get(n)
            if o is None:
                print(f"{n}\tNOT ON BOARD")
                continue
            z = (o.get("dial") or {}).get("zones")
            if not isinstance(z, dict):
                print(f"{n}\t{NM}")
                continue
            cells = {(c["split"], c["key"], c["zone"]): c for c in z["cells"]}
            for zn in ZONES:
                c = cells.get(("all", "all", zn))
                if not c:
                    continue
                print("\t".join(str(x) for x in [
                    n, str(z.get("grid", "")).rsplit("/", 1)[-1], zn, c["n_pairs"],
                    c["inv_material"], f'{100 * (c["inv_rate"] or 0):.2f}', c["n_ladders"],
                    c["ladders_with_inv"], c["ladders_ends_backwards"],
                    NM if c["inv_mag_max"] is None else f'{c["inv_mag_max"]:.1f}']))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
