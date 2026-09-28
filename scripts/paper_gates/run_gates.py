#!/usr/bin/env python3
"""Paper gates lane (2026-09-22): put every peer metric and frozen zensim
profile through the dial-addressability gate (G-ADDR C1-C6 + A7r) on the
floor-dense ladder instrument, through the OWNERS:

* `scripts/v_next/dial_peer_cells.py` turns a per-pair score table into the
  `image_id codec q pred` cell table (key normalisation only);
* `bake_verdict` (peer mode, `--dial-peer-scores` / `--identity-peer-scores`)
  computes every statistic and every verdict, with the registry's operative
  floor rule and the two-reference inversion attribution — the same
  invocation `scripts/gaddr_board_ladder.py` uses for the board's peer rows,
  except `--corpora tid` (the carrier's rank panel only; TRAIN-only data, so
  no held-out human set is read by this lane).

This file computes NO statistic. It declares, per scorer, the ORIENTATION
MAPPING the gate needs (a distance is negated so higher = better) and one
optional affine rescale, then hands the numbers to the owners:

  native   pred = o * value            (o = +1 quality, -1 distance)
  s100     pred = 100 - 100 * (P - o*value) / S
           P = the scorer's perfect (identity) value in oriented units,
           S = p1..p99 span of o*value over the 9,593 ladder cells
           (numpy.percentile, the same span `scripts/aic2026_agreement.py`
           uses for its relative materiality).

`s100` exists because C1's 0.5-point materiality and C5's [97.5, 100] band are
0..100-dial constants: under `s100` 0.5 points is 0.5 % of each scorer's own
robust span on this instrument, and a scorer's perfect value lands on exactly
100. It is an affine map, so strict order (A7r, C2, the scale-free step table)
is identical under both readings.
"""
import argparse, csv, json, os, re, subprocess, sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAD = "/mnt/v/output/zensim/ladder-2026-09-05/instruments"
GRID = f"{LAD}/dial_grid_372col_ladder.parquet"
TRUTH = f"{LAD}/dialcells_ssim2_ladder.tsv"
REFTRUTH = f"{LAD}/reference_truth_ladder_pnorm3.tsv:pnorm3"
D0904 = "/mnt/v/output/zensim/dialgate-2026-09-04"
IDENT = f"{D0904}/identity_probe_372_2026-09-04.parquet"
NEGTAIL = f"{D0904}/negtail_probe_372_2026-09-04.parquet"
NEGTAIL_SSIM2 = f"{D0904}/repin/negtail_peer_ssim2.tsv"
CARRIER = f"{REPO}/zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin"
BAKES = {  # feature-path reads of the shipped 372-wide bakes (dial(0-vector) identity)
    "bake_B": f"{REPO}/zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin",
    "bake_D": f"{REPO}/zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin",
}
S = "/var/tmp/paper-gates/scores"
LI = "/var/tmp/paper-gates/lists"

# name -> (file stem under scores/, column, orientation, perfect value in NATIVE units)
SCORERS = {
    "peer_ssim2_registered": (None, None, +1, 100.0),  # the mentor's registered ladder table
    "peer_fast_ssim2": ("{set}_fastssim2.tsv", "ssim2", +1, 100.0),
    "peer_ssim2_zm": ("{set}_ssim2.parquet", "ssim2", +1, 100.0),
    "peer_butteraugli_pnorm3": ("{set}_butteraugli.parquet", "butteraugli_pnorm3", -1, 0.0),
    "peer_butteraugli_max": ("{set}_butteraugli.parquet", "butteraugli_max", -1, 0.0),
    "peer_dssim": ("{set}_dssim.parquet", "dssim", -1, 0.0),
    "peer_iwssim": ("{set}_iwssim.parquet", "iwssim_cpu_imazen_v0_1_0", +1, 1.0),
    "peer_cvvdp_standard_4k": ("{set}_cvvdp.parquet", "cvvdp_cpu_imazen_v0_1_0", +1, 10.0),
    "peer_cvvdp_standard_fhd": ("{set}_cvvdp_fhd.tsv", "cvvdp_cpu_imazen_v0_1_0_standard_fhd", +1, 10.0),
    "peer_gmsd": ("{set}_gmsd.parquet", "gmsd_cpu_imazen_v0_1_0", -1, 0.0),
    "peer_dvifmish_talk_faithful_luma": ("{set}_dvifmish/talk-faithful-luma.tsv", "distortion", -1, 0.0),
    "peer_dvifmish_serving_gate_ycbcr3": ("{set}_dvifmish/serving-gate-ycbcr3.tsv", "distortion", -1, 0.0),
    "peer_dvifmish_ours_full_luma": ("{set}_dvifmish/ours-full-luma.tsv", "distortion", -1, 0.0),
    "zensim_v0_2": ("{set}_zensim.parquet", "score_v0_2", +1, 100.0),
    "zensim_b": ("{set}_zensim.parquet", "score_b", +1, 100.0),
    "zensim_c": ("{set}_zensim.parquet", "score_c", +1, 100.0),
    "zensim_d": ("{set}_zensim.parquet", "score_d", +1, 100.0),
    "zensim_r915_fast": ("{set}_zensim.parquet", "score_r915_y60_h32_ens5", +1, 100.0),
    "zensim_r915_rich": ("{set}_zensim.parquet", "score_r915_basic228_h128_ens5", +1, 100.0),
}


def read_scores(stem: str, col: str, set_: str) -> pd.DataFrame:
    """Per-pair scores for one scorer, joined back to the pair list by dist path
    and the list's row index `q`; refuses unless every list row is covered once."""
    lst = pd.read_csv(f"{LI}/{set_}.tsv", sep="\t", dtype=str, keep_default_na=False)
    path = f"{S}/{stem.format(set=set_)}"
    if path.endswith(".parquet"):
        t = pq.read_table(path).to_pandas()
        key = t["image_path"].astype(str) + "#" + t["q"].astype(str)
    else:
        t = pd.read_csv(path, sep="\t", dtype={"q": str}, keep_default_na=False)
        if "row" in t.columns:  # dvifmish batch: row index into the input list
            # numpy object arrays throughout: under pyarrow-backed strings `.iloc[..].values`
            # is an ArrowStringArray, which has no `.values` for the index below.
            rows = t["row"].astype(int).to_numpy()
            key = pd.Series(lst["image_path"].to_numpy(dtype=object)[rows] + "#"
                            + lst["q"].to_numpy(dtype=object)[rows])
            if not (t["dist_path"].to_numpy(dtype=object) == lst["dist_path"].to_numpy(dtype=object)[rows]).all():
                sys.exit(f"{path}: row index does not point at the same dist_path")
        elif "image_path" in t.columns:
            key = t["image_path"].astype(str) + "#" + t["q"].astype(str)
        else:  # zenmetrics batch TSV: (ref_path, dist_path) are unique on every list here
            if (lst["dist_path"] + "|" + lst["ref_path"]).duplicated().any():
                sys.exit(f"{path}: (ref,dist) not unique in the list; cannot join")
            m = dict(zip(lst["dist_path"] + "|" + lst["ref_path"], lst["image_path"] + "#" + lst["q"]))
            key = (t["dist_path"] + "|" + t["ref_path"]).map(m)
    if col not in t.columns:  # NOT MEASURED for this scorer, never a crash of the whole run
        sys.exit(f"{path}: no column {col!r} (has {[c for c in t.columns][:12]})")
    v = pd.Series(pd.to_numeric(t[col], errors="coerce").to_numpy(), index=pd.Index(key.to_numpy(dtype=object)))
    if v.index.duplicated().any():
        sys.exit(f"{path}: duplicate keys")
    want = lst["image_path"] + "#" + lst["q"]
    miss = set(want) - set(v.index)
    if miss:
        sys.exit(f"{path}: {len(miss)} of {len(want)} list rows missing (e.g. {sorted(miss)[:2]})")
    out = lst.copy()
    out["value"] = v.loc[want.values].values
    if out["value"].isna().any():
        sys.exit(f"{path}: {int(out['value'].isna().sum())} NaN scores for {col}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bv", required=True, help="bake_verdict binary (lane build)")
    ap.add_argument("--out", default="/var/tmp/paper-gates/bv")
    ap.add_argument("--only", default="", help="comma list of scorer names")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/cells", exist_ok=True)
    os.makedirs(f"{a.out}/gaddr", exist_ok=True)
    os.makedirs(f"{a.out}/logs", exist_ok=True)
    probe_entries = pq.read_table(IDENT).column("entry").to_pylist()
    only = [x for x in a.only.split(",") if x]
    summary = {}
    runs = []
    for name, (stem, col, o, perfect) in SCORERS.items():
        if only and name not in only:
            continue
        rec = {"orientation": "quality" if o > 0 else "distance", "perfect_native": perfect}
        # ---- ladder cells, oriented
        if stem is None:
            cells = pd.read_csv(TRUTH, sep="\t")
            lad = None
            ov = cells["pred"].to_numpy(float)
            rec["source"] = TRUTH
        else:
            try:
                lad = read_scores(stem, col, "ladder")
            except SystemExit as e:
                rec["not_measured"] = f"ladder scores unavailable: {e}"
                summary[name] = rec
                print(f"{name}: NOT MEASURED — {e}")
                continue
            ov = o * lad["value"].to_numpy(float)
            rec["source"] = f"{S}/{stem.format(set='ladder')}:{col}"
        p1, p99 = np.percentile(ov, [1, 99])
        span = float(p99 - p1)
        P = o * perfect
        rec.update(p1_oriented=float(p1), p99_oriented=float(p99), span_S=span)
        readings = {"native": (1.0, 0.0), "s100": (100.0 / span, 100.0 - 100.0 * P / span)}
        # ---- probe values (identity + lsb) for the same scorer
        pr = None
        if stem is not None:
            try:
                pr = read_scores(stem, col, "probes")
            except SystemExit as e:
                rec["probes_not_measured"] = str(e)
        elif name == "peer_ssim2_registered":
            pr = None  # identity for the registered table is the registry's own peer table
        for rd, (k, c) in readings.items():
            lab = f"{name}__{rd}"
            cells_tsv = f"{a.out}/cells/{lab}.tsv"
            if stem is None:
                t = cells.copy()
                t["pred"] = k * ov + c
                t.to_csv(cells_tsv, sep="\t", index=False, float_format="%.17g")
            else:
                src = f"{a.out}/cells/{lab}.src.tsv"
                with open(src, "w") as f:
                    f.write("ref_path\tdist_path\tcodec\tq\tknob_tuple_json\tv\n")
                    for r, v in zip(lad.itertuples(), k * ov + c):
                        f.write(f"{r.ref_path}\t{r.dist_path}\t{r.codec}\t{r.grid_q}\t{{}}\t{float(v)!r}\n")
                subprocess.run([sys.executable, f"{REPO}/scripts/v_next/dial_peer_cells.py", "--tsv", src,
                                "--value-col", "v", "--grid", GRID, "--out", cells_tsv], check=True)
            cmd = [a.bv, "--bake", CARRIER, "--dial-peer-scores", f"{lab}={cells_tsv}",
                   "--dial-grid", GRID, "--corpora", "tid", "--gaddr-tail-pins", "product",
                   "--gaddr-value-pins", "report", "--gaddr-json", f"{a.out}/gaddr/{lab}.json",
                   "--gaddr-grid-truth", TRUTH, "--reference-truth", REFTRUTH,
                   "--output", f"{a.out}/logs/{lab}.md"]
            ident_tsv = None
            if pr is not None:
                idr = pr[pr["codec"] == "identity"].set_index("entry")
                ident_tsv = f"{a.out}/cells/{lab}.identity.tsv"
                with open(ident_tsv, "w") as f:
                    f.write("entry\tpred\n")
                    for e in probe_entries:
                        f.write(f"{e}\t{k * o * float(idr.loc[e, 'value']) + c!r}\n")
            elif name == "peer_ssim2_registered":
                ident_tsv = f"{a.out}/cells/{lab}.identity.tsv"
                reg = pd.read_csv(f"{D0904}/repin/identity_peer_ssim2.tsv", sep="\t")
                reg["pred"] = k * reg["pred"] + c
                reg.to_csv(ident_tsv, sep="\t", index=False, float_format="%.17g")
                neg = f"{a.out}/cells/{lab}.negtail.tsv"
                ng = pd.read_csv(NEGTAIL_SSIM2, sep="\t")
                ng["pred"] = k * ng["pred"] + c
                ng.to_csv(neg, sep="\t", index=False, float_format="%.17g")
                cmd += ["--negtail-probe", NEGTAIL, "--negtail-peer-scores", f"{lab}={neg}"]
            if ident_tsv:
                cmd += ["--identity-probe", IDENT, "--identity-peer-scores", f"{lab}={ident_tsv}"]
            runs.append((lab, cmd))
            rec.setdefault("readings", {})[rd] = {"scale": k, "offset": c, "cells": cells_tsv,
                                                  "identity": ident_tsv}
        # native probe values for the lane record (identity exactness, 1-LSB step)
        if pr is not None:
            rec["probe_values_native"] = {
                kind: [float(x) for x in pr[pr["codec"] == kind]["value"]]
                for kind in ("identity", "lsb-center-g", "lsb-all")}
            rec["probe_entries"] = list(pr[pr["codec"] == "identity"]["entry"])
        summary[name] = rec
    # ---- feature-path reads of the shipped B and D bakes (their dial(0-vector) identity)
    for name, bake in BAKES.items():
        if only and name not in only:
            continue
        lab = f"{name}__native"
        runs.append((lab, [a.bv, "--bake", bake, "--dial-grid", GRID, "--corpora", "tid",
                           "--negtail-probe", NEGTAIL, "--identity-probe", IDENT,
                           "--gaddr-tail-pins", "product", "--gaddr-value-pins", "report",
                           "--gaddr-json", f"{a.out}/gaddr/{lab}.json", "--gaddr-grid-truth", TRUTH,
                           "--reference-truth", REFTRUTH, "--output", f"{a.out}/logs/{lab}.md"]))
        summary[name] = {"bake": bake, "orientation": "quality", "perfect_native": 100.0}
    for lab, cmd in runs:
        with open(f"{a.out}/logs/{lab}.log", "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        name, rd = lab.split("__")
        md = f"{a.out}/logs/{lab}.md"
        res = {"rc": rc, "argv": cmd}
        gj = f"{a.out}/gaddr/{lab}.json"
        if rc == 0 and os.path.isfile(gj):
            g = json.load(open(gj))
            res["headline"] = g.get("headline")
            res["checks"] = {c["id"]: {"state": c["state"], "measured": c["measured"], "bar": c.get("bar"),
                                       "tier": c.get("tier")} for c in g["checks"]}
            res["codec_floor"] = {r["codec"]: {"represented_frac": r["represented_frac"],
                                               "bar": r["represented_frac_reference"], "state": r["state"],
                                               "n_fail_order": r["n_fail_order"], "n_fail_clamp": r["n_fail_clamp"],
                                               "n_ladders": r["n_ladders"]}
                                  for r in g["measured"].get("codec_floor") or []}
            res["grid"] = g["measured"].get("grid")
            res["identity"] = g["measured"].get("identity")
            res["negtail"] = g["measured"].get("negtail")
        if os.path.isfile(md):
            t = open(md).read()
            res["scale_free"] = {}
            for m in re.finditer(r"^\| (\S+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$", t, re.M):
                res["scale_free"][m.group(1)] = {"pairs": int(m.group(2)), "forward": int(m.group(3)),
                                                 "backwards": int(m.group(4)), "tie": int(m.group(5))}
            m = re.search(r"\| ↳ strict backwards \(any > 1e-9\) \| ([\d.]+) \|", t)
            res["strict_backwards_pooled"] = float(m.group(1)) if m else None
            m = re.search(r"\| ↳ charged to the ENCODER \(both refs agree\) \| (\d+) \|", t)
            res["encoder_attributed"] = int(m.group(1)) if m else None
        summary[name].setdefault("gaddr", {})[rd] = res
        print(f"{lab}: rc={rc} {res.get('headline')}")
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1, default=float)
    print(f"summary -> {a.out}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
