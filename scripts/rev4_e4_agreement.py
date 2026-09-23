#!/usr/bin/env python3
"""Rev4 lane E4 — multimetric (two-reference) agreement as a SELECTION GATE.

Retrospective analysis over EXISTING board candidates. Nothing here trains,
extracts features or re-encodes. Scoring goes through the Rust serving
surface (`ensemble_score_rows`, bit-for-bit with `bake_verdict::score_row`);
every rank statistic goes through the statistics owner (`panel`, via
`scripts/lib/zen_stats`). The only arithmetic done here is counting ladder
pairs (reproduced against the owner's stored C1 before use), rank-averaging
two predictors, and the partial-correlation identity applied to owner SROCCs.

Subcommands (run in order):
  ladder    score every graded ladder-board cell on its ladder instrument,
            reproduce the stored C1 (mono_agree) and write the predictors.
  outcomes  (after the prereg commit) CID22-A(25) SROCC per candidate from
            A-only rows, stored AIC-3 / KonJND-504 / CSIQ, the composite-A,
            dial-contract outcomes.
  analyze   family-clustered bootstrap of predictor↔outcome SROCCs,
            incremental (partial) SROCC over the composite, gate confusion.

Prereg: benchmarks/rev4_e4_prereg_2026-09-23.md.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from lib import zen_stats  # noqa: E402

BOARD = "/mnt/v/output/zensim/ladder-board-2026-09-06"
INSTR = "/mnt/v/output/zensim/ladder-2026-09-05/instruments"
REF_TRUTH = f"{INSTR}/reference_truth_ladder_pnorm3.tsv"
WORK = "/var/tmp/rev4-e4"
GRID_DUMMY = {
    f"{INSTR}/dial_grid_944col_ladder.parquet": f"{WORK}/dial_grid_944col_ladder_dummytarget.parquet",
    f"{INSTR}/dial_grid_372col_ladder.parquet": f"{INSTR}/dial_grid_372col_ladder_dummytarget.parquet",
}
ESR = os.environ.get("ESR_BIN", f"{WORK}/target/release/ensemble_score_rows")

# Owner constants (bake_verdict MATERIAL_INV_PT; dial_addressability
# ENCODER_SSIM2_MARGIN_PT and ButteraugliVariant::Pnorm3.margin()).
MATERIAL_PT = 0.5
SSIM2_MARGIN = 0.5
BUTTER_MARGIN = 0.05
BOTTOM_K = 3  # A7r window: K lowest steps + the step into K+1


# Lineage = the resampling unit (prereg §4). Coarse on purpose: seeds, λ/epoch
# sweeps, repacks and ensembles of one recipe family share a lineage.
LINEAGE_RULES = [
    (r"^(sota944_)?ens_", "sota944_C"), (r"^(sota944_)?C_(co|em944|ensk|nt944)", "sota944_C"),
    (r"^H_co3abpg", "sota944_C"), (r"^sota944_nt223", "sota944_C"),
    (r"^(sota944_)?(winner_)?A_", "sota944_A"), (r"^sota944_B", "sota944_B"), (r"^sota944_Q_", "sota944_Q"),
    (r"^(R1_)?(CS|GL|PILOT)", "sparsehf"), (r"^R1_", "sparsehf"), (r"^sota944_FS_", "sparsehf"),
    (r"^(CTL_[AB]_)?LSTAR", "LSTAR"), (r"^lstar", "LSTAR"),
    (r"^HDR944", "HDR944"), (r"^KFG", "KFG"), (r"^PH_", "PH"), (r"^T_appT", "T_appT"),
    (r"^W10L", "W10"), (r"^(sota944_)?W11", "W11"), (r"^w11_", "W11"), (r"^W12", "W12"),
    (r"^W8", "W8"), (r"^W9", "W9"), (r"^BAL_", "BAL"), (r"^A1foldapp2|^A5_r4", "waver4_A"),
    (r"^copperline", "copperline"), (r"^jewelerloupe", "jewelerloupe"),
    (r"^BOA_", "BOA"), (r"^fc2_372", "fc2_372"), (r"^B_(im26|kon|oldanchor|safesyn)", "im26lane"),
    (r"^(D_shipped|d_id100)", "did100lane"), (r"^A2|^A6_", "waver4_A"), (r"^ADD156", "ADD156"),
    (r"^Dpeaks", "Dpeaks"), (r"^SADD", "SADD"), (r"^b_sdr", "b_sdr"), (r"^bhdr", "bhdr"),
    (r"^cl_tfm", "cl_tfm"), (r"^mlp_2L", "mlp_2L"), (r"^v02_", "v02"), (r"^v47", "v47"),
]


def lineage(name):
    n = name.split("@")[0]
    for pat, lab in LINEAGE_RULES:
        if re.search(pat, n):
            return lab
    return "UNMAPPED:" + n  # only out-of-population cells; analyze refuses these


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def argval(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def load_ref_truth():
    t = {}
    with open(REF_TRUTH) as f:
        for line in f:
            if line.startswith("#") or line.startswith("image_id"):
                continue
            img, codec, q, s2, bu = line.rstrip("\n").split("\t")
            t[(img, codec, round(float(q), 4))] = (float(s2), float(bu))
    return t


def load_grid(path):
    tb = pq.read_table(path, columns=["image_id", "codec", "q"]).to_pydict()
    return tb["image_id"], tb["codec"], tb["q"]


def cell_bakes(argv):
    ens = argval(argv, "--ensemble")
    if ens:
        bakes = ens.split(",")
        w = argval(argv, "--ensemble-weights")
        return bakes, (w.split(",") if w else None)
    return [argval(argv, "--bake")], None


def score_cmd(argv, out_tsv):
    grid = argval(argv, "--dial-grid")
    dummy = GRID_DUMMY[grid]
    bakes, weights = cell_bakes(argv)
    cmd = [ESR]
    for b in bakes:
        cmd += ["--bake", b]
    if weights:
        cmd += ["--weights", ",".join(weights)]
    cmd += ["--parquet", dummy, "--output", out_tsv]
    return cmd


def score_cell(name, argv, out_tsv):
    cmd = score_cmd(argv, out_tsv)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return cmd


def read_scores(tsv):
    s = []
    with open(tsv) as f:
        next(f)
        for line in f:
            s.append(float(line.rstrip("\n").split("\t")[2]))
    return s


def ladder_stats(img, codec, q, scores, truth):
    """Count adjacent-pair outcomes per the owner's windows (curves keyed by
    (image, codec), sorted by q). Returns a dict of counts."""
    curves = defaultdict(list)
    for i in range(len(scores)):
        curves[(img[i], codec[i])].append((q[i], scores[i]))
    c = defaultdict(int)
    for (im, co), pts in curves.items():
        pts.sort(key=lambda t: t[0])
        for j, ((q0, s0), (q1, s1)) in enumerate(zip(pts, pts[1:])):
            bottom = j < BOTTOM_K  # pair touches the A7r floor window
            d = s1 - s0
            c["pairs"] += 1
            r0 = truth.get((im, co, round(q0, 4)))
            r1 = truth.get((im, co, round(q1, 4)))
            known = r0 is not None and r1 is not None
            ds = r1[0] - r0[0] if known else None
            db = r1[1] - r0[1] if known else None
            enc = known and ds <= -SSIM2_MARGIN and db >= BUTTER_MARGIN
            if d < -MATERIAL_PT:
                if enc:
                    c["inv_encoder"] += 1
                else:
                    c["inv_dial"] += 1
                    if not known:
                        c["inv_unknown"] += 1
            if not known:
                c["ref_unknown"] += 1
                continue
            fwd_agree = ds >= SSIM2_MARGIN and db <= -BUTTER_MARGIN
            bwd_agree = ds <= -SSIM2_MARGIN and db >= BUTTER_MARGIN
            for tag, sub in (("all", True), ("nofloor", not bottom)):
                if not sub:
                    continue
                if fwd_agree or bwd_agree:
                    c[f"agree_{tag}"] += 1
                    if (fwd_agree and d < -MATERIAL_PT) or (bwd_agree and d > MATERIAL_PT):
                        c[f"disagree_{tag}"] += 1
                if abs(db) >= BUTTER_MARGIN:
                    c[f"bu_{tag}"] += 1
                    if (db <= -BUTTER_MARGIN and d < -MATERIAL_PT) or (db >= BUTTER_MARGIN and d > MATERIAL_PT):
                        c[f"bu_rev_{tag}"] += 1
                if abs(ds) >= SSIM2_MARGIN:
                    c[f"s2_{tag}"] += 1
                    if (ds >= SSIM2_MARGIN and d < -MATERIAL_PT) or (ds <= -SSIM2_MARGIN and d > MATERIAL_PT):
                        c[f"s2_rev_{tag}"] += 1
    return dict(c)


def cmd_ladder(a):
    os.makedirs(f"{WORK}/ladder_scores", exist_ok=True)
    cells = json.load(open(f"{BOARD}/cells.json"))
    truth = load_ref_truth()
    grids = {}
    out = []
    failed = []
    for x in cells:
        argv = x.get("argv")
        if not argv or x.get("kind") != "bake":
            continue
        name = x["name"]
        if a.only and name not in a.only.split(","):
            continue
        gpath = argval(argv, "--dial-grid")
        if gpath not in grids:
            grids[gpath] = load_grid(GRID_DUMMY[gpath])
        img, codec, q = grids[gpath]
        tsv = f"{WORK}/ladder_scores/{name}.tsv"
        cmd = None
        if not os.path.exists(tsv):
            try:
                cmd = score_cell(name, argv, tsv)
            except subprocess.CalledProcessError as e:
                failed.append((name, e.stderr.decode(errors="replace")[-300:]))
                print(f"{name:<48} SCORE FAILED", flush=True)
                continue
        scores = read_scores(tsv)
        if len(scores) != len(img):
            raise SystemExit(f"{name}: {len(scores)} scores vs {len(img)} grid rows")
        st = ladder_stats(img, codec, q, scores, truth)
        g = json.load(open(f"{BOARD}/gaddr/{name}.json"))
        chk = {c["id"]: c for c in g["checks"]}
        c1 = chk["C1"]["measured"]
        mono = 1.0 - st.get("inv_dial", 0) / st["pairs"]
        rec = {
            "name": name,
            "width": x["width"],
            "era": x["era"],
            "grid": gpath,
            "grid_sha256": x.get("grid_sha256"),
            "scores_tsv": tsv,
            "scores_sha256": sha256(tsv),
            "counts": st,
            "mono_agree_repro": mono,
            "mono_agree_stored": c1,
            "repro_abs_err": abs(mono - c1) if c1 is not None else None,
            "gaddr": {
                "contract": g.get("contract"),
                "regression": g.get("regression"),
                "checks": {k: {"state": v["state"], "measured": v["measured"]} for k, v in chk.items()},
                "codec_floor": [
                    {"codec": f["codec"], "represented_frac": f["represented_frac"],
                     "reference": f["represented_frac_reference"], "state": f["state"]}
                    for f in (g["measured"].get("codec_floor") or [])
                ],
                "tied": g["measured"]["grid"].get("tied"),
            },
            "fulleval": x.get("fulleval"),
            "bakes": cell_bakes(argv)[0],
            "weights": cell_bakes(argv)[1],
            "score_cmd": score_cmd(argv, tsv),
            "lineage": lineage(name),
        }
        out.append(rec)
        print(f"{name:<48} mono repro {mono:.12f} stored {c1:.12f} "
              f"|Δ|={rec['repro_abs_err']:.2e} agree_pairs={st.get('agree_all',0)} "
              f"disagree={st.get('disagree_all',0)}", flush=True)
    json.dump({"cells": out, "score_failed": failed}, open(a.out, "w"), indent=1)
    bad = [r["name"] for r in out if r["repro_abs_err"] is None or r["repro_abs_err"] > 1e-12]
    print(f"cells={len(out)} reproduced={len(out) - len(bad)} not_reproduced={len(bad)} {bad[:20]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ladder")
    p.add_argument("--out", default=f"{WORK}/ladder_predictors.json")
    p.add_argument("--only", default=None)
    a = ap.parse_args()
    {"ladder": cmd_ladder}[a.cmd](a)


if __name__ == "__main__":
    main()
