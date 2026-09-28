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


# ---------------------------------------------------------------- outcomes --
# CID22-A(25): docs/DATASET_HISTORY.md 2026-09-19 (seed 20260919), `.png` dropped
# to match the parquet's `ref_basename`.
CID22_A = [
    "1189261", "1531677", "159550", "1624487", "162520", "164595", "2079234",
    "21169144185_3f7977cb5a_o", "225228", "2389166", "2936831", "3316926", "3653963",
    "373965", "3762075", "4215100", "6078297", "6292444", "70497", "7062219", "844297",
    "pexels-photo-2686358", "pexels-photo-2802032", "pexels-photo-4210863",
    "ularapi_Semarang_City_Logo",
]
BV = os.environ.get("BV_BIN", f"{WORK}/bin/bake_verdict")
SLOTS = {  # corpus -> (944/720 slot, 372 slot); bake_verdict's own maps
    "cid22": ("ext_cid22val.parquet", "cid22_features_372col_2026-05-15.parquet"),
    "kadid": ("ext_kadid.parquet", "kadid_features_372col_2026-05-15.parquet"),
}
FE_NAMES = {"cid22": "CID22", "kadid": "KADIK10k"}


def main_population(cells):
    return [c for c in cells if c["width"] == 944 and c["era"].startswith("immune")]


def secondary_population(cells):
    return [c for c in cells if c["width"] == 372]


def resolve_root(cell, fe):
    fr = fe.get("features_root") or {}
    if fr.get("path"):
        return fr["path"], "fulleval.features_root.path"
    regime = str(fe.get("regime") or cell["width"])
    cmd = [BV, "--bake", cell["bakes"][0], "--regime", regime, "--print-features-root"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    root = out[-1] if r.returncode == 0 and out and os.path.isdir(out[-1]) else None
    return root, " ".join(cmd) + " :: " + r.stderr.strip().splitlines()[-1] if r.stderr.strip() else " ".join(cmd)


def slot_file(root, corpus, fe, width):
    fr = fe.get("features_root") or {}
    for cf in fr.get("corpus_files") or []:
        if cf.get("name") == FE_NAMES[corpus]:
            return os.path.join(root, cf["file"])
    s944, s372 = SLOTS[corpus]
    for cand in ((s372, s944) if width == 372 else (s944, s372)):
        if os.path.exists(os.path.join(root, cand)):
            return os.path.join(root, cand)
    return None


_A_CACHE = {}


def cid22_a_file(src):
    """A-only copy of a CID22 feature parquet. Rows are selected by
    `ref_basename` membership inside the reader (filters=), so no CID22-B row
    is ever handed to Python or to a statistic."""
    if src in _A_CACHE:
        return _A_CACHE[src]
    os.makedirs(f"{WORK}/cid22A", exist_ok=True)
    key = hashlib.sha256(src.encode()).hexdigest()[:16]
    dst = f"{WORK}/cid22A/{key}_cid22A.parquet"
    if not os.path.exists(dst):
        t = pq.read_table(src, filters=[("ref_basename", "in", CID22_A)])
        refs = set(t.column("ref_basename").to_pylist())
        if refs != set(CID22_A):
            raise SystemExit(f"{src}: A-filter found {len(refs)} refs, expected 25")
        pq.write_table(t, dst, compression="zstd")
    refs = pq.read_table(dst, columns=["ref_basename"]).column(0).to_pylist()
    _A_CACHE[src] = (dst, sha256(dst), refs)
    return _A_CACHE[src]


def run_esr(cell, parquet, out_tsv):
    cmd = [ESR]
    for b in cell["bakes"]:
        cmd += ["--bake", b]
    if cell["weights"]:
        cmd += ["--weights", ",".join(cell["weights"])]
    cmd += ["--parquet", parquet, "--output", out_tsv]
    if not os.path.exists(out_tsv):
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    hs, sc = [], []
    with open(out_tsv) as f:
        next(f)
        for line in f:
            _, h, v = line.rstrip("\n").split("\t")
            hs.append(float(h))
            sc.append(float(v))
    return cmd, hs, sc


ROOTS_BY_WIDTH = {  # zensim_validate::eval_roots, tried after the bake-resolved root
    944: ["/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01",
          "/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01"],
    720: ["/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22"],
    372: ["/mnt/v/zen/zensim-training/2026-08-30-full-features-372",
          "/mnt/v/zen/zensim-training/2026-05-15-full-features",
          "/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC"],
}


def cmd_outcomes(a):
    """Root selection is by the TRAIN-role KADID identity check only: the first
    candidate root whose KADID SROCC reproduces the cell's stored value to
    1e-6 is the root its stored rank block came from. CID22-A is scored only
    on that root, and only after it is chosen."""
    pred = json.load(open(a.predictors))["cells"]
    pop = main_population(pred) + secondary_population(pred)
    os.makedirs(f"{WORK}/cid22A_scores", exist_ok=True)
    os.makedirs(f"{WORK}/kadid_scores", exist_ok=True)
    recs, jobs_c = [], []
    for cell in pop:
        name = cell["name"]
        fe = json.load(open(cell["fulleval"]))
        width = int(fe.get("n_inputs") or cell["width"])
        width = 944 if width > 720 else (720 if width > 372 else 372)
        rec = {"name": name, "fulleval": cell["fulleval"], "fulleval_sha256": sha256(cell["fulleval"]),
               "bake_sha256": fe.get("bake_sha256"), "regime_field": fe.get("regime"),
               "n_inputs": fe.get("n_inputs"), "root_tries": []}
        rk = fe.get("rank") or {}
        rec["stored"] = {c: {k: (rk.get(c) or {}).get(k) for k in ("srocc", "srocc_signed", "n")}
                         for c in ("kadid", "aic3", "konjnd", "csiq", "imazen26", "nonphoto", "live", "aic4")}
        kst = rec["stored"]["kadid"]["srocc"]
        cands = []
        root0, how = resolve_root(cell, fe)
        rec["root_how"] = how
        if root0:
            cands.append(root0)
        cands += [r for r in ROOTS_BY_WIDTH[width] if r not in cands]
        chosen = None
        for root in cands:
            kfile = slot_file(root, "kadid", fe, width)
            cfile = slot_file(root, "cid22", fe, width)
            if not (kfile and cfile):
                rec["root_tries"].append([root, "slot missing"])
                continue
            tag = hashlib.sha256(root.encode()).hexdigest()[:8]
            try:
                kcmd, kh, ks = run_esr(cell, kfile, f"{WORK}/kadid_scores/{name}__{tag}.tsv")
            except subprocess.CalledProcessError as e:
                rec["root_tries"].append([root, "score failed: " + e.stderr.decode(errors="replace")[-120:]])
                continue
            kr = zen_stats.panel_batch([("k", ks, kh)], stats="srocc")[0]["srocc"]
            err = abs(abs(kr) - abs(kst)) if kst is not None else None
            rec["root_tries"].append([root, err])
            if err is not None and err <= 1e-6:
                chosen = (root, kfile, cfile, kcmd, kr, err)
                break
        if chosen is None:
            rec["excluded"] = "no candidate root reproduces stored KADID SROCC to 1e-6"
            recs.append(rec)
            print(f"{name:<48} EXCLUDED {rec['root_tries']}", flush=True)
            continue
        root, kfile, cfile, kcmd, kr, err = chosen
        rec.update({"root": root, "kadid_src": kfile, "cid22_src": cfile, "kadid_cmd": kcmd,
                    "kadid_srocc_repro": kr, "kadid_abs_err": err})
        afile, asha, arefs = cid22_a_file(cfile)
        rec["cid22A_file"], rec["cid22A_sha256"], rec["cid22A_n"] = afile, asha, len(arefs)
        ccmd, ch, cs = run_esr(cell, afile, f"{WORK}/cid22A_scores/{name}__{hashlib.sha256(root.encode()).hexdigest()[:8]}.tsv")
        rec["cid22A_cmd"] = ccmd
        jobs_c.append((name, cs, ch))
        recs.append(rec)
        print(f"{name:<48} root={os.path.basename(root)} kadid|Δ|={err:.1e} nA={len(arefs)}", flush=True)
    rows_c = zen_stats.panel_batch(jobs_c, stats="srocc")
    bc = {r["label"]: r for r in rows_c}
    for rec in recs:
        if "excluded" not in rec:
            rec["cid22A_srocc_signed"] = bc[rec["name"]].get("srocc_signed", bc[rec["name"]]["srocc"])
    json.dump({"cells": recs}, open(a.out, "w"), indent=1)
    ex = [r["name"] for r in recs if "excluded" in r]
    print(f"cells={len(recs)} kept={len(recs) - len(ex)} excluded={len(ex)} {ex}")


# ----------------------------------------------------------------- analyze --
PREDICTORS = {
    "P_dis": ("disagree_all", "agree_all"),
    "P_dial": None,  # 1 - mono_agree
    "P_bu": ("bu_rev_all", "bu_all"),
    "P_s2": ("s2_rev_all", "s2_all"),
    "P_dis_nofloor": ("disagree_nofloor", "agree_nofloor"),
}
COMPOSITE_W = {  # freeze_check::balanced (band-tail omitted: CID22-49 statistic)
    "cid22A": 1.00, "imazen26": 0.50, "nonphoto": 0.30, "konjnd": 0.20,
    "csiq": 0.15, "live": 0.15, "aic3": 0.10, "aic4": 0.05,
}
OUTCOMES = ["H1_cid22A", "H2_aic3", "H3_konjnd504", "H4_csiq", "D1_contract", "D2_floors"]
LOO_TERM = {"H1_cid22A": "cid22A", "H2_aic3": "aic3", "H3_konjnd504": "konjnd", "H4_csiq": "csiq"}
TAU = 0.009938837920489297  # prereg §7: median P_dis over MAIN
B_BOOT, SEED = 4000, 20260923
ALPHA_BONF = 0.05 / 6


def avg_ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def composite(terms, drop=None):
    num = den = 0.0
    for k, w in COMPOSITE_W.items():
        if k == drop or terms.get(k) is None:
            continue
        num += abs(terms[k]) * w
        den += w
    return num / den if den > 0 else None


def build_table(pred, outc, pop_fn):
    ob = {r["name"]: r for r in outc["cells"]}
    rows = []
    for c in pop_fn(pred):
        o = ob.get(c["name"])
        if o is None or "excluded" in o:
            continue
        if c["lineage"].startswith("UNMAPPED"):
            raise SystemExit(f"population cell without lineage: {c['name']}")
        cnt = c["counts"]
        P = {}
        for k, v in PREDICTORS.items():
            P[k] = (1.0 - c["mono_agree_repro"]) if v is None else cnt.get(v[0], 0) / cnt[v[1]]
        st = o["stored"]
        if st["konjnd"]["n"] != 504:
            raise SystemExit(f"{c['name']}: KonJND n={st['konjnd']['n']} (prereg requires 504)")
        terms = {"cid22A": o["cid22A_srocc_signed"]}
        for k in ("imazen26", "nonphoto", "konjnd", "csiq", "live", "aic3", "aic4"):
            terms[k] = st[k]["srocc"]
        chk = c["gaddr"]["checks"]
        d1 = 0 if any(chk[i]["state"] == "fail" for i in ("C2", "C3", "C4", "C5", "C6") if i in chk) else 1
        fl = [f["represented_frac"] for f in c["gaddr"]["codec_floor"]]
        Y = {
            "H1_cid22A": o["cid22A_srocc_signed"],
            "H2_aic3": st["aic3"]["srocc_signed"],
            "H3_konjnd504": -st["konjnd"]["srocc_signed"],
            "H4_csiq": st["csiq"]["srocc_signed"],
            "D1_contract": d1,
            "D2_floors": sum(fl) / len(fl) if len(fl) == 5 else None,
        }
        C = {y: composite(terms, LOO_TERM.get(y)) for y in OUTCOMES}
        C["C_A"] = composite(terms)
        rows.append({"name": c["name"], "lineage": c["lineage"], "P": P, "Y": Y, "C": C,
                     "terms": terms})
    return rows


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(q * len(v))))]


def boot_outcome(rows, y, rng_seed):
    import random
    rr = [r for r in rows if r["Y"][y] is not None and r["C"][y] is not None]
    n = len(rr)
    lin = defaultdict(list)
    for i, r in enumerate(rr):
        lin[r["lineage"]].append(i)
    lk = sorted(lin)
    bases = {"Y": [r["Y"][y] for r in rr], "C": [r["C"][y] for r in rr]}
    for p in PREDICTORS:
        bases[p] = [-r["P"][p] for r in rr]
    rc = avg_ranks(bases["C"])
    rp = avg_ranks(bases["P_dis"])
    bases["Combo"] = [(a + b) / 2.0 for a, b in zip(rc, rp)]
    rng = random.Random(rng_seed)
    idx_sets = [None]
    for _ in range(B_BOOT):
        idx = []
        for _l in range(len(lk)):
            idx += lin[lk[rng.randrange(len(lk))]]
        idx_sets.append(idx)
    jobs = []
    for b, idx in enumerate(idx_sets):
        for p in PREDICTORS:
            jobs.append((f"{b}|{p}|Y", p, "Y", idx))
            jobs.append((f"{b}|{p}|C", p, "C", idx))
        jobs.append((f"{b}|C|Y", "C", "Y", idx))
        jobs.append((f"{b}|Combo|Y", "Combo", "Y", idx))
    res = zen_stats.panel_batch_indexed(bases, jobs, stats="srocc")
    v = {r["label"]: r.get("srocc_signed", r["srocc"]) for r in res}
    out = {"outcome": y, "n_cells": n, "n_lineages": len(lk), "B": B_BOOT, "seed": rng_seed}

    def part(b, p):
        a, c, d = v[f"{b}|{p}|Y"], v[f"{b}|{p}|C"], v[f"{b}|C|Y"]
        den = ((1 - c * c) * (1 - d * d)) ** 0.5
        return (a - c * d) / den if den > 0 else float("nan")

    def summ(vals, pt):
        import math
        vals = [x for x in vals if not math.isnan(x)]
        return {"point": pt, "ci95": [pct(vals, 0.025), pct(vals, 0.975)],
                "ci_bonf": [pct(vals, ALPHA_BONF / 2), pct(vals, 1 - ALPHA_BONF / 2)],
                "n_valid": len(vals)}

    rs = range(1, B_BOOT + 1)
    out["rho_C"] = summ([v[f"{b}|C|Y"] for b in rs], v["0|C|Y"])
    out["rho_Combo"] = summ([v[f"{b}|Combo|Y"] for b in rs], v["0|Combo|Y"])
    out["delta_Combo_minus_C"] = summ([v[f"{b}|Combo|Y"] - v[f"{b}|C|Y"] for b in rs],
                                      v["0|Combo|Y"] - v["0|C|Y"])
    for p in PREDICTORS:
        out[f"rho_{p}"] = summ([v[f"{b}|{p}|Y"] for b in rs], v[f"0|{p}|Y"])
        out[f"rho_{p}_vs_C"] = v[f"0|{p}|C"]
        out[f"partial_{p}"] = summ([part(b, p) for b in rs], part(0, p))
    return out


def gate_view(rows):
    rr = [r for r in rows if all(r["Y"][y] is not None for y in OUTCOMES[:4])]
    n = len(rr)
    pr = {y: avg_ranks([r["Y"][y] for r in rr]) for y in OUTCOMES[:4]}
    hr = [sum(pr[y][i] for y in pr) / (4.0 * n) for i in range(n)]
    cut = sorted(hr)[n // 3]
    cmed = sorted(r["C"]["C_A"] for r in rr)[n // 2]
    tabs = {}
    for gate, rej in (("agreement P_dis > tau", lambda r: r["P"]["P_dis"] > TAU),
                      ("composite C_A < median", lambda r: r["C"]["C_A"] < cmed)):
        t = {"reject_failed": 0, "reject_ok": 0, "keep_failed": 0, "keep_ok": 0}
        for i, r in enumerate(rr):
            failed = r["Y"]["D1_contract"] == 0 or hr[i] < cut
            t[("reject_" if rej(r) else "keep_") + ("failed" if failed else "ok")] += 1
        tabs[gate] = t
    return {"n": n, "tau": TAU, "hr_bottom_tertile_cut": cut, "composite_median": cmed,
            "failed_def": "D1 measured contract failure (C2-C6) OR held-out mean pct-rank (H1-H4) in bottom tertile",
            "tables": tabs}


def verdict(res):
    pos, neg = [], []
    for y, r in res.items():
        lo, hi = r["partial_P_dis"]["ci_bonf"]
        if lo > 0:
            pos.append(y)
        if hi < 0:
            neg.append(y)
    if pos and not neg:
        circ = [y for y in pos if res[y]["partial_P_bu"]["ci95"][0] > 0]
        return ("ADOPT" if circ else "ADOPT, circularity unresolved"), pos, neg
    return "NEGATIVE", pos, neg


def cmd_analyze(a):
    pred = json.load(open(a.predictors))["cells"]
    outc = json.load(open(a.outcomes))
    report = {"predictors_sha256": sha256(a.predictors), "outcomes_sha256": sha256(a.outcomes)}
    for label, fn, seed in (("MAIN", main_population, SEED), ("SECONDARY", secondary_population, SEED)):
        rows = build_table(pred, outc, fn)
        if label == "SECONDARY":  # prereg §4 dedupe of re-root duplicates by bake sha
            ob = {r["name"]: r for r in outc["cells"]}
            keep = {}
            for r in sorted(rows, key=lambda r: r["name"]):
                sha = ob[r["name"]]["bake_sha256"]
                cur = "2026-08-30-full-features-372" in (ob[r["name"]]["root"] or "")
                if sha not in keep or (cur and "2026-08-30" not in (ob[keep[sha]["name"]]["root"] or "")):
                    keep[sha] = r
            rows = sorted(keep.values(), key=lambda r: r["name"])
        json.dump(rows, open(f"{WORK}/table_{label}.json", "w"), indent=1)
        res = {y: boot_outcome(rows, y, seed) for y in OUTCOMES}
        report[label] = {"n_cells": len(rows),
                         "n_lineages": len({r["lineage"] for r in rows}),
                         "lineage_sizes": dict(sorted(defaultdict(int, {k: sum(1 for r in rows if r["lineage"] == k) for k in {r["lineage"] for r in rows}}).items())),
                         "outcomes": res, "gate_view": gate_view(rows)}
        if label == "MAIN":
            report[label]["verdict"] = verdict(res)
        print(label, report[label]["n_cells"], report[label]["n_lineages"], flush=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print("wrote", a.out, sha256(a.out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ladder")
    p.add_argument("--out", default=f"{WORK}/ladder_predictors.json")
    p.add_argument("--only", default=None)
    p = sub.add_parser("outcomes")
    p.add_argument("--predictors", default=f"{WORK}/ladder_predictors.json")
    p.add_argument("--out", default=f"{WORK}/outcomes.json")
    p = sub.add_parser("analyze")
    p.add_argument("--predictors", default=f"{WORK}/ladder_predictors.json")
    p.add_argument("--outcomes", default=f"{WORK}/outcomes.json")
    p.add_argument("--out", default=f"{WORK}/analysis.json")
    a = ap.parse_args()
    {"ladder": cmd_ladder, "outcomes": cmd_outcomes, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
