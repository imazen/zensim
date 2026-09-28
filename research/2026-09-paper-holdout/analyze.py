#!/usr/bin/env python3
"""paper-holdout lane: assemble per-pair vectors and compute the table rows.

Computes no statistic itself: every SROCC/PLCC/KROCC comes from
`zen_stats.panel_batch` (the Rust `panel` binary, zenstats), and every CI from
`benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py` (the reference-
clustered paired bootstrap owner), which this script only feeds.

Subcommands
  prep      comparator vectors that need no new scoring: fast-ssim2 on AIC-4
            at full resolution (extractor audit channel, same decoded
            buffers) and frozen B on AIC-4 crops / full resolution / KonJND /
            CID22 / CSIQ / AIC-3 (board and verified stored dumps).
  assemble  join the new GMSD and DVIFM-ish scores, write the owner's
            pp_<ARM>_<corpus>.tsv inputs, per-arm tables for
            build_peer_fullevals --admitted-manifest, and stats.json.

Row alignment: every vector is in the board table's row order (build_pairs.py);
each stored vector is accepted only if its own target column equals the pair
list's target index-wise (1e-9), and extractor outputs are mapped through the
extractor's stable reference-basename sort with a per-row ref+label check.
"""
import csv, json, math, os, sys, hashlib, statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts", "lib"))
from zen_stats import panel_batch  # noqa: E402

A = os.environ.get("OUT", "/var/tmp/paper-holdout/a")
RM = "/mnt/v/output/zensim/reports/refmetrics"
FE = "/mnt/v/output/zensim/reports/fulleval"
BAR = "/mnt/v/output/zensim/ssim2-bar-2026-08-31"
DVP = "/var/tmp/dvifmish/peers"
DVPAIRS = "/mnt/v/output/zensim/dvifmish-eval-2026-09-22/pairs"
KONJND_ORDER = ("/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/"
                "ext_konjnd_jpeg_val.parquet")
B_BOARD = f"{FE}/b_sdr_linear_cid80_inclwinsor_dense_dial@cur372.fulleval.json"

# CID22 statistics on CID22-A(25) ONLY (coordinator ruling 2026-09-23): the
# full 4,292-pair `cid22` list is scored and aligned (pixels, target equality)
# but never correlated; `cid22a` rows are selected from it by `src_row`.
CORPORA = ["cid22a", "csiq", "aic3", "aic4", "aic4full", "sdr25", "konjnd"]
# label divisor used by the bootstrap owner, and target orientation.
DIV = {"cid22": 100.0, "cid22a": 100.0}
ORIENT = {"cid22a": "quality (MCOS)", "csiq": "quality (1-DMOS)",
          "aic3": "quality (-0.25 x design JND level)",
          "aic4": "distortion (reconstructed JND)",
          "aic4full": "distortion (reconstructed JND)",
          "sdr25": "distortion (q_jnd)",
          "konjnd": "construct: PJND = JPEG quality at the 50% visibility threshold"}
EXPECT_SIGN = {"cid22a": 1, "csiq": 1, "aic3": 1, "aic4": -1, "aic4full": -1,
               "sdr25": -1, "konjnd": None}
# board per-pair peer tables (context arms; each read with its own target)
BOARD_SSIM2 = {"cid22": ("cid22_ssim2.tsv", "ssim2"),
               "csiq": ("csiq_ssim2_gpu.tsv", "ssim2_gpu"),
               "aic3": ("aic3_ssim2_heldout.tsv", "ssim2_gpu"),
               "aic4": ("aic4_ssim2_gpu.tsv", "ssim2_gpu"),
               "sdr25": ("sdr25_ssim2.tsv", "ssim2"),
               "konjnd": ("konjnd_ssim2_heldout.tsv", "ssim2_gpu")}
PRESETS = {"DVtalk": "talk-faithful-luma", "DVours": "ours-full-luma",
           "DVgate": "serving-gate-ycbcr3"}


def read_tsv(p, delim="\t"):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter=delim))


def sha(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pairs(c):
    return read_tsv(f"{A}/pairs/{c}.tsv")


def src_rows(c):
    """Row indices into the parent list (cid22a -> cid22); identity otherwise."""
    return [int(r["src_row"]) for r in pairs(c)]


def parent(c):
    return "cid22" if c == "cid22a" else c


def check_targets(name, c, got, want, tol=1e-9):
    assert len(got) == len(want), f"{name}/{c}: {len(got)} rows != {len(want)}"
    d = max(abs(a - b) for a, b in zip(got, want))
    assert d <= tol, f"{name}/{c}: targets differ index-wise by {d}"
    return d


def extractor_perm(pair_rows, ref_col="ref_path"):
    """Row j of a Rev3-extractor output = pair row perm[j] (stable sort by
    reference basename; measured on cid22_49/nncd by the paper-holdout lane)."""
    return sorted(range(len(pair_rows)),
                  key=lambda i: os.path.basename(pair_rows[i][ref_col]))


def rejoin_bake(pair_rows, target_key, parquet, bake_tsv):
    import pyarrow.parquet as pq
    feat = pq.read_table(parquet, columns=["ref_basename", "human_score"]).to_pylist()
    bake = read_tsv(bake_tsv)
    perm = extractor_perm(pair_rows)
    assert len(feat) == len(bake) == len(pair_rows)
    out = [None] * len(pair_rows)
    for j, (f, b) in enumerate(zip(feat, bake)):
        i = perm[j]
        assert int(b["idx"]) == j
        assert f["ref_basename"] == os.path.basename(pair_rows[i]["ref_path"]), (j, i)
        assert abs(f["human_score"] - float(pair_rows[i][target_key])) < 1e-9, (j, i)
        assert abs(float(b["human"]) - f["human_score"]) < 1e-9, (j, i)
        out[i] = float(b["score"])
    assert all(v is not None for v in out)
    return out


# ---------------------------------------------------------------- prep
def prep():
    os.makedirs(f"{A}/refmetrics", exist_ok=True)
    os.makedirs(f"{A}/vec", exist_ok=True)
    notes = {}
    # fast-ssim2 at AIC-4 full resolution from the extractor audit channel
    full = pairs("aic4full")
    aud = {}
    with open(f"{DVP}/audit_aic4_full.jsonl") as f:
        for line in f:
            e = json.loads(line)
            aud[(e["reference"], e["distorted"])] = e
    p = f"{A}/refmetrics/aic4full_ssim2.tsv"
    with open(p, "w") as f:
        f.write("ref_path\tdist_path\thuman_score\tssim2\n")
        for r in full:
            e = aud[(r["ref_path"], r["dist_path"])]
            assert abs(e["human_score"] - float(r["target"])) < 1e-9
            f.write(f"{r['ref_path']}\t{r['dist_path']}\t{r['target']}\t"
                    f"{e['peer_ssim2']['score']!r}\n")
    notes["aic4full_ssim2"] = dict(source=f"{DVP}/audit_aic4_full.jsonl",
                                   sha256=sha(f"{DVP}/audit_aic4_full.jsonl"),
                                   implementation="fast-ssim2 (extractor audit, same decoded RGB8 buffers)")
    # CID22-A(25) peer table for the bootstrap owner: the board's cid22 ssim2
    # rows restricted to the A references, board order kept.
    rows = read_tsv(f"{RM}/cid22_ssim2.tsv")
    sr = src_rows("cid22a")
    with open(f"{A}/refmetrics/cid22a_ssim2.tsv", "w") as f:
        f.write("\t".join(rows[0].keys()) + "\n")
        for i in sr:
            f.write("\t".join(rows[i].values()) + "\n")
    # frozen B vectors
    B = {}
    board = json.load(open(B_BOARD))
    notes["B_board"] = dict(path=B_BOARD, sha256=sha(B_BOARD), bake=board.get("bake"),
                            bake_sha256=board.get("bake_sha256"))
    for c in ("cid22a", "csiq", "aic3"):
        src = parent(c)
        rows = read_tsv(f"{BAR}/pp_B_{src}.tsv")
        if c == "cid22a":   # only CID22-A rows are touched (ruling 2026-09-23)
            rows = [rows[i] for i in src_rows(c)]
        pr = pairs(c)
        div = DIV.get(c, 1.0)
        check_targets("B", c, [float(x["human"]) for x in rows],
                      [float(x["target"]) / div for x in pr])
        B[c] = [float(x["pred"]) for x in rows]
        notes[f"B_{c}"] = dict(source=f"{BAR}/pp_B_{src}.tsv", sha256=sha(f"{BAR}/pp_B_{src}.tsv"),
                               era="current 372 root (2026-08-30), ssim2-bar dump",
                               **({"subset": "CID22-A(25) rows"} if c == "cid22a" else {}))
    for c, n in (("aic4", 300), ("konjnd", 504)):
        pp = board["per_pair"][c]
        pr = pairs(c)
        tgt = pp["jnd"][:n]
        check_targets("B", c, tgt, [float(x["target"]) for x in pr])
        B[c] = pp["pred"][:n]
        notes[f"B_{c}"] = dict(source=f"{B_BOARD} per_pair.{c}[:{n}]",
                               era="current 372 root (2026-08-30), board row")
    # AIC-4 full resolution: the DVIFM-ish lane's Rev3-extractor B, rejoined here
    dvfull = read_tsv(f"{DVPAIRS}/aic4_full.tsv")
    vals = rejoin_bake(dvfull, "human_score", f"{DVP}/features_aic4_full.parquet",
                       f"{DVP}/zensim_b_aic4_full.tsv")
    by_path = {r["dist_path"]: v for r, v in zip(dvfull, vals)}
    B["aic4full"] = [by_path[r["dist_path"]] for r in full]
    notes["B_aic4full"] = dict(source=f"{DVP}/zensim_b_aic4_full.tsv",
                               sha256=sha(f"{DVP}/zensim_b_aic4_full.tsv"),
                               era="Rev3 public-human-eval extractor (ZENSIM_FORMULA_REV=3) + by-id B bake")
    # era cross-check on the crops: board B vs the same Rev3-extractor pipeline
    dvptc = read_tsv(f"{DVPAIRS}/aic4_ptc.tsv")
    vptc = rejoin_bake(dvptc, "human_score", f"{DVP}/features_aic4_ptc.parquet",
                       f"{DVP}/zensim_b_aic4_ptc.tsv")
    by_path = {r["dist_path"]: v for r, v in zip(dvptc, vptc)}
    B_rev3_aic4 = [by_path[r["dist_path"]] for r in pairs("aic4")]
    # SDR25: this lane's Rev3-extractor run
    sdr = pairs("sdr25")
    B["sdr25"] = rejoin_bake(sdr, "target", f"{A}/sdr25b/features.parquet",
                             f"{A}/sdr25b/zensim_b.tsv")
    notes["B_sdr25"] = dict(source=f"{A}/sdr25b/zensim_b.tsv",
                            era="Rev3 public-human-eval extractor (ZENSIM_FORMULA_REV=3) + by-id B bake")
    json.dump(dict(B=B, B_rev3_aic4=B_rev3_aic4, notes=notes),
              open(f"{A}/vec/comparators.json", "w"), indent=0)
    print("prep: comparators written", {k: len(v) for k, v in B.items()})


# ------------------------------------------------------------ assemble
def load_gmsd(c, tag=None):
    if c == "cid22a":
        v, col = load_gmsd("cid22")
        return [v[i] for i in src_rows(c)], col
    import pyarrow.parquet as pq
    t = pq.read_table(f"{A}/scores/gmsd_{tag or c}.parquet").to_pylist()
    col = next(k for k in t[0] if k.startswith("gmsd_cpu_imazen"))
    by = {}
    for r in t:
        i = json.loads(r["knob_tuple_json"])["row"]
        assert i not in by
        by[i] = r[col]
    n = len(pairs(c))
    assert sorted(by) == list(range(n)), f"gmsd {c}: rows {len(by)} != {n}"
    v = [by[i] for i in range(n)]
    assert all(math.isfinite(x) for x in v), f"gmsd {c}: non-finite"
    return v, col


def load_dvifmish(c, preset):
    if c == "cid22a":
        got = load_dvifmish("cid22", preset)
        if got is None:
            return None
        sr = src_rows(c)
        return [got[0][i] for i in sr], [got[1][i] for i in sr]
    p = f"{A}/scores/dvifmish_{c}/{preset}.tsv"
    if not os.path.exists(p):
        return None
    rows = read_tsv(p)
    pr = pairs(c)
    assert len(rows) == len(pr)
    for r, q in zip(rows, pr):
        assert int(r["row"]) >= 0
    by = {int(r["row"]): r for r in rows}
    out_q, out_e = [], []
    for i, q in enumerate(pr):
        r = by[i]
        assert r["ref_path"] == q["ref_path"] and r["dist_path"] == q["dist_path"], i
        out_q.append(float(r["quality"]))
        out_e.append(-float(r["distortion"]))
    return out_q, out_e


def ssim2_vec(c):
    if c == "aic4full":
        rows = read_tsv(f"{A}/refmetrics/aic4full_ssim2.tsv")
        return [float(r["ssim2"]) for r in rows], [float(r["human_score"]) for r in rows]
    tbl, col = BOARD_SSIM2[parent(c)]
    rows = read_tsv(f"{RM}/{tbl}")
    if c == "cid22a":
        rows = [rows[i] for i in src_rows(c)]
    if c == "konjnd":
        rows = [r for r in rows if "/jpeg/" in r["dist_path"]]
    pr = pairs(c)
    assert [r["dist_path"] for r in rows] == [q["orig_dist_path"] for q in pr], c
    return [float(r[col]) for r in rows], None


def context_arms(c):
    """Board peer tables (implementation labels in the record), own targets."""
    out = {}
    if c in ("aic4full", "cid22a"):   # cid22a: the paper cites the board's 49-ref peer rows
        return out
    stems = {"cid22": "cid22", "csiq": "csiq", "aic3": "aic3", "aic4": "aic4",
             "sdr25": "sdr25", "konjnd": "konjnd"}[c]
    cands = {
        "butteraugli": [f"{stems}_butter.tsv", f"{stems}_butteraugli_gpu.tsv",
                        f"{stems}_butteraugli_heldout.tsv"],
        "iwssim": [f"{stems}_iwssim.tsv", f"{stems}_iwssim_gpu.tsv", f"{stems}_iwssim_heldout.tsv"],
        "cvvdp_4k": [f"{stems}_cvvdp.tsv", f"{stems}_cvvdp_heldout.tsv"],
        "cvvdp_fhd": [f"{stems}_cvvdp_standard_fhd.tsv"] if c == "aic4" else [],
    }
    hc = ["MCOS", "human_score", "jnd", "q_jnd", "pjnd"]
    for fam, files in cands.items():
        f = next((x for x in files if os.path.exists(f"{RM}/{x}")), None)
        if f is None:
            continue
        rows = read_tsv(f"{RM}/{f}")
        if c == "konjnd":
            rows = [r for r in rows if "/jpeg/" in r["dist_path"]]
        h = next(k for k in hc if k in rows[0])
        cols = [k for k in rows[0] if k not in ("ref_path", "dist_path", "dist_id", h,
                                                  "img_num", "dlevel")]
        for col in cols:
            sign = -1.0 if "butter" in col else 1.0
            name = fam if fam != "butteraugli" else (
                "butteraugli_max" if "max" in col else "butteraugli_p3")
            out[name] = dict(pred=[sign * float(r[col]) for r in rows],
                             target=[float(r[h]) for r in rows],
                             refs=[os.path.basename(r["ref_path"]) for r in rows],
                             source=f"{RM}/{f}:{col}" + (" (negated)" if sign < 0 else ""))
    return out


def per_ref(pred, tgt, refs):
    groups = defaultdict(list)
    for i, r in enumerate(refs):
        groups[r].append(i)
    jobs = []
    for k in sorted(groups):
        idx = groups[k]
        if len(idx) < 3:
            continue
        xs = [pred[i] for i in idx]; ys = [tgt[i] for i in idx]
        if all(x == xs[0] for x in xs) or all(y == ys[0] for y in ys):
            continue
        jobs.append((f"g{len(jobs)}", xs, ys))
    if len(jobs) < 2:
        return None, 0
    res = panel_batch(jobs, stats="full")
    return statistics.fmean(r["srocc_signed"] for r in res), len(jobs)


def assemble():
    comp = json.load(open(f"{A}/vec/comparators.json"))
    os.makedirs(f"{A}/pp", exist_ok=True)
    os.makedirs(f"{A}/tables", exist_ok=True)
    stats = {}
    checks = {}
    import pyarrow.parquet as pq
    kon_order = pq.read_table(KONJND_ORDER, columns=["ref_basename"]).column(0).to_pylist()
    for c in CORPORA:
        pr = pairs(c)
        tgt = [float(r["target"]) for r in pr]
        refs = [r["ref_key"] for r in pr]
        div = DIV.get(c, 1.0)
        arms = {}
        s2, s2t = ssim2_vec(c)
        if s2t is not None:
            check_targets("ssim2", c, s2t, tgt)
        arms["ssim2"] = dict(pred=s2, source="board table" if c != "aic4full" else
                             "fast-ssim2 extractor audit (full resolution)")
        arms["B"] = dict(pred=comp["B"][c], source=comp["notes"].get(f"B_{c}", {}).get("source"))
        g, col = load_gmsd(c)
        arms["GMSD"] = dict(pred=[-x for x in g], source=f"{A}/scores/gmsd_{c}.parquet:{col} (negated)")
        for arm, preset in PRESETS.items():
            got = load_dvifmish(c, preset)
            if got is None:
                continue
            q, e = got
            arms[arm] = dict(pred=q, source=f"{A}/scores/dvifmish_{c}/{preset}.tsv:quality",
                             neg_distortion=e)
        # the bootstrap owner's inputs (pred, human, ref) in board order
        for arm, d in arms.items():
            if arm == "ssim2":
                continue
            rows = list(zip(d["pred"], [t / div for t in tgt], refs))
            if c == "konjnd":   # owner's JOIN mode: rows in the eval parquet's order
                pos = {r["ref_key"].rsplit(".", 1)[0]: i for i, r in enumerate(pr)}
                rows = [rows[pos[k]] for k in kon_order]
            with open(f"{A}/pp/pp_{arm}_{c}.tsv", "w") as f:
                f.write("pred\thuman\tref\n")
                for p_, h_, r_ in rows:
                    f.write(f"{p_!r}\t{h_!r}\t{r_}\n")
            with open(f"{A}/tables/{arm}_{c}.tsv", "w") as f:
                f.write("ref_path\tdist_path\ttarget\tmetric\n")
                for r, p_ in zip(pr, d["pred"]):
                    f.write(f"{r['ref_path']}\t{r['dist_path']}\t{r['target']}\t{p_!r}\n")
        ctx = context_arms(c)
        # canonical panel, one batch per corpus
        jobs = [(a, d["pred"], tgt) for a, d in arms.items()]
        jobs += [(f"ctx:{a}", d["pred"], d["target"]) for a, d in ctx.items()]
        jobs += [(f"negE:{a}", d["neg_distortion"], tgt) for a, d in arms.items()
                 if "neg_distortion" in d]
        res = {x["label"]: x for x in panel_batch(jobs, stats="full")}
        cs = {}
        for a, d in list(arms.items()) + [(f"ctx:{k}", v) for k, v in ctx.items()]:
            r = res[a]
            rr = d.get("refs", refs)
            t_ = d.get("target", tgt)
            pm, ng = (None, 0) if c == "konjnd" else per_ref(d["pred"], t_, rr)
            cs[a] = dict(srocc=r["srocc"], srocc_signed=r["srocc_signed"], plcc=r["plcc"],
                         krocc=r["krocc"], z_rmse=r["z_rmse"], n=r["n"],
                         n_dropped=r["n_dropped"], per_ref_mean_signed=pm, per_ref_groups=ng,
                         source=d["source"])
        for a, d in arms.items():
            if "neg_distortion" in d:
                checks[f"{c}:{a}:srocc(quality)-srocc(-E)"] = res[a]["srocc"] - res[f"negE:{a}"]["srocc"]
        stats[c] = dict(n=len(pr), refs=len(set(refs)), target_orientation=ORIENT[c],
                        expected_sign=EXPECT_SIGN[c], arms=cs)
        print(c, {a: round(v["srocc"], 4) for a, v in cs.items()})
    # AIC-4 crop era cross-check (board B vs Rev3-extractor B)
    tgt = [float(r["target"]) for r in pairs("aic4")]
    x = panel_batch([("board", comp["B"]["aic4"], tgt), ("rev3", comp["B_rev3_aic4"], tgt),
                     ("agree", comp["B"]["aic4"], comp["B_rev3_aic4"])], stats="full")
    checks["B_era_aic4_crop"] = dict(board_srocc=x[0]["srocc"], rev3_srocc=x[1]["srocc"],
                                     srocc_between=x[2]["srocc"])
    # decoder sensitivity (KonJND): GMSD on zenmetrics' own JPEG decode vs decode-owner PNGs
    gp, _ = load_gmsd("konjnd")
    gj, _ = load_gmsd("konjnd", tag="konjnd_rawjpg")
    kt = [float(r["target"]) for r in pairs("konjnd")]
    y = panel_batch([("png", gp, kt), ("jpg", gj, kt)], stats="full")
    checks["gmsd_konjnd_decoder"] = dict(png_srocc_signed=-y[0]["srocc_signed"],
                                         rawjpg_srocc_signed=-y[1]["srocc_signed"],
                                         max_abs_score_diff=max(abs(a - b) for a, b in zip(gp, gj)))
    json.dump(dict(stats=stats, checks=checks), open(f"{A}/stats.json", "w"), indent=1)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    {"prep": prep, "assemble": assemble}[sys.argv[1]]()
