"""Data files of the AIC2026 TRAIN dashboard. Exploration record: benchmarks/byconstruction_2026-10-22.md.

results.json holds every summary table in long form (`table` names it, `id` keys a row); predictions.json one row per
WG1 public_train crop (official JND, B score, out-of-fold T and Tga outputs, butteraugli 3-norm), without stimulus
names or levels. Everything is read from the study's saved outputs; the E21 scores, mixture report and panel are
regenerated through their owners (byc_cells.py score, mixture_test.py report, aic_panel.py). Nothing new is computed
on AIC2026 public_test: the panel's frozen-B row reuses the one read, and public_test label statistics are dropped.

  python3 aic_dashboard_data.py --out DIR --arms Tga,Ma,Bga,... [--commits-since 322930e4]
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aic_train as A  # noqa: E402

REPO = HERE.parents[1]
O = Path("/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train")
F = O / "folds_train"
CELLS = Path.home() / "tmp/byc/cells"
BIN = os.environ.get("AIC_BIN", "/var/tmp/byconstruction-bin/58a4d100ad36")
GROOT = "/mnt/v/output/zensim/byconstruction-2026-10-22/ringhpm/root_globaff"
MIX = "/mnt/v/output/zensim/byconstruction-2026-10-22/mixture"
SCORES = Path("/home/lilith/tmp/aic-full-repro/scores")
LOGS = Path.home() / "tmp/aictrain"

# Combined E21 + AIC arms: label, class, AIC weight.
ARMS = {
    "Tga": ("Tga", "monotone in damage inputs, descriptor sign-free", 32),
    "Ma": ("Ma (certified max pairs)", "certified convex monotone max pairs", 32),
    "Bga": ("Bga (B's signed recipe)", "pin only (signed)", 32),
    "Ta": ("Ta (certified hinges)", "certified convex monotone", 32),
    "Ts1a": ("Ts1a (no scale 0)", "monotone in damage inputs, descriptor sign-free, no scale-0 inputs", 32),
    "Mgfa": ("Mgfa (max pairs + free descriptor)", "convex monotone in damage inputs, descriptor sign-free", 32),
    "Tga128": ("Tga128 (AIC weight 128)", "monotone in damage inputs, descriptor sign-free", 128),
}
# Same model classes without the AIC leg, for the arms table.
CONTROLS = {"Tghpm": "Tghpm (no AIC leg)", "Bghpm": "Bghpm (B + descriptor, no AIC leg)"}
COMMIT_SUMMARY = {
    "d24c2ec6": "AIC2026 feature tables, source-held-out fits, band-feature prototypes",
    "31e3ba97": "AIC2026 as a research training leg through the Rust trainer (strict refusal kept)",
    "67e7659e": "Found the official split (WG1 public_train); fold tables; scale-0-free input set",
    "805242d9": "Trainer: max pairs accept a sign-free input (new monotonicity test)",
    "b7aba684": "First E21 + AIC arm (Tga); engine skip attempt recorded, not landed",
    "bf828d6c": "Full-data research fit route; per-type bias by arm",
    "4da1d051": "Six-statistic panel per dataset and split; Ma arm; AIC test-data ledger",
}
# Every use of AIC2026 public_test or its 14 sources on 2026-10-23, counted from the commands run (record section
# "AIC2026 TRAIN"). Edit only when a new use happens.
EXPOSURE = [
    ("Before the split was found", "Overlay labels (JND_curve), 42-source pool",
     "Frozen models scored, no fit: B, full-data T and full-data M", 1, "score only"),
    ("Before the split was found", "Overlay labels (JND_curve), 42-source pool",
     "Descriptive label reads: label statistics and supplied-metric ranking; confidence-interval ceiling; pointwise vs "
     "curve; single-feature rankings (4 feature sets); cross-ladder agreement; one ladder inspected by eye (2)", 10, "description"),
    ("Before the split was found", "Overlay labels (JND_curve), 42-source pool",
     "Python source-fold fits with the 14 public_test sources inside the folds: fold fits, peer scans, learning curves (5), "
     "prototypes, oracles (2), target transforms (3), scale tests (3), diagnostics", 32, "fit and score"),
    ("Before the split was found", "Overlay labels (JND_curve), 42-source pool",
     "Rust trainer: T and M on 6 source folds each, one 2-epoch smoke fit, one E21 smoke cell", 4, "fit and score"),
    ("After the split was found", "Official labels (JND_score), public_test sources",
     "Frozen B scored once, with the reconstruction-agreement check in the same script", 1, "score only"),
    ("After the split was found", "Official labels (JND_score), public_test sources",
     "Per-source JND statistics computed for the dashboard, then removed from it", 1, "description"),
    ("After the split was found", "Official labels (JND_score), public_test sources",
     "Six-statistic panel on the same frozen-B predictions (the dashboard)", 1, "score only"),
    ("After the split was found", "Official labels (JND_score), public_test sources",
     "Fits, model choices or candidate reads", 0, "none"),
]


def slug(s: str, n: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:n]


def last_result(path: Path):
    r = None
    for line in path.read_text().splitlines():
        if line.startswith("test result"):
            m = re.search(r"(\d+) passed; (\d+) failed; (\d+) ignored", line)
            r = tuple(map(int, m.groups()))
    return r


def oof_rust(arm: str, keys) -> pd.Series:
    s = pd.Series(np.nan, index=keys)
    for k in range(7):
        kk = pd.read_parquet(F / f"aic_k{k}_test.parquet", columns=["key"]).key.astype(str).to_numpy()
        s.loc[kk] = pd.read_csv(O / "rust/folds_train" / arm / f"k{k}" / "pred.tsv", sep="\t").pred.to_numpy()
    return s


def oof_cells(arm: str, keys) -> pd.Series:
    by = {}
    for rp in glob.glob(str(CELLS / arm / "*/byc_record.json")):
        r = json.loads(Path(rp).read_text())
        by.setdefault(r["aic_fold"], []).append(
            pd.read_csv(Path(rp).parent / f"pred_aic_k{r['aic_fold']}.tsv", sep="\t").pred.to_numpy())
    s = pd.Series(np.nan, index=keys)
    for k, ps in by.items():
        kk = pd.read_parquet(F / f"aic_k{k}_test.parquet", columns=["key"]).key.astype(str).to_numpy()
        s.loc[kk] = np.mean(ps, 0)
    return s


def run_json(cmd: list[str], env: dict) -> dict:
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout
    return json.loads(out[out.index("{"):])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", required=True, help="finished combined arms, comma-separated")
    ap.add_argument("--commits-since", default="322930e4")
    a = ap.parse_args()
    arms = a.arms.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, REV4_V2_BIN_DIR=BIN, BYC_ROOT=GROOT)
    rows = []

    def add(table, **kw):
        rows.append({"table": table, **kw})

    ptr = pd.read_csv(F / "pairs_train.tsv", sep="\t")
    pte = pd.read_csv(F / "pairs_test.tsv", sep="\t")
    keys = ptr.key.astype(str)
    y_tr = ptr.set_index("key").jnd
    q = lambda pred: float(spearmanr(np.asarray(pred, float), -y_tr.loc[keys]).correlation)  # noqa: E731
    b = pd.read_csv("/mnt/v/output/zensim/jnd-reconstruction-2026-10-19-b/JND_curve.scored.csv").set_index("distorted").score_b
    curve = pd.read_csv("/mnt/v/output/zensim/jnd-reconstruction-2026-10-19-b/JND_curve.csv").set_index("distorted").JND_curve
    T, S = oof_rust("T", keys), oof_rust("S", keys)
    tga = oof_cells("Tga", keys)
    allp = json.loads((O / "allpeers_train.json").read_text())["results"]
    ceil = max(r["oof_global"] for r in allp)

    add("headline", id="b", label="B today (zero-shot)", value=q(b.loc[keys]))
    add("headline", id="t", label="Convex monotone T, AIC-only, out of fold", value=q(T))
    add("headline", id="tga", label="Tga, E21 + AIC, out of fold", value=q(tga))
    add("headline", id="ceiling", label="Every available measurement, out of fold", value=ceil)
    add("headline", id="labels", label="Two WG1-derived reconstructions agree",
        value=float(spearmanr(curve.loc[keys], y_tr.loc[keys]).correlation))
    add("headline", id="target", label="Goal", value=0.99)

    zs = json.loads((O / "zeroshot_train_wg1.json").read_text())["models"]
    for name, lab, g in (("B", "B (diff+frag-820)", "pin only"), ("Tfull", "full-data T (old data, no AIC)", "convex monotone"),
                         ("Mfull", "full-data M (old data, no AIC)", "convex monotone, max pairs")):
        m = zs[name]
        add("models", model=lab, fit="none (zero-shot)", guarantee=g, global_=m["global"],
            within_source=m["within_ref_mean"], within_ladder=m["within_ladder_mean"])
    cv = json.loads((O / "cv_train_wg1_fx1g.json").read_text())["models"]
    for name, lab, g in (("nnls", "NNLS linear (Python)", "monotone"), ("gamcvx", "additive hinges (Python)", "convex monotone"),
                         ("hgbmono", "gradient boosting (Python)", "monotone"), ("hgb", "gradient boosting (Python)", "none")):
        m = cv[name]
        add("models", model=lab, fit="out of fold, 7 source folds x 3", guarantee=g, global_=m["global_mean"],
            within_source=m["within_ref_mean"], within_ladder=m["within_ladder_mean"])
    for arm, lab, g in (("T", "T (Rust trainer, AIC only)", "convex monotone, certified"),
                        ("M", "M (Rust trainer, AIC only)", "convex monotone max pairs, certified"),
                        ("S", "S (Rust trainer, AIC only)", "pin only")):
        m = json.loads((O / "rust/folds_train" / arm / "oof_panel.json").read_text())
        add("models", model=lab, fit="out of fold, 7 source folds", guarantee=g, global_=m["global"],
            within_source=m["within_ref_mean"], within_ladder=m["within_ladder_mean"])
    aicscore = {}
    for arm in arms:
        subprocess.run([sys.executable, str(REPO / "scripts/rev4_featpot/byc_cells.py"), "aicscore", "--arms", arm,
                        "--out", str(O / f"aicscore_{arm}.json")], check=True, capture_output=True, env=env)
        aicscore[arm] = json.loads((O / f"aicscore_{arm}.json").read_text())[arm]
        m = aicscore[arm]
        add("models", model=f"{ARMS[arm][0]}, E21 + AIC leg", fit="out of fold, rotating source fold",
            guarantee=ARMS[arm][1], global_=m["global"], within_source=m["within_ref_mean"],
            within_ladder=m["within_ladder_mean"])
    for r in allp:
        if r["inputs"] != "fx1g only":
            add("models", model=f"ceiling: {r['inputs']} ({'monotone ' if r['model'] == 'hgbmono' else ''}boosting)",
                fit="out of fold, 7 source folds", guarantee="monotone" if r["model"] == "hgbmono" else "none",
                global_=r["oof_global"], within_source=None, within_ladder=None)

    order = pd.read_parquet(F / "aic_train_all.parquet", columns=["key"]).key.astype(str)
    ins = pd.read_csv(O / "rust/folds_train/T_full/pred_insample.tsv", sep="\t").pred.to_numpy()
    add("overfit", id="t_aic", model="T, AIC only", in_sample=float(spearmanr(ins, -y_tr.loc[order]).correlation), out_of_fold=q(T))
    tz = [pd.read_csv(f, sep="\t").pred.to_numpy() for f in sorted(glob.glob(str(LOGS / "zs_tghpm/*.tsv")))]
    add("overfit", id="tghpm_zero", model="Tghpm cells, no AIC leg (zero-shot mean)", in_sample=None,
        out_of_fold=float(np.mean([spearmanr(p, -y_tr.loc[order]).correlation for p in tz])))
    for k, refs in ((2, 21), (3, 28), (6, 35), (14, 39), (42, 41)):
        j = json.loads((LOGS / f"lc_{k}.json").read_text())["models"]
        add("learning", id=f"n{refs}", train_sources=refs, hgbmono=j["hgbmono"]["global_mean"], nnls=j["nnls"]["global_mean"])
    an = json.loads((O / "analyze_fx1g_hgbmono.json").read_text())
    steps = {"remove mean residual by codec": "Remove each codec’s mean error",
             "remove mean residual by group": "Remove each source’s mean error",
             "remove mean residual by group+codec": "Remove each ladder’s mean error (source × codec)",
             "perfect within-ladder shape, ladder offsets kept": "Perfect shape inside each ladder, ladder offsets kept"}
    add("oracle", id="none", step="Out of fold, no correction", srocc=an["oof_global"])
    for k, v in an["oracle"].items():
        add("oracle", id=slug(k), step=steps[k], srocc=v)
    for i, (k, v) in enumerate((k, v) for k, v in an["discordance"].items() if k.startswith("|dJND|")):
        add("discord", id=f"g{i}", gap=k.replace("|dJND| ", "").replace("[", "").replace(")", "").replace(",", "–"),
            pair_share=v["pair_share"], discordant=v["discordant"], share_of_discordant=v["share_of_discordant"])

    # E21 + mixture for every arm with cells
    have = [x for x in list(CONTROLS) + arms if (CELLS / x).is_dir()]
    sc = run_json([sys.executable, str(REPO / "scripts/rev4_featpot/byc_cells.py"), "score", "--arms", ",".join(have)], env)
    mix = run_json([sys.executable, str(HERE / "mixture_test.py"), "report", "--out", MIX], env)
    for arm in have:
        s = sc[arm]
        cells = [v for k, v in mix.items() if k.startswith(arm + "_")]
        lab = CONTROLS.get(arm) or ARMS[arm][0]
        add("arms", id=slug(arm), arm=lab, aic_weight=None if arm in CONTROLS else ARMS[arm][2], e21_delta=s["delta"],
            e21_se=s["seed_unit_se"], **{d: s["per_dataset_delta"][d] for d in ("kadid", "tid2013", "konfig", "cid22_a25")},
            aic_oof=None if arm in CONTROLS else aicscore[arm]["global"],
            mix_min=min(v["inversions"] for v in cells) if cells else None,
            mix_max=max(v["inversions"] for v in cells) if cells else None,
            mix_max_rise=max(v["max_rise"] for v in cells) if cells else None, status="done")

    pc_b = zs["B"]["per_codec"]
    pc_t = json.loads((O / "rust/folds_train/T/oof_panel.json").read_text())["per_codec"]
    pc_g = aicscore["Tga"]["per_codec"]
    for c in sorted(set(ptr.codec) | set(pte.codec)):
        tr_, te_ = ptr[ptr.codec == c], pte[pte.codec == c]
        add("codecs", id=c, codec=c, train_rows=int(len(tr_)), test_rows=int(len(te_)), train_sources=int(tr_.group.nunique()),
            test_sources=int(te_.group.nunique()), mean_jnd_train=float(tr_.jnd.mean()) if len(tr_) else None,
            b=pc_b.get(c), t=pc_t.get(c), tga=pc_g.get(c))
    meta = pd.read_csv("/mnt/v/datasets/aic2026/AIC2026_source_images_metadata_and_attribution.csv")
    meta["src"] = meta.source.str[:3]
    catcol = [c for c in meta.columns if c.startswith("category")][0]
    pt = ptr.assign(b=b.loc[keys].to_numpy(), t=T.to_numpy(), tga=tga.to_numpy())
    for g_, gr in pt.groupby("group"):
        src = g_[4:7]
        add("sources", id=src, source=src, split="public_train", rows=int(len(gr)), codecs=int(gr.codec.nunique()),
            category=int(meta[meta.src == src].iloc[0][catcol]), mean_jnd=float(gr.jnd.mean()), max_jnd=float(gr.jnd.max()),
            b=float(spearmanr(gr.b, -gr.jnd).correlation), t=float(spearmanr(gr.t, -gr.jnd).correlation),
            tga=float(spearmanr(gr.tga, -gr.jnd).correlation))
    for g_, gr in pte.groupby("group"):  # no label statistics for public_test
        src = g_[4:7]
        add("sources", id=src, source=src, split="public_test", rows=int(len(gr)), codecs=int(gr.codec.nunique()),
            category=int(meta[meta.src == src].iloc[0][catcol]), mean_jnd=None, max_jnd=None, b=None, t=None, tga=None)

    def keyed(df):
        col = "distorted" if "distorted" in df.columns else "dist_path"
        return df.assign(key=df[col].astype(str).str.split("/").str[-1].str.removesuffix(".png")).set_index("key")
    bu, s2 = keyed(pd.read_csv(SCORES / "butteraugli.tsv", sep="\t")), keyed(pd.read_csv(SCORES / "ssim2.tsv", sep="\t"))
    gm, dv = keyed(pd.read_csv(SCORES / "gmsd.tsv", sep="\t")), keyed(pd.read_csv(SCORES / "dvifm-v03-rs.tsv", sep="\t"))
    sup = pd.read_csv(A.PEER_TABLE, low_memory=False)
    sup = sup[sup.view == "cropped"]
    sup.index = sup.distorted.str.removesuffix(".png")
    third = "third-party software may not label training data"
    for pid, lab, v, sign, owner, elig, note in (
            ("butter3", "butteraugli 3-norm (imazen crate)", bu.loc[keys].butteraugli_pnorm3, -1, "imazen", True, ""),
            ("buttermax", "butteraugli max (imazen crate)", bu.loc[keys].butteraugli_max, -1, "imazen", True, ""),
            ("ssim2", "SSIMULACRA2 (fast-ssim2)", s2.loc[keys].ssim2, 1, "imazen", True, "the teacher of the existing SafeSyn and CID22 legs"),
            ("gmsd", "GMSD (imazen crate)", gm.loc[keys].gmsd_cpu_imazen_v0_1_0, -1, "imazen", True, ""),
            ("dvifm03", "DVIFM 0.3 (dvifmish)", dv.loc[keys].jnd, -1, "imazen", False, "author constants trained on AIC2026: not an independent teacher"),
            ("fsimc", "FSIMc (supplied column)", sup.loc[keys, "FSIMc"], 1, "third party", False, third),
            ("haarpsi", "HaarPSI (supplied column)", sup.loc[keys, "HaarPSI"], 1, "third party", False, third),
            ("cvvdp", "CVVDP (supplied column)", sup.loc[keys, "CVVDP"], 1, "third party", False, "levels were placed with CVVDP"),
            ("aimos", "AIMOS (supplied column)", sup.loc[keys, "AIMOS"], 1, "third party", False, third),
            ("b", "B (zensim diff+frag-820)", b.loc[keys], 1, "imazen", True, "today's profile"),
            ("t_oof", "T AIC-only, out of fold", T, 1, "imazen", True, "our own AIC-trained monotone model")):
        add("peers", id=pid, metric=lab, owner=owner, teacher_allowed=elig, srocc=q(sign * np.asarray(v, float)), note=note)
    pscan = json.loads((O / "peerscan.json").read_text())
    fx = [r["oof_global"] for r in allp if r["inputs"] == "fx1g only" and r["model"] == "hgb"][0]
    add("teacher", id="one_peer", question="Largest gain from adding any one peer metric to our inputs (42-source pool)",
        value=float(max(d for d, c, a_ in pscan["gains"])))
    add("teacher", id="all_peers", question="Gain from adding all supplied peers to our inputs (public_train)", value=ceil - fx)

    for tid, suite, log, state in (
            ("validate", "zensim-validate library (trainer: max pairs + sign-free mask)", LOGS / "validate_lib_tests.log", "current tree"),
            ("zensim", "zensim library", LOGS / "zensim_lib_tests_current.log", "current tree"),
            ("zensim_attempt", "zensim library, per-scale skip attempt", LOGS / "zensim_lib_tests2.log", "attempt, reverted"),
            ("zensim_allfeat_attempt", "zensim all features, per-scale skip attempt (lib binary)", LOGS / "zensim_allfeat_tests.log", "attempt, reverted")):
        p, f_, i = last_result(log)
        add("tests", id=tid, suite=suite, passed=p, failed=f_, ignored=i, state=state)
    add("tests", id="new_free_pairs", suite="new test: max pairs with a sign-free input stay monotone", passed=1, failed=0, ignored=0, state="current tree")
    lint = subprocess.run([sys.executable, str(REPO / "scripts/lint_scripts.py")], capture_output=True, text=True, cwd=REPO).stdout
    m_ = re.search(r"(\d+) scripts checked, all runnable", lint)
    add("tests", id="lint", suite="script lint (lint_scripts.py)", passed=int(m_.group(1)) if m_ else 0,
        failed=0 if m_ else 1, ignored=0, state="current tree")
    add("tests", id="table_route", suite="table route reproduces stored B scores bit for bit", passed=5768, failed=0, ignored=0, state="parity gate")
    add("tests", id="crop_hashes", suite="AIC2026 crop files SHA-256-equal to the 2026-10-19 admission", passed=5810, failed=0, ignored=0, state="parity gate")

    log = subprocess.run(["jj", "log", "-r", f"{a.commits_since}..explore/byconstruction@origin", "--no-graph", "-T",
                          'commit_id.short(8) ++ "\\t" ++ committer.timestamp().format("%Y-%m-%d %H:%M") ++ "\\t" ++ description.first_line() ++ "\\n"'],
                         capture_output=True, text=True, cwd=REPO, check=True).stdout
    for line in log.splitlines():
        c, t, d = line.split("\t")
        add("changes", id=c, commit=c, when=t, summary=COMMIT_SUMMARY.get(c, d), message=d)

    panel_json = O / "panel_mohammadi.json"
    subprocess.run([sys.executable, str(HERE / "aic_panel.py"), "--out", str(panel_json), "--arms",
                    ",".join(["base-c", "Thpm", "Mhpm"] + [x for x in have])], check=True, capture_output=True, env=env)
    names = {"kadid": "KADID", "tid2013": "TID2013", "konfig": "KonFiG", "cid22_a25": "CID22-A",
             "aic2026_public_train": "AIC2026 public_train", "aic2026_public_test": "AIC2026 public_test"}
    models = {"base-c": "B recipe (base-c)", "Thpm": "Thpm (convex monotone)", "Mhpm": "Mhpm (max pairs)", **CONTROLS,
              **{k: v[0] for k, v in ARMS.items()}}
    for x in json.loads(panel_json.read_text())["rows"]:
        add("panel", id=slug(x["dataset"] + " " + x["model"] + " " + x["sample"]), dataset=names[x["dataset"]],
            model=models.get(x["model"], x["model"]), sample=x["sample"],
            kind="out" if x["sample"].startswith(("out", "zero")) else "in", n=x["n"], seeds=x["seeds"],
            srocc=x["srocc_signed"], plcc=x["plcc"], krocc=x["krocc"], **{"or": x["or"]}, pwrc=x["pwrc"], z_rmse=x["z_rmse"])
    for i, (ph, labels, what, runs, use) in enumerate(EXPOSURE):
        add("exposure", id=f"e{i}", phase=ph, labels=labels, what=what, runs=runs, use=use)

    for t in {r["table"] for r in rows}:
        ids = [r.get("id") or slug(r.get("model", "") + " " + r.get("guarantee", "")) for r in rows if r["table"] == t]
        for r, i in zip((r for r in rows if r["table"] == t), ids):
            r["id"] = i
        if len(ids) != len(set(ids)):
            raise ValueError(f"{t}: duplicate row ids")
    (out / "results.json").write_text(json.dumps(rows, indent=0))
    pr = pd.DataFrame({"jnd": y_tr.loc[keys].round(3).to_numpy(), "codec": ptr.codec.to_numpy(),
                       "b": b.loc[keys].round(4).to_numpy(), "t": T.round(5).to_numpy(), "tga": tga.round(5).to_numpy(),
                       "butter": bu.loc[keys].butteraugli_pnorm3.round(4).to_numpy()})
    pr.sample(frac=1, random_state=0).reset_index(drop=True).to_json(out / "predictions.json", orient="records")
    print(f"{len(rows)} result rows; {len(pr)} prediction rows; arms {have}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
