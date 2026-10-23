"""Mohammadi 2025 panel (SROCC, PLCC, KROCC, OR, PWRC, Z-RMSE through the native `panel --batch`) per dataset, model
and sample role, for the by-construction AIC2026 TRAIN study (benchmarks/byconstruction_2026-10-22.md).

Sample roles:
- E21 sets (KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL, CID22-A 25 refs): `out of sample` is each LODO cell's
  prediction of the dataset it was trained without (seeds 0-2, statistics averaged); `in sample` is the seed-0 cell of
  another fold predicting this dataset, which it trained on (fit and development rows).
- AIC2026 public_train (WG1 JND_score, quality = -JND): `zero-shot` (no AIC fit), `out of fold` (source-held-out
  folds), `in sample` (fitted on all of public_train).
- AIC2026 public_test: frozen B only (the one read of 2026-10-23); no candidate is read.

  python3 aic_panel.py --out PANEL.json [--arms base-c,Tghpm,Tga,Thpm,Mhpm]
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from lib.zen_stats import panel_batch  # noqa: E402

BIN = Path(os.environ.get("AIC_BIN", "/var/tmp/byconstruction-bin/58a4d100ad36"))
os.environ.setdefault("REV4_V2_BIN_DIR", str(BIN))
os.environ.setdefault("ZEN_PANEL_BIN", str(BIN / "panel"))
OUT = Path("/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train")
F = OUT / "folds_train"
CELLS = Path.home() / "tmp/byc/cells"
E34 = Path("/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json")
E34_CELLS = Path("/var/tmp/rev4-featpot/e34-controlc-results/cells/sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1__N")
ROOTS = {"raw": Path("/mnt/v/output/zensim/v40r4-2026-10-08/v2e29"),
         "hpm": Path("/mnt/v/output/zensim/byconstruction-2026-10-22/ringhpm/root"),
         "g": Path("/mnt/v/output/zensim/byconstruction-2026-10-22/ringhpm/root_globaff")}


def arm_root(arm: str, spec: str) -> str:
    """The training root of an arm's cells: B's control is the raw root; fx1g/fx1gs1 arms the hpm + descriptor root."""
    if arm == "base-c":
        return "raw"
    return "g" if research_decl(spec) is not None else "hpm"
FOLDS = ("kadid", "tid2013", "konfig", "cid22_a25")
POP = {"kadid": "KADID TRAIN+SELECT", "tid2013": "TID2013 (train-only by ruling)", "konfig": "KonFiG TRAIN+VAL",
       "cid22_a25": "CID22-A, 25 references"}
STATS = ("srocc_signed", "plcc", "krocc", "or", "pwrc", "z_rmse")


def research_decl(spec: str):
    for tok in ("fx1gs1", "fx1g"):
        if f":{tok}" in spec:
            return HERE.parent / "rev4_featpot" / f"byc_{tok}_declaration.json"
    return None


def predict(bake: Path, table: Path, spec: str) -> np.ndarray:
    with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as td:
        decl = research_decl(spec)
        if decl is not None:
            subprocess.run([str(BIN / "byc_predict"), "--bake", str(bake), "--decl", str(decl), "--corpus", str(table),
                            "--out", f"{td}/p.tsv"], check=True, capture_output=True)
            return pd.read_csv(f"{td}/p.tsv", sep="\t").pred.to_numpy(np.float64)
        sys.path.insert(0, str(HERE.parent / "rev4_featpot"))
        from v2_lodo_mlp import predict as owner_predict  # noqa: E402
        return owner_predict(bake, table, Path(td) / "p.tsv")


def e21_jobs(arms: list[str]):
    panel_in = {d["name"]: d for d in json.loads(E34.read_text())["datasets"]}
    jobs, meta = [], []
    for fold in FOLDS:
        y = np.asarray(panel_in[fold]["target"], np.float64)
        for arm in arms:
            for s in (0, 1, 2):  # out of sample: the cell trained without this dataset
                if arm == "base-c":
                    x = np.asarray(panel_in[fold]["predictions"]["base-c"][s], np.float64)
                else:
                    p = CELLS / arm / f"without_{fold}_s{s}" / f"pred_{fold}.tsv"
                    if not p.is_file():
                        continue
                    x = pd.read_csv(p, sep="\t").pred.to_numpy(np.float64)
                jobs.append((f"{fold}|{arm}|out|{s}", x, y))
                meta.append(dict(dataset=fold, model=arm, sample="out of sample (dataset held out)", seed=s))
            other = next(f for f in FOLDS if f != fold)  # in sample: a seed-0 cell that trained on this dataset
            cell = (E34_CELLS if arm == "base-c" else CELLS / arm) / f"without_{other}_s0"
            bake = cell / "refit" / "last.bin"
            if not bake.is_file():
                continue
            spec = "" if arm == "base-c" else json.loads((cell / "byc_record.json").read_text())["spec"]
            table = ROOTS[arm_root(arm, spec)] / "wide/main/real" / f"{fold}.parquet"
            x = predict(bake, table, spec)
            jobs.append((f"{fold}|{arm}|in|0", x, y))
            meta.append(dict(dataset=fold, model=arm, sample=f"in sample (cell without {other}, seed 0)", seed=0))
    return jobs, meta


def aic_jobs():
    ptr = pd.read_csv(F / "pairs_train.tsv", sep="\t")
    pte = pd.read_csv(F / "pairs_test.tsv", sep="\t")
    keys = ptr.key.astype(str)
    y = -ptr.jnd.to_numpy(np.float64)
    b = pd.read_csv("/mnt/v/output/zensim/jnd-reconstruction-2026-10-19-b/JND_curve.scored.csv").set_index("distorted").score_b
    jobs = [("aic_train|B|zero", b.loc[keys].to_numpy(), y)]
    meta = [dict(dataset="aic2026_public_train", model="B", sample="zero-shot (no AIC fit)", seed=None)]
    full = pd.read_parquet(OUT / "table_raw/table.parquet", columns=["key"]).key.astype(str).tolist()
    for name, path in (("Tfull", "full/mono-thr_s0/refit/production-f16.bin"),
                       ("Mfull", "full/mono-thr-mp32_s0/refit/production-f16.bin")):
        sys.path.insert(0, str(HERE.parent / "rev4_featpot"))
        from v2_lodo_mlp import predict as owner_predict  # noqa: E402
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as td:
            pr = owner_predict(Path("/mnt/v/output/zensim/byconstruction-2026-10-22") / path,
                               OUT / "table_raw/table.parquet", Path(td) / "p.tsv")
        jobs.append((f"aic_train|{name}|zero", pd.Series(pr, index=full).loc[keys].to_numpy(), y))
        meta.append(dict(dataset="aic2026_public_train", model=f"{name} (old data, no AIC)", sample="zero-shot (no AIC fit)", seed=None))
    for arm in ("T", "M", "S"):
        d = OUT / "rust/folds_train" / arm
        if not (d / "oof_panel.json").is_file():
            continue
        s = pd.Series(np.nan, index=keys)
        for k in range(7):
            kk = pd.read_parquet(F / f"aic_k{k}_test.parquet", columns=["key"]).key.astype(str).to_numpy()
            s.loc[kk] = pd.read_csv(d / f"k{k}" / "pred.tsv", sep="\t").pred.to_numpy()
        jobs.append((f"aic_train|{arm}|oof", s.to_numpy(), y))
        meta.append(dict(dataset="aic2026_public_train", model=f"{arm} (AIC only)", sample="out of fold (7 source folds)", seed=None))
    ins = OUT / "rust/folds_train/T_full/pred_insample.tsv"
    if ins.is_file():
        order = pd.read_parquet(F / "aic_train_all.parquet", columns=["key"]).key.astype(str)
        s = pd.Series(pd.read_csv(ins, sep="\t").pred.to_numpy(), index=order).loc[keys]
        jobs.append(("aic_train|T|in", s.to_numpy(), y))
        meta.append(dict(dataset="aic2026_public_train", model="T (AIC only)", sample="in sample (fitted on all of it)", seed=None))
    for arm in sorted(p.name for p in CELLS.iterdir() if (p / "without_kadid_s0" / "byc_record.json").is_file()):
        recs = [json.loads(Path(r).read_text()) for r in glob.glob(str(CELLS / arm / "*/byc_record.json"))]
        if not recs or "aic_fold" not in recs[0] or len(recs) < 12:
            continue
        by = {}
        for r in recs:
            by.setdefault(r["aic_fold"], []).append(
                pd.read_csv(CELLS / arm / f"without_{r['fold']}_s{r['seed']}" / f"pred_aic_k{r['aic_fold']}.tsv",
                            sep="\t").pred.to_numpy())
        s = pd.Series(np.nan, index=keys)
        for k, ps in by.items():
            kk = pd.read_parquet(F / f"aic_k{k}_test.parquet", columns=["key"]).key.astype(str).to_numpy()
            s.loc[kk] = np.mean(ps, 0)
        jobs.append((f"aic_train|{arm}|oof", s.to_numpy(), y))
        meta.append(dict(dataset="aic2026_public_train", model=f"{arm} (E21 + AIC)", sample="out of fold (rotating source fold)", seed=None))
    tkeys = pte.key.astype(str)
    jobs.append(("aic_test|B|zero", b.loc[tkeys].to_numpy(), -pte.jnd.to_numpy(np.float64)))
    meta.append(dict(dataset="aic2026_public_test", model="B", sample="zero-shot, frozen; the one public_test read", seed=None))
    return jobs, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="base-c,Tghpm,Tga,Thpm,Mhpm")
    a = ap.parse_args()
    j1, m1 = e21_jobs(a.arms.split(","))
    j2, m2 = aic_jobs()
    res = panel_batch(j1 + j2, stats="full")
    rows = []
    for m, r, (label, x, _) in zip(m1 + m2, res, j1 + j2):
        rows.append({**m, "n": int(r["n"]), **{k: float(r[k]) for k in STATS}})
    # E21 out of sample: mean over seeds
    df = pd.DataFrame(rows)
    out = []
    for (ds, model, sample), g in df.groupby(["dataset", "model", "sample"], sort=False):
        out.append({"dataset": ds, "model": model, "sample": sample, "n": int(g.n.iloc[0]), "seeds": int(len(g)),
                    **{k: float(g[k].mean()) for k in STATS}})
    Path(a.out).write_text(json.dumps({"stats": list(STATS), "rows": out, "per_seed": rows}, indent=1) + "\n")
    for r in out:
        print(f"{r['dataset']:22s} {r['model']:24s} {r['sample'][:38]:38s} n={r['n']:5d} srocc={r['srocc_signed']:+.4f} "
              f"plcc={r['plcc']:.4f} krocc={r['krocc']:.4f} or={r['or']:.4f} pwrc={r['pwrc']:.4f} zrmse={r['z_rmse']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
