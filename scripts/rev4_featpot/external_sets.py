"""External held-out evaluation sets for the featpot instrument (registered 2026-10-03; DATA_SPLITS §3e).

  python3 external_sets.py extract --root ROOT --set nits|live|mciqa   # Rev4 canon f0-f1824 + decoded-pixel audit
  python3 external_sets.py table   --root ROOT --set ...               # -> <root>/external/<set>.parquet (+ keys, manifests)
  python3 external_sets.py score   --root ROOT --specs SPEC[,SPEC...] [--seeds 0-4] [--out NAME]

Sets are OPEN external reads (never training inputs; T0 eval-only like LIVE): NITS-IQA (405 pairs, 9 distortion types incl.
contrast change and pixelate), LIVE release 2 (779 pairs, 5 types) and MCIQA-2K (2,000 colorized vs their COCO originals;
no-reference by design, so an exploratory read of global naturalness / colour smearing / semantic misalignment). Pairs come
from `scripts/canonical_corpus/build_fr_corpus_pairs.py` (the owner); features from the r4 bank's extractor and arguments
(as E14/E15); tables follow the v2c wide layout (f0-f1824 real, f1825+ NaN — bakes reading only f0-f1824 are served).
Every cell of a spec is scored on every external set (each LODO bake is held out from these sets by construction), and arms
are seed-paired against the control spec.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

import e14_kadis_ordinal as e14
from v2_common import FITBIN, SOURCE_ORDER, V2, dense_bake

PAIRS = {
    "nits": Path("/mnt/v/datasets/nits-iqa_extracted/nits_iqa_pairs.tsv"),
    "live": Path("/mnt/v/datasets/LIVE/live_r2_pairs.tsv"),
    "mciqa": Path("/mnt/v/datasets/mciqa-2k_extracted/mciqa_2k_pairs.tsv"),
}
# Content-overlap audit (2026-10-03, benchmarks/external_sets_2026-10-03.md §2): these LIVE references are Kodak scenes that
# contain a TID2013 reference (multi-scale template match NCC 0.90-0.995, runner-up <= 0.82; the other 10 references <= 0.64),
# and 17 of them also contain a SafeSyn `<kodak>_512sq.png` source. TID2013 trains 4 of 5 LODO folds and SafeSyn every fold,
# so those references are not unseen content. The dHash owner missed them (crop + rescale). Every LIVE read is reported
# overall AND on the 10 content-disjoint references (`disjoint:` metrics); the subset is fixed by the audit, not by any score.
CONTENT_OVERLAP = {"live": frozenset({
    "bikes.bmp", "buildings.bmp", "caps.bmp", "house.bmp", "lighthouse.bmp", "lighthouse2.bmp", "ocean.bmp", "paintedhouse.bmp",
    "parrots.bmp", "plane.bmp", "rapids.bmp", "sailing1.bmp", "sailing2.bmp", "sailing3.bmp", "sailing4.bmp", "statue.bmp",
    "stream.bmp", "woman.bmp", "womanhat.bmp"})}
# Per-set grouping columns for per-type reads.
GROUPS = {"nits": ["distortion"], "live": ["distortion"], "mciqa": ["model"]}
NITS_TYPES = {"D1": "gaussian blur", "D2": "chromatic gaussian noise", "D3": "chromatic uniform noise", "D4": "contrast change",
              "D5": "pixelate mosaic", "D6": "motion blur", "D7": "jpeg", "D8": "jpeg2000", "D9": "jpeg-xt"}


def work(name: str) -> Path:
    return V2 / "external" / name


def load_pairs(name: str) -> pd.DataFrame:
    df = pd.read_csv(PAIRS[name], sep="\t")
    if name == "live":  # distortion = the LIVE folder of the distorted image
        df["distortion"] = [Path(p).parent.name for p in df.dist_path]
    return df


def cmd_extract(args) -> int:
    if e14.sha256(e14.BIN) != e14.BIN_SHA:
        raise ValueError("not the r4 bank's extractor")
    df = load_pairs(args.set)
    d = work(args.set)
    d.mkdir(parents=True, exist_ok=True)
    tsv, csv = d / "pairs.tsv", d / "features.csv"
    with tsv.open("w") as f:
        f.write("ref_path\tdist_path\thuman_score\trow_id\n")
        for i, (r, p) in enumerate(zip(df.ref_path, df.dist_path)):
            f.write(f"{r}\t{p}\t0\t{i}\n")
    t0 = time.time()
    with (d / "extract.log").open("w") as log:
        rc = subprocess.run([str(e14.BIN), "--corpus", "pairs-tsv", "--path", str(tsv), "--out", str(csv), *e14.EXTRACT_ARGS,
                             "--audit-jsonl", str(d / "audit.jsonl")], env={**os.environ, **e14.EXTRACT_ENV},
                            stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"extractor rc={rc}; see {d / 'extract.log'}")
    man = json.loads(Path(f"{csv}.manifest.json").read_text())
    if man["era_label"] != e14.ERA or man["formula_revision"] != "4":
        raise ValueError("extractor manifest: wrong era or revision")
    print(json.dumps({"set": args.set, "rows": len(df), "wall_s": round(time.time() - t0, 1), "feature_set_id": man["feature_set_id"]}))
    return 0


def cmd_table(args) -> int:
    df = load_pairs(args.set)
    d = work(args.set)
    W = e14.WIDTH
    names = ["row_id"] + [f"f{i}" for i in range(W)]
    conv = pacsv.ConvertOptions(column_types={"row_id": pa.int64(), **{f"f{i}": pa.float64() for i in range(W)}}, include_columns=names)
    tab = pacsv.read_csv(d / "features.csv", convert_options=conv)
    rid = tab.column("row_id").to_numpy()
    order = np.argsort(rid, kind="stable")
    if not (rid[order] == np.arange(len(df))).all():
        raise ValueError("row_id join failed")
    feats = np.stack([tab.column(f"f{i}").to_numpy()[order] for i in range(W)], axis=1)
    if not np.isfinite(feats).all():
        raise ValueError("nonfinite features")
    audit = [json.loads(ln) for ln in (d / "audit.jsonl").read_text().splitlines() if ln.strip()]
    if len(audit) != len(df) or any(a["distorted"] != p for a, p in zip(audit, df.dist_path)):
        raise ValueError("audit rows do not match the pairs in order")
    keys = df.copy()
    keys["ref_pixels_sha256"] = [a["reference_pixels_sha256"] for a in audit]
    keys["dist_pixels_sha256"] = [a["distorted_pixels_sha256"] for a in audit]
    keys["pixels_identical"] = [bool(a["pixels_identical"]) for a in audit]
    keys["pair_key"] = [hashlib.sha256((r + p + "legacy-rgb8").encode()).hexdigest()
                        for r, p in zip(keys.ref_pixels_sha256, keys.dist_pixels_sha256)]
    if keys.pixels_identical.any():
        raise ValueError(f"{int(keys.pixels_identical.sum())} pixel-identical pairs in an evaluation set")
    wide = json.loads((V2 / "wide" / "main" / "real" / "receipt.json").read_text())["width"]
    cols = {"ref_basename": pa.array([Path(p).name for p in df.ref_path]), "human_score": pa.array(df.human_score.to_numpy(np.float64))}
    cols.update({f"f{i}": pa.array(feats[:, i].astype(np.float32)) for i in range(W)})
    nan = pa.array(np.full(len(df), np.nan, np.float32))
    cols.update({f"f{i}": nan for i in range(W, wide)})
    out = d.parent / f"{args.set}.parquet"
    pq.write_table(pa.table(cols), out, compression="zstd", use_byte_stream_split=[f"f{i}" for i in range(wide)])
    Path(f"{out}.manifest.json").write_text(json.dumps({
        "source_bank_feature_set_id": "basic+peaks+masked+iw+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype"
                                      "+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/tiercanon_c3negfold#d57e9571",
        "composite": f"Rev4 POTENTIAL external held-out set {args.set} (f0-f1824 Rev4 bank arithmetic; f1825-f{wide - 1} NaN); "
                     "evaluation only", "formula_revision": 4}, indent=1) + "\n")
    keys.to_parquet(d.parent / f"{args.set}.keys.parquet")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canonical_corpus"))
    import check_target_orientation as cto
    gate = cto.check(str(out), corpus=args.set) if args.set in ("nits", "mciqa") else cto.check(str(out), corpus="live")
    man = {"schema": "rev4-featpot-external-v1", "label": "POTENTIAL — external held-out evaluation, never a training input",
           "set": args.set, "rows": len(df), "pairs_source": str(PAIRS[args.set]), "pairs_sha256": e14.sha256(PAIRS[args.set]),
           "extractor_sha256": e14.BIN_SHA, "extract_args": e14.EXTRACT_ARGS, "era": e14.ERA, "formula_revision": 4,
           "target_orientation": gate, "sha256": e14.sha256(out), "keys_sha256": e14.sha256(d.parent / f"{args.set}.keys.parquet")}
    (d.parent / f"{args.set}.manifest.json").write_text(json.dumps(man, indent=1, default=str) + "\n")
    print(json.dumps({"set": args.set, "rows": len(df), "orientation": gate.get("verdict"), "signed_srocc": gate.get("signed_srocc")}))
    return 0 if gate.get("verdict") in ("OK", "SKIPPED") else 1


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() and ry.std() else float("nan")


UNREAD_FROM = 1825  # external tables carry f1825+ as NaN (texgain/satsign not extracted, as the E14/E15 legs)


def reads_unextracted(cell: Path) -> bool:
    return max(int(x) for x in (cell / "keep_features.txt").read_text().split()) >= UNREAD_FROM


def predict(cell: Path, table: Path, cache: Path) -> np.ndarray:
    if reads_unextracted(cell):
        raise ValueError(f"{cell} reads f{UNREAD_FROM}+; external tables carry those columns as NaN")
    bake = Path(json.loads((cell / "result.json").read_text())["selected_bake"])
    dense = dense_bake(bake, cache)
    out = cache / f"{dense.stem}_{table.stem}.tsv"
    if not out.is_file():
        tmp = out.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        subprocess.run([str(FITBIN), "predict", "--bake", str(dense), "--corpus", str(table), "--score-units", "--out", str(tmp)],
                       check=True, capture_output=True)
        tmp.rename(out)
    p = pd.read_csv(out, sep="\t").pred.to_numpy(np.float64)
    if not np.isfinite(p).all():
        raise ValueError(f"non-finite predictions from {dense} on {table}")
    return p


def cmd_score(args) -> int:
    specs = args.specs.split(",")
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = range(lo, hi + 1)
    sets = [s for s in PAIRS if (V2 / "external" / f"{s}.parquet").is_file()]
    cache = V2 / "external" / "pred"
    cache.mkdir(parents=True, exist_ok=True)
    refused = [sp for sp in specs if any(reads_unextracted(c.parent) for c in (V2 / "cells" / f"{sp}__N").glob("without_*/result.json"))]
    if specs[0] in refused:
        raise ValueError(f"control {specs[0]} reads f{UNREAD_FROM}+, which the external tables do not carry")
    specs = [sp for sp in specs if sp not in refused]  # reported, never silently dropped
    jobs = []  # warm the prediction cache in parallel; the loop below then reads it
    for sp in specs:
        for fold in SOURCE_ORDER:
            for i in seeds:
                cell = V2 / "cells" / f"{sp}__N" / f"without_{fold}_s{i}"
                if (cell / "result.json").is_file():
                    jobs += [(cell, V2 / "external" / f"{st}.parquet") for st in sets]
    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(lambda j: predict(j[0], j[1], cache), jobs))
    res = {}  # (spec, set) -> list over (fold, seed) of {"all": r, groups...}
    for sp in specs:
        for st in sets:
            table = V2 / "external" / f"{st}.parquet"
            keys = pd.read_parquet(V2 / "external" / f"{st}.keys.parquet")
            y = keys.human_score.to_numpy(np.float64)
            gcol = GROUPS[st][0]
            rows = []
            for fold in SOURCE_ORDER:
                for i in seeds:
                    cell = V2 / "cells" / f"{sp}__N" / f"without_{fold}_s{i}"
                    if not (cell / "result.json").is_file():
                        continue
                    p = predict(cell, table, cache)
                    r = {"fold": fold, "seed": i, "all": spearman(p, y)}
                    for g, idx in keys.groupby(gcol).indices.items():
                        r[f"{gcol}:{g}"] = spearman(p[idx], y[idx])
                    if st in CONTENT_OVERLAP:
                        dis = ~keys.ref_path.map(lambda q: Path(q).name).isin(CONTENT_OVERLAP[st]).to_numpy()
                        r["disjoint:all"] = spearman(p[dis], y[dis])
                        for g, idx in keys[dis].groupby(gcol).indices.items():
                            sel = np.flatnonzero(dis)[idx]
                            r[f"disjoint:{gcol}:{g}"] = spearman(p[sel], y[sel])
                    if st == "mciqa":
                        for dim in ("cs_z", "scm_z", "gn_z"):
                            r[f"dim:{dim}"] = spearman(p, keys[dim].to_numpy(np.float64))
                    rows.append(r)
            res[(sp, st)] = rows
    ctl = specs[0]
    report = {"schema": "rev4-featpot-external-score-v1", "control": ctl, "seeds": list(seeds), "sets": sets, "specs": {},
              "refused_reads_unextracted_columns": refused}
    for sp in specs:
        report["specs"][sp] = {}
        for st in sets:
            rows = {(r["fold"], r["seed"]): r for r in res[(sp, st)]}
            crow = {(r["fold"], r["seed"]): r for r in res[(ctl, st)]}
            common = sorted(set(rows) & set(crow))
            metrics = sorted({k for r in rows.values() for k in r if k not in ("fold", "seed")})
            out = {}
            for m in metrics:
                a = np.array([rows[k][m] for k in common])
                b = np.array([crow[k][m] for k in common])
                out[m] = {"mean": float(np.nanmean(a)), "delta": float(np.nanmean(a - b)) if sp != ctl else 0.0,
                          "se": float(np.nanstd(a - b, ddof=1) / math.sqrt(len(common))) if sp != ctl and len(common) > 1 else None,
                          "n": len(common)}
            report["specs"][sp][st] = out
    name = args.out or "external_score"
    (V2 / "compare").mkdir(exist_ok=True)
    (V2 / "compare" / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
    if refused:
        print(f"REFUSED (read f{UNREAD_FROM}+, not extracted for external sets): {', '.join(refused)}")
    for st in sets:
        print(f"== {st}")
        for sp in specs:
            r = report["specs"][sp][st]
            head = f"  {sp:44s} SROCC {r['all']['mean']:.4f}" + ("" if sp == ctl else f" Δ {r['all']['delta']:+.4f}±{r['all']['se']:.4f}")
            worst = sorted(((k, v) for k, v in r.items() if k.startswith(GROUPS[st][0] + ":")), key=lambda kv: kv[1]["mean"])[:3]
            print(head + "  worst: " + "  ".join(f"{k.split(':')[1]} {v['mean']:.3f}" + ("" if sp == ctl else f"{v['delta']:+.3f}") for k, v in worst))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract", "table", "score"])
    ap.add_argument("--root")
    ap.add_argument("--set", choices=sorted(PAIRS))
    ap.add_argument("--specs")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    return {"extract": cmd_extract, "table": cmd_table, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
