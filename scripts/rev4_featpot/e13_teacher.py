"""Design log E13 (registered 2026-10-03, before any E13 cell runs): which SafeSyn teacher subsets are poor training data?

  python3 e13_teacher.py strata --root ROOT              # -> <root>/e13/safesyn_fit_strata.npz (codec, quality per fit row)
  python3 e13_teacher.py grid   --root ROOT --out SPEC.json --program-sha P --data-sha D
  python3 e13_teacher.py score  --root ROOT              # -> <root>/compare/e13_teacher.json

Arms: v2 + basic at the registered recipe (@h32:H128, head N, seeds 0-4, five design folds) with the SafeSyn fit leg curated by
one rule (v2_common.TEACHER_SUBSETS); control = the same set without a curation token, same seeds. Scored seed-paired against
the control on the held-out sources: signed SROCC (mean and worst source) plus worst-case measures:
  W1  per-reference SROCC, 10th percentile over the held-out source's references;
  W2  KADID-10k / TID2013 per-distortion-type SROCC: minimum and mean of the worst three types;
  W3  the panel's z_rmse and outlier ratio;
  W4  bounding, label-free: on SafeSyn dev (reference-disjoint from every fit) the share of consecutive-quality steps where the
      model's score falls by > 2 as quality rises (teacher-free monotonicity), and the share of held-out predictions below 0.
Adoption rule (registered): mean Δ >= 0, Δ >= -0.003 on every source, and W1 or W2 better by >= 2 SE (seed-paired, mean over
sources/sets) with neither worse by >= 2 SE. Adopted rules are combined in a registered second round.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v2_common import FITBIN, SOURCE_ORDER, TEACHER_SUBSETS, V2
from v2_teacher import STRATA_SCHEMA

SEEDS, HEAD, BASE = range(5), "N", "set:v2+basic@h32:H128"
BANK = Path("/var/tmp/rev4-featbank/bank")
TYPE_SOURCES = {"kadid": ("kadid_train", "kadid_select"), "tid2013": ("tid2013",)}
MEAN_FLOOR, SOURCE_FLOOR = 0.0, -0.003


def spec(rule: str | None) -> str:
    return BASE if rule is None else f"{BASE}:ts{rule}"


def cell(rule: str | None, source: str, seed: int) -> Path:
    return cell_of(spec(rule), source, seed)


def cell_of(spec_: str, source: str, seed: int) -> Path:
    return V2 / "cells" / f"{spec_}__{HEAD}" / f"without_{source}_s{seed}" / "result.json"


def cmd_strata(args) -> int:
    real = V2 / "wide" / "main" / "real"
    receipt = json.loads((real / "receipt.json").read_text())
    keys_path = real / "safesyn_fit.keys.parquet"
    keys_sha = hashlib.sha256(keys_path.read_bytes()).hexdigest()
    if keys_sha != receipt["legs"]["safesyn"]["fit"]["keys_sha256"]:
        raise ValueError("safesyn_fit.keys.parquet differs from the receipt")
    keys = pq.read_table(keys_path, columns=["pair_key"]).to_pandas()
    bank = pq.read_table(BANK / "safesyn" / "keys.parquet", columns=["pair_key", "codec", "knob"]).to_pandas()
    idx = pd.Index(bank.pair_key).get_indexer(keys.pair_key)
    if (idx < 0).any():
        raise ValueError("fit rows missing from the SafeSyn bank keys")
    codec = bank.codec.to_numpy()[idx].astype(str)
    quality = bank.knob.str.removeprefix("q").astype(int).to_numpy()[idx]
    names = np.array(sorted(set(codec)))
    out = V2 / "e13" / "safesyn_fit_strata.npz"
    out.parent.mkdir(exist_ok=True)
    with out.open("wb") as f:
        np.savez_compressed(f, schema=np.array(STRATA_SCHEMA), keys_sha256=np.array(keys_sha), codec_names=names,
                            codec_idx=np.searchsorted(names, codec).astype(np.uint8), quality=quality.astype(np.uint8))
    print(json.dumps({"out": str(out), "rows": len(codec), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                      "bytes": out.stat().st_size, "codecs": names.tolist()}))
    return 0


def cmd_grid(args) -> int:
    cells = []
    for rule in TEACHER_SUBSETS:
        for s in SOURCE_ORDER:
            for i in SEEDS:
                cells.append({"name": f"{spec(rule)}__{HEAD}/without_{s}_s{i}",
                              "argv": ["v2_lodo_mlp.py", "--spec", spec(rule), "--head", HEAD, "--heldout", s,
                                       "--seed-index", str(i), "--root", str(V2)]})
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    missing_ctl = [f"{s}_s{i}" for s in SOURCE_ORDER for i in SEEDS if not cell(None, s, i).is_file()]
    if missing_ctl:
        raise ValueError(f"control cells missing: {missing_ctl}")
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo)}))
    return 0


# ------------------------------------------------------------------ scoring
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def heldout_meta(source: str) -> pd.DataFrame:
    keys = pq.read_table(V2 / "wide" / "main" / "real" / f"{source}.keys.parquet").to_pandas()
    if source in TYPE_SOURCES:
        paths = pd.concat([pq.read_table(BANK / s / "keys.parquet", columns=["pair_key", "dist_path"]).to_pandas()
                           for s in TYPE_SOURCES[source]]).drop_duplicates("pair_key")
        idx = pd.Index(paths.pair_key).get_indexer(keys.pair_key)
        if (idx < 0).any():
            raise ValueError(f"{source}: rows without a distorted path")
        keys["dtype"] = [Path(p).stem.split("_")[1] for p in paths.dist_path.to_numpy()[idx]]
    return keys


def worst_case(result: Path, meta: pd.DataFrame) -> dict:
    r = json.loads(result.read_text())
    pred, y = np.asarray(r["prediction"], dtype=np.float64), meta.target.to_numpy(dtype=np.float64)
    sign = 1.0 if spearman(pred, y) >= 0 else -1.0  # human scales differ in orientation (AIC-3 is negative-oriented)
    per_ref = [sign * spearman(pred[g], y[g]) for g in meta.groupby("ref_basename").indices.values() if len(g) >= 4]
    out = {"signed": float(r["score"]["srocc_signed"]) if "srocc_signed" in r["score"] else None,
           "w1_ref_p10": float(np.nanquantile(per_ref, 0.10)),
           "w3_z_rmse": float(r["score"]["z_rmse"]), "w3_or": float(r["score"]["or"]),
           "w4_neg_share": float((pred < 0).mean())}
    if "dtype" in meta:
        per_type = sorted(sign * spearman(pred[g], y[g]) for g in meta.groupby("dtype").indices.values())
        out["w2_type_min"], out["w2_type_worst3"] = float(per_type[0]), float(np.mean(per_type[:3]))
    return out


def dev_monotonicity(bake: Path, cache: Path) -> float:
    """Share of consecutive-quality steps on SafeSyn dev where the bake's score falls by > 2 as quality rises."""
    out = cache / (hashlib.sha256(bake.read_bytes()).hexdigest()[:16] + ".tsv")
    if not out.is_file():
        real = V2 / "wide" / "main" / "real"
        subprocess.run([str(FITBIN), "predict", "--bake", str(bake), "--corpus", str(real / "safesyn_dev.parquet"),
                        "--score-units", "--out", str(out)], check=True, capture_output=True)
    pred = pd.read_csv(out, sep="\t").pred.to_numpy()
    frame = dev_series_frame()
    frame["p"] = pred
    d = frame.sort_values(["ref", "codec", "q"]).groupby(["ref", "codec"], sort=False).p.diff().dropna()
    return float((d < -2.0).mean())


_DEV = None


def dev_series_frame() -> pd.DataFrame:
    global _DEV
    if _DEV is None:
        real = V2 / "wide" / "main" / "real"
        keys = pq.read_table(real / "safesyn_dev.keys.parquet", columns=["pair_key", "ref_basename"]).to_pandas()
        bank = pq.read_table(BANK / "safesyn" / "keys.parquet", columns=["pair_key", "codec", "knob"]).to_pandas()
        idx = pd.Index(bank.pair_key).get_indexer(keys.pair_key)
        _DEV = pd.DataFrame({"ref": keys.ref_basename.to_numpy(), "codec": bank.codec.to_numpy()[idx],
                             "q": bank.knob.str.removeprefix("q").astype(int).to_numpy()[idx]})
    return _DEV.copy()


def paired(a: list, b: list) -> dict:
    d = np.asarray(a) - np.asarray(b)
    return {"delta": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None, "n": len(d)}


def cmd_score(args) -> int:
    return score_arms([(rule, spec(rule)) for rule in TEACHER_SUBSETS], "e13_teacher", args.monotonicity)


def score_arms(arms: list, out_name: str, monotonicity: bool) -> int:
    """Seed-paired worst-case scoring of `arms` [(label, spec)] against the uncurated control (BASE); E13 and E14 share it."""
    metas = {s: heldout_meta(s) for s in SOURCE_ORDER}
    cache = V2 / "e13" / "devpred"
    cache.mkdir(parents=True, exist_ok=True)
    stats, missing = {}, 0
    for label, sp in [(None, BASE), *arms]:
        for s in SOURCE_ORDER:
            for i in SEEDS:
                p = cell_of(sp, s, i)
                if not p.is_file():
                    missing += 1
                    continue
                st = worst_case(p, metas[s])
                if monotonicity:
                    st["w4_dev_mono"] = dev_monotonicity(Path(json.loads(p.read_text())["selected_bake"]), cache)
                stats[(label, s, i)] = st
    rows = {}
    for label, sp in arms:
        row = {"spec": sp, "per_source": {}}
        for s in SOURCE_ORDER:
            seeds = [i for i in SEEDS if (label, s, i) in stats and (None, s, i) in stats]
            if not seeds:
                continue
            row["per_source"][s] = {m: paired([stats[(label, s, i)][m] for i in seeds], [stats[(None, s, i)][m] for i in seeds])
                                    for m in stats[(label, s, seeds[0])] if stats[(label, s, seeds[0])][m] is not None}
        ps = row["per_source"]
        if len(ps) == len(SOURCE_ORDER):
            for m in ("signed", "w1_ref_p10", "w3_z_rmse", "w4_neg_share", *(("w4_dev_mono",) if monotonicity else ())):
                d = [ps[s][m]["delta"] for s in SOURCE_ORDER]
                row[m] = {"mean": float(np.mean(d)), "worst": float(min(d)) if m in ("signed", "w1_ref_p10") else float(max(d)),
                          "se": math.sqrt(sum(ps[s][m]["se"] ** 2 for s in SOURCE_ORDER)) / len(SOURCE_ORDER)}
            w2 = [ps[s]["w2_type_worst3"] for s in TYPE_SOURCES]
            row["w2_type_worst3"] = {"mean": float(np.mean([x["delta"] for x in w2])),
                                     "se": math.sqrt(sum(x["se"] ** 2 for x in w2)) / len(w2)}
            sig, w1, w2r = row["signed"], row["w1_ref_p10"], row["w2_type_worst3"]
            better = any(x["mean"] >= 2 * x["se"] for x in (w1, w2r))
            worse = any(x["mean"] <= -2 * x["se"] for x in (w1, w2r))
            row["adopt"] = bool(sig["mean"] >= MEAN_FLOOR and sig["worst"] >= SOURCE_FLOOR and better and not worse)
        rows[label] = row
    out = {"schema": f"rev4-featpot-{out_name.replace('_', '-')}-v1", "control": BASE, "seeds": list(SEEDS),
           "rule": {"mean_floor": MEAN_FLOOR, "source_floor": SOURCE_FLOOR, "worst_case": "W1 or W2 >= +2 SE, neither <= -2 SE"},
           "rows": rows, "control_stats": {f"{s}_s{i}": stats[(None, s, i)] for s in SOURCE_ORDER for i in SEEDS
                                            if (None, s, i) in stats},
           "missing_cells": missing, "status": "INCOMPLETE" if missing else "complete"}
    (V2 / "compare").mkdir(exist_ok=True)
    (V2 / "compare" / f"{out_name}.json").write_text(json.dumps(out, indent=1) + "\n")
    for label, r in rows.items():
        if "signed" not in r:
            print(f"  {label:9s} incomplete")
            continue
        print(f"  {label:9s} signed {r['signed']['mean']:+.4f}±{r['signed']['se']:.4f} (worst {r['signed']['worst']:+.4f})  "
              f"W1 {r['w1_ref_p10']['mean']:+.4f}±{r['w1_ref_p10']['se']:.4f}  W2 {r['w2_type_worst3']['mean']:+.4f}"
              f"±{r['w2_type_worst3']['se']:.4f}  neg {r['w4_neg_share']['mean']:+.4f}" + ("  ADOPT" if r["adopt"] else ""))
    print(json.dumps({"status": out["status"], "missing_cells": missing}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["strata", "grid", "score"])
    ap.add_argument("--root")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    ap.add_argument("--monotonicity", action="store_true", help="also W4 dev monotonicity (predicts SafeSyn dev per bake)")
    args = ap.parse_args()
    return {"strata": cmd_strata, "grid": cmd_grid, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
