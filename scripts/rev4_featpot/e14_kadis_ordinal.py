"""Design log E14 (registered 2026-10-03 04:55 MT, before any E14 data or cell): a teacher-free ordinal coverage leg built from
KADIS-700k distortion ladders.

  python3 e14_kadis_ordinal.py select  --root ROOT        # subset + fetch persisted distortions (R2) -> <root>/e14/
  python3 e14_kadis_ordinal.py extract --root ROOT        # Rev4 canon f0-f1824 with the r4 bank's extractor and arguments
  python3 e14_kadis_ordinal.py table   --root ROOT        # -> <root>/e14/kadis_ordinal.parquet (+ manifest)
  python3 e14_kadis_ordinal.py grid    --root ROOT --out SPEC.json --program-sha P --data-sha D
  python3 e14_kadis_ordinal.py score   --root ROOT        # -> <root>/compare/e14_kadis_ordinal.json

Subset: KADIS train split (source_id % 10 < 8), 100 references per assigned type (lowest sha256(source_filename)), all 5
levels. Ladder = (reference, type, sign of dist_param); target = -|dist_param|, ranked only within a ladder (group mode
`withinref,rank`), so signed types 7/18/25 never compare across directions. Pixel-identical pairs are dropped.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from v2_common import SOURCE_ORDER, V2

KADIS = Path("/mnt/v/datasets/kadis700k")
CANON = KADIS / "canonical" / "kadis700k_canonical_gpu_2026-07-01.parquet"
CANON_SHA = "c9a6fd56f8f5a73106438c325f77a0cc78dffb25bcfeefe91d8a06c1cdd5e779"
REFS = KADIS / "refs"
PER_TYPE = 100
# The r4 bank's extractor and arguments (/var/tmp/reextract/assemble_set.py; zensim 259045b0 + era-label lane patch).
BIN = Path("/var/tmp/reextract/target/release/examples/extract_features_372col")
BIN_SHA = "8c6f4c03695660fbb4fdce46e05859b40d9080497622cd726a7db75fd8323aa8"
ERA, WIDTH = "tiercanon_c3negfold", 1825
EXTRACT_ARGS = ["--restore-cuts", "prefix,mapdev,z1max,gmsnative,dvifmgate", "--input-contract", "legacy-rgb8",
                "--era-label", ERA, "--force-tier", "native"]
EXTRACT_ENV = {"ZENSIM_FORMULA_REV": "4", "ZENSIM_ROOT_FORM": "sqrt", "RAYON_NUM_THREADS": "8"}
WEIGHTS = (1, 4, 16)
SEEDS, HEAD, BASE = range(5), "N", "set:v2+basic@h32:H128"


def work() -> Path:
    return V2 / "e14"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def cmd_select(args) -> int:
    if sha256(CANON) != CANON_SHA:
        raise ValueError(f"{CANON}: sha256 differs from the registered canonical")
    t = pq.read_table(CANON, columns=["source_id", "source_filename", "dist_type", "dist_name", "severity_level",
                                      "dist_param", "distorted_url"]).to_pandas()
    t = t[t.source_id % 10 < 8]
    refs = t.drop_duplicates("source_filename")[["source_filename", "dist_type"]].copy()
    refs["h"] = [hashlib.sha256(s.encode()).hexdigest() for s in refs.source_filename]
    pick = refs.sort_values(["dist_type", "h"]).groupby("dist_type").head(PER_TYPE).source_filename
    sel = t[t.source_filename.isin(set(pick))].sort_values(["dist_type", "source_filename", "severity_level"]).reset_index(drop=True)
    if len(sel) != 24 * PER_TYPE * 5:
        raise ValueError(f"selected {len(sel)} rows, expected {24 * PER_TYPE * 5}")
    dest = work() / "dist"
    dest.mkdir(parents=True, exist_ok=True)
    sel["dist_path"] = [str(dest / u.rsplit("/", 1)[1]) for u in sel.distorted_url]
    sel["ref_path"] = [str(REFS / s) for s in sel.source_filename]
    missing_refs = [p for p in sel.ref_path.unique() if not Path(p).is_file()]
    if missing_refs:
        raise ValueError(f"{len(missing_refs)} references missing under {REFS}")
    todo = [u for u, p in zip(sel.distorted_url, sel.dist_path) if not Path(p).is_file()]
    if todo:
        listing = work() / "fetch.txt"
        listing.write_text("".join(f"cp {u} {dest}/\n" for u in todo))
        env = {"HOME": os.environ["HOME"], "PATH": os.environ["PATH"], "ZEN_STORE": "r2"}
        shell = f". {Path.home()}/.config/zen/s3env.sh >/dev/null 2>&1; s5cmd --endpoint-url \"$EP\" run {listing}"
        subprocess.run(["bash", "-c", shell], env=env, check=True, capture_output=True)
    still = [p for p in sel.dist_path if not Path(p).is_file()]
    if still:
        raise ValueError(f"{len(still)} distortions failed to fetch")
    sel.drop(columns=["distorted_url"]).to_parquet(work() / "selection.parquet")
    print(json.dumps({"rows": len(sel), "references": int(sel.source_filename.nunique()), "fetched": len(todo)}))
    return 0


def cmd_extract(args) -> int:
    if sha256(BIN) != BIN_SHA:
        raise ValueError(f"{BIN}: not the r4 bank's extractor")
    sel = pd.read_parquet(work() / "selection.parquet")
    run = work() / "extract"
    run.mkdir(exist_ok=True)
    tsv, csv = run / "pairs.tsv", run / "features.csv"
    with tsv.open("w") as f:
        f.write("ref_path\tdist_path\thuman_score\trow_id\n")
        for i, (r, d) in enumerate(zip(sel.ref_path, sel.dist_path)):
            f.write(f"{r}\t{d}\t0\t{i}\n")
    t0 = time.time()
    with (run / "extract.log").open("w") as log:
        rc = subprocess.run([str(BIN), "--corpus", "pairs-tsv", "--path", str(tsv), "--out", str(csv), *EXTRACT_ARGS],
                            env={**os.environ, **EXTRACT_ENV}, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"extractor rc={rc}; see {run / 'extract.log'}")
    man = json.loads(Path(f"{csv}.manifest.json").read_text())
    if man["era_label"] != ERA or man["formula_revision"] != "4":
        raise ValueError("extractor manifest: wrong era or revision")
    print(json.dumps({"rows": len(sel), "extract_wall_s": round(time.time() - t0, 1), "feature_set_id": man["feature_set_id"]}))
    return 0


def ladders(sel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(ladder key, ordinal target) per row: within (reference, type, sign), target = -|dist_param|."""
    sign = np.sign(sel.dist_param.to_numpy())
    key = [f"kadis:{s}|t{t}|{'+' if g > 0 else '-' if g < 0 else '0'}"
           for s, t, g in zip(sel.source_filename, sel.dist_type, sign)]
    return np.array(key), -np.abs(sel.dist_param.to_numpy(dtype=np.float64))


def pixels_identical(ref: str, dist: str) -> bool:
    from PIL import Image
    a, b = Image.open(ref).convert("RGB"), Image.open(dist).convert("RGB")
    return a.size == b.size and a.tobytes() == b.tobytes()


def cmd_table(args) -> int:
    sel = pd.read_parquet(work() / "selection.parquet")
    run = work() / "extract"
    names = ["row_id"] + [f"f{i}" for i in range(WIDTH)]
    conv = pacsv.ConvertOptions(column_types={"row_id": pa.int64(), **{f"f{i}": pa.float64() for i in range(WIDTH)}},
                                include_columns=names)
    tab = pacsv.read_csv(run / "features.csv", convert_options=conv)
    rid = tab.column("row_id").to_numpy()
    order = np.argsort(rid, kind="stable")
    if not (rid[order] == np.arange(len(sel))).all():
        raise ValueError("row_id join failed")
    feats = np.stack([tab.column(f"f{i}").to_numpy()[order] for i in range(WIDTH)], axis=1)
    if not np.isfinite(feats).all():
        raise ValueError("nonfinite features")
    ident = np.array([pixels_identical(r, d) for r, d in zip(sel.ref_path, sel.dist_path)])
    key, target = ladders(sel)
    keep = ~ident
    # A ladder needs two rungs to make a within-ladder pair; single-rung ladders (a signed type's lone direction rung after an
    # identical level is dropped) carry no rank information and are dropped too.
    counts = pd.Series(key[keep]).value_counts()
    keep &= np.isin(key, counts.index[counts >= 2])
    cols = {"ref_basename": pa.array(key[keep]), "human_score": pa.array(target[keep])}
    cols.update({f"f{i}": pa.array(feats[keep, i].astype(np.float32)) for i in range(WIDTH)})
    # The trainer needs every group at the wide tables' width (1853); f1825.. (the texgain/satsign sidecars) were not
    # extracted for KADIS, so they are NaN and v2_lodo_mlp refuses any keep list that reads them with this leg.
    wide = json.loads((V2 / "wide" / "main" / "real" / "receipt.json").read_text())["width"]
    nan = pa.array(np.full(int(keep.sum()), np.nan, np.float32))
    cols.update({f"f{i}": nan for i in range(WIDTH, wide)})
    out = work() / "kadis_ordinal.parquet"
    pq.write_table(pa.table(cols), out, compression="zstd", use_byte_stream_split=[f"f{i}" for i in range(wide)])
    Path(f"{out}.manifest.json").write_text(json.dumps({
        "source_bank_feature_set_id": "basic+peaks+masked+iw+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype"
                                      "+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/tiercanon_c3negfold#d57e9571",
        "composite": f"Rev4 POTENTIAL E14 KADIS ordinal ladders (f0-f1824 Rev4 bank arithmetic; f1825-f{wide - 1} NaN, not "
                     "extracted); diagnostic only", "formula_revision": 4}, indent=1) + "\n")
    keys = sel.loc[keep, ["source_id", "source_filename", "dist_type", "dist_name", "severity_level", "dist_param"]].copy()
    keys.insert(0, "ladder", key[keep])
    keys.to_parquet(work() / "kadis_ordinal.keys.parquet")
    man = {"schema": "rev4-featpot-e14-kadis-ordinal-v1", "label": "POTENTIAL — ceiling, not a model score",
           "rows": int(keep.sum()), "ladders": int(len(set(key[keep]))), "dropped_identical": int(ident.sum()),
           "nan_columns": f"f{WIDTH}-f{wide - 1}",
           "dropped_single_rung": int((~ident).sum() - keep.sum()), "width": WIDTH, "era": ERA, "formula_revision": 4,
           "extractor_sha256": BIN_SHA, "extract_args": EXTRACT_ARGS, "extract_env": EXTRACT_ENV,
           "canonical_sha256": CANON_SHA, "split": "source_id % 10 < 8", "per_type": PER_TYPE,
           "target": "-|dist_param|, ranked within (reference, type, sign) only",
           "sha256": sha256(out), "keys_sha256": sha256(work() / "kadis_ordinal.keys.parquet")}
    (work() / "kadis_ordinal.manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("rows", "ladders", "dropped_identical", "dropped_single_rung", "sha256")}))
    return 0


def spec(w: int | None) -> str:
    return BASE if w is None else f"{BASE}:ko{w}"


def cmd_grid(args) -> int:
    cells = [{"name": f"{spec(w)}__{HEAD}/without_{s}_s{i}",
              "argv": ["v2_lodo_mlp.py", "--spec", spec(w), "--head", HEAD, "--heldout", s, "--seed-index", str(i),
                       "--root", str(V2)]}
             for w in WEIGHTS for s in SOURCE_ORDER for i in SEEDS]
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo)}))
    return 0


def cmd_score(args) -> int:
    import e13_teacher as e13
    return e13.score_arms([(f"ko{w}", spec(w)) for w in WEIGHTS], "e14_kadis_ordinal", args.monotonicity)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "extract", "table", "grid", "score"])
    ap.add_argument("--root")
    ap.add_argument("--out")
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    ap.add_argument("--monotonicity", action="store_true")
    args = ap.parse_args()
    return {"select": cmd_select, "extract": cmd_extract, "table": cmd_table, "grid": cmd_grid, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
