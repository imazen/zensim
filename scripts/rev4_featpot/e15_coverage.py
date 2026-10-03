"""Design log E15 (registered 2026-10-03 05:00 MT, before any E15 data or cell): which training coverage helps?

  python3 e15_coverage.py select   --root ROOT      # KADIS 20 types x 400 train refs (+ fetch) and 2 x 400 refs for new types
  python3 e15_coverage.py generate --root ROOT      # our numpy TID-style types: lbw (local block-wise), cab (chromatic aberration)
  python3 e15_coverage.py extract  --root ROOT      # Rev4 canon f0-f1824, the r4 bank extractor (as E14)
  python3 e15_coverage.py table    --root ROOT      # -> <root>/e15/coverage_pool.parquet (+ keys, manifest)
  python3 e15_coverage.py grid     --root ROOT --weight W --out SPEC.json --program-sha P --data-sha D
  python3 e15_coverage.py score    --root ROOT --weight W

Every row is a rung of a teacher-free ordinal ladder (reference x type x direction); target = -severity, ranked only within its
ladder (`withinref,rank`). A cell selects families by a mask (`cf<hex>`) and weights the leg by `cv<w>`.
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
import pyarrow.parquet as pq

import e14_kadis_ordinal as e14
from v2_common import COVERAGE_FAMILIES, SOURCE_ORDER, V2

PER_TYPE = 400
NEW_TYPES = ("lbw", "cab")
LBW_BLOCKS = (2, 4, 8, 16, 32)   # nested: level L applies the first LBW_BLOCKS[L-1] blocks
CAB_SHIFT = (1, 2, 3, 5, 8)      # px: red right, blue left
SEEDS, HEAD, BASE = range(3), "N", "set:v2+basic@h32:H128"


def work() -> Path:
    return V2 / "e15"


def family_of(t) -> str:
    for name, types in COVERAGE_FAMILIES.items():
        if t in types:
            return name
    raise ValueError(f"type {t!r} has no family")


def cmd_select(args) -> int:
    if e14.sha256(e14.CANON) != e14.CANON_SHA:
        raise ValueError("KADIS canonical sha256 differs from the registered one")
    t = pq.read_table(e14.CANON, columns=["source_id", "source_filename", "dist_type", "dist_name", "severity_level",
                                          "dist_param", "distorted_url"]).to_pandas()
    t = t[(t.source_id % 10 < 8)]
    refs = t.drop_duplicates("source_filename")[["source_filename", "dist_type"]].copy()
    refs["h"] = [hashlib.sha256(s.encode()).hexdigest() for s in refs.source_filename]
    kad = refs[~refs.dist_type.isin(e14.EXCLUDED_TYPES)]
    pick = set(kad.sort_values(["dist_type", "h"]).groupby("dist_type").head(PER_TYPE).source_filename)
    sel = t[t.source_filename.isin(pick)].sort_values(["dist_type", "source_filename", "severity_level"]).reset_index(drop=True)
    n_types = 24 - len(e14.EXCLUDED_TYPES)
    if len(sel) != n_types * PER_TYPE * 5:
        raise ValueError(f"selected {len(sel)} KADIS rows, expected {n_types * PER_TYPE * 5}")
    # New types: further train references (any assigned KADIS type; only the pristine image is used), lowest sha first.
    rest = refs[~refs.source_filename.isin(pick)].sort_values("h").source_filename.tolist()
    new_refs = {nt: rest[i * PER_TYPE:(i + 1) * PER_TYPE] for i, nt in enumerate(NEW_TYPES)}
    dest = work() / "dist"
    dest.mkdir(parents=True, exist_ok=True)
    sel["dist_path"] = [str(e14.work() / "dist" / u.rsplit("/", 1)[1]) for u in sel.distorted_url]
    local = [Path(p).is_file() for p in sel.dist_path]
    sel.loc[[not x for x in local], "dist_path"] = [str(dest / u.rsplit("/", 1)[1])
                                                    for u, x in zip(sel.distorted_url, local) if not x]
    sel["ref_path"] = [str(e14.REFS / s) for s in sel.source_filename]
    todo = [u for u, p in zip(sel.distorted_url, sel.dist_path) if not Path(p).is_file()]
    if todo:
        listing = work() / "fetch.txt"
        listing.write_text("".join(f"cp {u} {dest}/\n" for u in todo))
        env = {"HOME": os.environ["HOME"], "PATH": os.environ["PATH"], "ZEN_STORE": "r2"}
        shell = f". {Path.home()}/.config/zen/s3env.sh >/dev/null 2>&1; s5cmd --endpoint-url \"$EP\" run {listing}"
        subprocess.run(["bash", "-c", shell], env=env, check=True, capture_output=True)
    if not all(Path(p).is_file() for p in sel.dist_path):
        raise ValueError("some KADIS distortions failed to fetch")
    sel["type"] = sel.dist_type.astype(str)
    sel["severity"] = np.abs(sel.dist_param.to_numpy(dtype=np.float64))
    sel["sign"] = np.sign(sel.dist_param.to_numpy())
    rows = [sel[["source_filename", "type", "severity_level", "severity", "sign", "ref_path", "dist_path"]]]
    for nt, names in new_refs.items():
        for s in names:
            for lv in range(1, 6):
                rows.append(pd.DataFrame({"source_filename": [s], "type": [nt], "severity_level": [lv], "severity": [float(lv)],
                                          "sign": [1.0], "ref_path": [str(e14.REFS / s)],
                                          "dist_path": [str(dest / f"{Path(s).stem}__{nt}_L{lv}.png")]}))
    out = pd.concat(rows, ignore_index=True)
    out["family"] = [family_of(int(t) if t.isdigit() else t) for t in out.type]
    out.to_parquet(work() / "selection.parquet")
    print(json.dumps({"rows": len(out), "kadis_rows": len(sel), "fetched": len(todo),
                      "families": out.family.value_counts().to_dict()}))
    return 0


def _seed(name: str, nt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"e15|{nt}|{name}".encode()).digest()[:8], "little")


def local_block_wise(img: np.ndarray, level: int, seed: int) -> np.ndarray:
    """TID-15-style: uniform blocks at a shifted intensity; level L applies the first LBW_BLOCKS[L-1] of 32 fixed blocks."""
    rng = np.random.default_rng(seed)
    h, w, _ = img.shape
    out = img.astype(np.float64)
    for k in range(LBW_BLOCKS[-1]):
        bs = int(rng.choice([16, 24, 32]))
        y, x = int(rng.integers(0, h - bs)), int(rng.integers(0, w - bs))
        delta = float(rng.choice([-1.0, 1.0]) * rng.uniform(32, 96))
        if k < LBW_BLOCKS[level - 1]:
            mean = img[y:y + bs, x:x + bs].reshape(-1, 3).mean(axis=0)
            out[y:y + bs, x:x + bs] = np.clip(mean + delta, 0, 255)
    return np.round(out).astype(np.uint8)


def chromatic_aberration(img: np.ndarray, level: int) -> np.ndarray:
    """Lateral chromatic aberration: red shifted right and blue left by CAB_SHIFT[L-1] px (edge-replicated)."""
    d = CAB_SHIFT[level - 1]
    out = img.copy()
    r = np.pad(img[:, :, 0], ((0, 0), (d, 0)), mode="edge")[:, :img.shape[1]]
    b = np.pad(img[:, :, 2], ((0, 0), (0, d)), mode="edge")[:, d:]
    out[:, :, 0], out[:, :, 2] = r, b
    return out


def cmd_generate(args) -> int:
    from PIL import Image
    sel = pd.read_parquet(work() / "selection.parquet")
    made = 0
    for row in sel[sel.type.isin(NEW_TYPES)].itertuples():
        p = Path(row.dist_path)
        if p.is_file():
            continue
        img = np.asarray(Image.open(row.ref_path).convert("RGB"))
        out = (local_block_wise(img, row.severity_level, _seed(row.source_filename, row.type)) if row.type == "lbw"
               else chromatic_aberration(img, row.severity_level))
        Image.fromarray(out).save(p, compress_level=1)
        made += 1
    print(json.dumps({"generated": made}))
    return 0


def cmd_extract(args) -> int:
    if e14.sha256(e14.BIN) != e14.BIN_SHA:
        raise ValueError("not the r4 bank's extractor")
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
        rc = subprocess.run([str(e14.BIN), "--corpus", "pairs-tsv", "--path", str(tsv), "--out", str(csv), *e14.EXTRACT_ARGS],
                            env={**os.environ, **e14.EXTRACT_ENV}, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"extractor rc={rc}; see {run / 'extract.log'}")
    man = json.loads(Path(f"{csv}.manifest.json").read_text())
    if man["era_label"] != e14.ERA or man["formula_revision"] != "4":
        raise ValueError("extractor manifest: wrong era or revision")
    print(json.dumps({"rows": len(sel), "extract_wall_s": round(time.time() - t0, 1), "feature_set_id": man["feature_set_id"]}))
    return 0


def cmd_table(args) -> int:
    import pyarrow.csv as pacsv
    sel = pd.read_parquet(work() / "selection.parquet")
    W = e14.WIDTH
    names = ["row_id"] + [f"f{i}" for i in range(W)]
    conv = pacsv.ConvertOptions(column_types={"row_id": pa.int64(), **{f"f{i}": pa.float64() for i in range(W)}},
                                include_columns=names)
    tab = pacsv.read_csv(work() / "extract" / "features.csv", convert_options=conv)
    rid = tab.column("row_id").to_numpy()
    order = np.argsort(rid, kind="stable")
    if not (rid[order] == np.arange(len(sel))).all():
        raise ValueError("row_id join failed")
    feats = np.stack([tab.column(f"f{i}").to_numpy()[order] for i in range(W)], axis=1)
    if not np.isfinite(feats).all():
        raise ValueError("nonfinite features")
    ident = np.array([e14.pixels_identical(r, d) for r, d in zip(sel.ref_path, sel.dist_path)])
    sign = sel["sign"].to_numpy()
    key = np.array([f"{'kadis' if t.isdigit() else 'gen'}:{s}|t{t}|{'+' if g > 0 else '-' if g < 0 else '0'}"
                    for s, t, g in zip(sel.source_filename, sel.type, sign)])
    target = -sel.severity.to_numpy(dtype=np.float64)
    keep = ~ident
    counts = pd.Series(key[keep]).value_counts()
    keep &= np.isin(key, counts.index[counts >= 2])
    wide = json.loads((V2 / "wide" / "main" / "real" / "receipt.json").read_text())["width"]
    cols = {"ref_basename": pa.array(key[keep]), "human_score": pa.array(target[keep])}
    cols.update({f"f{i}": pa.array(feats[keep, i].astype(np.float32)) for i in range(W)})
    nan = pa.array(np.full(int(keep.sum()), np.nan, np.float32))
    cols.update({f"f{i}": nan for i in range(W, wide)})
    out = work() / "coverage_pool.parquet"
    pq.write_table(pa.table(cols), out, compression="zstd", use_byte_stream_split=[f"f{i}" for i in range(wide)])
    Path(f"{out}.manifest.json").write_text(json.dumps({
        "source_bank_feature_set_id": "basic+peaks+masked+iw+v2+append+append2+csfw+dvifm+gridblk+ringbasis+tailhist+arttype"
                                      "+gmsbank+mapdev+z1max+gmsnative+dvifmgate@w1825/tiercanon_c3negfold#d57e9571",
        "composite": f"Rev4 POTENTIAL E15 coverage pool (ordinal ladders; f0-f1824 Rev4 bank arithmetic; f1825-f{wide - 1} NaN, "
                     "not extracted); diagnostic only", "formula_revision": 4}, indent=1) + "\n")
    keys = sel.loc[keep, ["source_filename", "type", "family", "severity_level", "severity", "sign"]].copy()
    keys.insert(0, "ladder", key[keep])
    keys.to_parquet(work() / "coverage_pool.keys.parquet")
    man = {"schema": "rev4-featpot-e15-coverage-pool-v1", "label": "POTENTIAL — ceiling, not a model score",
           "rows": int(keep.sum()), "ladders": int(len(set(key[keep]))), "dropped_identical": int(ident.sum()),
           "dropped_single_rung": int((~ident).sum() - keep.sum()), "width": W, "nan_columns": f"f{W}-f{wide - 1}",
           "per_type": PER_TYPE, "excluded_kadis_types": sorted(e14.EXCLUDED_TYPES), "new_types": list(NEW_TYPES),
           "families": {k: list(map(str, v)) for k, v in COVERAGE_FAMILIES.items()},
           "rows_by_family": keys.family.value_counts().to_dict(), "extractor_sha256": e14.BIN_SHA, "era": e14.ERA,
           "sha256": e14.sha256(out), "keys_sha256": e14.sha256(work() / "coverage_pool.keys.parquet")}
    (work() / "coverage_pool.manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("rows", "ladders", "dropped_identical", "dropped_single_rung", "rows_by_family", "sha256")}))
    return 0


def arms() -> list:
    """(label, family mask) of the registered E15 arms: ALL, KADIS-only, add-one x8, leave-one-out x8."""
    fams = list(COVERAGE_FAMILIES)
    full = (1 << len(fams)) - 1
    out = [("all", full), ("kadis", full & ~(1 << fams.index("new")))]
    out += [(f"only_{f}", 1 << i) for i, f in enumerate(fams)]
    out += [(f"drop_{f}", full & ~(1 << i)) for i, f in enumerate(fams)]
    seen, uniq = set(), []
    for label, mask in out:  # drop_new is the KADIS-only mask: keep the first label for each mask
        if mask not in seen:
            seen.add(mask)
            uniq.append((label, mask))
    return uniq


def spec(weight: float, mask: int) -> str:
    return f"{BASE}:cv{weight:g}:cf{mask:x}"


def cmd_grid(args) -> int:
    cells = [{"name": f"{spec(args.weight, m)}__{HEAD}/without_{s}_s{i}",
              "argv": ["v2_lodo_mlp.py", "--spec", spec(args.weight, m), "--head", HEAD, "--heldout", s, "--seed-index", str(i),
                       "--root", str(V2)]}
             for _, m in arms() for s in SOURCE_ORDER for i in SEEDS]
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo), "arms": len(arms())}))
    return 0


def cmd_score(args) -> int:
    import e13_teacher as e13
    e13.SEEDS = SEEDS
    labelled = [(label, spec(args.weight, m)) for label, m in arms()]
    rc = e13.score_arms(labelled, "e15_coverage", args.monotonicity)
    table = e13.type_table(labelled)
    (V2 / "compare" / "e15_coverage_types.json").write_text(json.dumps(table, indent=1) + "\n")
    return rc


# Design log E16 (registered 2026-10-03 10:43 MT): confirm and size the coverage recipe from E15.
E16_ARMS = [("fd_w1_confirm", 1.0, 0xfd, range(3, 10)),
            *[(f"fd_w{w:g}", w, 0xfd, range(5)) for w in (4.0, 16.0)],
            *[(f"bd_w{w:g}", w, 0xbd, range(5)) for w in (1.0, 4.0, 16.0)],
            *[(f"98_w{w:g}", w, 0x98, range(5)) for w in (4.0, 16.0)]]


def cmd_grid16(args) -> int:
    cells = [{"name": f"{spec(w, m)}__{HEAD}/without_{s}_s{i}",
              "argv": ["v2_lodo_mlp.py", "--spec", spec(w, m), "--head", HEAD, "--heldout", s, "--seed-index", str(i),
                       "--root", str(V2)]}
             for _, w, m, seeds in E16_ARMS for s in SOURCE_ORDER for i in seeds]
    todo = [c for c in cells if not (V2 / "cells" / c["name"] / "result.json").is_file()]
    Path(args.out).write_text(json.dumps({"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": todo}, indent=1))
    print(json.dumps({"cells": len(cells), "to_run": len(todo), "arms": len(E16_ARMS)}))
    return 0


def cmd_score16(args) -> int:
    import e13_teacher as e13
    rc = 0
    for label, w, m, seeds in E16_ARMS:  # each arm against the control at its own seeds
        e13.SEEDS = seeds
        rc |= e13.score_arms([(label, spec(w, m))], f"e16_{label}", args.monotonicity)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "generate", "extract", "table", "grid", "score", "grid16", "score16"])
    ap.add_argument("--root")
    ap.add_argument("--out")
    ap.add_argument("--weight", type=float, default=4.0)
    ap.add_argument("--program-sha", default="")
    ap.add_argument("--data-sha", default="")
    ap.add_argument("--monotonicity", action="store_true")
    args = ap.parse_args()
    return {"select": cmd_select, "generate": cmd_generate, "extract": cmd_extract, "table": cmd_table, "grid": cmd_grid,
            "score": cmd_score, "grid16": cmd_grid16, "score16": cmd_score16}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
