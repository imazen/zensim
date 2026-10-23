"""Table route for the label-free damage-mixture test: score bakes that read research columns the pixel scorer does not
compute (the byc_globaff descriptor at f1867-f1890).

Exploration record: benchmarks/byconstruction_2026-10-22.md. For every image of the mixture manifest, the computed Rev5
vector comes from `score_pair_with_bake --dump-features-to` (B's served bake reads every id the fx1 family reads; the
ring form is whatever ZENSIM_RING_FORM selects) and the descriptor from `byc_globaff`. One wide table holds them in
manifest image order; `bake_dial_refit predict --score-units` scores it. The route is checked against the pixel route on
a bake that reads no research column (`check`), which must agree before any table score is reported.

  build   --mixture DIR --bin-dir BIN --bake B.bin --globaff BIN --out DIR [--jobs 16]
  score   --mixture DIR --table DIR --bin-dir BIN --bake NAME=PATH [--bake ...]
  check   --mixture DIR --table DIR --pixel-scores scores_X.json --table-scores scores_Y.json
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

WIDTH = 1891
GLOBAFF = list(range(1867, 1891))


def images(mix: Path) -> list[tuple[str, str]]:
    man = json.loads((mix / "manifest.json").read_text())
    return sorted({(r["ref"], r["base"]) for r in man["rows"]} | {(r["ref"], r["more"]) for r in man["rows"]})


def cmd_build(a) -> int:
    mix, out = Path(a.mixture), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    pairs = images(mix)
    dumps = out / "dumps"
    dumps.mkdir()

    def one(i):
        ref, dist = pairs[i]
        dest = dumps / f"{i:05d}.f32"
        subprocess.run([str(Path(a.bin_dir) / "score_pair_with_bake"), "--bake", a.bake, "--bake-post", "raw", "--ref",
                        str(mix / "img" / ref), "--dist", str(mix / "img" / dist), "--dump-features-to", str(dest)],
                       check=True, capture_output=True)
        b = dest.read_bytes()
        n = int.from_bytes(b[:4], "little")
        return np.frombuffer(b[8:8 + 4 * n], dtype="<f4")

    with cf.ThreadPoolExecutor(a.jobs) as pool:
        vecs = list(pool.map(one, range(len(pairs))))
    n = {len(v) for v in vecs}
    if len(n) != 1 or next(iter(n)) < 720:
        raise ValueError(f"unexpected dump widths {n}")
    tsv = out / "pairs.tsv"
    tsv.write_text("ref_path\tdist_path\thuman_score\trow_id\n" +
                   "".join(f"{mix / 'img' / r}\t{mix / 'img' / d}\t0\t{i}\n" for i, (r, d) in enumerate(pairs)))
    gcsv = out / "globaff.csv"
    subprocess.run([a.globaff, "--pairs", str(tsv), "--out", str(gcsv)], check=True,
                   env=dict(os.environ, RAYON_NUM_THREADS=str(a.jobs)))
    g = pacsv.read_csv(gcsv)
    order = np.argsort(g.column("row_id").to_numpy())
    G = np.stack([g.column(f"g{i}").to_numpy()[order] for i in range(len(GLOBAFF))], axis=1).astype(np.float32)
    X = np.full((len(pairs), WIDTH), np.nan, np.float32)
    k = next(iter(n))
    X[:, :min(k, 720)] = np.stack(vecs)[:, :min(k, 720)]
    X[:, GLOBAFF] = G
    cols = {"ref_basename": pa.array([r for r, _ in pairs]), "human_score": pa.array(np.zeros(len(pairs)))}
    cols.update({f"f{i}": pa.array(X[:, i]) for i in range(WIDTH)})
    pq.write_table(pa.table(cols), out / "table.parquet", compression="zstd")
    meta = {"images": [d for _, d in pairs], "ring_form": os.environ.get("ZENSIM_RING_FORM", "raw"), "dump_bake": a.bake,
            "dump_bake_sha256": hashlib.sha256(Path(a.bake).read_bytes()).hexdigest(), "dump_width": k,
            "globaff_sha256": hashlib.sha256(Path(a.globaff).read_bytes()).hexdigest()}
    (out / "table.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(pairs)} images, dump width {k}")
    return 0


def cmd_score(a) -> int:
    mix, table = Path(a.mixture), Path(a.table)
    meta = json.loads((table / "table.json").read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
    os.environ.setdefault("REV4_V2_BIN_DIR", a.bin_dir)
    from v2_lodo_mlp import predict  # noqa: E402
    for spec in a.bake:
        name, path = spec.split("=", 1)
        dest = mix / f"scores_{name}.json"
        if dest.exists():
            print(f"exists: {dest}")
            continue
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as td:
            pred = predict(Path(path), table / "table.parquet", Path(td) / "pred.tsv")
        scores = dict(zip(meta["images"], (float(round(p, 6)) for p in pred)))
        dest.write_text(json.dumps({"bake": path, "post": "table-score-units", "route": "table",
                                    "ring_form": meta["ring_form"], "table": str(table),
                                    "bake_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                    "scores": scores}, indent=1) + "\n")
        print(f"{name}: {len(scores)} images scored (table route)")
    return 0


def cmd_check(a) -> int:
    p = json.loads(Path(a.pixel_scores).read_text())["scores"]
    t = json.loads(Path(a.table_scores).read_text())["scores"]
    d = np.array([abs(p[k] - t[k]) for k in p])
    print(json.dumps({"images": len(d), "max_abs_diff": float(d.max()), "identical": int((d == 0).sum())}))
    return 0 if d.max() <= a.tol else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--mixture", required=True)
    b.add_argument("--bin-dir", required=True)
    b.add_argument("--bake", required=True)
    b.add_argument("--globaff", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--jobs", type=int, default=16)
    b.set_defaults(fn=cmd_build)
    s = sub.add_parser("score")
    s.add_argument("--mixture", required=True)
    s.add_argument("--table", required=True)
    s.add_argument("--bin-dir", required=True)
    s.add_argument("--bake", action="append", required=True)
    s.set_defaults(fn=cmd_score)
    c = sub.add_parser("check")
    c.add_argument("--mixture", required=True)
    c.add_argument("--table", required=True)
    c.add_argument("--pixel-scores", required=True)
    c.add_argument("--table-scores", required=True)
    c.add_argument("--tol", type=float, default=1e-4)
    c.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
