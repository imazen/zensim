#!/usr/bin/env python3
"""STEERCODEC — the steering map as a per-block quality allocator for zenjpeg (4:4:4), JPEG XL (zenjxl) and zqi lossy encodes.

  python3 steer_codec.py swap    --bake NAME=PATH[,REV] ...  [--exact 0,1] [--jobs N]
  python3 steer_codec.py compose
  python3 steer_codec.py score   --bake NAME=PATH[,REV] ...
  python3 steer_codec.py summary --out RESULT.json

Per case (codec, image, quality pair lo->hi, bake, neighbour-exact on/off): `diffmap_block_coherence src lo --bake B --block 8`
with `ZENSIM_REPAIR_SOURCE=hi` gives every 8x8 block's map prediction (`refinement_gain`) and true zensim change from pasting
the hi decode into it (`score_delta`), plus M2/M3f. `compose` upgrades 25 % of the blocks of the lo decode to the hi decode,
chosen by the map, by the oracle (true single-block change), at random, or by the lowest map gain (anti). `score` scores lo,
hi and every composite with the bake (zensim, `serve_custom_bake --pairs`) and with SSIMULACRA2 and butteraugli (zenmetrics
`score-pairs`, independent judges). The headline is the map's share of the full-upgrade gain at 25 % of blocks, against oracle
and random, under all three judges. Block swaps of two independent encodes stand in for per-block quantization decisions;
nothing here is an encoder RD result.

Inputs: /var/tmp/steercodec/src/<img>.png (512x512 crops of CLIC 2025 tuning photos, `src_manifest.json`), decodes at
/var/tmp/steercodec/dec/<codec>/<img>_q<q>.png.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/var/tmp/steercodec")
CODECS = ("zenjpeg", "zenjxl", "zqi")
SWAPS = ((30, 60), (60, 85))
FRAC = 0.25
RANKINGS = ("map", "oracle", "random", "anti")
REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "target/release/examples/diffmap_block_coherence"
SCORER = REPO / "target/release/examples/serve_custom_bake"
ZM_IMAGE = "ghcr.io/imazen/zenfleet-worker:exec-cvvdp-0a61830d"


def images() -> list[str]:
    return sorted(p.stem for p in (ROOT / "src").glob("*.png"))


def parse_bakes(specs: list[str]) -> dict:
    out = {}
    for s in specs:
        name, _, rest = s.partition("=")
        path, _, rev = rest.partition(",")
        out[name] = (path, rev or "4")
    return out


def case_id(codec, img, lo, hi, bake, exact) -> str:
    return f"{codec}-{img}-q{lo}to{hi}-{bake}-x{exact}"


def run_swap(args):
    codec, img, lo, hi, bake, (path, rev), exact = args
    out = ROOT / "swap" / f"{case_id(codec, img, lo, hi, bake, exact)}.json"
    if out.is_file():
        return str(out), 0
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ZENSIM_REPAIR_SOURCE=str(ROOT / "dec" / codec / f"{img}_q{hi}.png"), ZENSIM_PREPARED_STEERING="1",
               ZENSIM_FORMULA_REV=rev, RAYON_NUM_THREADS="1")
    env.pop("ZENSIM_NEIGHBOUR_EXACT", None)
    if exact:
        env["ZENSIM_NEIGHBOUR_EXACT"] = "1"
    cmd = ["nice", "-n", "19", str(TOOL), str(ROOT / "src" / f"{img}.png"), str(ROOT / "dec" / codec / f"{img}_q{lo}.png"),
           "--bake", path, "--block", "8", "--json", str(out)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        (out.with_suffix(".err")).write_text(r.stdout + r.stderr)
    return str(out), r.returncode


def cmd_swap(a) -> int:
    bakes = parse_bakes(a.bake)
    exacts = [int(x) for x in a.exact.split(",")]
    jobs = [(c, i, lo, hi, b, bakes[b], x) for c in CODECS for i in images() for lo, hi in SWAPS for b in bakes for x in exacts]
    bad = 0
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for path, rc in ex.map(run_swap, jobs):
            bad += rc != 0
    print(json.dumps({"cases": len(jobs), "failed": bad}))
    return 1 if bad else 0


def cmd_compose(a) -> int:
    n_out = 0
    for js in sorted((ROOT / "swap").glob("*.json")):
        codec, img, qq = js.stem.split("-")[:3]
        lo_q, hi_q = qq[1:].split("to")
        lo = np.array(Image.open(ROOT / "dec" / codec / f"{img}_q{lo_q}.png").convert("RGB"))
        hi = np.array(Image.open(ROOT / "dec" / codec / f"{img}_q{hi_q}.png").convert("RGB"))
        d = json.loads(js.read_text())
        blocks = d["blocks"]
        k = int(round(FRAC * len(blocks)))
        keys = {"map": [-b["refinement_gain"] for b in blocks], "oracle": [-b["score_delta"] for b in blocks],
                "random": list(np.random.default_rng(20261004).permutation(len(blocks))),
                "anti": [b["refinement_gain"] for b in blocks]}
        for name in RANKINGS:
            dst = ROOT / "comp" / f"{js.stem}-{name}.png"
            if dst.is_file():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            out = lo.copy()
            for j in np.argsort(keys[name], kind="stable")[:k]:
                x0, y0, x1, y1 = blocks[j]["bounds"]
                out[y0:y1, x0:x1] = hi[y0:y1, x0:x1]
            Image.fromarray(out).save(dst)
            n_out += 1
    print(json.dumps({"composites_written": n_out}))
    return 0


def all_pairs() -> list[tuple[str, str]]:
    """(ref, dist) for every lo, hi and composite."""
    pairs = []
    for c in CODECS:
        for i in images():
            for q in sorted({q for s in SWAPS for q in s}):
                pairs.append((str(ROOT / "src" / f"{i}.png"), str(ROOT / "dec" / c / f"{i}_q{q}.png")))
    for p in sorted((ROOT / "comp").glob("*.png")):
        pairs.append((str(ROOT / "src" / f"{p.stem.split('-')[1]}.png"), str(p)))
    return pairs


def cmd_score(a) -> int:
    pairs = all_pairs()
    (ROOT / "score").mkdir(exist_ok=True)
    tsv = ROOT / "score" / "pairs.tsv"
    with tsv.open("w") as f:
        f.write("ref_path\tdist_path\n")
        for r, d in pairs:
            f.write(f"{r}\t{d}\n")
    for name, (path, rev) in parse_bakes(a.bake).items():
        out = ROOT / "score" / f"zensim-{name}.tsv"
        with out.open("w") as f:
            subprocess.run(["nice", "-n", "19", str(SCORER), "--pairs", str(tsv), path], stdout=f, check=True,
                           env=dict(os.environ, ZENSIM_FORMULA_REV=rev, RAYON_NUM_THREADS="8"))
    for metric in ("ssim2", "butteraugli"):
        out = ROOT / "score" / f"{metric}.parquet"
        if out.is_file():
            continue
        subprocess.run(["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}", "--cpus", "8", "-v", f"{ROOT}:{ROOT}",
                        "--entrypoint", "zenmetrics", ZM_IMAGE, "score-pairs", "--metric", metric, "--pairs-tsv", str(tsv),
                        "--out-parquet", str(out)], check=True)
    print(json.dumps({"pairs": len(pairs)}))
    return 0


def load_scores() -> dict:
    """judge -> {dist_path: score, oriented higher = better}."""
    import pyarrow.parquet as pq
    out = {}
    for f in (ROOT / "score").glob("zensim-*.tsv"):
        with f.open() as fh:
            rd = csv.reader(fh, delimiter="\t")
            next(rd)
            out[f"zensim:{f.stem[7:]}"] = {r[1]: float(r[2]) for r in rd}
    for metric, col, sign in (("ssim2", None, 1.0), ("butteraugli", "pnorm3", -1.0)):
        p = ROOT / "score" / f"{metric}.parquet"
        if not p.is_file():
            continue
        t = pq.read_table(p).to_pandas()
        dcol = next(c for c in t.columns if "dist" in c)
        scol = next(c for c in t.columns if (col and col in c) or (not col and c.startswith("score")))
        out[metric] = dict(zip(t[dcol], sign * t[scol].astype(float)))
    return out


def cmd_summary(a) -> int:
    scores = load_scores()
    rows = []
    for js in sorted((ROOT / "swap").glob("*.json")):
        codec, img, qq, bake, xx = js.stem.split("-")
        lo_q, hi_q = qq[1:].split("to")
        d = json.loads(js.read_text())
        lo = str(ROOT / "dec" / codec / f"{img}_q{lo_q}.png")
        hi = str(ROOT / "dec" / codec / f"{img}_q{hi_q}.png")
        row = {"codec": codec, "img": img, "swap": qq, "bake": bake, "exact": int(xx[1:]), "m2": d.get("m2"), "m3f": d.get("m3f")}
        for judge, sc in scores.items():
            if judge.startswith("zensim:") and judge != f"zensim:{bake}":
                continue
            full = sc[hi] - sc[lo]
            for name in RANKINGS:
                comp = str(ROOT / "comp" / f"{js.stem}-{name}.png")
                row[f"{judge.split(':')[0]}_{name}_share"] = (sc[comp] - sc[lo]) / full if full else float("nan")
        rows.append(row)
    groups = {}
    for r in rows:
        groups.setdefault((r["codec"], r["swap"], r["bake"], r["exact"]), []).append(r)
    summary = []
    for (codec, swap, bake, exact), rs in sorted(groups.items()):
        s = {"codec": codec, "swap": swap, "bake": bake, "neighbour_exact": exact, "n": len(rs),
             "m3f_median": st.median(r["m3f"] for r in rs), "m3f_min": min(r["m3f"] for r in rs),
             "m2_min": min(r["m2"] for r in rs)}
        for k in rs[0]:
            if k.endswith("_share"):
                s[k] = st.median(r[k] for r in rs)
        summary.append(s)
    Path(a.out).write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    for s in summary:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["swap", "compose", "score", "summary"])
    ap.add_argument("--bake", action="append", default=[])
    ap.add_argument("--exact", default="0,1")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "summary.json"))
    a = ap.parse_args()
    return {"swap": cmd_swap, "compose": cmd_compose, "score": cmd_score, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
