"""By-construction exploration: run E21 LODO cells for recipe arms and score them against E34's stored control.

Exploration harness only (benchmarks/byconstruction_2026-10-22.md). Each cell is the registered `v2_lodo_mlp.py` strict
training-only route (same tables, seeds, recipe owner) with the arm's recipe tokens appended to B's C-family spec, followed
by a held-out prediction through `v2_lodo_mlp.predict` (dense bake + `bake_dial_refit predict --score-units`). Scoring
pairs every arm with E34's stored `base-c` predictions for the same fold and seed index (E34 `panel-input.json`), which
this harness reproduced bit for bit (kadid seed 0, 7,869 rows).

  run    --arm TOKENS --folds kadid,tid2013,konfig,cid22_a25 --seeds 0,1,2 --jobs N
  score  --arms NAME[,NAME...] [--seeds 0,1,2]

Point estimates only: the E21 statistic (mean over the four D1 sources of signed SROCC, arm minus paired control,
averaged over seeds) with its seed-unit SE. Registered intervals and tests belong to the `panel e34` owner.
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/v/output/zensim/v40r4-2026-10-08/v2e29")
BASE_SPEC = "sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1"
E34_PANEL = Path("/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json")
FOLDS = ("kadid", "tid2013", "konfig", "cid22_a25")
HERE = Path(__file__).resolve().parent
CELLS = Path(os.environ.get("BYC_CELLS", str(Path.home() / "tmp/byc/cells")))
COLS = Path(os.environ.get("BYC_COLS", str(Path.home() / "tmp/byc/cols_c.txt")))
# v2_common resolves its root and binaries at import; this harness binds both before importing it.
os.environ.setdefault("REV4_V2_ROOT", str(ROOT))
os.environ.setdefault("REV4_V2_BIN_DIR", "/var/tmp/byconstruction-target/release")
sys.path.insert(0, str(HERE))
from v2_common import sha  # noqa: E402
from v2_lodo_mlp import predict, read_curve  # noqa: E402


def spec_of(tokens: str) -> str:
    """B's C-family spec with the arm's tokens. `H<n>` replaces H128; other tokens are appended."""
    toks = [t for t in tokens.split(":") if t]
    base = BASE_SPEC
    hidden = [t for t in toks if t.startswith("H")]
    if hidden:
        base = base.replace(":H128:", f":{hidden[0]}:")
    rest = [t for t in toks if not t.startswith("H")]
    return base + "".join(f":{t}" for t in rest)


def arm_name(tokens: str) -> str:
    return tokens.replace(":", "-") or "control"


def run_cell(tokens: str, fold: str, seed: int) -> dict:
    dest = CELLS / arm_name(tokens) / f"without_{fold}_s{seed}"
    record = dest / "byc_record.json"
    if record.is_file():
        return json.loads(record.read_text())
    if dest.exists():
        raise ValueError(f"{dest}: incomplete earlier attempt; inspect and move it aside")
    scratch = Path.home() / "tmp/byc/scratch" / arm_name(tokens) / f"{fold}_s{seed}"
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(scratch), ZENSIM_MAX_TIER="v3")
    cmd = [sys.executable, str(HERE / "v2_lodo_mlp.py"), "--root", str(ROOT), "--spec", spec_of(tokens),
           "--columns", COLS.read_text().strip(), "--head", "N", "--heldout", fold, "--seed-index", str(seed),
           "--strict-admission", "--data-role-decision", str(ROOT / "human_role_decision.json"),
           "--dest", str(dest), "--train-only"]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    train_s = time.time() - t0
    if proc.returncode:
        raise RuntimeError(f"{dest}: rc={proc.returncode}\n{proc.stderr[-2000:]}")
    pred = predict(dest / "refit/last.bin", ROOT / "wide/main/real" / f"{fold}.parquet", dest / f"pred_{fold}.tsv")
    rec = {"tokens": tokens, "spec": spec_of(tokens), "fold": fold, "seed": seed, "train_seconds": round(train_s, 1),
           "last_bin_sha256": sha(dest / "refit/last.bin"), "pred_sha256": sha(dest / f"pred_{fold}.tsv"),
           "rows": int(len(pred)), "curve": read_curve(dest / "train.log")}
    for line in (dest / "train.log").read_text().splitlines():
        if line.startswith("  epoch 119 "):
            rec["epoch119"] = line.split("| t=")[0].strip()
    record.write_text(json.dumps(rec, indent=1) + "\n")
    for leftover in scratch.glob("*"):
        leftover.unlink()
    return rec


def cmd_run(args) -> int:
    folds = args.folds.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    cells = [(f, s) for s in seeds for f in folds]
    failures = 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_cell, args.arm, f, s): (f, s) for f, s in cells}
        for fut in cf.as_completed(futures):
            f, s = futures[fut]
            try:
                rec = fut.result()
                print(json.dumps({"arm": arm_name(args.arm), "fold": f, "seed": s, "train_s": rec["train_seconds"]}),
                      flush=True)
            except Exception as e:  # report every cell; the run fails at the end
                failures += 1
                print(json.dumps({"arm": arm_name(args.arm), "fold": f, "seed": s, "error": str(e)[-1500:]}), flush=True)
    return 1 if failures else 0


def srocc(x, y) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def cmd_score(args) -> int:
    import numpy as np
    import pandas as pd
    panel = json.loads(E34_PANEL.read_text())
    data = {d["name"]: d for d in panel["datasets"]}
    seeds = [int(s) for s in args.seeds.split(",")]
    out = {}
    for arm in args.arms.split(","):
        per_seed, per_ds, absolute = [], {f: [] for f in FOLDS}, {f: [] for f in FOLDS}
        for s in seeds:
            deltas = []
            for f in FOLDS:
                d = data[f]
                target = np.asarray(d["target"], dtype=float)
                orient = float(d["orientation"])
                if arm in d["predictions"]:
                    pa = np.asarray(d["predictions"][arm][s], dtype=float)
                else:
                    t = pd.read_csv(CELLS / arm / f"without_{f}_s{s}" / f"pred_{f}.tsv", sep="\t")
                    pa = t.pred.to_numpy(dtype=float)
                pc = np.asarray(d["predictions"][args.control][s], dtype=float)
                if len(pa) != len(target):
                    raise ValueError(f"{arm}/{f}/s{s}: {len(pa)} predictions for {len(target)} targets")
                ra, rc = orient * srocc(pa, target), orient * srocc(pc, target)
                deltas.append(ra - rc)
                per_ds[f].append(ra - rc)
                absolute[f].append(ra)
            per_seed.append(float(np.mean(deltas)))
        n = len(per_seed)
        mean = float(np.mean(per_seed))
        se = float(np.std(per_seed, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        out[arm] = {"control": args.control, "seeds": seeds, "delta": mean, "seed_unit_se": se,
                    "per_seed_delta": per_seed,
                    "per_dataset_delta": {f: float(np.mean(v)) for f, v in per_ds.items()},
                    "per_dataset_srocc": {f: float(np.mean(v)) for f, v in absolute.items()},
                    "e33_point_form": bool(mean >= -0.002 and all(np.mean(v) >= -0.005 for v in per_ds.values()))}
    print(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, help="recipe tokens, e.g. mono:thr or H1:mono")
    r.add_argument("--folds", default=",".join(FOLDS))
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--jobs", type=int, default=12)
    s = sub.add_parser("score")
    s.add_argument("--arms", required=True)
    s.add_argument("--control", default="base-c")
    s.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    return {"run": cmd_run, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
