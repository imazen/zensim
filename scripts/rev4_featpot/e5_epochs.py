#!/usr/bin/env python3
"""Design log E5 (exploratory, amendment R2): which epoch-selection rule should a v2 cell use?

Runs one v2 cell exactly as `v2_lodo_mlp.py` does, but in a scratch root and with a checkpoint dumped every epoch,
then scores every checkpoint on the held-out source (signed SROCC from the panel owner). `summarise` compares rules:
  cur    argmax of the registered dev aggregate val(geomean3) (today's rule)
  hdev   argmax of the human_development geomean of (SROCC, PLCC, PWRC)
  last   the final epoch
  late   mean held-out SROCC of epochs 60-119 (diagnostic: what a late-epoch average would sit near)
  best   max held-out SROCC (diagnostic upper bound; not a rule)

  e5_epochs.py cell --spec oracle_hi@h8 --head N --heldout aic3 --seed-index 0
  e5_epochs.py summarise
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2_lodo_mlp  # noqa: E402
from v2_common import V2  # noqa: E402

ROOT = Path(os.environ.get("E5_ROOT", "/var/tmp/rev4-featpot/e5"))
EPOCH = re.compile(r"epoch\s+(\d+)\s+\|.*?val\(geomean3\)=([+-]?\d+\.\d+).*?human_development: srocc=([0-9.]+) "
                   r"plcc=([0-9.]+) pwrc=([0-9.]+)")


def cell(args) -> None:
    scratch = ROOT / "v2"
    (scratch / "cells").mkdir(parents=True, exist_ok=True)
    if not (scratch / "wide").exists():
        (scratch / "wide").symlink_to(V2 / "wide")
    dest = scratch / "cells" / f"{args.spec}__{args.head}" / f"without_{args.heldout}_s{args.seed_index}"
    ckpt = dest / "ckpt"
    real_run = v2_lodo_mlp.run

    def run(cmd, log):
        if Path(cmd[0]).name == "zensim_mlp_train":
            ckpt.mkdir(parents=True, exist_ok=True)
            cmd = cmd + ["--dump-checkpoints-every", "1", "--dump-checkpoints-dir", str(ckpt)]
        real_run(cmd, log)

    v2_lodo_mlp.run = run
    v2_lodo_mlp.V2 = scratch
    sys.argv = ["v2_lodo_mlp.py", "--spec", args.spec, "--head", args.head, "--heldout", args.heldout,
                "--seed-index", str(args.seed_index)]
    v2_lodo_mlp.main()
    result = json.loads((dest / "result.json").read_text())
    keys = v2_lodo_mlp.pq.read_table(V2 / "wide" / result["family"] / result["variant"] /
                                     f"{args.heldout}.keys.parquet", columns=["target"]).to_pandas()
    y = keys.target.to_numpy(np.float64)
    receipt = json.loads((V2 / "wide" / result["family"] / result["variant"] / "receipt.json").read_text())
    table = Path(receipt["legs"][args.heldout]["full"]["path"])
    per_epoch = {}
    for path in sorted(ckpt.glob("ckpt_epoch*.bin")):
        e = int(path.stem[len("ckpt_epoch"):])
        pred = v2_lodo_mlp.predict(path, table, ckpt / f"pred_{e:03d}.tsv")
        per_epoch[e] = pred
        (ckpt / f"pred_{e:03d}.tsv").unlink()
    jobs = [(str(e), p, y) for e, p in sorted(per_epoch.items())]
    rows = v2_lodo_mlp.panel_batch(jobs, stats="full")
    heldout = {int(r["label"]): r["srocc_signed"] for r in rows}
    curve = {}
    for m in EPOCH.finditer((dest / "train.log").read_text()):
        e, agg, s, p, w = int(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])
        curve[e] = {"agg": agg, "hdev": (s * p * w) ** (1 / 3)}
    (dest / "e5.json").write_text(json.dumps({"heldout_srocc": heldout, "dev": curve,
                                              "registered_best_epoch": result["best_epoch_by_curve"],
                                              "registered_srocc": result["score"]["srocc_signed"]}) + "\n")
    for path in ckpt.glob("ckpt_epoch*.bin"):
        path.unlink()
    print(json.dumps({"cell": str(dest), "epochs": len(heldout)}))


def rules(rec: dict) -> dict:
    h = {int(k): v for k, v in rec["heldout_srocc"].items()}
    d = {int(k): v for k, v in rec["dev"].items()}
    epochs = sorted(set(h) & set(d))
    cur = max(epochs, key=lambda e: (d[e]["agg"], -e))
    hdev = max(epochs, key=lambda e: (d[e]["hdev"], -e))
    late = [h[e] for e in epochs if 60 <= e <= 119]
    return {"cur": h[cur], "hdev": h[hdev], "last": h[max(epochs)], "late": float(np.mean(late)),
            "best": max(h[e] for e in epochs), "cur_epoch": cur, "hdev_epoch": hdev}


def summarise(_args) -> None:
    out = {}
    for f in sorted((ROOT / "v2" / "cells").glob("*__*/without_*_s*/e5.json")):
        spec_head, cellname = f.parent.parent.name, f.parent.name
        fold, seed = re.fullmatch(r"without_(.+)_s(\d+)", cellname).groups()
        out.setdefault(spec_head, {})[(fold, int(seed))] = rules(json.loads(f.read_text()))
    names = ("cur", "hdev", "last", "late", "best")
    table = {}
    for spec_head, cells in sorted(out.items()):
        folds = sorted({k[0] for k in cells})
        per = {}
        for r in names:
            means = [np.mean([v[r] for k, v in cells.items() if k[0] == fo]) for fo in folds]
            sds = [np.std([v[r] for k, v in cells.items() if k[0] == fo], ddof=1)
                   for fo in folds if sum(1 for k in cells if k[0] == fo) > 1]
            per[r] = {"mean": float(np.mean(means)), "seed_sd": float(np.mean(sds)) if sds else math.nan}
        per["n"] = len(cells)
        per["cur_epochs"] = sorted(v["cur_epoch"] for v in cells.values())
        per["hdev_epochs"] = sorted(v["hdev_epoch"] for v in cells.values())
        table[spec_head] = per
    for head in ("N", "F"):
        hi, p1 = out.get(f"oracle_hi@h8__{head}", {}), out.get(f"oracle_hi~p1@h8__{head}", {})
        both = sorted(set(hi) & set(p1))
        if both:
            table[f"detection__{head}"] = {r: float(np.mean([hi[k][r] - p1[k][r] for k in both])) for r in names[:4]}
            table[f"detection__{head}"]["n_pairs"] = len(both)
    (ROOT / "e5_summary.json").write_text(json.dumps(table, indent=1) + "\n")
    for k, v in table.items():
        print(k, json.dumps(v)[:400])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--spec", required=True)
    c.add_argument("--head", required=True)
    c.add_argument("--heldout", required=True)
    c.add_argument("--seed-index", type=int, required=True)
    sub.add_parser("summarise")
    args = ap.parse_args()
    {"cell": cell, "summarise": summarise}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
