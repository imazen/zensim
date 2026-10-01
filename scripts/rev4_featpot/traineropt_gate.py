"""TRAINEROPT gate: run one v2 cell through several `zensim_mlp_train` binaries and compare byte for byte.

For each variant `label=trainer_path[:log_every]` the cell is trained with the v2 recipe (`v2_lodo_mlp.train_command`,
final-epoch checkpoint dump), then the final-epoch weights (`harvest_fit_cells.weights_sha` semantics: the checkpoint
minus its trailing run-metadata JSON) and the held-out predictions (`bake_dial_refit predict`) are hashed. The first
variant is the reference; every other variant must match it on the weights hash, the prediction hash and the dev
values at the epochs both logged. Wall time is recorded per variant (pin the process with `taskset`).

Not an instrument cell: it writes only under --out and never into the instrument root.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / ".." / "zenmetrics" / "scripts" / "jobsys"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--head", choices=("N", "F"), required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--seed-index", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--fitbin", required=True, help="bake_dial_refit used to predict for every variant")
    ap.add_argument("--variant", action="append", required=True, help="label=trainer[:log_every]; first is the reference")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sys.argv = [sys.argv[0], "--root", args.root]  # v2_common reads the root from argv at import
    import v2_lodo_mlp as lodo
    from harvest_fit_cells import weights_sha

    lodo.EPOCHS = args.epochs
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    core_spec, human_w = lodo.split_weight(args.spec)
    lists = json.loads((lodo.V2 / "wide" / "keep_lists.json").read_text())
    entry = lists["specs"][core_spec]
    family, variant, keep = entry["family"], entry["variant"], entry["keep"]
    vdir = lodo.V2 / "wide" / family / variant
    receipt = json.loads((vdir / "receipt.json").read_text())
    width, legs = receipt["width"], receipt["legs"]
    keep_file = out / "keep_features.txt"
    keep_file.write_text("\n".join(map(str, keep)) + "\n")
    groups, weights = [], {}
    for leg, (_, _, val_w) in lodo.TEACHERS.items():
        fit, dev = lodo.checked(legs[leg]["fit"]), lodo.checked(legs[leg]["dev"])
        weights[leg] = lodo.acceptance_weight(lodo.NOMINAL_WEIGHT[leg], lodo.refs_of(fit))
        groups += [(leg, fit, weights[leg], 0, "withinref,both"), (f"{leg}_development", dev, 0, val_w, "withinref,both")]
    hl = legs[f"human_without_{args.heldout}"]
    hfit, hdev = lodo.checked(hl["fit"]), lodo.checked(hl["dev"])
    weights["human"] = lodo.acceptance_weight(lodo.NOMINAL_WEIGHT["human"] if human_w is None else human_w,
                                               lodo.refs_of(hfit))
    groups += [("human", hfit, weights["human"], 0, "withinref,rank"),
               ("human_development", hdev, 0, lodo.HUMAN_VAL_WEIGHT, "withinref,rank")]
    init_seed, sample_seed = lodo.seeds(args.heldout, args.seed_index)
    table = lodo.checked(legs[args.heldout]["full"])
    lodo.FITBIN = Path(args.fitbin)

    env = {**os.environ, "ZENSIM_MAX_TIER": "v3", "RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    results = []
    for spec in args.variant:
        label, _, rest = spec.partition("=")
        trainer, _, le = rest.partition(":")
        log_every = int(le) if le else 1
        d = out / label
        d.mkdir(exist_ok=True)
        ckpt = d / "ckpt"
        ckpt.mkdir(exist_ok=True)
        lodo.TRAINER = Path(trainer)
        cmd = lodo.train_command(groups, init_seed, sample_seed, width, keep_file, args.head, d / "best.bin")
        cmd[cmd.index("--log-every") + 1] = str(log_every)
        cmd += ["--dump-checkpoints-every", str(args.epochs - 1), "--dump-checkpoints-dir", str(ckpt)]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        wall = time.perf_counter() - t0
        (d / "train.log").write_text("$ " + " ".join(cmd) + "\n" + proc.stdout)
        if proc.returncode:
            raise RuntimeError(f"{label}: rc={proc.returncode}; see {d / 'train.log'}")
        final = ckpt / f"ckpt_epoch{args.epochs - 1:03d}.bin"
        curve = {int(e): v for e, v in re.findall(r"epoch\s+(\d+)\s+\|.*?val\(geomean3\)=([+-]?\d+\.\d+)", proc.stdout)}
        pred = d / "eval_preds.tsv"
        subprocess.run([args.fitbin, "predict", "--bake", str(final), "--corpus", str(table), "--score-units",
                        "--out", str(pred)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        results.append({"label": label, "trainer": trainer, "log_every": log_every, "wall_s": round(wall, 2),
                        "weights_sha": weights_sha(final), "pred_sha": hashlib.sha256(pred.read_bytes()).hexdigest(),
                        "curve": curve})
    ref = results[0]
    for r in results[1:]:
        shared = sorted(set(ref["curve"]) & set(r["curve"]))
        r["match"] = {"weights": r["weights_sha"] == ref["weights_sha"], "pred": r["pred_sha"] == ref["pred_sha"],
                      "dev_epochs_compared": len(shared),
                      "dev": all(ref["curve"][e] == r["curve"][e] for e in shared)}
    rec = {"spec": args.spec, "head": args.head, "heldout": args.heldout, "seed_index": args.seed_index,
           "epochs": args.epochs, "kept": len(keep), "width": width, "variants": results}
    (out / "gate.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: [{kk: vv for kk, vv in r.items() if kk != "curve"} for r in v] if k == "variants" else v
                      for k, v in rec.items()}))
    if any(not all(r["match"].values()) for r in results[1:]):
        sys.exit(3)


if __name__ == "__main__":
    main()
