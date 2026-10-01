"""One Instrument v2 cell (R915 sampling): (arm spec, head, held-out human source, seed index).

Governing record: benchmarks/rev4_featpot_v2_amendment_2026-09-30.md, revision R1. One training run:
SafeSyn and CID22-train teacher legs (MSE + rank, within-reference pairs) and the human leg of the four
non-held-out sources (rank only, within-reference), each with its reference-disjoint dev group; the trainer
exports the best epoch by the weighted mean dev geomean3 (R915's --val-policy mean). The arm is the
--keep-features subset of the shared wide table of its variant. Predict and score the held-out source.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch  # noqa: E402
from v2_common import (EPOCH_RULE, EPOCHS, FITBIN, HEADS, HIDDEN, HUMAN_VAL_WEIGHT, NOMINAL_WEIGHT, PAIRS_PER_EPOCH, split_weight,
                       PANEL, REPLAY, SOURCE_ORDER, TEACHERS, TRAINER, V2, WIDTH, acceptance_weight,
                       parse_spec, seeds, sha, table_path)

WIDE_SCHEMAS = ("rev4-featpot-v2-wide-v2", "rev4-featpot-v2c-wide-v1")
# Under the final-epoch rule no per-epoch dev score selects anything, so the dev panels run every 17th epoch (17 divides
# EPOCHS - 1 = 119: epoch 0, every 17th, and the final epoch, which is also where --dump-checkpoints-every EPOCHS-1
# fires). Evaluation is pure (no RNG, no state, LR depends on the epoch index only), so the trajectory and the final
# weights do not depend on this. best_dev still needs every epoch.
LOG_EVERY = 17 if EPOCH_RULE == "last" else 1
if (EPOCHS - 1) % LOG_EVERY:
    raise ValueError(f"LOG_EVERY={LOG_EVERY} must divide EPOCHS-1={EPOCHS - 1}")
EPOCH_RE = re.compile(r"epoch\s+(\d+)\s+\|.*?val\(geomean3\)=([+-]?\d+\.\d+)")


def run(cmd: list[str], log: Path) -> None:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text("$ " + " ".join(cmd) + "\n" + proc.stdout)
    if proc.returncode:
        raise RuntimeError(f"{cmd[0]} rc={proc.returncode}; see {log}")


def checked(record: dict) -> Path:
    path = table_path(record)
    if sha(path) != record["sha256"] or sha(Path(f"{path}.manifest.json")) != record["manifest_sha256"]:
        raise ValueError(f"{path}: table changed after its receipt")
    return path


def refs_of(path: Path) -> list[str]:
    return pq.read_table(path, columns=["ref_basename"])["ref_basename"].to_pylist()


def predict(bake: Path, table: Path, out: Path) -> np.ndarray:
    run([str(FITBIN), "predict", "--bake", str(bake), "--corpus", str(table), "--score-units",
         "--out", str(out)], out.with_suffix(".log"))
    result = pd.read_csv(out, sep="\t")
    if result.columns.tolist() != ["row_idx", "pred"] or not np.array_equal(
            result.row_idx.to_numpy(), np.arange(len(result))):
        raise ValueError(f"{out}: unexpected prediction row order")
    pred = result.pred.to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError(f"{out}: nonfinite prediction")
    return pred


def train_command(groups: list, init_seed: int, sample_seed: int, width: int, keep_file: Path, head: str,
                  out: Path) -> list[str]:
    """The trainer argv of one v2 cell; `groups` = (name, path, train_weight, val_weight, mode). Shared with
    v2_confirm_fit.py, which trains on the same recipe over the full-data legs."""
    cmd = [str(TRAINER)]
    for name, path, tw, vw, mode in groups:
        cmd += ["--group", f"{name}:{path}:{tw!r}:{vw!r}:{mode}"]
    cmd += ["--target-column", "human_score", "--target-scale", "1", "--hidden", str(HIDDEN),
            "--epochs", str(EPOCHS), "--pairs-per-epoch", str(PAIRS_PER_EPOCH),
            "--init-seed", str(init_seed), "--sample-seed", str(sample_seed),
            "--pair-sampling", "uniform", "--max-features", str(width), "--keep-features", str(keep_file),
            "--mse-weight", "1", "--early-stop-patience", "0", "--val-policy", "mean",
            "--val-aggregate", "geomean3", "--out-dtype", "f32", "--log-every", str(LOG_EVERY), "--no-auto-eval",
            "--historical-replay", REPLAY, "--out", str(out)]
    if head == "N":
        cmd.append("--nonneg-distance")
    return cmd


def train_and_select(groups: list, init_seed: int, sample_seed: int, width: int, keep_file: Path, head: str,
                     dest: Path) -> tuple[Path, dict[int, float], dict]:
    """Train one cell and return (selected bake, dev curve, selection record) under EPOCH_RULE.

    best_dev: the trainer's own best-validation bake (refit/best.bin); the recorded epoch is the argmax of the log's
    4-decimal curve, which can differ from the trainer's full-precision pick on ties (label only; the bake is the
    trainer's). last: the trainer also dumps the final epoch's weights (--dump-checkpoints-every EPOCHS-1 fires at epoch 0
    and EPOCHS-1) and that checkpoint is the selected bake (refit/last.bin)."""
    (dest / "refit").mkdir(exist_ok=True)
    cmd = train_command(groups, init_seed, sample_seed, width, keep_file, head, dest / "refit" / "best.bin")
    ckpt = dest / "ckpt"
    if EPOCH_RULE == "last":
        ckpt.mkdir(exist_ok=True)
        cmd += ["--dump-checkpoints-every", str(EPOCHS - 1), "--dump-checkpoints-dir", str(ckpt)]
    elif EPOCH_RULE != "best_dev":
        raise ValueError(f"unknown EPOCH_RULE {EPOCH_RULE!r}")
    run(cmd, dest / "train.log")
    curve = read_curve(dest / "train.log")
    best = max(curve, key=curve.get)
    if EPOCH_RULE == "last":
        final = ckpt / f"ckpt_epoch{EPOCHS - 1:03d}.bin"
        if not final.is_file():
            raise ValueError(f"{final}: final-epoch checkpoint missing")
        bake = dest / "refit" / "last.bin"
        shutil.copyfile(final, bake)
        shutil.rmtree(ckpt)
        selected = EPOCHS - 1
    else:
        bake, selected = dest / "refit" / "best.bin", best
    return bake, curve, {"epoch_rule": EPOCH_RULE, "selected_epoch": selected, "best_epoch_by_curve": best}


def read_curve(log: Path) -> dict[int, float]:
    curve = {int(e): float(v) for e, v in EPOCH_RE.findall(log.read_text())}
    expected = sorted(set(range(0, EPOCHS, LOG_EVERY)) | {EPOCHS - 1})
    if sorted(curve) != expected:
        raise ValueError(f"validation curve incomplete: {len(curve)} of {len(expected)} evaluated epochs")
    return curve


def main() -> None:
    os.environ.setdefault("ZEN_PANEL_BIN", str(PANEL))  # the fit-cell executor sets it; local runs may not
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--head", choices=HEADS, required=True)
    ap.add_argument("--heldout", choices=SOURCE_ORDER, required=True)
    ap.add_argument("--seed-index", type=int, choices=range(10), required=True)
    ap.add_argument("--root", help="instrument root (default: the Rev3 v2 root); read by v2_common from argv")
    args = ap.parse_args()
    parse_spec(args.spec)
    core_spec, human_w = split_weight(args.spec)
    lists = json.loads((V2 / "wide" / "keep_lists.json").read_text())
    if lists["schema"] != "rev4-featpot-v2-keeplists-v2":
        raise ValueError("keep-list schema mismatch")
    entry = lists["specs"][core_spec]
    family, variant, keep = entry["family"], entry["variant"], entry["keep"]
    vdir = V2 / "wide" / family / variant
    receipt_path = vdir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    # v2c (CANONTAB): the canon receipt's width may exceed 1825 when sidecar families are appended after f1824.
    width = receipt["width"]
    if (receipt["schema"] not in WIDE_SCHEMAS or receipt["family"] != family or receipt["variant"] != variant
            or width < WIDTH or (receipt["schema"] == WIDE_SCHEMAS[0] and width != WIDTH)):
        raise ValueError("wide receipt identity mismatch")
    # Two-part cell path under v2/cells (the fit-cell executor's destination contract).
    dest = V2 / "cells" / f"{args.spec}__{args.head}" / f"without_{args.heldout}_s{args.seed_index}"
    if (dest / "result.json").is_file():
        print(json.dumps({"skip": str(dest / "result.json")}))
        return
    dest.mkdir(parents=True, exist_ok=True)
    keep_file = dest / "keep_features.txt"
    keep_file.write_text("\n".join(map(str, keep)) + "\n")
    legs = receipt["legs"]
    groups = []  # (name, path, train_weight, val_weight, mode)
    weights = {}
    for leg, (_, _, val_w) in TEACHERS.items():
        fit, dev = checked(legs[leg]["fit"]), checked(legs[leg]["dev"])
        weights[leg] = acceptance_weight(NOMINAL_WEIGHT[leg], refs_of(fit))
        groups += [(leg, fit, weights[leg], 0, "withinref,both"),
                   (f"{leg}_development", dev, 0, val_w, "withinref,both")]
    hfit = checked(legs[f"human_without_{args.heldout}"]["fit"])
    hdev = checked(legs[f"human_without_{args.heldout}"]["dev"])
    weights["human"] = acceptance_weight(NOMINAL_WEIGHT["human"] if human_w is None else human_w, refs_of(hfit))
    groups += [("human", hfit, weights["human"], 0, "withinref,rank"),
               ("human_development", hdev, 0, HUMAN_VAL_WEIGHT, "withinref,rank")]
    init_seed, sample_seed = seeds(args.heldout, args.seed_index)
    bake, curve, selection = train_and_select(groups, init_seed, sample_seed, width, keep_file, args.head, dest)
    best_epoch = selection["selected_epoch"]
    heldout = legs[args.heldout]
    table = checked(heldout["full"])
    pred = predict(bake, table, dest / "eval_preds.tsv")
    keys = pq.read_table(vdir / f"{args.heldout}.keys.parquet").to_pandas()
    if sha(vdir / f"{args.heldout}.keys.parquet") != heldout["keys_sha256"] or len(keys) != len(pred):
        raise ValueError("held-out keys changed or length mismatch")
    y = keys.target.to_numpy(dtype=np.float64)
    score = panel_batch([(args.heldout, pred, y)], stats="full")[0]
    out = {"schema": "rev4-featpot-v2-cell-v2", "label": "POTENTIAL — ceiling, not a model score",
           "recipe": "R915 sampling (amendment revision R1, layout R1.1)", "spec": args.spec,
           "human_nominal_weight": NOMINAL_WEIGHT["human"] if human_w is None else human_w,
           "family": family, "variant": variant,
           "kept_features": len(keep), "head": args.head, "heldout": args.heldout,
           "seed_index": args.seed_index, "init_seed": init_seed, "sample_seed": sample_seed,
           "train_weights": weights, "hidden": HIDDEN, "epochs": EPOCHS, "pairs_per_epoch": PAIRS_PER_EPOCH,
           "wide_receipt_sha256": sha(receipt_path), "table_receipt_sha256": sha(receipt_path),
           "keep_lists_sha256": sha(V2 / "wide" / "keep_lists.json"), "binaries": {p.name: sha(p) for p in (TRAINER, FITBIN, PANEL)},
           "dev_geomean3_by_epoch": curve, **selection,
           "selected_bake": str(bake), "selected_bake_sha256": sha(bake), "rows": len(y),
           "references": int(keys.ref_basename.nunique()), "keys_sha256": heldout["keys_sha256"],
           "prediction": pred.tolist(), "score": score}
    (dest / "result.json").write_text(json.dumps(out) + "\n")
    print(json.dumps({"result": str(dest / "result.json"), "best_epoch": best_epoch,
                      "srocc": score.get("srocc")}), flush=True)


if __name__ == "__main__":
    main()
