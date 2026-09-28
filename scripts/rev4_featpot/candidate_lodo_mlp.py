"""One D2 source-held-out MLP replicate for C1-C4 or matched permutation."""

import argparse
import json
from pathlib import Path

import numpy as np

import restore_data
from candidate_data import ARMS, ROOT, columns, load, sha
from lodo_bvls import EVAL, SOURCES
from mlp_probe import INIT_SEEDS, SAMPLE_SEEDS, EPOCHS, PAIRS_PER_EPOCH, LOG_EVERY
from candidate_mlp import TRAINER, predict, run, validation_scores
from linear_probe import BIN, panel_batch


def table_receipt(arm):
    path = ROOT / "candidates/d2_mlp/tables" / arm / "receipt.json"
    value = json.loads(path.read_text())
    if value["schema"] != "rev4-featpot-c1c4-d2-mlp-tables-v1" or value["arm"] != arm:
        raise ValueError("D2 MLP table receipt identity mismatch")
    return value, sha(path)


def check_table(record, kind):
    item = record["tables"][kind]
    path = Path(item["path"])
    if sha(path) != item["sha256"] or sha(Path(f"{path}.manifest.json")) != item["manifest_sha256"]:
        raise ValueError(f"{path}: table or provenance sidecar hash changed")
    return path


def train(arm, hidden, source_paths, validation, dest, init_seed, sample_seed, dump, keep_file=None):
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [str(TRAINER)]
    for name, path in source_paths:
        cmd += ["--group", f"{name}:{path}:1:0"]
    if validation is not None:
        name, path = validation
        cmd += ["--group", f"inner_{name}:{path}:0:1"]
    cmd += ["--target-column", "human_score", "--target-scale", "100",
            "--max-features", str(len(columns(arm)[0])),
            "--hidden", str(hidden), "--epochs", str(EPOCHS),
            "--pairs-per-epoch", str(PAIRS_PER_EPOCH), "--log-every", str(LOG_EVERY),
            "--early-stop-patience", "0", "--val-aggregate", "geomean3",
            "--nonneg-distance", "--init-seed", str(init_seed),
            "--sample-seed", str(sample_seed), "--no-auto-eval",
            "--out", str(dest / "best.bin")]
    cmd += ["--historical-replay",
            "C1-C4 D2 candidate diagnostic: pinned Rev3 944 plus pinned Part B sidecar; never ship"]
    if keep_file is not None:
        cmd += ["--keep-features", str(keep_file)]  # A1w: population drop list removed
    if dump:
        cmd += ["--dump-checkpoints-every", str(LOG_EVERY),
                "--dump-checkpoints-dir", str(dest)]
    run(cmd, dest / "train.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=[*ARMS, *(f"{name}_perm" for name in ARMS)], required=True)
    parser.add_argument("--hidden", type=int, choices=[32, 128], required=True)
    parser.add_argument("--heldout", choices=SOURCES, required=True)
    parser.add_argument("--rep", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    if not TRAINER.is_file() or not BIN.is_file():
        raise FileNotFoundError("build the Rust trainer and predictor through heavy first")
    preflight = json.loads((ROOT / "sampler_preflight.json").read_text())
    if len(preflight) != 5 or [row["seed"] for row in preflight] != list(SAMPLE_SEEDS):
        raise ValueError("sample seeds were not preflighted")
    if any(row["epochs"] != EPOCHS or row["pairs_per_epoch"] != PAIRS_PER_EPOCH
           or row["disjoint_sampler_window_words"] != 4 * EPOCHS * PAIRS_PER_EPOCH
           for row in preflight):
        raise ValueError("sample window preflight does not cover this training budget")
    receipt, receipt_sha = table_receipt(args.arm)
    included = [name for name in SOURCES if name != args.heldout]
    eval_name = EVAL.get(args.heldout, args.heldout)
    paths = {name: check_table(receipt["tables"][name], "fit") for name in included}
    eval_path = check_table(receipt["tables"][eval_name], "eval")
    fold_i = SOURCES.index(args.heldout)
    keep_file = (restore_data.a1w_slice(f"D2/without_{args.heldout}", len(columns(args.arm)[0]))
                 if restore_data.is_a1w(args.arm) else None)
    init_seed = INIT_SEEDS[args.rep]
    sample_seed = SAMPLE_SEEDS[(args.rep + fold_i) % 5]
    dest = ROOT / "candidates/d2_mlp" / f"LODO_{args.arm}_mlp{args.hidden}" / f"without_{args.heldout}_r{args.rep}"
    dest.mkdir(parents=True, exist_ok=True)
    inner_scores = []
    inner_receipts = []
    for validation in included:
        training = [(name, paths[name]) for name in included if name != validation]
        inner = dest / f"inner_{validation}"
        train(args.arm, args.hidden, training, (validation, paths[validation]),
              inner, init_seed, sample_seed, dump=False, keep_file=keep_file)
        scores = validation_scores(inner / "train.log")
        inner_scores.append(scores)
        inner_receipts.append({"validation_source": validation,
                               "fit_sources": [name for name, _ in training],
                               "validation_geomean3_by_epoch": scores,
                               "log_sha256": sha(inner / "train.log")})
        print(json.dumps({"inner_source": validation,
                          "best_epoch": max(scores, key=scores.get)}), flush=True)
    mean_by_epoch = {epoch: float(np.mean([scores[epoch] for scores in inner_scores]))
                     for epoch in inner_scores[0]}
    selected_epoch = max(mean_by_epoch, key=mean_by_epoch.get)
    final = dest / "refit"
    train(args.arm, args.hidden, [(name, paths[name]) for name in included],
          None, final, init_seed, sample_seed, dump=True, keep_file=keep_file)
    bake = final / f"ckpt_epoch{selected_epoch:03}.bin"
    if not bake.is_file():
        raise FileNotFoundError(f"inner-selected checkpoint {bake} missing")
    pred = predict(bake, eval_path, final / "eval_preds.tsv")
    data, label_meta = load(eval_name, args.arm, features=False)
    y = data.target.to_numpy(dtype=np.float64)
    if len(pred) != len(y):
        raise ValueError("D2 MLP prediction/label row mismatch")
    score = panel_batch([(args.heldout, pred, y)], stats="full")[0]
    output = {"schema": "rev4-featpot-c1c4-d2-mlp-replicate-v1", "label": "POTENTIAL — ceiling, not a model score",
              "arm": args.arm, "hidden": args.hidden, "heldout": args.heldout,
              "canonical_candidate_ids": columns(args.arm)[1],
              "eval_set": eval_name, "rep": args.rep, "init_seed": init_seed,
              "sample_seed": sample_seed, "fit_sources": included,
              "equal_train_weight_per_source": 1.0,
              "table_receipt_sha256": receipt_sha,
              "source_bounds": {name: receipt["tables"][name]["bounds"] for name in included},
              "inner": inner_receipts, "inner_mean_geomean3_by_epoch": mean_by_epoch,
              "selected_epoch": selected_epoch,
              "selected_bake": str(bake), "selected_bake_sha256": sha(bake),
              "preflight_sha256": sha(ROOT / "sampler_preflight.json"),
              "rows": len(y), "references": data.ref_basename.nunique(),
              "label_source": label_meta, "prediction": pred.tolist(), "score": score}
    result = dest / "result.json"
    result.write_text(json.dumps(output) + "\n")
    print(json.dumps({"result": str(result), "sha256": sha(result),
                      "heldout": args.heldout, "eval_set": eval_name,
                      "rows": len(y), "selected_epoch": selected_epoch,
                      "srocc": score["srocc"]}), flush=True)


if __name__ == "__main__":
    main()
