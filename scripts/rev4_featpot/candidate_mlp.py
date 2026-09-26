"""One nested-CV H32/H128 MLP replicate for a pinned C1-C4 candidate arm.

Run under the shared heavy wrapper. Four inner reference folds choose a
checkpoint epoch; the outer-fit run trains on every allowed fit reference and
uses its dump at that epoch. All bakes and temporary label-bearing fit tables
stay under /var/tmp/rev4-featpot/candidates/mlp/POT_*. No board or auto-evaluation runs.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from linear_probe import BIN, ROOT, panel_batch, sha
import restore_data
from candidate_data import ARMS, columns, load


TRAINER = Path(os.environ.get("REV4_TRAINER_BIN", str(ROOT / "target/debug/zensim_mlp_train")))
FEATURES = []
CANONICAL_IDS = []
INIT_SEEDS = (1101, 1103, 1107, 1109, 1117)
SAMPLE_SEEDS = (101, 100000101, 200000101, 300000101, 400000101)
EPOCHS = 60
PAIRS_PER_EPOCH = 50000
LOG_EVERY = 5
VAL_RE = re.compile(r"epoch\s+(\d+)\s+\|.*?val\(geomean3\)=([+-]?\d+\.\d+)")


def run(cmd: list[str], log: Path) -> None:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text("$ " + " ".join(cmd) + "\n" + proc.stdout)
    if proc.returncode:
        raise RuntimeError(f"{cmd[0]} rc={proc.returncode}; see {log}")


def table(data: pd.DataFrame, indices: np.ndarray, bounds: tuple[float, float],
          path: Path) -> None:
    lo, hi = bounds
    target = data.loc[indices, "target"].to_numpy(dtype=np.float64)
    view = data.loc[indices, ["ref_basename"] + FEATURES].copy()
    view.insert(1, "human_score", np.clip((target - lo) / (hi - lo), 0.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(view, preserve_index=False), path, compression="zstd")
    # This packed diagnostic composite has no production feature-set ID.
    # Canonical candidate IDs remain in the receipt and are never shipped.
    Path(f"{path}.manifest.json").write_text(json.dumps({
        "source_bank_feature_set_id": "basic+peaks+masked+iw+v2+append+append2@w944/ceiling_rev3#b782e349",
        "composite": "C1-C4 POTENTIAL diagnostic: bank f0-f943 plus packed candidate IDs",
        "canonical_candidate_ids": CANONICAL_IDS,
        "packed_candidate_ids": list(range(944, len(FEATURES))),
        "formula_revision": 3,
    }) + "\n")


def fit_bounds(data: pd.DataFrame, indices: np.ndarray) -> tuple[float, float]:
    values = data.loc[indices, "target"].to_numpy(dtype=np.float64)
    lo, hi = np.quantile(values, [0.001, 0.999])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError(f"bad target bounds {lo}, {hi}")
    return float(lo), float(hi)


def train(train_path: Path, val_path: Path | None, dest: Path,
          hidden: int, arm: str, init_seed: int, sample_seed: int,
          slice_file: Path, dump: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [str(TRAINER), "--group", f"fit:{train_path}:1:0"]
    if val_path is not None:
        cmd += ["--group", f"inner:{val_path}:0:1"]
    cmd += ["--target-column", "human_score", "--target-scale", "100",
            "--max-features", str(len(FEATURES)), "--hidden", str(hidden),
            "--epochs", str(EPOCHS), "--pairs-per-epoch", str(PAIRS_PER_EPOCH),
            "--log-every", str(LOG_EVERY), "--early-stop-patience", "0",
            "--val-aggregate", "geomean3", "--nonneg-distance",
            "--init-seed", str(init_seed), "--sample-seed", str(sample_seed),
            "--historical-replay", "C1-C4 candidate diagnostic: pinned Rev3 944 plus pinned Part B sidecar; never ship",
            "--no-auto-eval", "--out", str(dest / "best.bin")]
    if restore_data.is_a1w(arm):
        cmd += ["--keep-features", str(slice_file)]  # A1w: population drop list removed
    if dump:
        cmd += ["--dump-checkpoints-every", str(LOG_EVERY),
                "--dump-checkpoints-dir", str(dest)]
    run(cmd, dest / "train.log")


def validation_scores(log: Path) -> dict[int, float]:
    values = {int(epoch): float(score) for epoch, score in VAL_RE.findall(log.read_text())}
    expected = set(range(0, EPOCHS, LOG_EVERY))
    if not expected.issubset(values):
        raise ValueError(f"{log}: missing checkpoint validation epochs {expected - values.keys()}")
    return {epoch: values[epoch] for epoch in sorted(expected)}


def predict(bake: Path, table_path: Path, output: Path) -> np.ndarray:
    run([str(BIN), "predict", "--bake", str(bake), "--corpus", str(table_path),
         "--score-units", "--out", str(output)], output.with_suffix(".log"))
    result = pd.read_csv(output, sep="\t")
    if result.columns.tolist() != ["row_idx", "pred"] or not np.array_equal(
        result.row_idx.to_numpy(), np.arange(len(result))
    ):
        raise ValueError(f"{output}: unexpected prediction row order")
    values = result.pred.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{output}: nonfinite prediction")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--arm", choices=[*ARMS, *(f"{name}_perm" for name in ARMS)], required=True)
    ap.add_argument("--hidden", type=int, choices=[32, 128], required=True)
    ap.add_argument("--outer", type=int, choices=range(5), help="omit for full in-sample fit")
    ap.add_argument("--rep", type=int, choices=range(5), required=True)
    args = ap.parse_args()
    if not TRAINER.is_file() or not BIN.is_file():
        raise FileNotFoundError("build the Rust trainer and predictor through heavy first")
    preflight = json.loads((ROOT / "sampler_preflight.json").read_text())
    if len(preflight) != 5 or [row["seed"] for row in preflight] != list(SAMPLE_SEEDS):
        raise ValueError("sample seeds were not preflighted")
    if any(row["epochs"] != EPOCHS or row["pairs_per_epoch"] != PAIRS_PER_EPOCH
           or row["disjoint_sampler_window_words"] != 4 * EPOCHS * PAIRS_PER_EPOCH
           for row in preflight):
        raise ValueError("sample window preflight does not cover this training budget")
    if (ROOT / "folds.json").is_file() is False:
        raise FileNotFoundError("reference-grouped folds missing")
    folds = json.loads((ROOT / "folds.json").read_text())["sets"][args.set]
    global FEATURES, CANONICAL_IDS
    FEATURES, CANONICAL_IDS = columns(args.arm)
    data, label_meta = load(args.set, args.arm)
    refs = data.ref_basename.astype(str).to_numpy()
    if args.outer is None:
        fit_refs = folds["full_refs"]
        inner_folds = folds["full_inner_val_refs"]
        test_idx = np.arange(len(data))
        tag = "full"
        sample_seed = SAMPLE_SEEDS[args.rep]
    else:
        outer = folds["outer"][args.outer]
        fit_refs = sorted(set(folds["full_refs"]) - set(outer["test_refs"]))
        inner_folds = outer["inner_val_refs"]
        test_idx = np.flatnonzero(np.isin(refs, outer["test_refs"]))
        tag = f"o{args.outer}"
        sample_seed = SAMPLE_SEEDS[(args.rep + args.outer) % 5]
    init_seed = INIT_SEEDS[args.rep]
    dest = ROOT / "candidates" / "mlp" / f"POT_{args.set}_{args.arm}_mlp{args.hidden}" / f"{tag}_r{args.rep}"
    dest.mkdir(parents=True, exist_ok=True)
    slice_file = (restore_data.a1w_slice(f"D1/{args.set}/{tag}", len(FEATURES))
                  if restore_data.is_a1w(args.arm) else dest / "unused_slice.txt")
    inner_scores = []
    inner_receipts = []
    for inner_i, val_refs in enumerate(inner_folds):
        train_refs = sorted(set(fit_refs) - set(val_refs))
        train_idx = np.flatnonzero(np.isin(refs, train_refs))
        val_idx = np.flatnonzero(np.isin(refs, val_refs))
        stem = dest / f"inner{inner_i}"
        bounds = fit_bounds(data, train_idx)
        train_path, val_path = stem / "fit.parquet", stem / "val.parquet"
        table(data, train_idx, bounds, train_path)
        table(data, val_idx, bounds, val_path)
        train(train_path, val_path, stem, args.hidden, args.arm,
              init_seed, sample_seed, slice_file, dump=False)
        scores = validation_scores(stem / "train.log")
        inner_scores.append(scores)
        inner_receipts.append({"fit_rows": len(train_idx), "val_rows": len(val_idx),
                               "bounds": bounds, "log_sha256": sha(stem / "train.log"),
                               "validation_geomean3_by_epoch": scores})
        print(json.dumps({"inner": inner_i, "fit_rows": len(train_idx),
                          "val_rows": len(val_idx), "best_log_epoch": max(scores, key=scores.get)}),
              flush=True)
    mean_by_epoch = {epoch: float(np.mean([scores[epoch] for scores in inner_scores]))
                     for epoch in inner_scores[0]}
    selected_epoch = max(mean_by_epoch, key=mean_by_epoch.get)
    fit_idx = np.flatnonzero(np.isin(refs, fit_refs))
    bounds = fit_bounds(data, fit_idx)
    final = dest / "refit"
    fit_path, test_path = final / "fit.parquet", final / "test.parquet"
    table(data, fit_idx, bounds, fit_path)
    table(data, test_idx, bounds, test_path)
    train(fit_path, None, final, args.hidden, args.arm,
          init_seed, sample_seed, slice_file, dump=True)
    selected_bake = final / f"ckpt_epoch{selected_epoch:03}.bin"
    if not selected_bake.is_file():
        raise FileNotFoundError(f"inner-selected checkpoint {selected_bake} missing")
    pred = predict(selected_bake, test_path, final / "test_preds.tsv")
    y = data.loc[test_idx, "target"].to_numpy(dtype=np.float64)
    score = panel_batch([(tag, pred, y)], stats="full")[0]
    result = {"schema": "rev4-featpot-c1c4-mlp-replicate-v1", "label": "POTENTIAL — ceiling, not a model score",
              "set": args.set, "arm": args.arm, "hidden": args.hidden,
              "canonical_candidate_ids": CANONICAL_IDS,
              "outer": args.outer, "rep": args.rep, "init_seed": init_seed,
              "sample_seed": sample_seed, "inner": inner_receipts,
              "inner_mean_geomean3_by_epoch": mean_by_epoch,
              "selected_epoch": selected_epoch, "fit_rows": len(fit_idx),
              "test_rows": len(test_idx), "test_index": test_idx.tolist(),
              "bounds": bounds, "score": score, "prediction": pred.tolist(),
              "label_source": label_meta,
              "selected_bake": str(selected_bake), "selected_bake_sha256": sha(selected_bake),
              "preflight_sha256": sha(ROOT / "sampler_preflight.json")}
    output = dest / "result.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(output), "sha256": sha(output),
                      "selected_epoch": selected_epoch, "score": score}), flush=True)


if __name__ == "__main__":
    main()
