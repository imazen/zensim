"""Seven-fold D2 BVLS LODO POTENTIAL transfer, equal source-set weights.

Run under the shared heavy wrapper. Each source set's q0.001/q0.999 target
transform is fitted on that source's TRAIN rows only. KonFiG and BPG VAL are
reference-disjoint evaluation views; BPG's target is an SSIM2 oracle /100.
"""

import argparse
import json

import numpy as np

from data import load
from linear_probe import BIN, FEATURES, REPO, ROOT, fold_table, gram, panel_batch, run, sha


SOURCES = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
           "cid22_a25", "aic3", "kadid_select")
EVAL = {"konfig_train": "konfig_val", "konjnd_bpg_train": "konjnd_bpg_val"}


def label_for(name: str) -> str:
    return "ssim2_oracle" if name.startswith("konjnd_bpg") else "human_score"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["r0", "minus_basic"], required=True)
    args = ap.parse_args()
    if not BIN.is_file():
        raise FileNotFoundError(f"build {BIN} through the heavy runner first")
    root = ROOT / "lodo" / f"LODO_{args.arm}_bvls"
    root.mkdir(parents=True, exist_ok=True)
    slice_file = root / "keep_f228_to_f943.txt"
    slice_file.write_text("\n".join(map(str, range(228, 944))) + "\n")
    source_grams = {}
    normalization = {}
    for name in SOURCES:
        data, label_meta = load(name)
        target = label_meta["target"]
        table = root / f"source_{name}.parquet"
        bounds = fold_table(data, np.arange(len(data)), table)
        g = root / f"source_{name}.npz"
        gram(table, g, per_ref=True)
        source_grams[name] = {"path": str(g), "rows": len(data), "sha256": sha(g)}
        normalization[name] = {"raw_target": target, "scale": "ssim2/100" if target == "ssim2_oracle" else "native human quality",
                               "q001": bounds[0], "q999": bounds[1]}
        print(json.dumps({"source": name, "rows": len(data), "bounds": bounds,
                          "gram_sha256": sha(g)}), flush=True)

    results = {"schema": "rev4-featpot-lodo-bvls-v1", "arm": args.arm,
               "label": "POTENTIAL — ceiling, not a model score",
               "source_grams": source_grams, "normalization": normalization, "folds": {}}
    for heldout in SOURCES:
        eval_name = EVAL.get(heldout, heldout)
        target = label_for(eval_name)
        test, eval_label_meta = load(eval_name)
        x = test[FEATURES].to_numpy(dtype=np.float64, copy=False)
        y = test["target"].to_numpy(dtype=np.float64)
        cmd = [str(BIN), "fit-lasso", "--space", "raw", "--target", "target__mm01",
               "--solver", "bvls", "--lam", "0",
               "--bounds-tsv", str(REPO / "benchmarks/feature_sign_mask_2026-05-26.tsv")]
        included = [name for name in SOURCES if name != heldout]
        for name in included:
            cmd += ["--gram", source_grams[name]["path"]]
        for name in included:
            cmd += ["--weight", repr(1.0 / source_grams[name]["rows"])]
        out = root / f"fit_without_{heldout}.npz"
        cmd += ["--emit-fit-npz", str(out), "--diagnostic-fit-only",
                "--out", str(out.with_suffix(".unused.bin"))]
        if args.arm == "minus_basic":
            cmd += ["--slice-file", str(slice_file)]
        run(cmd, out.with_suffix(".log"))
        with np.load(out) as fit:
            pred = (x - fit["mu"]) / fit["sd"] @ fit["w"] + fit["bias"].item()
        score = panel_batch([(heldout, pred, y)], stats="full")[0]
        results["folds"][heldout] = {"eval_set": eval_name, "eval_target": target,
                                     "label_source": eval_label_meta,
                                     "eval_scale": "ssim2/100" if target == "ssim2_oracle" else "native human quality",
                                     "rows": len(test), "references": test["ref_basename"].nunique(),
                                     "source_sets": included, "score": score,
                                     "prediction": pred.tolist(), "fit_sha256": sha(out)}
        print(json.dumps({"heldout": heldout, "eval": eval_name, "rows": len(test),
                          "references": test["ref_basename"].nunique(),
                          "srocc": score["srocc"], "fit_sha256": sha(out)}), flush=True)
    output = root / "result.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"result": str(output), "sha256": sha(output)}), flush=True)


if __name__ == "__main__":
    main()
