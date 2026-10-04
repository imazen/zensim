"""One full-data confirmatory fit cell (amendment R2): (arm spec, head, seed index) -> predictions, no labels.

Governing record: benchmarks/rev4_featpot_v2_amendment_2026-09-30.md, revision R2. Trains exactly as v2_lodo_mlp.py
(R915 teacher legs + the human leg, shared `train_command`; heads N and F; H = 32; 120 epochs x 50,000 pairs; best epoch
by the weighted dev geomean3) but the human leg is `human_all` (the union of the five exploratory sources; no source is
held out), then predicts every confirmatory feature table of the spec's family with the selected bake through
`bake_dial_refit predict --score-units`. `result.json` holds per-set predictions, the keys hashes they align to and the
receipts. It never reads a label: a confirmatory set has none in these tables.

  python v2_confirm_fit.py --spec c1 --head N --seed-index 0 [--root /var/tmp/canontab/v2c]

The null is MATCHED (coordinator decision 2026-10-01): a spec's variant selects its confirmatory tables, so `<arm>~pk` is
evaluated on the within-reference permuted confirmatory columns (built label-free by `v2c_wide.py confirm`), as v2 did.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from v2_common import (load_frozen, recipe_of, EPOCHS, FITBIN, HEADS, HIDDEN, HUMAN_VAL_WEIGHT, NOMINAL_WEIGHT, PANEL, PAIRS_PER_EPOCH, TEACHERS,
                       TRAINER, V2, WIDTH, acceptance_weight, confirm_seeds, parse_spec, sha, split_weight, table_path)
from v2_lodo_mlp import WIDE_SCHEMAS, checked, predict, refs_of, resolve_keep, train_and_select

CONFIRM_SCHEMA = "rev4-featpot-v2c-confirm-v1"
RESULT_SCHEMA = "rev4-featpot-v2c-confirm-cell-v1"


def confirm_tables(family: str, variant: str, receipt: dict) -> dict[str, dict]:
    """{set: table record} for the spec's family; the table record carries its path, hashes and keys hash."""
    want = variant
    out = {}
    for name, rec in receipt["sets"].items():
        by_variant = rec["tables"].get(family, {})
        if want not in by_variant:
            raise ValueError(f"{name}: no {family}/{want} confirmatory table (build it with v2c_wide.py confirm)")
        out[name] = by_variant[want]
    return out


def cell_dir(spec: str, head: str, seed_index: int) -> Path:
    return V2 / "confirm" / "cells" / f"{spec}__{head}" / f"full_s{seed_index}"


def main() -> None:
    os.environ.setdefault("ZEN_PANEL_BIN", str(PANEL))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--head", choices=HEADS, required=True)
    ap.add_argument("--seed-index", type=int, choices=range(10), required=True)
    ap.add_argument("--dest", type=Path, help="write here instead of <root>/confirm/cells/... (determinism checks)")
    ap.add_argument("--root", help="instrument root (default: the Rev3 v2 root); read by v2_common from argv")
    ap.add_argument("--columns", help="comma-separated sorted wide columns of a sel:<id> spec (as v2_lodo_mlp)")
    args = ap.parse_args()
    parse_spec(args.spec)
    core_spec, human_w = split_weight(args.spec)
    lists = json.loads((V2 / "wide" / "keep_lists.json").read_text())
    if lists["schema"] != "rev4-featpot-v2-keeplists-v2":
        raise ValueError("keep-list schema mismatch")
    # 2026-10-04 (set-compare confirmatory read): registered arms, E9 block/set specs and sel: subsets resolve exactly as in the
    # LODO cells (v2_lodo_mlp.resolve_keep); a registered arm's keep list is unchanged.
    family, variant, keep = resolve_keep(core_spec, args.columns, lists)
    vdir = V2 / "wide" / family / variant
    receipt_path = vdir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    width = receipt["width"]
    if (receipt["schema"] != WIDE_SCHEMAS[1] or receipt["family"] != family or receipt["variant"] != variant
            or width < WIDTH or not receipt.get("complete")):
        raise ValueError("wide receipt identity mismatch or incomplete (confirmatory fits need the canon tables, all legs)")
    frozen, frozen_sha = load_frozen(V2)  # refuses an unfrozen root or any receipt changed since the freeze
    if frozen["wide_receipts"].get(f"{family}/{variant}") != sha(receipt_path):
        raise ValueError("wide receipt is not the frozen one")
    confirm_path = V2 / "wide" / "confirm" / "receipt.json"
    confirm = json.loads(confirm_path.read_text())
    if confirm["schema"] != CONFIRM_SCHEMA or confirm["width"] != width or confirm["feature_set_id"] != receipt["feature_set_id"]:
        raise ValueError("confirmatory receipt does not match the wide receipt (schema, width or feature_set_id)")
    tables = confirm_tables(family, variant, confirm)
    dest = args.dest or cell_dir(args.spec, args.head, args.seed_index)
    if (dest / "result.json").is_file():
        print(json.dumps({"skip": str(dest / "result.json")}))
        return
    dest.mkdir(parents=True, exist_ok=True)
    keep_file = dest / "keep_features.txt"
    keep_file.write_text("\n".join(map(str, keep)) + "\n")
    legs = receipt["legs"]
    groups, weights = [], {}
    for leg, (_, _, val_w) in TEACHERS.items():
        fit, dev = checked(legs[leg]["fit"]), checked(legs[leg]["dev"])
        weights[leg] = acceptance_weight(NOMINAL_WEIGHT[leg], refs_of(fit))
        groups += [(leg, fit, weights[leg], 0, "withinref,both"), (f"{leg}_development", dev, 0, val_w, "withinref,both")]
    recipe = recipe_of(args.spec)
    coverage_record, curated_extra = None, None
    if "coverage_weight" in recipe:  # design log E15/E17: the coverage leg exactly as v2_lodo_mlp builds it (label-free ordinal pool)
        import v2_teacher
        if max(keep) >= v2_teacher.ORDINAL_WIDTH:
            raise ValueError(f"{core_spec}: the coverage pool has no f{v2_teacher.ORDINAL_WIDTH}+ (NaN); refusing this keep list")
        cpath, coverage_record = v2_teacher.coverage_leg(recipe["coverage_mask"], Path(os.environ.get("TMPDIR") or dest))
        curated_extra = cpath
        weights["coverage"] = acceptance_weight(recipe["coverage_weight"], refs_of(cpath))
        groups.append(("coverage", cpath, weights["coverage"], 0, "withinref,rank"))
    if "kadis_ordinal" in recipe or recipe.get("teacher_subset"):
        raise ValueError(f"{args.spec}: the confirm fitter implements the coverage leg only (no ko/ts recipe tokens)")
    hfit, hdev = checked(legs["human_all"]["fit"]), checked(legs["human_all"]["dev"])
    weights["human"] = acceptance_weight(NOMINAL_WEIGHT["human"] if human_w is None else human_w, refs_of(hfit))
    groups += [("human", hfit, weights["human"], 0, "withinref,rank"),
               ("human_development", hdev, 0, HUMAN_VAL_WEIGHT, "withinref,rank")]
    init_seed, sample_seed = confirm_seeds(args.seed_index)
    try:
        bake, curve, selection = train_and_select(groups, init_seed, sample_seed, width, keep_file, args.head, dest, recipe)
    finally:
        if curated_extra is not None:
            curated_extra.unlink(missing_ok=True)
    best_epoch = selection["selected_epoch"]
    predictions = {}
    for name, table in tables.items():
        path = table_path(table)
        if sha(path) != table["sha256"] or sha(Path(f"{path}.manifest.json")) != table["manifest_sha256"]:
            raise ValueError(f"{name}: confirmatory table changed after its receipt")
        pred = predict(bake, path, dest / f"preds_{name}.tsv")
        rows = confirm["sets"][name]["rows"]
        if len(pred) != rows:
            raise ValueError(f"{name}: {len(pred)} predictions for {rows} rows")
        predictions[name] = {"rows": rows, "keys_sha256": table["keys_sha256"], "table_sha256": table["sha256"],
                             "pred": pred.tolist()}
    out = {"schema": RESULT_SCHEMA, "label": "POTENTIAL — ceiling, not a model score; features only, no label read",
           "recipe": "R915 sampling (amendment R1/R1.1) + full-data human_all leg (amendment R2)", "spec": args.spec,
           "human_nominal_weight": NOMINAL_WEIGHT["human"] if human_w is None else human_w, "family": family,
           "variant": variant, "eval_variant": variant, "kept_features": len(keep), "head": args.head,
           "seed_index": args.seed_index, "init_seed": init_seed, "sample_seed": sample_seed, "train_weights": weights,
           "hidden": recipe.get("hidden", HIDDEN), "epochs": EPOCHS, "recipe_tokens": recipe,
           **({"coverage_leg": coverage_record} if coverage_record else {}), "pairs_per_epoch": PAIRS_PER_EPOCH, "width": width,
           "wide_receipt_sha256": sha(receipt_path), "frozen_sha256": frozen_sha, "confirm_receipt_sha256": sha(confirm_path),
           "keep_lists_sha256": sha(V2 / "wide" / "keep_lists.json"),
           "binaries": {p.name: sha(p) for p in (TRAINER, FITBIN, PANEL)},
           "dev_geomean3_by_epoch": curve, **selection,
           "selected_bake": str(bake), "selected_bake_sha256": sha(bake),
           "predictions": predictions}
    (dest / "result.json").write_text(json.dumps(out) + "\n")
    print(json.dumps({"result": str(dest / "result.json"), "best_epoch": best_epoch,
                      "sets": {k: v["rows"] for k, v in predictions.items()}}), flush=True)


if __name__ == "__main__":
    main()
