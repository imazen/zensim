"""R0 LODO JPEG-vs-other cross-format ordering contrast.

Use only E1b-classifiable codec sets and the already admitted bank-label
adapter. Scores and bootstrap accuracies come from panel --pairwise. This is a
descriptive transfer contrast; peer-relative JPEG residuals remain separate.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data import load
from enrich_stats import pairwise_owner
from linear_probe import ROOT, sha
from lodo_bvls import EVAL, SOURCES


FORMATS = {
    "cid22_a25": {
        "aom": "AVIF", "cld_avif": "AVIF", "vis_avif": "AVIF",
        "cld_heic": "HEIC", "cld_jp2": "JP2", "cld_webp": "WebP",
        "libjxl": "JXL", "mozjpeg": "JPEG",
    },
    "aic3": {
        "avif": "AVIF", "hm": "HEVC", "jpeg-1": "JPEG",
        "jpeg-2000": "JP2", "jpegxl": "JXL", "vvc": "VVC",
    },
    "tid2013": {"tid_10": "JPEG", "tid_11": "JP2"},
}


def owner_draws(path: Path, b: int) -> dict[int, float]:
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        values = {row["label"]: float(row["acc_response"]) for row in rows}
    return {i: values[f"B{i}"] for i in range(b) if f"B{i}" in values}


def contrast(heldout: str, fold: dict, model_root: Path) -> dict:
    eval_set = EVAL.get(heldout, heldout)
    if eval_set not in FORMATS:
        return {"status": "NOT_CLASSIFIABLE", "eval_set": eval_set,
                "reason": "E1b has no JPEG/other codec-format classifier for this view"}
    data, label_meta = load(eval_set, features=False, with_codec=True)
    pred = np.asarray(fold["prediction"], dtype=np.float64)
    if len(pred) != len(data) or not np.isfinite(pred).all():
        raise ValueError(f"{heldout}: prediction coverage/nonfinite mismatch")
    if fold["eval_set"] != eval_set or fold["rows"] != len(data):
        raise ValueError(f"{heldout}: LODO evaluation receipt changed")
    mapping = FORMATS[eval_set]
    unknown = set(data.codec.astype(str)) - set(mapping)
    if eval_set in ("cid22_a25", "aic3") and unknown:
        raise ValueError(f"{heldout}: unclassified codecs {sorted(unknown)}")
    y = data.target.to_numpy(dtype=np.float64)
    refs = data.ref_basename.astype(str).to_numpy()
    formats = [mapping.get(str(codec)) for codec in data.codec]
    by_ref: dict[str, list[int]] = defaultdict(list)
    for i, ref in enumerate(refs):
        if formats[i] is not None:
            by_ref[ref].append(i)
    classified: dict[str, list[tuple[str, float, float, str]]] = {
        "jpeg_cross": [], "other_cross": []}
    ties = {key: 0 for key in classified}
    for ref, indices in sorted(by_ref.items()):
        for pos, i in enumerate(indices):
            for j in indices[pos + 1:]:
                if formats[i] == formats[j]:
                    continue
                category = "jpeg_cross" if "JPEG" in (formats[i], formats[j]) else "other_cross"
                if y[i] == y[j]:
                    ties[category] += 1
                    continue
                choice = "left" if y[i] < y[j] else "right"
                classified[category].append((ref, float(pred[i]), float(pred[j]), choice))
    result = {"status": "MISSING_COMPARATOR", "eval_set": eval_set,
              "label_source": label_meta, "rows": len(data),
              "references": len(by_ref), "format_map": mapping,
              "pairs": {k: len(v) for k, v in classified.items()},
              "target_ties_dropped": ties}
    if not all(classified.values()):
        result["reason"] = "JPEG and non-JPEG cross-format pair strata both required"
        return result
    unique_refs = sorted(set(refs))
    rng = np.random.default_rng(20260923)
    draws = [[unique_refs[i] for i in rng.integers(0, len(unique_refs), len(unique_refs))]
             for _ in range(2000)]
    records = {}
    boot = {}
    for category, pairs in classified.items():
        stem = model_root / f"{heldout}_{category}"
        record = pairwise_owner(stem, pairs, draws)
        if record["status"] != "MEASURED":
            raise ValueError(f"{heldout}/{category}: owner did not score")
        records[category] = record
        boot[category] = owner_draws(Path(record["panel_output"]), len(draws))
    shared = sorted(set(boot["jpeg_cross"]) & set(boot["other_cross"]))
    delta = np.asarray([boot["jpeg_cross"][i] - boot["other_cross"][i]
                        for i in shared], dtype=np.float64)
    finite = delta[np.isfinite(delta)]
    if len(finite) < 1900:
        raise ValueError(f"{heldout}: only {len(finite)} finite paired bootstrap draws")
    result.update({"status": "MEASURED", "B": 2000, "seed": 20260923,
                   "unit": "reference", "owner": "panel --pairwise",
                   "jpeg_accuracy": records["jpeg_cross"]["point"],
                   "other_accuracy": records["other_cross"]["point"],
                   "jpeg_minus_other": (records["jpeg_cross"]["point"]
                                        - records["other_cross"]["point"]),
                   "ci95_jpeg_minus_other": np.quantile(finite, [0.025, 0.975]).tolist(),
                   "bootstrap_finite": len(finite), "owner_receipts": records})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("bvls", "linear"), required=True)
    args = parser.parse_args()
    model_root = ROOT / "lodo" / f"LODO_r0_{args.model}"
    source = model_root / "result.json"
    value = json.loads(source.read_text())
    if set(value["folds"]) != set(SOURCES):
        raise ValueError("D2 LODO fold list changed")
    output_root = ROOT / "lodo_jpeg" / f"LODO_r0_{args.model}"
    output_root.mkdir(parents=True, exist_ok=True)
    result = {"schema": "rev4-featpot-lodo-jpeg-cross-contrast-v1",
              "label": "POTENTIAL — ceiling, not a model score",
              "model": args.model, "source_result_sha256": sha(source),
              "definition": "JPEG-involving minus non-JPEG cross-format within-reference ordering accuracy",
              "scope": "E1b codec-format classifier; descriptive, not peer-relative",
              "folds": {}}
    for heldout in SOURCES:
        item = contrast(heldout, value["folds"][heldout], output_root)
        result["folds"][heldout] = item
        print(json.dumps({"heldout": heldout, "status": item["status"],
                          "pairs": item.get("pairs"),
                          "delta": item.get("jpeg_minus_other")}), flush=True)
    output = output_root / "result.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha(output)}), flush=True)


if __name__ == "__main__":
    main()
