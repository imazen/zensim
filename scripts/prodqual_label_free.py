#!/usr/bin/env python3
"""Summarize serve_custom_bake --prodqual evidence, without reading labels."""
import argparse
import json
from pathlib import Path


def summarize(native, wasm):
    assert native["schema"] == 2, "cache-feature audit requires schema 2"
    if wasm is not None:
        assert wasm["schema"] == 2
    results = []
    for seed in native["seeds"]:
        permutations = seed["permutations"]
        baseline = permutations[0]["rows"]
        rows = [row for permutation in permutations for row in permutation["rows"]]
        identity = [row for row in baseline if row["dropped_bits"] == 0]
        ladders = []
        for width, height in native["geometries"]:
            ladder = [r for r in baseline if (r["width"], r["height"]) == (width, height)]
            ladders.append({"width": width, "height": height,
                            "scores": [r["pixel_score"] for r in ladder],
                            "nonincreasing": all(a["pixel_score"] >= b["pixel_score"]
                                                 for a, b in zip(ladder, ladder[1:]))})
        keys = ("pixel_bits", "cached_bits", "feature_bits", "identity_aware_bits", "read_bits",
                "feature_mismatches", "cached_feature_mismatches", "finite", "density_cells")
        tier_differences = sum(any(row[k] != base[k] for k in keys)
                               for permutation in permutations[1:]
                               for row, base in zip(permutation["rows"], baseline, strict=True))
        wasm_differences = None
        wasm_rows = []
        if wasm is not None:
            other = wasm["seeds"][seed["seed"]]
            assert other["declared_reads"] == seed["declared_reads"]
            wasm_rows = [r for p in other["permutations"] for r in p["rows"]]
            wasm_differences = sum(any(row[k] != base[k] for k in keys)
                                   for permutation in other["permutations"]
                                   for row, base in zip(permutation["rows"], baseline, strict=True))
        results.append({
            "seed": seed["seed"], "caller_width": seed["caller_width"],
            "declared_reads": seed["declared_reads"], "pairs_per_permutation": len(baseline),
            "permutations": [p["label"] for p in permutations],
            "finite": all(r["finite"] for r in rows),
            "pixel_cache_mismatches": sum(r["pixel_bits"] != r["cached_bits"] for r in rows),
            "pixel_identity_aware_feature_mismatches": sum(r["pixel_bits"] != r["identity_aware_bits"] for r in rows),
            "consumed_feature_mismatches": sum(len(r["feature_mismatches"]) for r in rows),
            "cached_feature_mismatches": sum(len(r["cached_feature_mismatches"]) for r in rows),
            "native_tier_row_mismatches": tier_differences,
            "wasm_row_mismatches": wasm_differences,
            "wasm_finite": all(r["finite"] for r in wasm_rows) if wasm is not None else None,
            "wasm_consumed_feature_mismatches": sum(len(r["feature_mismatches"]) for r in wasm_rows) if wasm is not None else None,
            "wasm_cached_feature_mismatches": sum(len(r["cached_feature_mismatches"]) for r in wasm_rows) if wasm is not None else None,
            "pixel_identity_exact_100": all(r["pixel_score"] == 100.0 for r in identity),
            "feature_identity_scores": [r["feature_score"] for r in identity],
            "feature_identity_band_97_5_to_100": all(97.5 <= r["feature_score"] <= 100.0 for r in identity),
            "no_distortion_above_identity": all(r["pixel_score"] <= 100.0 for r in baseline),
            "ladders_report_only": ladders,
        })
    return {"schema": 2, "scope": "three independent packed seeds; label-free synthetic SDR only",
            "geometries": native["geometries"], "dropped_bits": native["dropped_bits"], "seeds": results}


def prepare_steerfix(root):
    """Freeze the existing engineering roster, never a corpus/label reader."""
    import hashlib
    prior = Path("/mnt/v/output/zensim/prodqual-b-2026-10-07")
    adjudicate = Path("/mnt/v/output/zensim/adjudicate-2026-10-07")
    sha = lambda data: hashlib.sha256(data).hexdigest()
    source = prior / "STEERING_PREREAD.json"
    roster = json.loads(source.read_text())
    assert len(roster["cases"]) == 135 and roster["rules"] == {"m2": .99, "m3f": .7}
    failures = json.loads((adjudicate / "STEERING.json").read_text())
    keys = {r["case"] for r in failures if r["model"] == "seed0"}
    assert len(keys) == 7
    dest = root / "inputs"
    dest.mkdir()
    models = {}
    pins = json.loads((adjudicate / "FINAL_PROVENANCE.json").read_text())["models"]
    for name in ("seed0.bin", "dense.bin"):
        blob = (adjudicate / "models" / name).read_bytes()
        assert sha(blob) == pins[name]
        (dest / name).write_bytes(blob)
        models[name] = (str(dest / name), pins[name])
    copied = {}
    base = []
    for row in roster["cases"]:
        paths = []
        for field, pin in (("ref_path", "reference_file_sha256"),
                           ("dist_path", "distorted_file_sha256")):
            path = Path(row[field])
            assert path.parent.parent == Path("/var/tmp/shippath7/delivery")
            digest = row[pin]
            target = dest / (digest + ".png")
            if digest not in copied:
                blob = path.read_bytes()
                assert sha(blob) == digest
                target.write_bytes(blob)
                copied[digest] = str(target)
            paths.append(str(target))
        base.append({"key": row["case"], "reference": paths[0], "distorted": paths[1],
                     "reference_sha256": row["reference_file_sha256"],
                     "distorted_sha256": row["distorted_file_sha256"],
                     "reference_pixels_sha256": row["reference_pixels_sha256"],
                     "distorted_pixels_sha256": row["distorted_pixels_sha256"],
                     "block": row["block"]})
    selected = [r for r in base if r["key"] in keys]
    assert len(selected) == 7
    diagnostic = []
    for row in selected:
        for objective in ("served", "pre-floor", "smooth-floor"):
            model, pin = models["seed0.bin"]
            diagnostic.append(dict(row, model=model, model_sha256=pin, objective=objective))
        model, pin = models["dense.bin"]
        diagnostic.append(dict(row, model=model, model_sha256=pin, objective="served"))
    model, pin = models["seed0.bin"]
    full = [dict(r, model=model, model_sha256=pin, objective="served") for r in base]
    for name, rows, engine in (("diagnostic-before", diagnostic, False),
                               ("engine", diagnostic, True),
                               ("full-before", full, False), ("full", full, True)):
        packet = {"cases": rows, "engine": engine, "floor_recovery": name == "full",
                  "output": str(root / (name + ".json"))}
        (root / (name + "-packet.json")).write_text(json.dumps(packet, indent=2) + "\n")
    receipt = {"engineering_roster_sha256": sha(source.read_bytes()), "models": models,
               "input_file_hashes": copied, "cases": roster["cases"], "labels_read": False,
               "case_exclusions": [], "bars": roster["rules"]}
    (root / "INPUT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"STEERFIX: admitted 135 cases, seven failures, {len(copied)} pinned files")


def report_steerfix(root):
    """Join native engineering receipts; never score pixels or read labels."""
    import csv
    import hashlib
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    read = lambda name: json.loads((root / name).read_text())
    admitted = read("INPUT_RECEIPT.json")
    diagnostic = read("diagnostic-before.json")["rows"]
    replay = read("engine.json")["rows"]
    before = read("full-before.json")["rows"]
    after = read("full.json")["rows"]
    roster = [r["case"] for r in admitted["cases"]]
    assert len(roster) == 135 and len(set(roster)) == 135
    assert [r["key"] for r in before] == roster == [r["key"] for r in after]
    assert len(diagnostic) == len(replay) == 28
    assert admitted["bars"] == {"m2": .99, "m3f": .7}
    packed = admitted["models"]["seed0.bin"][1]
    dense = admitted["models"]["dense.bin"][1]
    index = lambda rows: {(r["key"], r["model"], r["objective"]): r for r in rows}
    original, exact = index(diagnostic), index(replay)
    assert original.keys() == exact.keys()
    score_checks = 0
    for a, b in zip(before, after, strict=True):
        assert a["model"] == b["model"] == packed
        assert a["served_base_bits"] == b["served_base_bits"]
        assert a["served_repair_bits"] == b["served_repair_bits"]
        score_checks += 1 + len(a["served_repair_bits"])
    for key in original:
        assert original[key]["served_base_bits"] == exact[key]["served_base_bits"]
        assert original[key]["served_repair_bits"] == exact[key]["served_repair_bits"]
    assert all(r["sensitivity_independent_probe_exact"] for r in before + after + diagnostic + replay)
    disagreements = sum(r["engine_feature_disagreements"] for r in replay)
    comparisons = sum(r["engine_feature_comparisons"] for r in replay)
    full_disagreements = sum(r["engine_feature_disagreements"] for r in after)
    assert disagreements == full_disagreements == 0
    assert all(r["engine_feature_comparisons"] > 0 for r in replay + after)
    verdicts = []
    for key in dict.fromkeys(r["key"] for r in diagnostic):
        served = original[key, packed, "Served"]
        raw = original[key, packed, "PreFloor"]
        smooth = original[key, packed, "SmoothFloor"]
        neighbour = exact[key, packed, "PreFloor"]
        dense_old, dense_new = original[key, dense, "Served"], exact[key, dense, "Served"]
        fixed = next(r for r in after if r["key"] == key)
        floor = (served["base_score"] != raw["base_score"]
                 and all(b["score_delta"] == 0.0 for b in served["blocks"])
                 and any(b["score_delta"] != 0.0 for b in raw["blocks"]))
        frozen_map = raw["m3f"] < .7 <= neighbour["m3f"]
        code = floor or frozen_map
        model = not neighbour["pass"]
        verdict = "BOTH" if code and model else "CODE" if code else "MODEL"
        if floor:
            assert fixed["sensitivity_objective"] == "PreFloor"
            assert [b["refinement_gain"] for b in fixed["blocks"]] == [b["refinement_gain"] for b in neighbour["blocks"]]
            assert [b["linearized_gain"] for b in fixed["blocks"]] == [b["linearized_gain"] for b in neighbour["blocks"]]
        verdicts.append({"case": key, "verdict": verdict, "floor_signal_loss": floor,
                         "frozen_map_loss": frozen_map,
                         "served": [served["m2"], served["m3f"]],
                         "pre_floor": [raw["m2"], raw["m3f"]],
                         "smooth_floor": [smooth["m2"], smooth["m3f"]],
                         "pre_floor_replay": [neighbour["m2"], neighbour["m3f"]],
                         "dense_before": [dense_old["m2"], dense_old["m3f"]],
                         "dense_replay": [dense_new["m2"], dense_new["m3f"]],
                         "fixed_served": [fixed["m2"], fixed["m3f"]],
                         "feature_comparisons": neighbour["engine_feature_comparisons"],
                         "feature_disagreements": neighbour["engine_feature_disagreements"]})
    before_failures = [r["key"] for r in before if not r["pass"]]
    after_failures = [r["key"] for r in after if not r["pass"]]
    assert before_failures == after_failures, "new G-STEER failure requires investigation"
    artifacts = ["INPUT_RECEIPT.json", "diagnostic-before.json", "engine.json", "full-before.json", "full.json",
                 "provenance/instrument-before.json", "provenance/instrument-after.json"]
    summary = {"base": "462f7fe5", "labels_read": False, "case_exclusions": [],
               "bars": admitted["bars"], "verdicts": verdicts,
               "score_bit_checks": score_checks, "score_bit_mismatches": 0,
               "seven_case_feature_comparisons": comparisons, "seven_case_feature_disagreements": disagreements,
               "full_feature_comparisons": sum(r["engine_feature_comparisons"] for r in after),
               "full_feature_disagreements": full_disagreements,
               "g_steer_before_pass": len(before) - len(before_failures),
               "g_steer_after_pass": len(after) - len(after_failures),
               "remaining_g_steer_failures": after_failures, "model_qualified": False,
               "raw_sha256": {name: sha(root / name) for name in artifacts}}
    with (root / "SUMMARY.json").open("x") as out:
        json.dump(summary, out, indent=2)
        out.write("\n")
    with Path("benchmarks/steerfix_cases_2026-10-09.csv").open("w") as out:
        writer = csv.writer(out)
        writer.writerow(["case", "before_m2", "before_m3f", "after_m2", "after_m3f", "pass", "sensitivity_objective",
                         "engine_feature_comparisons", "engine_feature_disagreements"])
        for a, b in zip(before, after, strict=True):
            writer.writerow([a["key"], a["m2"], a["m3f"], b["m2"], b["m3f"], b["pass"], b["sensitivity_objective"],
                             b["engine_feature_comparisons"], b["engine_feature_disagreements"]])
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--steerfix-prepare", type=Path)
    parser.add_argument("--steerfix-report", type=Path)
    parser.add_argument("--wasm", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steerfix_report is not None:
        assert args.native is None and args.output is None and args.wasm is None and args.steerfix_prepare is None
        report_steerfix(args.steerfix_report)
        return
    if args.steerfix_prepare is not None:
        assert args.native is None and args.output is None and args.wasm is None
        prepare_steerfix(args.steerfix_prepare)
        return
    if args.native is None or args.output is None:
        parser.error("--native and --output are required for summary")
    native = json.loads(args.native.read_text())
    wasm = json.loads(args.wasm.read_text()) if args.wasm else None
    if wasm is not None:
        assert native["geometries"] == wasm["geometries"]
        assert native["dropped_bits"] == wasm["dropped_bits"]
        assert len(native["seeds"]) == len(wasm["seeds"])
    summary = summarize(native, wasm)
    with args.output.open("x") as out:
        json.dump(summary, out, indent=2)
        out.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
