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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--wasm", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
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
