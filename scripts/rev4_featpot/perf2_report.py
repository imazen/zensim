#!/usr/bin/env python3
"""Render qualified PERF2 receipts; statistics remain owned by SPEEDQ."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def render(root):
    matrix = read(root / "matched-band-final/SUMMARY.json")
    options = read(root / "matched-band-final-options/SUMMARY.json")
    movement = read(root / "matched-band-final-options/MOVEMENT.json")
    runtime = read(root / "matched-band-final-verify/RUNTIME_PARITY.json")
    native = read(root / "native-band-final/original-packets/FULL_NATIVE_CACHE_PARITY.json")
    assert len(matrix) == 36 and len(options) == 24 and len(movement) == 9
    assert runtime["status"] == native["status"] == "PASS"
    assert sum(row["rows"] for row in native["models"].values()) == 270
    assert all(row["every_output_row_field_equal"] for row in native["models"].values())
    final_profiles = read(root / "profiles-band-final/TOP15.json")
    before_profiles = read(root / "profiles-isolated-before-v2/TOP15.json")
    assert len(final_profiles) == len(before_profiles) == 6
    def cell(rows, side, threads, arm):
        return next(row for row in rows if row["size"] == f"{side}x{side}"
                    and row["threads"] == threads and row["arm"] == arm)
    out = ["# Production candidate timing and threading (2026-10-18)", "",
        "Measured on r7900x (Ryzen 9 7900X, Zen 4), Rust 1.99.1, symbolized release, thin LTO. "
        "No target-cpu=native. Frozen diff+frag-820 model SHA256 "
        "`89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e`.", "",
        "SPEEDQ owns the unchanged quiet gate, paired rounds, retries and percentiles: first 32 clean rounds "
        "from at most 64, shared exclusion indices. All raw rounds, failed attempts, source pins and binaries "
        "are retained. p95 describes these rounds; it is not a service tail-latency guarantee. "
        "Inputs are deterministic engineering pairs; fresh offsets11/23 add independent parity cases. "
        "Native fixture roles are 96 TRAIN, 12 design-released SELECT and 27 historical M3 per model. "
        "No labels were read; these diagnostics do not establish held-out model quality.", "",
        "## Complete action timing", "",
        "Scalar means an uncached candidate pair. Map means a prepared cached reference plus complete "
        "score, full-resolution density, SAT and finite-refinement snapshot. There is no complete cached-scalar "
        "BakeScorer entry; that cell is N/A. The lower-level fold cache is not a complete candidate scorer.", "",
        "Values are p50 / p95 in ms. Efficiency = (1-thread p50 / current p50) / requested threads. "
        "16 workers cover 11 physical cores plus 5 SMT siblings; 8 workers cover 8 physical cores. "
        "The reserved physical core and its sibling are excluded.", "",
        "| Side | Threads | Scalar before | Scalar after | Map before | Map after | Scalar efficiency | Map efficiency |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for side in [256, 1024, 2048]:
        for n in [1, 8, 16]:
            rows = [cell(matrix, side, n, arm) for arm in ["e34_control", "e34_candidate",
                    "e34_candidate_map_before", "e34_candidate_map_retained"]]
            values = [f"{r['p50_ms']:.3f} / {r['p95_ms']:.3f}" for r in rows]
            out.append(f"|{side}|{n}|" + "|".join(values) +
                       f"|{rows[1]['efficiency']:.1%}|{rows[3]['efficiency']:.1%}|")
    out += ["", "## What limits threading", "",
        "The retained map extraction takes the split channel path in `feature_v2.rs`: phase A over three "
        "channels, serial retention copies, then phase B. Retention disables both fused channels and the "
        "bounded independent-strip queue. Some inner kernels use row bands, but each strip still crosses "
        "these barriers. The scalar path can use queued strip jobs. This limits available work even on large images.", "",
        "Density processes scales and channel contributions in order. The original bin fold was serial. "
        "SAT construction retains horizontal and vertical recurrence dependencies; zero fills, retained-plane "
        "copies and output allocation add memory traffic. This is verified source structure. Cache misses alone "
        "do not prove DRAM bandwidth saturation.", "",
        "The exact changes reuse the temporary f64 bin canvas, bound combine slices once, and process "
        "independent bin rows in 64-row bands with runtime SIMD dispatch once per band. Each pixel retains "
        "the original addition/multiply/zero-handling order; returned density and SAT buffers remain independent. "
        "The retained-extraction queue/barrier limitation remains.", "",
        "## Lower-resolution options", "",
        "Existing `bin=2` gives half width and half height. `bin=8`/`16` give coarser grids. These change "
        "map representation and unaligned rectangle interpolation, while preserving scalar score bits. "
        "Feature extraction and much of density evaluation still run at the original scales. Defaults remain bin1.", "",
        "Values are p50 / p95 ms; savings are paired-arm median differences from the same run.", "",
        "|Side|Threads|Full map|Half grid|bin8|bin16|Half saving|bin8 saving|bin16 saving|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for side in [256, 1024, 2048]:
        for n in [1, 16]:
            rows = [cell(options, side, n, arm) for arm in ["e34_candidate_map_retained",
                    "e34_candidate_map_bin2", "e34_candidate_map_bin8", "e34_candidate_map_bin16"]]
            values = [f"{r['p50_ms']:.3f} / {r['p95_ms']:.3f}" for r in rows]
            savings = [f"{rows[0]['p50_ms'] - r['p50_ms']:.3f}" for r in rows[1:]]
            out.append(f"|{side}|{n}|" + "|".join(values + savings) + "|")
    out += ["", "Measured output movement uses 13x19 source-pixel rectangles beginning at 3 and stepping 17 "
        "in each axis over the whole input. Maxima are sampled errors in score-gain units, not universal bounds "
        "or finite-edit accuracy guarantees. The same grid checks finite-refinement movement.", "",
        "|Side|Bin|Rectangles|max absolute query error|mean absolute query error|max refinement error|",
        "|---:|---:|---:|---:|---:|---:|"]
    for row in movement:
        m = row["movement"]
        assert m["score_bits_equal"] and m["finite_rectangles"] == m["unaligned_rectangles"]
        out.append(f"|{row['side']}|{row['bin']}|{m['unaligned_rectangles']}|"
                   f"{m['query_max_abs']:.9g}|{m['query_mean_abs']:.9g}|{m['refinement_max_abs']:.9g}|")
    out += ["", "Cheaper density formulas, family pruning and reduced feature precision have no qualified "
        "alternate implementation measured here. No saving or output movement is assigned to those ideas.", "",
        "## Isolated profiles", "",
        "perf sampled 499Hz with DWARF stacks. Acknowledged enable/disable commands isolate scalar extraction, "
        "retained map extraction and density. First three calls are warmup; samples cover 248 regions per capture. "
        "Percentages are sampled CPU cycles within that region, not wall-time fractions. "
        "The captures assert actual Rayon pool sizes and retain interference/admission receipts. "
        "No `_scalar` dispatch symbol appeared in the six before captures; inspect the six after lists below. "
        "Sparse blockiness, ordered accumulation/SAT, copies and allocation include scalar/serial work; "
        "the hot dense, gradient and blur kernels use SIMD."]
    for name, profiles in [("Before", before_profiles), ("After", final_profiles)]:
        if name == "Before":
            out += ["", "### Baseline feature families", ""]
            for key, profile in sorted(profiles.items()):
                out.append(key + ": " + "; ".join(f"{family} {value:.2f}%" for family, value
                    in sorted(profile["families"].items(), key=lambda item: -item[1])) + ".")
                out.append("")
            out.append("All baseline top-15 symbols remain in `profiles-isolated-before-v2/TOP15.json` in the archive.")
            continue
        for key, profile in sorted(profiles.items()):
            out += ["", f"### {name}: {key}", "", "|Rank|Cycles|Symbol|", "|---:|---:|---|"]
            for rank, row in enumerate(profile["top15"], 1):
                out.append(f"|{rank}|{row['percent']:.2f}%|`{row['symbol']}`|")
            out += ["", "Families: " + "; ".join(f"{family} {value:.2f}%" for family, value
                    in sorted(profile["families"].items(), key=lambda item: -item[1])) + ".",
                    "Named scalar dispatch: " + (", ".join(row["symbol"] for row in profile["scalar_named_symbols"]) or "none sampled") + "."]
    out += ["", "## Qualification and artifacts", "",
        "270/270 original native-cache rows compare equal in every field, including the original cache path. "
        "The existing scorer/refinement/subset gates pass. 54 matched runtime audits cover three models, "
        "1/8/16 threads, four standard sizes and two fresh odd-size pairs; full feature/gradient bits, "
        "density/query/refinement hashes and scores match. This includes the required 12 full-map cells. "
        "Before/after dependency and feature inventories match.", "",
        "Raw evidence: `/mnt/v/output/zensim/perf2-2026-10-18/remote/` on dev; "
        "mirror `/mnt/tower/output/zensim-perf2-archive-2026-10-18/`. "
        "The tracked pointer records hashes and archive verification. "
        "Remote source receipts bind every compiled input and frozen binary SHA256.", ""]
    out += ["", "## Resource wrapper records", "",
        "These are complete command scopes, including setup and workers, not per-call heap measurements.", ""]
    for name in ["band-oracles.log", "final-validation.log", "after-release-band-checked-build.log",
                 "profile-after-band-build.log", "matched-band-final-compare-controller.log",
                 "matched-band-final-options-options-controller.log", "profiles-band-final-profile-controller.log",
                 "profiles-band-final-profile_stat-controller.log"]:
        lines = [line for line in (root / name).read_text().splitlines() if line.startswith("run-heavy: done")]
        assert len(lines) == 1, (name, lines)
        out.append(f"- `{name}`: `{lines[0]}`")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = render(args.root)
    args.output.write_text(report)
    print(json.dumps(dict(path=str(args.output), bytes=len(report.encode()),
                          sha256=hashlib.sha256(report.encode()).hexdigest())))


if __name__ == "__main__":
    main()
