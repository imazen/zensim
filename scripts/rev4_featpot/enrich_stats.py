"""E1-band and within-reference POTENTIAL diagnostics via the panel owners.

This script defines target-only bands and within-reference comparison rows.
All rank and agreement statistics come from panel --batch / --pairwise.
Run it under the shared heavy wrapper; inputs are admitted tables only.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from data import load

ROOT = Path("/var/tmp/rev4-featpot")
TMP = ROOT / "tmp"
TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TMP)
tempfile.tempdir = str(TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch_indexed  # noqa: E402


def bands_for(name: str, y: np.ndarray) -> tuple[np.ndarray, dict]:
    if name == "aic3":
        distance = -y
        return np.where(distance < 1, "NT", np.where(distance < 2, "MID", "LOW")), {
            "NT": "d<1 JND", "MID": "1<=d<2", "LOW": "d>=2", "d": "-human_score"}
    if name.startswith("konfig"):
        distance = (1 - y) * 3.2
        return np.where(distance < 1, "NT", np.where(distance < 2, "MID", "LOW")), {
            "NT": "d<1 JND", "MID": "1<=d<2", "LOW": "d>=2", "d": "(1-human_score)*3.2"}
    if name.startswith("konjnd_bpg"):
        return np.repeat("ORACLE", len(y)), {"ORACLE": "SSIMULACRA2 /100 target; no human E1 band"}
    edges = np.percentile(y, [20, 40, 60, 80])
    bands = np.asarray([f"Q{1 + sum(value > edge for edge in edges)}" for value in y])
    return bands, {"edges_p20_p40_p60_p80": edges.tolist(),
                   "Q5": "near-threshold/highest human quality",
                   "Q4,Q3": "mid", "Q2,Q1": "low"}


def pairwise_owner(path: Path, rows: list[tuple[str, float, float, str]], draws: list[list[str]]) -> dict:
    if not rows:
        return {"status": "MISSING", "reason": "no distinct-target within-reference pairs"}
    table = path.with_suffix(".pairs.tsv")
    with table.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["group", "s_left", "s_right", "choice", "weight"])
        for row in rows:
            writer.writerow((*row, 1))
    groups = list(dict.fromkeys(row[0] for row in rows))
    group_idx = {ref: i for i, ref in enumerate(groups)}
    resamples = path.with_suffix(".resamples.tsv")
    with resamples.open("w") as stream:
        stream.write("POINT\t*\n")
        for b, sample in enumerate(draws):
            indices = [str(group_idx[ref]) for ref in sample if ref in group_idx]
            if indices:
                stream.write(f"B{b}\t{','.join(indices)}\n")
    panel_bin = os.environ.get("ZEN_PANEL_BIN", str(ROOT / "target/debug/panel"))
    proc = subprocess.run([panel_bin, "--pairwise", str(table), "--resample", str(resamples)],
                          text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"panel --pairwise rc={proc.returncode}: {proc.stderr[-2000:]}")
    output = path.with_suffix(".pairwise.tsv")
    output.write_text(proc.stdout)
    values = {record["label"]: float(record["acc_response"])
              for record in csv.DictReader(proc.stdout.splitlines(), delimiter="\t")}
    boot = np.asarray([values[f"B{b}"] for b in range(len(draws)) if f"B{b}" in values])
    boot = boot[np.isfinite(boot)]
    return {"status": "MEASURED", "pairs": len(rows), "references": len(groups),
            "point": values["POINT"],
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist() if len(boot) else None,
            "bootstrap_finite": len(boot), "panel_output": str(output),
            "panel_output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--model", choices=["linear", "bvls"], required=True)
    ap.add_argument("--arm", choices=["r0", "minus_basic"], required=True)
    args = ap.parse_args()
    fit_result = ROOT / "fits" / f"POT_{args.set}_{args.arm}_{args.model}" / "result.json"
    value = json.loads(fit_result.read_text())
    target = "ssim2_oracle" if args.set.startswith("konjnd_bpg") else "human_score"
    data, label_meta = load(args.set, features=False)
    refs = data["ref_basename"].astype(str).to_numpy()
    y = data["target"].to_numpy(dtype=np.float64)
    bands, definition = bands_for(args.set, y)
    unique_refs = sorted(set(refs))
    rng = np.random.default_rng(20260923)
    draws = [[unique_refs[i] for i in rng.integers(0, len(unique_refs), len(unique_refs))]
             for _ in range(2000)]
    result = {"schema": "rev4-featpot-e1-stats-v1", "set": args.set,
              "model": args.model, "arm": args.arm, "target": target,
              "label_source": label_meta,
              "band_definition": definition, "B": 2000, "seed": 20260923,
              "label": "POTENTIAL — ceiling, not a model score", "phases": {}}
    for phase in ("nested", "in_sample"):
        p = np.asarray(value[phase]["prediction"], dtype=np.float64)
        if len(p) != len(y) or not np.isfinite(p).all():
            raise ValueError(f"{phase}: prediction length/nonfinite mismatch")
        phase_result = {}
        for band in ["ALL"] + sorted(set(bands)):
            ii = np.arange(len(y)) if band == "ALL" else np.flatnonzero(bands == band)
            if len(ii) < 3:
                phase_result[band] = {"status": "MISSING", "reason": "<3 rows"}
                continue
            by_ref = {ref: ii[refs[ii] == ref] for ref in unique_refs}
            bases = {"p": p[ii], "y": y[ii]}
            positions = {ref: np.flatnonzero(refs[ii] == ref) for ref in unique_refs}
            jobs = [("POINT", "p", "y", None)]
            for b, sample in enumerate(draws):
                sampled = [positions[ref] for ref in sample if len(positions[ref])]
                if not sampled:
                    continue
                selection = np.concatenate(sampled)
                if len(selection) >= 3:
                    jobs.append((f"B{b}", "p", "y", selection))
            panel_rows = panel_batch_indexed(bases, jobs, stats="srocc", timeout=7200)
            ranks = {row["label"]: row["srocc_signed"] for row in panel_rows}
            boot = np.asarray([ranks[f"B{b}"] for b in range(2000) if f"B{b}" in ranks],
                              dtype=np.float64)
            finite = boot[np.isfinite(boot)]
            tail = {"point_signed_srocc": ranks["POINT"], "bootstrap_finite": len(finite),
                    "ci95": np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None}
            pairs = []
            for ref, ref_idx in by_ref.items():
                for a in range(len(ref_idx)):
                    for b in range(a + 1, len(ref_idx)):
                        i, j = ref_idx[a], ref_idx[b]
                        if y[i] != y[j]:
                            pairs.append((ref, p[i], p[j], "left" if y[i] < y[j] else "right"))
            stem = ROOT / "stats" / f"{args.set}_{args.arm}_{args.model}_{phase}_{band}"
            stem.parent.mkdir(parents=True, exist_ok=True)
            ordering = pairwise_owner(stem, pairs, draws)
            phase_result[band] = {"status": "MEASURED", "rows": len(ii),
                                  "references": len({refs[i] for i in ii}),
                                  "tail_srocc": tail, "within_ref_ordering": ordering}
            print(json.dumps({"phase": phase, "band": band, "rows": len(ii),
                              "tail_srocc": tail["point_signed_srocc"],
                              "ordering": ordering.get("point")}), flush=True)
        result["phases"][phase] = phase_result
    output = ROOT / "stats" / f"{args.set}_{args.arm}_{args.model}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(output),
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
