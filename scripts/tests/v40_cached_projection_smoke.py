"""Compare cached f64-wire prediction to the canonical admitted Parquet loader.

Uses only the approved D1 KADID TRAIN+SELECT table and local smoke models.
No external, HDR VAL, UPIQ development or terminal population is accessed.
"""

import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--harvest-attempt", type=int, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.bundle / "committed-tools"))
    from harvest_fit_cells import blob_root

    keep = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "benchmarks/e32_palette_feature_ids_2026-10-07.json"
        ).read_text()
    )["arm_ids"]
    selection = json.loads((a.bundle / "SMOKE_SELECTION.json").read_text())
    report = []
    for arm, modes in selection.items():
        key = f"{arm}-bounded-{modes['bounded']}"
        job = json.loads((a.bundle / f"smoke-manifest-{key}.json").read_text())[0]
        # The refusal owner already verified these bytes without installing.
        cell = (
            a.bundle
            / f"harvest-refusal-{a.harvest_attempt}-{key}"
            / blob_root(job["kind"])
            / job["cell"]["image_path"]
        )
        root = a.bundle / ("v2e32" if arm == "palette" else "v2e29")
        table = root / "wide/main/real/kadid.parquet"
        ids = keep if arm == "palette" else keep[:420]
        width = 1867 if arm == "palette" else 1825
        columns = [f"palette_f{i}" if i >= 1825 else f"f{i}" for i in ids]
        values = pq.read_table(table, columns=columns).slice(0, 12)
        matrix = np.full((12, width), np.nan, dtype="<f8")
        for i, column in zip(ids, columns):
            matrix[:, i] = (
                values[column].to_numpy().astype(np.float32).astype(np.float64)
            )
        if not np.isfinite(matrix[:, ids]).all():
            raise ValueError("requested projection values unmeasured")
        dest = a.out / arm
        dest.mkdir()
        wire, dense = dest / "features.f64.wire", dest / "dense.bin"
        wire.write_bytes(struct.pack("<II", width, 12) + matrix.tobytes())
        if arm == "palette":
            refused = subprocess.run([
                str(a.bundle / "bin/bake_dial_refit"), "densify",
                "--in", str(cell / "refit/last.bin"), "--out", str(dest / "unflagged.bin")
            ], capture_output=True, text=True)
            if refused.returncode == 0 or (dest / "unflagged.bin").exists() or "unavailable to the extraction plan" not in refused.stderr:
                raise AssertionError("unflagged palette serving/densify must remain refused")
        subprocess.run(
            [
                str(a.bundle / "bin/bake_dial_refit"),
                "densify",
                "--in",
                str(cell / "refit/last.bin"),
                "--out",
                str(dense),
                *(["--research-palette-cached"] if arm == "palette" else []),
            ],
            check=True,
            capture_output=True,
        )
        output = dest / "canonical.tsv"
        subprocess.run(
            [
                str(a.bundle / "bin/bake_dial_refit"),
                "predict",
                "--bake",
                str(dense),
                "--corpus",
                str(table),
                "--score-units",
                "--out",
                str(output),
                *(["--research-palette-cached"] if arm == "palette" else []),
            ],
            check=True,
            capture_output=True,
        )
        canonical = pd.read_csv(
            output, sep="\t", float_precision="round_trip"
        ).pred.to_numpy()[:12]
        raw = subprocess.check_output(
            [
                str(a.bundle / "bin/predict_features_with_bake"),
                "--bake",
                str(dense),
                "--features-file",
                str(wire),
                "--f64-wire",
                "--production",
                *(["--research-palette-cached"] if arm == "palette" else []),
            ],
            text=True,
        )
        cached = np.array([float(v) for v in raw.split()])
        if canonical.tobytes() != cached.tobytes():
            raise AssertionError((arm, canonical.tolist(), cached.tolist()))
        report.append(
            dict(
                arm=arm,
                rows=12,
                status="PASS",
                prediction_bytes_bit_equal=True,
                maximum_absolute_delta=float(np.max(np.abs(canonical - cached))),
            )
        )
    (a.out / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
