"""Bounded refusal checks of E30's unchanged trainer, without fitting a cell."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--upiq", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    args.dest.mkdir()  # preserve prior evidence
    trainer = args.bundle / "bin/zensim_mlp_train"
    inventory = json.loads((args.bundle / "PROGRAM_INVENTORY.json").read_text())
    digest = hashlib.sha256(trainer.read_bytes()).hexdigest()
    if digest != inventory["bin/zensim_mlp_train"]:
        raise ValueError("pinned E30 trainer bytes changed")
    upiq = args.upiq / "upiq380_fit.parquet"
    manifest = Path(f"{upiq}.manifest.json")
    declaration = json.loads(manifest.read_text())
    if declaration["requested_ids"] != list(range(13, 26)) + list(
        range(39, 156)
    ) + list(range(401, 430)) + list(range(459, 720)):
        raise ValueError("not the registered by_v2fy 420-ID transport")
    keep = args.dest / "keep_features.txt"
    keep.write_text("\n".join(map(str, declaration["requested_ids"])) + "\n")
    base = [
        str(trainer),
        "--target-column",
        "human_score",
        "--target-scale",
        "1",
        "--hidden",
        "128",
        "--epochs",
        "120",
        "--pairs-per-epoch",
        "50000",
        "--init-seed",
        "1101",
        "--sample-seed",
        "101",
        "--pair-sampling",
        "uniform",
        "--max-features",
        "1853",
        "--keep-features",
        str(keep),
        "--mse-weight",
        "1",
        "--early-stop-patience",
        "0",
        "--val-policy",
        "mean",
        "--val-aggregate",
        "geomean3",
        "--out-dtype",
        "f32",
        "--log-every",
        "17",
        "--no-auto-eval",
        "--nonneg-distance",
    ]
    records = []

    def check(name, groups, expected, forbidden_payloads):
        output = args.dest / f"{name}.bin"
        trace = args.dest / f"{name}.strace"
        argv = (
            base
            + sum((["--group", group] for group in groups), [])
            + ["--out", str(output)]
        )
        result = subprocess.run(
            ["strace", "-f", "-e", "trace=openat", "-o", str(trace), *argv],
            text=True,
            capture_output=True,
            timeout=30,
        )
        (args.dest / f"{name}.stdout").write_text(result.stdout)
        (args.dest / f"{name}.stderr").write_text(result.stderr)
        opens = [
            line
            for line in trace.read_text().splitlines()
            if any(f'"{path}"' in line for path in forbidden_payloads)
        ]
        if (
            result.returncode != 2
            or expected not in result.stderr
            or output.exists()
            or opens
        ):
            raise ValueError(
                f"{name}: refusal/payload tripwire failed: {result.returncode}, {result.stderr}, {opens}"
            )
        records.append(
            {
                "name": name,
                "status": "REFUSED",
                "returncode": result.returncode,
                "argv": argv,
                "stderr": result.stderr,
                "forbidden_payload_open_calls": opens,
            }
        )

    # Put the new leg first to exercise its metadata refusal before any payload.
    # The second, existing teacher retains the absolute term the CLI requires.
    teacher = args.bundle / "v2d1/wide/main/real/safesyn_fit.parquet"
    check(
        "registered-upiq-null-id",
        [
            f"hdr:{upiq}:4.34410740924913:0:rank",
            f"safesyn:{teacher}:1.0168526508775275:0:withinref,both",
        ],
        "malformed feature_set_id",
        [upiq, teacher],
    )

    # Fully synthetic tables isolate the mixed-decoder rule, independently of
    # UPIQ's null ID. No original producer declaration is rewritten.
    paths = []
    for name, decoder in [
        ("sdr", "fixture-rgb8-decoder"),
        ("hdr", "fixture-zenexr-bt709-absolute-nits"),
    ]:
        path = args.dest / f"{name}.parquet"
        paths.append(path)
        pq.write_table(
            pa.table(
                {
                    **{f"f{i}": [0.0, 1.0] for i in range(1853)},
                    "human_score": [0.0, 1.0],
                    "ref_basename": ["r0", "r1"],
                }
            ),
            path,
        )
        Path(f"{path}.manifest.json").write_text(
            json.dumps(
                {
                    "feature_set_id": "basic+peaks+v2@w1825/rev5_localwin#36c3f3af",
                    "formula_revision": 5,
                    "decoder_era": decoder,
                }
            )
        )
    check(
        "mixed-decoder-fixture",
        [f"sdr:{paths[0]}:1:0:both", f"hdr:{paths[1]}:4.34410740924913:0:rank"],
        "mixed decoder declarations",
        [upiq, teacher],
    )
    receipt = {
        "schema": "e31-pinned-trainer-admission-v1",
        "status": "BLOCKED",
        "trainer_sha256": digest,
        "upiq_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "tests": records,
        "scope": "metadata/fixture refusals only; no training or model output",
    }
    (args.dest / "PINNED_TRAINER_ADMISSION.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
