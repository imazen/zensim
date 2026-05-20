#!/usr/bin/env python3
"""Strip `zentrain.output_calibration_spline` metadata from a ZNPR v3 bake.

Used by the V10 spline retrofit pipeline: TunerV3 already carries a V9
spline; to refit a V10 spline against the raw network output we need a
spline-less variant of the same network. The MLP layers, scaler, and
all other metadata (per-sample-alpha head, tanh-output head, etc.) are
preserved bit-exactly.

Output bake is a clean re-emission via `zenpredict bake` from a
reconstructed JSON BakeRequest. Bit-identical to the input except for
the dropped metadata entry.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ZENPREDICT_BIN_DEFAULT = Path(
    "/home/lilith/work/zen/zenanalyze/target/release/zenpredict"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bake", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--zenpredict-bin", type=Path, default=ZENPREDICT_BIN_DEFAULT
    )
    parser.add_argument(
        "--drop-metadata-key",
        default="zentrain.output_calibration_spline",
        help="Metadata key to remove (default: spline)",
    )
    args = parser.parse_args()

    proc = subprocess.run(
        [str(args.zenpredict_bin), "inspect", str(args.bake), "--weights"],
        capture_output=True,
        text=True,
        check=True,
    )
    inspected = json.loads(proc.stdout)
    scaler_mean = inspected["scaler_mean"]
    scaler_scale = inspected["scaler_scale"]
    out_layers = [
        {
            "in_dim": l["in_dim"],
            "out_dim": l["out_dim"],
            "activation": l["activation"],
            "dtype": l["dtype"],
            "weights": l["weights"],
            "biases": l["biases"],
        }
        for l in inspected["layers"]
    ]

    md_list = []
    for entry in inspected.get("metadata", []):
        if entry["key"] == args.drop_metadata_key:
            print(f"  dropping metadata key {entry['key']}")
            continue
        kind = entry["kind"]
        item = {"key": entry["key"], "type": kind}
        if "value_hex" in entry:
            item["hex"] = entry["value_hex"]
        elif "value_text" in entry:
            item["text"] = entry["value_text"]
        elif "value_f32_array" in entry:
            item["f32"] = entry["value_f32_array"]
        else:
            raise RuntimeError(f"unknown metadata encoding: {entry}")
        md_list.append(item)

    schema_hash_raw = inspected.get("schema_hash", 0)
    if isinstance(schema_hash_raw, str):
        schema_hash = (
            int(schema_hash_raw, 16)
            if schema_hash_raw.startswith("0x")
            else int(schema_hash_raw)
        )
    else:
        schema_hash = int(schema_hash_raw)

    req = {
        "schema_hash": schema_hash,
        "flags": 0,
        "compressed": True,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "layers": out_layers,
        "metadata": md_list,
    }
    json_path = args.out.with_suffix(".tmp.json")
    json_path.write_text(json.dumps(req))
    subprocess.run(
        [str(args.zenpredict_bin), "bake", str(json_path), str(args.out)],
        check=True,
    )
    json_path.unlink()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
