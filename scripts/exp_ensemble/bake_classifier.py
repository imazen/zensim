#!/usr/bin/env python3
"""
Convert the EXP-ENSEMBLE-V05 classifier JSON dump to a ZNPR v3 bake.

Per zensim/CLAUDE.md "JSON pipeline mandate": never write ZNPR v3 bytes
directly. Emit a BakeRequestJson and shell out to `zenpredict bake`.

The classifier is a 372 → 64 → 1 MLP with relu hidden activation and a
sigmoid output (the bake stores the pre-sigmoid logit). The runtime
applies sigmoid + threshold itself.

Standardization (subtract mean, divide by scale) is the bake's native
input transform — encoded via `scaler_mean` + `scaler_scale`.

Usage:
    python3 scripts/exp_ensemble/bake_classifier.py \\
        --classifier /tmp/exp_ensemble_classifier_weights.json \\
        --output zensim/weights/v05_ensemble_classifier_2026-05-18.bin
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ZENPREDICT = Path("/home/lilith/work/zen/zenanalyze/target/release/zenpredict")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-compress", action="store_true",
                    help="disable LZ4 + zerobias (debugging only)")
    args = ap.parse_args()

    classifier = json.loads(Path(args.classifier).read_text())
    assert classifier["kind"] == "mlp_1_hidden_relu_sigmoid"
    n_in = classifier["n_inputs"]
    n_hidden = classifier["n_hidden"]
    hidden_w = classifier["hidden_weights"]  # n_in * n_hidden, row-major (in × out)
    hidden_b = classifier["hidden_biases"]
    output_w = classifier["output_weights"]  # n_hidden * 1
    output_b = classifier["output_bias"]

    assert len(hidden_w) == n_in * n_hidden
    assert len(hidden_b) == n_hidden
    assert len(output_w) == n_hidden
    assert len(classifier["scaler_mean"]) == n_in
    assert len(classifier["scaler_scale"]) == n_in

    # Build BakeRequestJson.
    # Schema hash: arbitrary but stable per-bake identifier; tools use it
    # to enforce schema invariants. Use sha1 of the classifier kind +
    # n_in + n_hidden so multiple ensemble bakes don't collide.
    sha = hashlib.sha256(
        f"v05_ensemble_classifier:{n_in}:{n_hidden}".encode()
    ).digest()
    schema_hash = int.from_bytes(sha[:8], "little")

    request = {
        "schema_hash": schema_hash,
        "flags": 0,
        "scaler_mean": classifier["scaler_mean"],
        "scaler_scale": classifier["scaler_scale"],
        "layers": [
            {
                "in_dim": n_in,
                "out_dim": n_hidden,
                "activation": "relu",
                "dtype": "i8",
                "weights": hidden_w,
                "biases": hidden_b,
            },
            {
                "in_dim": n_hidden,
                "out_dim": 1,
                "activation": "identity",
                "dtype": "i8",
                "weights": output_w,
                "biases": [output_b],
            },
        ],
        "metadata": [
            {
                "key": "zensim.ensemble_classifier",
                "type": "utf8",
                "text": (
                    "EXP-ENSEMBLE-V05 routing classifier — outputs a pre-"
                    "sigmoid logit. Caller computes sigmoid + thresholds. "
                    "Route to compression bake if sigmoid(out[0]) > 0.5."
                ),
            },
            {
                "key": "zensim.ensemble_threshold",
                "type": "numeric",
                "f32": [classifier.get("threshold", 0.5)],
            },
        ],
    }
    if not args.no_compress:
        # Sweet-spot per zenpredict docs: zerobias_tau=0.005 + LZ4.
        request["zerobias_tau"] = 0.005
        request["compressed"] = True

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(request, f)
        tmp = Path(f.name)
    try:
        # zenpredict CLI uses subcommand form.
        r = subprocess.run(
            [str(ZENPREDICT), "bake", str(tmp), str(out)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(r.stdout)
        if r.stderr:
            print("STDERR:", r.stderr, file=sys.stderr)
    finally:
        tmp.unlink()

    size = out.stat().st_size
    md5 = hashlib.md5(out.read_bytes()).hexdigest()
    print(f"\nBake: {out}")
    print(f"  size_bytes={size}  md5={md5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
