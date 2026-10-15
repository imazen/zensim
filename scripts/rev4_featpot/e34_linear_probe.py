"""Prepare a label-free H1 fixture or bind the 135-case P2 packet to an L bake.

The fixture has deterministic positive weights, not fitted weights. Its result
checks the instrument and input-domain assumptions, not L's qualification.
"""
import argparse
import json
from pathlib import Path

from v2_common import fx1_declaration, sha

PACKET_SHA = "b2af8465520139da6ebeede71c0c73c688d0d579e52e512cbf53f8181aec5ed9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture")
    fixture.add_argument("--dtype", choices=("f32", "f16"), required=True)
    fixture.add_argument("--model", type=Path, required=True)
    fixture.add_argument("--output", type=Path, required=True)
    packet = commands.add_parser("packet")
    packet.add_argument("--source", type=Path, required=True)
    packet.add_argument("--model", type=Path, required=True)
    packet.add_argument("--result", type=Path, required=True)
    packet.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh output required")
    if args.command == "fixture":
        ids = fx1_declaration()["direct"]
        model = dict(schema_hash=1, scaler_mean=[0.] * len(ids), scaler_scale=[1.] * len(ids),
                     metadata=[dict(key="zentrain.feature_ids", type="utf8", text=" ".join(map(str, ids))),
                               dict(key="zentrain.formula_revision", type="utf8", text="5")],
                     layers=[dict(in_dim=len(ids), out_dim=1, activation="relu", dtype=args.dtype,
                                  weights=[0.002 + 0.0003 * ((i * 7) % 13) for i in ids], biases=[0.]),
                             dict(in_dim=1, out_dim=1, activation="identity", dtype=args.dtype,
                                  weights=[-1.], biases=[100.])])
        result = dict(model=model, output=str(args.model.resolve()), purpose="unfitted P2 fixture")
    else:
        if sha(args.source) != PACKET_SHA:
            raise ValueError("not the frozen 135-case engineering packet")
        result = json.loads(args.source.read_text())
        digest = sha(args.model)
        for case in result["cases"]:
            case.update(model=str(args.model.resolve()), model_sha256=digest, objective="pre-floor")
        result.update(engine=True, floor_recovery=False, linear_probe=True,
                      output=str(args.result.resolve()))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
