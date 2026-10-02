"""Input columns a bake actually reads: indices k whose layer-0 weight row is non-zero (design log E9′ method 2,
group-lasso survival). Uses `zenpredict inspect --weights` (zenanalyze zenpredict-bake). Layer 0 stores weights
input-major (in_dim × out_dim); the orientation is checked against `bake_dial_refit densify`'s count on the first call.

  python3 bake_survivors.py BAKE [BAKE ...]          # prints {bake: [ids]} JSON
"""
import json
import os
import subprocess
import sys

ZENPREDICT = os.environ.get("ZENPREDICT_BIN", "/mnt/data/fitv2-canon/zenpredict-target/release/zenpredict")


def survivors(bake: str) -> list[int]:
    d = json.loads(subprocess.run([ZENPREDICT, "inspect", bake, "--weights"], check=True, capture_output=True,
                                  text=True).stdout)
    l0 = d["layers"][0]
    n_in, n_out, w = l0["in_dim"], l0["out_dim"], l0["weights"]
    if len(w) != n_in * n_out:
        raise ValueError(f"{bake}: layer 0 has {len(w)} weights for {n_in}x{n_out}")
    return [k for k in range(n_in) if any(w[k * n_out + j] != 0.0 for j in range(n_out))]


if __name__ == "__main__":
    print(json.dumps({b: survivors(b) for b in sys.argv[1:]}))
