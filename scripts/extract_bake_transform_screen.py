#!/usr/bin/env python3
"""Reconstruct the `--transforms-tsv` screen a ZNPR bake was fit in, from the
bake's OWN `zentrain.feature_transform*` metadata.

Why this exists: the 2026-08-29 W-LIN 954 heads were fit ad hoc, so the screen
TSV they used was never committed and is not on disk. It IS, however, carried
verbatim in every bake they produced (the transform tokens + params are part of
the bake bytes and are what the runtime applies), so it is exactly recoverable.
Verified: a gram rebuilt with the reconstructed screen reproduces the frozen
Python gram to rel ~4e-9 (BLAS order), and `fit-lasso --parity-fit` reproduces
the stored head npz bit-exactly. See
`benchmarks/carrier_head_recipe_2026-08-30.md` §1.

Reads the metadata through the OWNER (`zenpredict inspect`); parses nothing out
of the bake bytes itself.

    extract_bake_transform_screen.py <bake.bin> <out.tsv> [--n-feat N] [--zenpredict PATH]
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

DEFAULT_ZP = "/home/lilith/work/zen/zenanalyze/target/release/zenpredict"
TOKS_KEY = "zentrain.feature_transforms"
PARS_KEY = "zentrain.feature_transform_params"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bake")
    ap.add_argument("out")
    ap.add_argument("--n-feat", type=int, default=None,
                    help="truncate to the first N features (e.g. 944 from a 954 bake)")
    ap.add_argument("--zenpredict", default=DEFAULT_ZP)
    a = ap.parse_args()

    raw = subprocess.run([a.zenpredict, "inspect", a.bake],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(raw)
    md = {m["key"]: m for m in d.get("metadata", [])}
    for k in (TOKS_KEY, PARS_KEY):
        if k not in md:
            print(f"ERROR: {a.bake} carries no {k}", file=sys.stderr)
            return 2
    toks = md[TOKS_KEY]["value_text"].split("\n")
    pars = md[PARS_KEY]["value_text"].split("\n")
    # the params blob may carry one trailing empty line for a trailing newline
    if len(pars) == len(toks) + 1 and pars[-1] == "":
        pars = pars[:-1]
    if len(toks) != len(pars):
        print(f"ERROR: {len(toks)} transform tokens vs {len(pars)} param rows", file=sys.stderr)
        return 2
    n = a.n_feat if a.n_feat is not None else len(toks)
    if n > len(toks):
        print(f"ERROR: --n-feat {n} > {len(toks)} in the bake", file=sys.stderr)
        return 2

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("feat_idx\tbest_transform\tparams_csv\n")
        for i in range(n):
            f.write(f"{i}\t{toks[i]}\t{pars[i]}\n")

    sha = hashlib.sha256(pathlib.Path(a.bake).read_bytes()).hexdigest()
    counts: dict[str, int] = {}
    for t in toks[:n]:
        counts[t] = counts.get(t, 0) + 1
    shaped = sorted(i for i in range(n) if toks[i] != "identity")
    print(f"source bake  {a.bake}\n  sha256 {sha}")
    print(f"wrote {out} ({n} rows): {counts}")
    if shaped:
        print(f"  shaped slots: {len(shaped)}, index range f{shaped[0]}..f{shaped[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
