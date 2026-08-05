#!/usr/bin/env python3
"""Appendix Q linear control — the BHdr-family baseline at 944 (Q.2).

Drives the pinned deterministic linear instrument
(`scripts/v_next/linear_projections_2026-07-03.py`, loaded verbatim — the
bandvis-loo `run_twin944.py` pattern) with the ONE registered difference: the
mix is the hdr944 leg alone (`hdr_v3mix944_traindigits`, human_score = the
cvvdp-mix target, weight 1.0), matching BHdr's own shape (fit on hdr_v3mix
alone). Everything else is the E2/LOO standard: ZLIN_NFEAT=944, shaped space
via `screen_720_merged_safe.tsv` (f720+ identity), BVLS with the shipped v1
sign mask, tau 0, f16 pack. No anchor npz exists in this scratch, so no
output spline is attached — the monotone spline is rank-invariant, and every
Q.3 read of this bake is rank-only (|SROCC|), so its absence cannot move any
registered number.

Deterministic: no seed, no SGD. Output: Q_lin944_hdr.bin under the Q.5
artifact dir.

  usage: hdrp1_linear_control.py
"""
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("HDRP1_OUT", "/mnt/v/output/zensim/bakes/hdrp1"))

os.environ["ZLIN_NFEAT"] = "944"
os.environ["ZLIN_SCRATCH"] = str(OUT_DIR / "linear-probe")
os.environ["ZLIN_SCREEN"] = str(
    REPO / "benchmarks/v2_transform_screen_2026-07-23/screen_720_merged_safe.tsv")

spec = importlib.util.spec_from_file_location(
    "lp", REPO / "scripts/v_next/linear_projections_2026-07-03.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)

HDR_LEG = Path("/mnt/v/output/zensim/hdr944-leg")
lp.GROUPS.update({
    "hdr944": (HDR_LEG / "hdr_v3mix944_traindigits_2026-08-03.parquet",
               ["human_score"]),
})
lp.MIXES_SDR["hdrmix944"] = [("hdr944", 1.0, "human_score")]


class GramArgs:
    force = False
    only = "hdr944"


class TwinArgs:
    mix = "hdrmix944"
    out = str(OUT_DIR / "Q_lin944_hdr.bin")
    tau = 0.0
    loo = None


def main() -> int:
    (OUT_DIR / "linear-probe").mkdir(parents=True, exist_ok=True)
    print("[hdrp1] gram over the hdr944 leg ...", flush=True)
    lp.cmd_gram(GramArgs())
    print("[hdrp1] BVLS twin (shaped, v1 sign mask, tau 0) ...", flush=True)
    lp.cmd_twin(TwinArgs())
    print("[hdrp1] done:", TwinArgs.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
