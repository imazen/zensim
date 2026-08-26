#!/usr/bin/env python3
"""hdrgrid372 linear control — the deterministic BVLS baseline on the NEW leg.

Mirror of `hdrp1_linear_control.py` (the pinned appendix-Q instrument driver),
with the ONE registered difference: the mix is the hdrgrid372 leg alone
(`hdrgrid372_v3mix_traindigits_2026-08-26.parquet`, human_score = the Appendix-Q
cvvdp-mix target, weight 1.0) at the leg's native width — ZLIN_NFEAT=372, no
ZLIN_SCREEN (the instrument's 372-screen + identity path; v1 sign mask).
Deterministic: no seed, no SGD. Output: hdrgrid372_lin372.bin.

This is a BASELINE measurement for the new-lineage leg (MODELS-criterion first
number), not a ship candidate.

  usage: hdrgrid372_linear_control.py
"""
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("HDRGRID372_OUT", "/mnt/v/output/zensim/bakes/hdrgrid372-probe"))

os.environ["ZLIN_NFEAT"] = "372"
os.environ["ZLIN_SCRATCH"] = str(OUT_DIR / "linear-probe")
os.environ.pop("ZLIN_SCREEN", None)  # 372-native: instrument's own 372 screen + identity

spec = importlib.util.spec_from_file_location(
    "lp", REPO / "scripts/v_next/linear_projections_2026-07-03.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)

LEG = Path("/mnt/v/output/zensim/hdrgrid372-leg")
lp.GROUPS.update({
    "hdrgrid372": (LEG / "hdrgrid372_v3mix_traindigits_2026-08-26.parquet",
                   ["human_score"]),
})
lp.MIXES_SDR["hdrgrid372mix"] = [("hdrgrid372", 1.0, "human_score")]


class GramArgs:
    force = False
    only = "hdrgrid372"


class TwinArgs:
    mix = "hdrgrid372mix"
    out = str(OUT_DIR / "hdrgrid372_lin372.bin")
    tau = 0.0
    loo = None


def main() -> int:
    (OUT_DIR / "linear-probe").mkdir(parents=True, exist_ok=True)
    print("[hdrgrid372] gram over the hdrgrid372 leg ...", flush=True)
    lp.cmd_gram(GramArgs())
    print("[hdrgrid372] BVLS twin (372 screen, v1 sign mask, tau 0) ...", flush=True)
    lp.cmd_twin(TwinArgs())
    print("[hdrgrid372] done:", TwinArgs.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
