#!/usr/bin/env python3
"""hdrgrid944 linear control — the deterministic BVLS baseline on the NEW leg.

Mirror of `hdrp1_linear_control.py` (the pinned appendix-Q instrument driver),
with the ONE registered difference: the mix is the hdrgrid944 leg alone
(`hdrgrid944_v3mix_traindigits_2026-08-26.parquet`, human_score = the Appendix-Q
cvvdp-mix target, weight 1.0) at the leg's native width — ZLIN_NFEAT=944, no
ZLIN_SCREEN (the instrument's 944-width identity path (no 372 screen)).
Deterministic: no seed, no SGD. Output: hdrgrid944_lin944.bin.

This is a BASELINE measurement for the new-lineage leg (MODELS-criterion first
number), not a ship candidate.

  usage: hdrgrid944_linear_control.py
"""
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("HDRGRID944_OUT", "/mnt/v/output/zensim/bakes/hdrgrid944-probe"))

os.environ["ZLIN_NFEAT"] = "944"
os.environ["ZLIN_SCRATCH"] = str(OUT_DIR / "linear-probe")
os.environ.pop("ZLIN_SCREEN", None)  # 372-native: instrument's own 372 screen + identity

spec = importlib.util.spec_from_file_location(
    "lp", REPO / "scripts/v_next/linear_projections_2026-07-03.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)

LEG = Path("/mnt/v/output/zensim/hdrgrid944-leg")
lp.GROUPS.update({
    "hdrgrid944": (LEG / "hdrgrid944_v3mix_traindigits_2026-08-26.parquet",
                   ["human_score"]),
})
lp.MIXES_SDR["hdrgrid944mix"] = [("hdrgrid944", 1.0, "human_score")]


class GramArgs:
    force = False
    only = "hdrgrid944"


class TwinArgs:
    mix = "hdrgrid944mix"
    out = str(OUT_DIR / "hdrgrid944_lin944.bin")
    tau = 0.0
    loo = None


def main() -> int:
    (OUT_DIR / "linear-probe").mkdir(parents=True, exist_ok=True)
    print("[hdrgrid944] gram over the hdrgrid944 leg ...", flush=True)
    lp.cmd_gram(GramArgs())
    print("[hdrgrid944] BVLS twin (944 identity, tau 0) ...", flush=True)
    lp.cmd_twin(TwinArgs())
    print("[hdrgrid944] done:", TwinArgs.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
