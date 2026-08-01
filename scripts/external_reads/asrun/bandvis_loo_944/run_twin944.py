#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/bandvis-loo-2026-07-28/harness/run_twin944.py
# sha256(source): d3325a9399b48ac2624f3e3198ba5d14d81db998a219d05ad55fa151241c6604
# build_commit:  b1d4bc257e57f7c3215ec8a237e9f87cdad8e35f
# Protocol doc:  benchmarks/bandvis_loo_944_2026-07-28.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""944 replication of the E2 deterministic linear BVLS twin + zero-ablation LOO
(e2_optimal_model_720_vs_372_2026-07-23.md `## Frontier` protocol), driven
through the ORIGINAL tool `scripts/v_next/linear_projections_2026-07-03.py`
loaded verbatim from the pinned origin/main archive
(b1d4bc257e57f7c3215ec8a237e9f87cdad8e35f). Nothing in the tool is edited;
this wrapper only registers the ext944 groups + the twinsdr944 mix (the exact
twinsdr recipe: safesyn 1.0 / cid201 1.5 / kadid 0.5 / tid 0.5 / konjnd 1.2)
and passes the 944 family map (11 v2 families verbatim + append2 + per-slot +
per-feature singles).

E2-exact settings: shaped space, ZLIN_SCREEN=screen_720_merged_safe.tsv
(f720..f943 default to identity — no screen exists for the append blocks; the
same treatment RAW v2 got before its screen), BVLS with the shipped v1 sign
mask (pin) and everything past f371 sign-free, tau=0, no ridge, no dial-mono.
"""
import os
import sys
from pathlib import Path

BT = "/home/lilith/tmp/loo944/build-tree"
OUTROOT = Path("/mnt/v/output/zensim/bandvis-loo-2026-07-28")
os.environ["ZLIN_NFEAT"] = "944"
os.environ["ZLIN_SCRATCH"] = str(OUTROOT / "linear-probe-944")
os.environ["ZLIN_SCREEN"] = (
    f"{BT}/benchmarks/v2_transform_screen_2026-07-23/screen_720_merged_safe.tsv"
)

import importlib.util

spec = importlib.util.spec_from_file_location(
    "lp", f"{BT}/scripts/v_next/linear_projections_2026-07-03.py"
)
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)

D944 = Path("/mnt/v/zen/zensim-training/ext944-instrument-2026-07-28")
lp.GROUPS.update(
    {
        "t944_safesyn": (D944 / "ext_safesyn_full.parquet", ["human_score"]),
        "t944_cid201": (D944 / "ext_cid22_train201.parquet", ["human_score"]),
        "t944_kadid": (D944 / "ext_kadid.parquet", ["human_score"]),
        "t944_tid": (D944 / "ext_tid.parquet", ["human_score"]),
        "t944_konjnd": (D944 / "ext_konjnd_jpeg_val.parquet", ["human_score"]),
    }
)
lp.MIXES_SDR["twinsdr944"] = [
    ("t944_safesyn", 1.0, "human_score"),
    ("t944_cid201", 1.5, "human_score"),
    ("t944_kadid", 0.5, "human_score"),
    ("t944_tid", 0.5, "human_score"),
    ("t944_konjnd", 1.2, "human_score"),
]


class GramArgs:
    force = False
    only = "t944_safesyn,t944_cid201,t944_kadid,t944_tid,t944_konjnd"


class TwinArgs:
    mix = "twinsdr944"
    out = str(OUTROOT / "bakes" / "lin944.bin")
    tau = 0.0
    loo = "/home/lilith/tmp/loo944/fam944.json"


def main() -> int:
    (OUTROOT / "bakes").mkdir(parents=True, exist_ok=True)
    print("[wrap] gram over ext944 groups ...", flush=True)
    lp.cmd_gram(GramArgs())
    print("[wrap] twin + LOO ...", flush=True)
    lp.cmd_twin(TwinArgs())
    print("[wrap] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
