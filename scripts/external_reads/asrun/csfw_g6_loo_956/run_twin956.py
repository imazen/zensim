#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/csfw-g6-loo-2026-07-29/harness/run_twin956.py
# sha256(source): fd4809a551f7b9e226ef922066b6542763b5bdb1fce7c2b3c0430f5e633292f3
# build_commit:  7bfd511de78f85e8fcd618df15716ca56575bb60
# Protocol doc:  benchmarks/csfw_g6_loo_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""956 replication of the E2 deterministic linear BVLS twin + zero-ablation LOO
(e2_optimal_model_720_vs_372_2026-07-23.md `## Frontier` protocol; the
bandvis-loo-2026-07-28 run_twin944.py adapted to 956), driven through the
ORIGINAL tool `scripts/v_next/linear_projections_2026-07-03.py` loaded
verbatim from a `git archive` of origin/main
7bfd511de78f85e8fcd618df15716ca56575bb60 (verified byte-identical to the
bandvis wave's pinned b1d4bc25 copy). Nothing in the tool is edited; this
wrapper only registers the ext956 groups + the twinsdr956 mix (the exact
twinsdr recipe: safesyn 1.0 / cid201 1.5 / kadid 0.5 / tid 0.5 / konjnd 1.2)
and passes the 956 family map (11 v2 families + append2 + a2 slots verbatim
+ csfw block + 3 csfw lane families + 12 csfw singles).

E2-exact settings: shaped space, ZLIN_SCREEN=screen_720_merged_safe.tsv
(f720..f955 default to identity — no screen exists for the append/csfw
blocks; the same treatment RAW v2 got before its screen), BVLS with the
shipped v1 sign mask (pin) and everything past f371 sign-free, tau=0, no
ridge, no dial-mono.
"""
import os
import sys
from pathlib import Path

BT = "/home/lilith/tmp/g6loo/tree"
OUTROOT = Path("/mnt/v/output/zensim/csfw-g6-loo-2026-07-29")
os.environ["ZLIN_NFEAT"] = "956"
os.environ["ZLIN_SCRATCH"] = str(OUTROOT / "linear-probe-956")
os.environ["ZLIN_SCREEN"] = (
    f"{BT}/benchmarks/v2_transform_screen_2026-07-23/screen_720_merged_safe.tsv"
)

import importlib.util

spec = importlib.util.spec_from_file_location(
    "lp", f"{BT}/scripts/v_next/linear_projections_2026-07-03.py"
)
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)

D956 = Path("/mnt/v/zen/zensim-training/ext956-instrument-2026-07-29")
lp.GROUPS.update(
    {
        "t956_safesyn": (D956 / "ext_safesyn_full.parquet", ["human_score"]),
        "t956_cid201": (D956 / "ext_cid22_train201.parquet", ["human_score"]),
        "t956_kadid": (D956 / "ext_kadid.parquet", ["human_score"]),
        "t956_tid": (D956 / "ext_tid.parquet", ["human_score"]),
        "t956_konjnd": (D956 / "ext_konjnd_jpeg_val.parquet", ["human_score"]),
    }
)
lp.MIXES_SDR["twinsdr956"] = [
    ("t956_safesyn", 1.0, "human_score"),
    ("t956_cid201", 1.5, "human_score"),
    ("t956_kadid", 0.5, "human_score"),
    ("t956_tid", 0.5, "human_score"),
    ("t956_konjnd", 1.2, "human_score"),
]


class GramArgs:
    force = False
    only = "t956_safesyn,t956_cid201,t956_kadid,t956_tid,t956_konjnd"


class TwinArgs:
    mix = "twinsdr956"
    out = str(OUTROOT / "bakes" / "lin956.bin")
    tau = 0.0
    loo = "/home/lilith/tmp/g6loo/fam956.json"


def main() -> int:
    (OUTROOT / "bakes").mkdir(parents=True, exist_ok=True)
    print("[wrap] gram over ext956 groups ...", flush=True)
    lp.cmd_gram(GramArgs())
    print("[wrap] twin + LOO ...", flush=True)
    lp.cmd_twin(TwinArgs())
    print("[wrap] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
