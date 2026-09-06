#!/usr/bin/env python3
"""R6 / R6b: turn one arm's extracted CSVs into (a) a drop-in `bake_verdict`
eval root and (b) the parquets the fit chain consumes.

WHICH lane is a parameter, so R6b (F17) reuses this rather than copying it:
`R6_ARM_SELECTOR_ENV` / `R6_ARM_SELECTOR_DESC` / `R6_LANE` / `R6_ARM_KEY` name
the knob and the lane in the emitted manifest, and `ZEN_BUILD_COMMIT` names the
commit the EXTRACTOR was built at (defaulting to R6's).

The eval root is built by `canonical_corpus/pack_eval372_root.py` — THE owner of
that job — in its `EVAL372_NO_STORED` mode, so no rev1 table is copied into an
arm's root. A copy would be a different arithmetic revision sitting inside a
directory that claims to be one arm; `bake_verdict` reporting a corpus as
missing is the honest alternative and costs nothing (none of the copied corpora
are in R6's decision set).

`OLD` is pointed at the registered postC 372 root on purpose: that makes the
packer's own positional row-alignment check and its per-slot drift table run
arm-vs-rev1 for free, and for the `ssim2` arm the drift must come out ZERO.

The three extra legs (safesyn / anchor / identity) are packed through the same
`read_fresh_csv` the eval root uses, so every table in the wave has one reader.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "canonical_corpus"))
import pyarrow.parquet as pq  # noqa: E402
from pack_eval372_root import read_fresh_csv, sha256  # noqa: E402

POSTC = "/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC"
EXTRA_LEGS = ("safesyn", "anchor", "identity")
# main@origin at the moment `zensim-validate` and
# `extract_features_372col` were built for this wave.
EXTRACTOR_BUILD_COMMIT = os.environ.get(
    "ZEN_BUILD_COMMIT", "ceb86c2dad6b856c1999392731635f14eba921f9"
)
ARM_KEY = os.environ.get("R6_ARM_KEY", "r6_f4_arm")
ARM_ENV = os.environ.get("R6_ARM_SELECTOR_ENV", "ZENSIM_SSIM_LUMA")
ARM_DESC = os.environ.get(
    "R6_ARM_SELECTOR_DESC",
    "ssim_form::SsimLumaForm; ssim2 == the shipped revision-1 form",
)
LANE = os.environ.get(
    "R6_LANE", "R6 F4-arm decision, docs/PLAN_FEATURE_REV2_2026-09-05.md §7"
)


def main() -> int:
    arm = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else "/mnt/v/output/zensim/rev2-2026-09-05/r6"
    tables = os.path.join(root, "tables", arm)
    evalroot = os.path.join(root, "evalroot", arm)

    for leg in EXTRA_LEGS:
        src = os.path.join(tables, f"{leg}.csv")
        if not os.path.exists(src):
            print(f"{arm}/{leg}: MISSING csv, skipped")
            continue
        dst = os.path.join(tables, f"{leg}.parquet")
        t = read_fresh_csv(src)
        pq.write_table(t, dst, compression="zstd")
        print(f"{arm}/{leg:9s} rows={t.num_rows:7d} sha256 {sha256(dst)[:16]}…")

    env = dict(os.environ)
    # The commit the EXTRACTOR BINARY was built at, not the packer's working
    # copy: one binary produced every arm (that is the whole control), and the
    # tree moves under it while the wave runs. pack_eval372_root.py honours
    # ZEN_BUILD_COMMIT ahead of git/jj for exactly this case.
    env.setdefault("ZEN_BUILD_COMMIT", EXTRACTOR_BUILD_COMMIT)
    env["EVAL372_NO_STORED"] = "1"
    env["EVAL372_SKIP"] = "pipal"
    env["EVAL372_MANIFEST_EXTRA"] = (
        '{"' + ARM_KEY + '": "%s",'
        ' "r6_arm_selector": "' + ARM_ENV + '=%s (' + ARM_DESC + ')",'
        ' "decoder_era": "zensim-bench/examples/shared/zen_decode.rs at this build_commit —'
        ' zencodec magic-byte detect then zenjpeg / zenpng / zenwebp / zenavif / zenjxl;'
        ' safesyn decodes .jpg 111068 / .avif 34001 / .jxl 26362 / .webp 24655 from'
        ' BITSTREAMS (the q<X>.png decode cache was deleted 2026-06-22)",'
        ' "lane": "' + LANE + '"}'
    ) % (arm, arm)
    r = subprocess.run(
        [sys.executable,
         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "canonical_corpus", "pack_eval372_root.py"),
         evalroot, tables, POSTC],
        env=env)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
