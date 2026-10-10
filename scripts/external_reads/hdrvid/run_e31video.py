#!/usr/bin/env python3
"""Run `v40_panels.py --mode e31video` inside the frozen V40 packet runtime.

The packet's own launcher (`<packet>/panels.py`) executes its frozen copy of
`v40_panels.py`, which predates the external-video mode. This launcher keeps
everything else frozen: the packet's `assessment-runtime/` modules (`complete`,
`dense_bake`, the canonical panel statistics, row-key identity), its `bin/`
tools (`panel`, `predict_features_with_bake`, `bake_dial_refit`) and the
control/arm cell verification. Only the report module is this repository's
`scripts/rev4_featpot/v40_panels.py`, whose SHA-256 the exposure freeze pins
(`assessment_source_sha256`).

  run_e31video.py --packet /mnt/v/output/zensim/v40r4-2026-10-08 -- <v40_panels args>
"""

import os
import runpy
import sys
from pathlib import Path


def main():
    argv = sys.argv[1:]
    if len(argv) < 3 or argv[0] != "--packet" or argv[2] != "--":
        raise SystemExit(__doc__)
    packet = Path(argv[1]).resolve()
    owner = Path(__file__).resolve().parents[2] / "rev4_featpot" / "v40_panels.py"
    os.environ.setdefault("REV4_V2_BIN_DIR", str(packet / "bin"))
    os.environ.setdefault("ZEN_PANEL_BIN", str(packet / "bin/panel"))
    os.environ.setdefault("TMPDIR", str(Path.home() / "tmp/v40"))
    runtime = packet / "assessment-runtime"
    sys.path[:0] = [str(runtime), str(runtime / "scripts"), str(runtime / "scripts/rev4_featpot")]
    sys.argv = [str(owner), *argv[3:]]
    runpy.run_path(str(owner), run_name="__main__")


if __name__ == "__main__":
    main()
