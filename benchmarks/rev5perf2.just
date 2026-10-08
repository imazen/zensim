# Tiny fixed-cost investigation; all heavy work uses the shared task envelope.
set working-directory := '..'

rev5perf2-profile binary dest mode="perf" calls="8192":
    python3 scripts/demos/rev5perf_profile.py --binary "{{binary}}" --dest "{{dest}}" --mode "{{mode}}" --only v4x-t1-64x64,v4x-t1-128x128,v4x-t2-64x64,v4-t1-64x64,v4-t1-128x128 --calls "{{calls}}"

rev5perf2-scalar-profile binary dest:
    python3 scripts/demos/rev5perf_profile.py --binary "{{binary}}" --dest "{{dest}}" --mode callgrind --only scalar-t1-64x64,scalar-t1-128x128 --calls 8

[positional-arguments]
rev5perf2-timing root analyzer:
    #!/usr/bin/env bash
    set -euo pipefail
    only=$(python3 -c 'import json; print(",".join(c["tag"] for c in json.load(open("benchmarks/rev5perf2_2026-10-08.cells.json"))["cells"]))')
    just speedq-timing --binary "$1/provenance/instrument" --dest "$1/timing" \
        --parity "$1/parity/PARITY_STRICT_PASS.json" --analyzer "$2" --only "$only" \
        --lock /home/lilith/tmp/zensim-paper/rev4/heavy.lock \
        --arms by_v2fy_r3 by_v2fy_r4 by_v2fy_r5 zensim_B fast_ssim2 butteraugli
