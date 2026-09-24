#!/usr/bin/env bash
set -euo pipefail
root=/var/tmp/gmsd-chroma
date -u +%FT%TZ
"$root/stats-env/bin/python" "$root/c8/scripts/numpy_reference.py" --chroma \
    "$root/c8/calibration/chroma_xyb8.tsv" "$root/c8/features8_v3.csv" \
    "$root/c8/calibration/chroma_report.json"
date -u +%FT%TZ
