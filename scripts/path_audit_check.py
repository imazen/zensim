#!/usr/bin/env python3
"""Exact-output and timing-packet contract for zenbench pair."""
import re
import sys
from pathlib import Path

from path_audit import METRIC_PATTERN, benchmark_median, packet


def checked(path):
    text = Path(path).read_text()
    result = packet(text)
    median = benchmark_median(result, path)
    values = re.findall(METRIC_PATTERN, text)
    if len(values) != 1 or float(values[0]) != median:
        raise RuntimeError(f"metric extraction disagrees with JSON: {path}")
    return result["fingerprint"]


if __name__ == "__main__":
    if checked(sys.argv[1]) != checked(sys.argv[2]):
        raise RuntimeError("output fingerprints differ")
