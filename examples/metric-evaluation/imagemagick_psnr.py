#!/usr/bin/env python3
"""Adapt ImageMagick PSNR to panel evaluate's JSON protocol.

Uses the supplied executable's default pixel/color interpretation, with no
resizing or color conversion. See https://imagemagick.org/compare/ .
Pin this script and the ImageMagick executable in the metric's artifacts.
"""
import argparse
import json
import math
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--magick', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--distorted', required=True)
    args = parser.parse_args()
    dimensions = []
    for path in (args.reference, args.distorted):
        result = subprocess.run(
            [args.magick, 'identify', '-format', '%w %h', path],
            capture_output=True, text=True, check=True)
        dimensions.append(result.stdout)
    if dimensions[0] != dimensions[1]:
        raise ValueError('reference and distorted dimensions differ')
    result = subprocess.run(
        [args.magick, 'compare', '-limit', 'thread', '1', '-metric', 'PSNR',
         args.reference, args.distorted, 'null:'],
        capture_output=True, text=True, check=False)
    sys.stderr.write(result.stderr)
    if result.returncode not in (0, 1):
        raise RuntimeError(f'ImageMagick compare exited {result.returncode}')
    # compare prints its scalar on stderr; status 1 means images differ.
    score = float(result.stderr.strip().split()[0])
    if not math.isfinite(score):
        raise ValueError('PSNR is not finite (including exact image identity)')
    print(json.dumps({'score': score}))


if __name__ == '__main__':
    main()
