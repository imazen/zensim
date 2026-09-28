#!/usr/bin/env python3
"""Independent NumPy C8 map/pool check from research XYB plane dumps.

Usage: numpy_reference.py <xyb8.tsv> <features.csv> <constants.json>
The first eight calibration pairs are extracted by the normal Rust owner in
the same order. A deliberately wrong c bank must fail the same comparator.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def one(ref, dist, constants):
    # The walk uses reflect-101 for true-image y borders and clamps x borders.
    # Central differences have no 1/2 divisor.
    def magnitude(a):
        left = np.concatenate((a[:, :1], a[:, :-1]), axis=1)
        right = np.concatenate((a[:, 1:], a[:, -1:]), axis=1)
        up = np.concatenate((a[1:2], a[:-1]), axis=0)
        down = np.concatenate((a[1:], a[-2:-1]), axis=0)
        dx = right.astype(np.float64) - left
        dy = down.astype(np.float64) - up
        mag = np.sqrt(dx * dx + dy * dy)
        # The production gradient pass evaluates complete interior V8
        # chunks in f32, then widens the magnitudes for C8's f64 pooling.
        # First/last columns and a trailing incomplete chunk use f64.
        chunk_end = 1 + ((a.shape[1] - 2) // 8) * 8
        if chunk_end > 1:
            dx32 = right[:, 1:chunk_end] - left[:, 1:chunk_end]
            dy32 = down[:, 1:chunk_end] - up[:, 1:chunk_end]
            mag[:, 1:chunk_end] = np.sqrt(dx32 * dx32 + dy32 * dy32)
        return mag

    mr = magnitude(ref)
    md = magnitude(dist)
    out = []
    for c in constants:
        delta = (mr - md) ** 2 / (mr * mr + md * md + c)
        out.extend((float(np.mean(np.where(md < mr, delta, 0))),
                    float(np.mean(np.where(md >= mr, delta, 0))),
                    float(np.std(delta, ddof=0))))
    return out


def main():
    if sys.argv[1] == 'prepare':
        root = Path('/var/tmp/gmsbank/calibration')
        with (root / 'pairs.tsv').open() as f:
            lines = [line.rstrip('\n').split('\t') for line in f][:8]
        assert len(lines) == 8 and all(len(row) == 5 for row in lines)
        with (root / 'first8_pairs.tsv').open('w') as f:
            f.write('ref_path\tdist_path\tpair_index\n')
            for i, row in enumerate(lines):
                f.write(f'{row[3]}\t{row[4]}\t{i}\n')
        print('prepared 8 TRAIN pairs')
        return
    chroma = sys.argv[1] == '--chroma'
    index, features, const_path = map(Path, sys.argv[2:] if chroma else sys.argv[1:])
    report = json.loads(const_path.read_text())
    constants = report['ratios'] if chroma else report['constants']
    root = index.parent
    planes = {}
    with index.open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            w, h = int(row['width']), int(row['height'])
            a = np.fromfile(root / row['file'], dtype='<f4')
            assert a.size == w * h
            planes[(row['pair_key'], int(row['side']), int(row['scale']), int(row['channel']))] = a.reshape(h, w)
    with features.open() as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(float(r['pair_index'])))
    assert len(rows) >= 8
    # This command is run on exactly the eight selected pairs, in the TSV
    # order. The source table carries no label and no metric target.
    keys = list(dict.fromkeys(k[0] for k in planes))
    assert len(keys) == 8 and len(rows) == 8
    assert [int(float(r['pair_index'])) for r in rows] == list(range(8))
    errors = []
    wrong_fails = 0
    for ri, key in enumerate(keys):
        values = []
        wrong = []
        for scale in range(4):
            for ch in range(3):
                if chroma and scale == 0 and ch != 1:
                    continue
                ref = planes[(key, 0, scale, ch)]
                dist = planes[(key, 1, scale, ch)]
                bank = constants if not chroma else (
                    # Frozen Y literals, byte-for-byte from its original calibration.
                    [0.00013321662567409982, 0.0005328665026963993,
                     0.002131466010785597, 0.008525864043142388,
                     0.03410345617256955]
                    if ch == 1 else constants['x_gradient' if ch == 0 else 'b_gradient']['constants'])
                values.extend(one(ref, dist, bank))
                wrong.extend(one(ref, dist, [c * 16 for c in bank]))
            if chroma and scale > 0:
                xr, xd = [planes[(key, side, scale, 0)].astype(np.float64) - float(np.float32(.42)) for side in (0, 1)]
                br, bd = [planes[(key, side, scale, 2)].astype(np.float64) - float(np.float32(.55)) for side in (0, 1)]
                for cx, cb in zip(constants['x_value']['constants'], constants['b_value']['constants']):
                    for target, multiplier in [(values, 1), (wrong, 16)]:
                        loss = ((xr-xd)**2/(cx*multiplier) + (br-bd)**2/(cb*multiplier)) / (
                            (xr*xr+xd*xd)/(cx*multiplier) + (br*br+bd*bd)/(cb*multiplier) + 1)
                        target.extend((float(np.mean(loss)), float(np.std(loss, ddof=0))))
        for local, want in enumerate(values):
            got = float(rows[ri][f'f{1322 + local}'])
            rel = abs(got - want) / max(abs(want), 1e-12)
            errors.append((rel, ri, local, got, want))
            bad_rel = abs(got - wrong[local]) / max(abs(wrong[local]), 1e-12)
            wrong_fails += bad_rel > 1e-6
    worst = max(errors)
    assert worst[0] <= 1e-6, f'independent reference mismatch: {worst}'
    assert wrong_fails > 0, 'negative control failed to reject wrong c'
    print(json.dumps({'pairs': 8, 'cells': len(errors), 'max_relative_error': worst[0],
                      'worst_pair': worst[1], 'worst_local_slot': worst[2],
                      'wrong_c_rejections': wrong_fails}, sort_keys=True))


if __name__ == '__main__':
    main()
