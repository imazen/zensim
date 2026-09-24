#!/usr/bin/env python3
"""TRAIN pixels-only C1/C3 registry calibration; reads feature columns only."""

import argparse
import csv
import math
from pathlib import Path
import struct

GATE = Path('/var/tmp/featbank-impl/evidence/gate')
CORPORA = {
    'cid22': GATE / 'cid22-64-t1-rev4.csv',
    'safesyn': GATE / 'safesyn-64-t1-rev4.csv',
    'kadid': GATE / 'kadid-train16-pairs-t1-rev4.csv',
}


def edge(top):
    return 10 ** (-6 + (31 / 32) * (6 + math.log10(top)))


def old_census():
    old_top = edge(1)
    print(f'old_top={old_top:.17g} candidate_2_top={edge(2):.17g}')
    for corpus, path in CORPORA.items():
        if corpus == 'kadid':
            continue  # C3 range is selected on SafeSyn/CID22 TRAIN only.
        saturation = [0] * 4
        upper_bound = [0] * 4
        cells = rows = 0
        for row in csv.DictReader(path.open()):
            rows += 1
            for cell in range(12):
                cells += 1
                for m in range(4):
                    base = 1154 + cell * 12 + m * 3
                    saturation[m] += float(row[f'f{base + 1}']) >= old_top
                    # p99 <= max, so max >= candidate top is a rigorous
                    # upper bound on p99 saturation at that top.
                    upper_bound[m] += float(row[f'f{base + 2}']) >= edge(2)
        print(f'C3 {corpus} rows={rows} cells={cells} '
              f'old_p99_saturated={saturation} '
              f'candidate_2_saturation_upper_bound={upper_bound}')
    counts = [0] * 6
    cells = 0
    for path in CORPORA.values():
        for row in csv.DictReader(path.open()):
            for cell in range(12):
                cells += 1
                for hat in range(6):
                    counts[hat] += float(row[f'f{986 + cell * 8 + hat}']) != 0
    print(f'C1 old_emitted_nonzero_cells={counts} total_cells={cells}')


def diag_census(paths):
    counts = [0] * 64
    lines = 0
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.startswith('REV4DIAGABS '):
                continue
            hist = [int(x) for x in line.rsplit(' ', 1)[1].split(',')]
            assert len(hist) == 64
            counts = [a + b for a, b in zip(counts, hist)]
            lines += 1
    total = sum(counts)
    print(f'C1 diag_lines={lines} positive_on_grid_samples={total}')
    for q in (0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99, 1.0):
        target = math.ceil(q * total)
        running = 0
        for bin_id, n in enumerate(counts):
            running += n
            if running >= target:
                lo = 10 ** (bin_id / 8 - 5)
                hi = 10 ** ((bin_id + 1) / 8 - 5)
                print(f'C1 quantile q={q:.2f} bin={bin_id} '
                      f'interval=[{lo:.6g},{hi:.6g}) count={n}')
                break
    print('C1 occupied_bins=' + ','.join(f'{i}:{n}' for i, n in enumerate(counts) if n))


def post_census(directory):
    top = struct.unpack('>d', bytes.fromhex('3ff455be4ebe49c1'))[0]
    all_nonzero = [0] * 6
    all_cells = 0
    for corpus, expected_rows in [('cid22', 64), ('safesyn', 64), ('kadid', 16)]:
        path = directory / f'{corpus}-t1-rev4.csv'
        top_count = [0] * 4
        nonzero = [0] * 6
        cells = rows = 0
        for row in csv.DictReader(path.open()):
            rows += 1
            for cell in range(12):
                cells += 1
                for hat in range(6):
                    hit = float(row[f'f{986 + cell * 8 + hat}']) != 0
                    nonzero[hat] += hit
                    all_nonzero[hat] += hit
                for m in range(4):
                    p99 = float(row[f'f{1154 + cell * 12 + m * 3 + 1}'])
                    top_count[m] += p99 >= top
        assert rows == expected_rows, f'{corpus} rows={rows}'
        if corpus != 'kadid':
            assert all(100 * n <= cells for n in top_count), (
                f'{corpus}: C3 p99 top-bin count exceeds 1%: {top_count}/{cells}'
            )
        all_cells += cells
        print(f'POST {corpus} rows={rows} cells={cells} '
              f'C1_nonzero={nonzero} C3_p99_saturated={top_count}')
    assert all(x > 0 for x in all_nonzero)
    print(f'POST C1_all_nonzero={all_nonzero} total_cells={all_cells}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--diag', type=Path, nargs='*', default=[])
    parser.add_argument('--post-dir', type=Path)
    args = parser.parse_args()
    old_census()
    if args.diag:
        diag_census(args.diag)
    if args.post_dir:
        post_census(args.post_dir)
