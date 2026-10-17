#!/usr/bin/env python3
"""Bound one cooperative heavy-lock window; never launch or signal workers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def completed(root, models, geometries):
    cells = set()
    for model in models:
        for geometry in geometries:
            path = root / 'results' / model / 'timing' / f'v4x-t1-{geometry}' / 'COMPLETE.json'
            if not path.exists():
                continue
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                # The owner publishes this file directly; observe it once complete.
                continue
            if record['status'] == 'PASS':
                cells.add((model, geometry))
    return cells


def hold(root, seconds, log, clock=time.monotonic, sleep=time.sleep):
    job_path = root / 'JOB.json'
    job = json.loads(job_path.read_text())
    expected = len(job['models']) * len(job['geometries'])
    initial = completed(root, job['models'], job['geometries'])
    start = clock()
    next_progress = start
    while True:
        current = completed(root, job['models'], job['geometries'])
        elapsed = clock() - start
        reason = ('complete' if len(current) == expected else
                  'one_cell_completed' if current - initial else
                  'window_expired' if elapsed >= seconds else None)
        if reason or clock() >= next_progress:
            record = dict(utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=elapsed,
                          initial_cells=sorted(initial), completed_cells=sorted(current),
                          expected_cells=expected, reason=reason or 'waiting',
                          job_sha256=hashlib.sha256(job_path.read_bytes()).hexdigest())
            log.write(json.dumps(record) + '\n')
            log.flush()
            print('cooperative lock window', record, flush=True)
            next_progress = clock() + 15
        if reason:
            return reason
        sleep(min(1, seconds - elapsed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    assert 0 < args.seconds <= 300, 'bounded administrative window, not a timing gate'
    with args.log.open('x') as log:
        hold(args.root, args.seconds, log)


if __name__ == '__main__':
    main()
