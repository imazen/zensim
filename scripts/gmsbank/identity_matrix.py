#!/usr/bin/env python3
"""Run the frozen 144-pair × 6-mode C8 prefix identity matrix.

Usage: identity_matrix.py <candidate-extractor> <corrected-base-extractor>
Both binaries decode the same path-only TRAIN TSVs. Every arm writes a CSV,
command log and SHA256 record under /var/tmp/gmsbank/identity/; f0..f1321
are compared as round-tripped IEEE f64 bits. No label input is opened.
"""
import csv
import datetime as dt
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path('/var/tmp/gmsbank/identity')
INPUT = Path('/var/tmp/featbank-impl/evidence/gate')
SETS = (('cid22', 'cid22-64.tsv', 64),
        ('safesyn', 'safesyn-64.tsv', 64),
        ('kadid', 'kadid-train16-pairs.tsv', 16))
WIDTH = 1322


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(binary, set_name, input_tsv, tier, threads, arm):
    stem = f'{set_name}_{tier}_mt{threads}_{arm}'
    csv_path = ROOT / f'{stem}.csv'
    log_path = ROOT / f'{stem}.log'
    command = [str(binary), '--corpus', 'pairs-tsv', '--path', str(input_tsv),
               '--out', str(csv_path), '--input-contract', 'legacy-rgb8',
               '--force-tier', tier,
               '--full-gmsbank' if arm == 'on' else '--full-rev4']
    env = {**os.environ, 'RAYON_NUM_THREADS': str(threads),
           'ZENSIM_FORMULA_REV': '3', 'ZENSIM_ROOT_FORM': 'sqrt'}
    start = now()
    with log_path.open('wb') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env, check=False)
    end = now()
    record = dict(start_utc=start, end_utc=end, cwd=os.getcwd(), command=command,
                  rayon_threads=threads, formula_revision='3', root_form='sqrt',
                  exit_code=result.returncode,
                  log=str(log_path), log_sha256=sha(log_path),
                  csv=str(csv_path), csv_sha256=sha(csv_path) if csv_path.is_file() else None)
    (ROOT / f'{stem}.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    if result.returncode:
        raise RuntimeError(f'{stem} failed: {record}')
    return csv_path


def load(path, expected_rows):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != expected_rows:
        raise ValueError(f'{path}: {len(rows)} != {expected_rows} rows')
    for row in rows:
        if any(f'f{i}' not in row for i in range(WIDTH)):
            raise ValueError(f'{path}: missing prefix column')
    return rows


def compare(a, b):
    differing = 0
    first = []
    for ri, (left, right) in enumerate(zip(a, b)):
        if left['ref_basename'] != right['ref_basename']:
            raise ValueError(f'row order changed at {ri}')
        for i in range(WIDTH):
            name = f'f{i}'
            if struct.pack('<d', float(left[name])) != struct.pack('<d', float(right[name])):
                differing += 1
                if len(first) < 8:
                    first.append((ri, i, left[name], right[name]))
    return differing, first


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == 'compare-existing':
        options = sys.argv[4:]
        assert all(s == '--stdout' or s.startswith('--base-attempt=') for s in options)
        stdout_only = '--stdout' in options
        root, attempt = Path(sys.argv[2]), sys.argv[3]
        base_attempt = next((s.split('=',1)[1] for s in options if s.startswith('--base-attempt=')), attempt)
        result = dict(schema='gmsbank-prefix-identity-v2', modes=[],
                      compared_cells=0, differing_cells=0)
        if base_attempt != attempt:
            baseline = root.parent/'binaries'/f'matrix_{base_attempt}_base'
            candidate = root.parent/'binaries'/f'matrix_{attempt}_on'
            result.update(baseline_binary_sha256=sha(baseline), candidate_binary_sha256=sha(candidate))
            assert result['baseline_binary_sha256'] != result['candidate_binary_sha256'], 'artifact collision'
        for tier in ('native', 'v3', 'scalar'):
            for threads in (1, 8):
                dirs = {arm: root/f'{attempt}_{arm}_{tier}_mt{threads}' for arm in ('base','off','on')}
                dirs['base'] = root/f'{base_attempt}_base_{tier}_mt{threads}'
                pixels = {arm: json.loads((path/'pixels.json').read_text()) for arm,path in dirs.items()}
                assert pixels['base'] == pixels['off'] == pixels['on'], 'decoded input changed'
                assert len(pixels['base']) == 144
                for set_name, _, n in SETS:
                    paths = {arm: path/f'{set_name}.csv' for arm,path in dirs.items()}
                    rows = {arm: load(path,n) for arm,path in paths.items()}
                    for left,right in [('base','off'),('off','on'),('base','on')]:
                        diff,first = compare(rows[left],rows[right])
                        result['compared_cells'] += n*WIDTH
                        result['differing_cells'] += diff
                        result['modes'].append(dict(set=set_name,tier=tier,threads=threads,
                            left=left,right=right,rows=n,cells=n*WIDTH,differing=diff,first=first,
                            left_sha256=sha(paths[left]),right_sha256=sha(paths[right])))
        if not stdout_only:
            with (root/f'report_{attempt}.json').open('x') as f:
                f.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
        print(json.dumps({k:result[k] for k in ('compared_cells','differing_cells')},sort_keys=True))
        assert result['differing_cells'] == 0
        return
    if len(sys.argv) != 3:
        raise SystemExit('identity_matrix.py <candidate-extractor> <corrected-base-extractor>')
    candidate, base = map(Path, sys.argv[1:])
    if not candidate.is_file() or not base.is_file():
        raise FileNotFoundError('candidate and corrected-base extractor binaries required')
    ROOT.mkdir(parents=True, exist_ok=True)
    result = dict(schema='gmsbank-prefix-identity-v1', candidate_binary_sha256=sha(candidate),
                  corrected_base_binary_sha256=sha(base), modes=[],
                  compared_cells=0, differing_cells=0)
    for set_name, name, n in SETS:
        source = INPUT / name
        header = source.open().readline().strip()
        if header != 'ref_path\tdist_path':
            raise ValueError(f'{source}: path-only input required')
        for tier in ('native', 'v3', 'scalar'):
            for threads in (1, 8):
                paths = {arm: run(base if arm == 'base' else candidate,
                                  set_name, source, tier, threads, arm)
                         for arm in ('base', 'off', 'on')}
                rows = {arm: load(path, n) for arm, path in paths.items()}
                for pair in (('base', 'off'), ('off', 'on'), ('base', 'on')):
                    diff, first = compare(rows[pair[0]], rows[pair[1]])
                    result['compared_cells'] += n * WIDTH
                    result['differing_cells'] += diff
                    result['modes'].append(dict(set=set_name, tier=tier, threads=threads,
                                                left=pair[0], right=pair[1], rows=n,
                                                cells=n * WIDTH, differing=diff, first=first))
                print(f'identity {set_name} {tier} mt{threads} complete', flush=True)
    (ROOT / 'report.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: result[k] for k in ('compared_cells', 'differing_cells')}, sort_keys=True))
    if result['differing_cells']:
        raise AssertionError('C8 or corrected base moved an f0..f1321 prefix cell')


if __name__ == '__main__':
    main()
